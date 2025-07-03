# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from collections.abc import Mapping
from operator import itemgetter

import torch
from einops import rearrange
from omegaconf import DictConfig
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import HeteroData

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.train.forecaster import GraphForecaster

LOGGER = logging.getLogger(__name__)


class GraphInterpolator(GraphForecaster):
    """Graph neural network interpolator for PyTorch Lightning."""

    def __init__(
        self,
        *,
        config: DictConfig,
        graph_data: HeteroData,
        statistics: dict,
        data_indices: IndexCollection,
        metadata: dict,
        supporting_arrays: dict,
        relative_date_indices: dict,
    ) -> None:
        """Initialize graph neural network interpolator.

        Parameters
        ----------
        config : DictConfig
            Job configuration
        graph_data : HeteroData
            Graph object
        statistics : dict
            Statistics of the training data
        data_indices : IndexCollection
            Indices of the training data,
        metadata : dict
            Provenance information

        """
        super().__init__(
            config=config,
            graph_data=graph_data,
            statistics=statistics,
            data_indices=data_indices,
            metadata=metadata,
            supporting_arrays=supporting_arrays,
            relative_date_indices=relative_date_indices,
        )
        self.known_future_variables = itemgetter(*config.training.known_future_variables)(
            data_indices.data.input.name_to_index,
        ) if len(config.training.known_future_variables) else []
        if isinstance(self.known_future_variables, int):
            self.known_future_variables = [self.known_future_variables]
        self.multi_step = getattr(self.config["training"], "multistep_input", 1)
        boundary_times = config.training.explicit_times.input
        self.boundary_times = [t + self.multi_step - 1 for t in boundary_times]
        interp_times = config.training.explicit_times.target
        

    def _step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
        validation_mode: bool = False,
    ) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:

        del batch_idx
        loss = torch.zeros(1, dtype=batch.dtype, device=self.device, requires_grad=False)
        metrics = {}
        y_preds = []
        interp_times = list(range(self.multi_step, self.boundary_times[-1]))
        probs = torch.exp(-0.07 * torch.tensor(interp_times))
        probs = probs / probs.sum()
        rd_interp_indices = torch.distributions.Categorical(probs).sample([6])
        rd_interp_times = sorted(torch.tensor(interp_times)[rd_interp_indices].tolist())
        sorted_indices = sorted(
            set(range(self.multi_step)).union(
                self.boundary_times,
                rd_interp_times,
            ),
        )
        imap = {data_index: batch_index for batch_index, data_index in enumerate(sorted_indices)}

        batch = self.model.pre_processors(batch)
        present, future = itemgetter(*self.boundary_times)(imap)
        obs = set([var.item() for var in self.data_indices.data.input.full]).difference(
            set(self.known_future_variables)
        )
        x_init = batch[:, : self.multi_step][..., list(obs)]
        x_init_nwp = batch[:, 1][..., self.known_future_variables]
        x_init = rearrange(x_init, "batch time ens grid var -> batch ens grid (var time)")
        x_future = batch[:, future][..., self.known_future_variables]  # adding future known vars to the input
        x_bound = torch.cat([x_init, x_init_nwp, x_future], dim=-1)
        target_forcing = torch.empty(
            batch.shape[0],
            batch.shape[2],
            batch.shape[3],
            len(self.known_future_variables) + 1,
            device=self.device,
            dtype=batch.dtype,
        )
        if hasattr(self.loss, "losses"):
            time_weights = self.loss.losses[0].loss.time_weights
            if time_weights is not None:
                time_weights = time_weights.detach().clone()
            original_weights = {specific_loss: specific_loss.loss.time_weights for specific_loss in self.loss.losses}
        else:
            time_weights = self.loss.loss.time_weights
            if time_weights is not None:
                time_weights = time_weights.detach().clone()
            original_weights = {self.loss: self.loss.loss.time_weights}
        for interp_step in rd_interp_times:
            if time_weights is not None:
                updated_time_weights = time_weights[imap[interp_step]].detach().clone()
            else:
                updated_time_weights = None
            # update time weights in loss function for this specific case
            for specific_loss in original_weights:
                specific_loss.loss.time_weights = updated_time_weights
            # get the forcing information for the target interpolation time:
            target_forcing[..., : len(self.known_future_variables)] = batch[
                :, imap[interp_step], :, :, self.known_future_variables
            ]
            target_forcing[..., -1] = (interp_step - future) / (future - present)
            x_with_intermediate_forcings = torch.cat([x_bound, target_forcing], dim=-1).unsqueeze(dim=1)
            y_pred = self(x_with_intermediate_forcings)
            y = batch[:, imap[interp_step], ...]
            loss += checkpoint(self.loss, y_pred, y, use_reentrant=False)

            metrics_next = {}
            if validation_mode:
                metrics_next = self.calculate_val_metrics(
                    y_pred,
                    y,
                    interp_step - 1,
                )
            metrics.update(metrics_next)
            y_preds.extend(y_pred)

        loss *= 1.0 / len(rd_interp_times)

        for specific_loss in original_weights:
            specific_loss.loss.time_weights = original_weights[specific_loss]
        return loss, metrics, y_preds

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x, self.model_comm_group)
