# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from __future__ import annotations

import logging

import torch
import torch.fft
from anemoi.training.losses.huber import WeightedHuberLoss
from anemoi.training.losses.weightedloss import BaseWeightedLoss, FunctionalWeightedLoss

LOGGER = logging.getLogger(__name__)


def amplitude(spectrum):
    return torch.sqrt(spectrum.real**2 + spectrum.imag**2)


def log_spectral_distance(fake_output: torch.Tensor, real_output: torch.Tensor, dims: tuple[int, int]) -> torch.Tensor:
    assert dims[0] * dims[1] == real_output.shape[2]
    dims_total = (*real_output.shape[:2], *dims, real_output.shape[-1])
    epsilon = torch.finfo(real_output.dtype).eps  # Small epsilon to avoid division by zero
    power_spectra_real_sq = amplitude(torch.fft.rfft2(real_output.reshape(dims_total), dim=(-2, -3))) ** 2
    power_spectra_fake_sq = amplitude(torch.fft.rfft2(fake_output.reshape(dims_total), dim=(-2, -3))) ** 2
    ratio = (power_spectra_real_sq + epsilon) / (power_spectra_fake_sq + epsilon)

    def log10(x: torch.Tensor) -> torch.Tensor:
        return torch.log(x) / torch.log(torch.tensor(10.0, device=x.device, dtype=x.dtype))

    log_10 = (10 * log10(ratio)) ** 2
    return torch.sqrt(torch.mean(log_10)) / 10


def fourier_correlation_loss(
    fake_output: torch.Tensor, real_output: torch.Tensor, dims: tuple[int, int]
) -> torch.Tensor:
    assert dims[0] * dims[1] == real_output.shape[2]
    dims_total = (*real_output.shape[:2], *dims, real_output.shape[-1])
    power_spectra_real = torch.fft.rfft2(real_output.reshape(dims_total), dim=(-2, -3))
    power_spectra_fake = torch.fft.rfft2(fake_output.reshape(dims_total), dim=(-2, -3))
    numerator = (1 / 2) * torch.sum(
        power_spectra_real * power_spectra_fake.conj() + power_spectra_fake * power_spectra_real.conj(),
        dim=(-2, -3),
    ).real
    epsilon = torch.finfo(real_output.dtype).eps  # Small epsilon to avoid division by zero
    denominator = torch.sqrt(
        torch.sum(amplitude(power_spectra_real) ** 2, dim=(-2, -3))
        * torch.sum(amplitude(power_spectra_fake) ** 2, dim=(-2, -3))
        + epsilon
    )
    return torch.mean(1 - numerator / denominator)


class LogSpectralDistance(FunctionalWeightedLoss):
    r"""The log spectral distance is used to compute the difference between spectra of two fields.

    It is also called log spectral distorsion.
    When it is expressed in discrete space with L2 norm, it is defined as:
    <math>D_{LS}={\left\{ \frac{1}{N} \sum_{n=1}^N \left[ \log P(n) - \log\hat{P}(n)\right]^2\right\\}}^{1/2} ,</math>.
    All scaling and weighting is handled by the parent class.
    """

    def __init__(
        self,
        node_weights: torch.Tensor,
        x_dim: int,
        y_dim: int,
        time_weights: torch.Tensor = None,
        ignore_nans: bool = False,
    ) -> None:
        super().__init__(node_weights, time_weights, ignore_nans)
        self.x_dim = x_dim
        self.y_dim = y_dim

    def calculate_difference(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return log_spectral_distance(pred, target, dims=(self.x_dim, self.y_dim))

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        *,
        scalar_indices: tuple[int, ...] | None = None,
        without_scalars: list[str] | list[int] | None = None,
    ) -> torch.Tensor:
        out = self.calculate_difference(pred, target)
        return self.scale_by_time_weights(out)


class FourierCorrelationLoss(FunctionalWeightedLoss):
    r"""The log spectral distance is used to compute the difference between spectra of two fields.

    See https://arxiv.org/pdf/2410.23159.pdf for more details.
    No scaling and weighting is performed for this loss.
    """

    def __init__(
        self,
        node_weights: torch.Tensor,
        x_dim: int,
        y_dim: int,
        time_weights: torch.Tensor = None,
        ignore_nans: bool = False,
    ) -> None:
        super().__init__(node_weights, time_weights, ignore_nans)
        self.x_dim = x_dim
        self.y_dim = y_dim

    def calculate_difference(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return fourier_correlation_loss(pred, target, dims=(self.x_dim, self.y_dim))

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        *,
        scalar_indices: tuple[int, ...] | None = None,
        without_scalars: list[str] | list[int] | None = None,
    ) -> torch.Tensor:
        out = self.calculate_difference(pred, target)
        return self.scale_by_time_weights(out)


class MultiLoss(BaseWeightedLoss):
    r"""Mix between both loss above.

    See https://arxiv.org/pdf/2410.23159.pdf for more details.
    """

    def __init__(
        self,
        node_weights: torch.Tensor,
        x_dim: int,
        y_dim: int,
        p=None,
        time_weights: torch.Tensor = None,
        ignore_nans: bool = False,
    ) -> None:
        super().__init__(node_weights, time_weights, ignore_nans)
        self.x_dim = x_dim
        self.y_dim = y_dim
        if p is None:
            p = torch.tensor([0.33, 0.33, 0.33])
        else:
            p = torch.tensor(p)
        self.d = torch.distributions.Categorical(p)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        *,
        scalar_indices: tuple[int, ...] | None = None,
        without_scalars: list[str] | list[int] | None = None,
    ) -> torch.Tensor:
        w = self.d.sample()
        w_array = torch.zeros(3, device=pred.device, dtype=pred.dtype)
        w_array[w] = 1
        return (
            w_array[0]
            * LogSpectralDistance(node_weights=self.node_weights, x_dim=self.x_dim, y_dim=self.y_dim)(
                pred, target, squash, scalar_indices=scalar_indices, without_scalars=without_scalars
            )
            + w_array[1]
            * FourierCorrelationLoss(node_weights=self.node_weights, x_dim=self.x_dim, y_dim=self.y_dim)(pred, target)
            + w_array[2]
            * torch.mean(
                WeightedHuberLoss(node_weights=self.node_weights, time_weights=self.time_weights)(
                    pred, target, squash, scalar_indices=scalar_indices, without_scalars=without_scalars
                )
            )
        )
