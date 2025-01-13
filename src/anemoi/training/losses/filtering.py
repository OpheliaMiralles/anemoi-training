import torch
from anemoi.models.data_indices.collection import IndexCollection


class FilteringLossWrapper(torch.nn.Module):

    def __init__(
        self,
        loss: torch.nn.Module,
        data_indices: IndexCollection,
        predicted_variables: list[str] | None = None,
        target_variables: list[str] | None = None,
        **loss_kwargs,
    ):
        """Loss wrapper to filter variables to compute the loss on.

        Parameters
        ----------
        loss : Union[Type[torch.nn.Module], Dict[str, Any]]
            wrapped loss
        predicted_variables : List[str] | None
            predicted variables to keep, if None, all variables are kept
        target_variables : List[str] | None
            target variables to keep, if None, all variables are kept
        """
        if predicted_variables and target_variables:
            assert len(predicted_variables) == len(
                target_variables
            ), "predicted and target variables must have the same length for loss computation"

        super().__init__()
        self.loss = loss
        self.predicted_variables = predicted_variables
        self.target_variables = target_variables
        self.loss_kwargs = loss_kwargs
        self.data_indices = data_indices
        name_to_index = data_indices.model.output.name_to_index
        output_indices = data_indices.internal_model.output.full
        if self.predicted_variables is not None:
            predicted_indices = [name_to_index[name] for name in self.predicted_variables]
        else:
            predicted_indices = output_indices
        if self.target_variables is not None:
            target_indices = [name_to_index[name] for name in self.target_variables]
        else:
            target_indices = output_indices

        assert len(predicted_indices) == len(
            target_indices
        ), "predicted and target variables must have the same length for loss computation"

        self.predicted_indices = predicted_indices
        self.target_indices = target_indices

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred[..., self.predicted_indices]
        target = target[..., self.target_indices]
        return self.loss(pred, target, **self.loss_kwargs)
