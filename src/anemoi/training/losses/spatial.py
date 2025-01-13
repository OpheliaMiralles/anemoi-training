from __future__ import annotations

import logging
import torch
import torch.fft

from anemoi.training.losses.weightedloss import FunctionalWeightedLoss

LOGGER = logging.getLogger(__name__)

def log_spectral_distance(real_output, fake_output):
    epsilon = torch.finfo(torch.float32).eps  # Small epsilon to avoid division by zero
    power_spectra_real = torch.abs(torch.fft.rfft2(real_output)) ** 2
    power_spectra_fake = torch.abs(torch.fft.rfft2(fake_output)) ** 2
    ratio = (power_spectra_real + epsilon) / (power_spectra_fake + epsilon)
    
    def log10(x):
        return torch.log(x) / torch.log(torch.tensor(10.0, device=x.device, dtype=x.dtype))
    
    result = (10 * log10(ratio)) ** 2
    lsd = torch.sqrt(torch.mean(result, dim=(-1, -2, -3)))  # Mean over last 3 dimensions
    lsd = torch.where(torch.isnan(lsd), torch.zeros_like(lsd), lsd)
    return lsd

class LogSpectralDistance(FunctionalWeightedLoss):
    """WeightedLoss which a user can subclass and provide `calculate_difference`.

    `calculate_difference` should calculate the difference between the prediction and target.
    All scaling and weighting is handled by the parent class.

    Example:
    --------
    ```python
    class MyLoss(FunctionalWeightedLoss):
        def calculate_difference(self, pred, target):
            return pred - target
    ```
    """

    def __init__(
        self,
        node_weights: torch.Tensor,
        ignore_nans: bool = False,
    ) -> None:
        super().__init__(node_weights, ignore_nans)

    def calculate_difference(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate Difference between prediction and target."""
        return log_spectral_distance(pred, target)