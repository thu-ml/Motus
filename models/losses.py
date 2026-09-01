"""Loss functions used by Motus models."""

import torch
import torch.nn.functional as F


def future_video_mse_loss(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Return the mean squared error over non-conditioning latent frames.

    Video tensors are expected in ``[batch, channels, time, height, width]``
    order, with the first latent frame supplied as a fixed condition.
    """
    if prediction.shape != target.shape:
        raise ValueError(
            "prediction and target must have matching shapes, "
            f"got {prediction.shape} and {target.shape}"
        )
    if prediction.ndim != 5:
        raise ValueError(
            "prediction and target must be 5D [batch, channels, time, height, width] tensors"
        )
    if prediction.shape[2] <= 1:
        raise ValueError(
            "video loss requires at least one non-conditioning latent frame"
        )

    return F.mse_loss(prediction[:, :, 1:], target[:, :, 1:], reduction="mean")
