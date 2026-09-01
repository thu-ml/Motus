import pytest
import torch

from models.losses import future_video_mse_loss


@pytest.mark.parametrize("num_future_frames", [1, 2])
def test_future_video_mse_excludes_condition_and_normalizes_valid_frames(
    num_future_frames: int,
) -> None:
    prediction = (
        torch.tensor([100.0] + [2.0] * num_future_frames)
        .reshape(1, 1, num_future_frames + 1, 1, 1)
        .requires_grad_()
    )
    target = torch.tensor([-100.0] + [0.0] * num_future_frames).reshape_as(prediction)

    loss = future_video_mse_loss(prediction, target)

    assert loss.item() == pytest.approx(4.0)

    loss.backward()
    assert prediction.grad is not None
    assert prediction.grad[:, :, 0].eq(0).all()
    assert prediction.grad[:, :, 1:].eq(4.0 / num_future_frames).all()


def test_future_video_mse_rejects_condition_only_video() -> None:
    condition_only = torch.zeros(1, 1, 1, 2, 2)

    with pytest.raises(ValueError, match="at least one non-conditioning latent frame"):
        future_video_mse_loss(condition_only, condition_only)
