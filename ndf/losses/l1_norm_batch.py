import torch


class BatchNormalizedL1(torch.nn.Module):
    def __init__(self, epsilon: float = 1e-7):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError("prediction and target must have the same shape")
        if prediction.size(-1) != 2:
            raise ValueError("expected last dimension to be 2 (real/imag)")
        diff = torch.abs(prediction - target)
        target_mag = torch.abs(target)
        diff_sum = diff.sum(dim=(1, 2, 3))
        target_sum = target_mag.sum(dim=(1, 2, 3))
        loss = diff_sum / (target_sum + self.epsilon)
        return loss.mean()
