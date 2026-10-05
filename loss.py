import torch


class WeightedMSE:

    def __init__(self, reduction='mean', device='cpu', weights=None):
        self.reduction = reduction
        self.device = device
        if weights is not None:
            self.weights = torch.tensor(weights, dtype=torch.float32).to(device)
        else:
            # Default weights for 10 target spatial points
            self.weights = torch.tensor([2.0, 2.0, 2.0, 2.0, 1.0, 0.8, 0.5, 0.5, 0.1, 0.1], dtype=torch.float32).to(device)

    def __call__(self, input, target):
        if input.shape[1] != self.weights.shape[0]:
            # Fallback if dimension does not match default 10 points
            weights = torch.ones(input.shape[1], dtype=torch.float32, device=self.device)
        else:
            weights = self.weights

        squared_error = ((input - target) ** 2) * weights

        if self.reduction == 'mean':
            return squared_error.mean()
        elif self.reduction == 'sum':
            return squared_error.sum()
        else:
            return squared_error


class HuberLoss:
    def __init__(self, delta: float = 1.0):
        self.delta = delta

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        error = torch.abs(y_true - y_pred)
        loss = torch.where(error <= self.delta,
                           0.5 * error ** 2,
                           self.delta * (error - 0.5 * self.delta))
        return loss.mean()

