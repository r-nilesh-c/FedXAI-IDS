import torch
import torch.nn as nn
import numpy as np


class IDS_MLP(nn.Module):
    """
    Deep Neural Network (MLP) for Intrusion Detection System (IDS) in FedXAI-IDS.

    Features:
    - Input BatchNorm1d layer to dynamically normalize tabular network flow features
      per mini-batch inside PyTorch, preventing feature misalignments across clients
      without exchanging global feature statistics (privacy-preserving).
    - Lightweight 3-layer architecture for fast local client training and FL communication efficiency.
    """

    # Input Layer(features) --> Hidden layers --> Output(class)
    def __init__(self, input_dim: int, num_classes: int = 7):
        super(IDS_MLP, self).__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes

        # Dynamic Batch Normalization directly on raw input features
        self.input_bn = nn.BatchNorm1d(input_dim)

        # Deep Feature Extractor
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle batch size 1 edge cases for BatchNorm1d
        if x.dim() == 2 and x.size(0) > 1:
            x = self.input_bn(x)
        logits = self.net(x)
        return logits


class LogitAdjustedLoss(nn.Module):
    """
    Mini-Batch Logit Adjustment Loss for Non-IID Label Skew in Federated Learning.

    Adjusts raw model output logits by adding class prior offsets:
    Adjusted Logits = f(x) + tau * log(pi)

    Prevents local client models from over-predicting dominant classes under
    Dirichlet Non-IID label skew (alpha = 0.5), enforcing balanced relative margins.
    """

    def __init__(self, cls_priors: np.ndarray, tau: float = 1.0):
        super(LogitAdjustedLoss, self).__init__()
        priors_tensor = torch.tensor(cls_priors, dtype=torch.float32)
        # Normalize priors to ensure valid probabilities
        priors_tensor = priors_tensor / torch.sum(priors_tensor)
        self.register_buffer("cls_priors", priors_tensor)
        self.tau = tau
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Compute logit adjustments: tau * log(pi + epsilon)
        log_priors = torch.log(self.cls_priors + 1e-12)
        adjusted_logits = logits + self.tau * log_priors
        return self.cross_entropy(adjusted_logits, targets)


if __name__ == "__main__":
    # Quick sanity check test
    model = IDS_MLP(input_dim=21, num_classes=7)
    dummy_x = torch.randn(32, 21)
    out = model(dummy_x)
    print(f"[MODEL CHECK] Output Logits Shape: {out.shape}")

    priors = np.array([0.28, 0.23, 0.17, 0.11, 0.11, 0.05, 0.05])
    criterion = LogitAdjustedLoss(priors, tau=1.0)
    dummy_y = torch.randint(0, 7, (32,))
    loss = criterion(out, dummy_y)
    print(f"[LOSS CHECK] Computed Logit Adjustment Loss: {loss.item():.4f}")


