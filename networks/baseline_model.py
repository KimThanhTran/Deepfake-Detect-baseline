import torch.nn as nn
from torchvision.models import resnet50


class BaselineModel(nn.Module):
    """Standard RGB-image baseline for binary deepfake detection."""

    def __init__(self, num_classes: int = 1):
        super().__init__()
        self.backbone = resnet50(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


def build_baseline_model(num_classes: int = 1) -> BaselineModel:
    return BaselineModel(num_classes=num_classes)
