import torch
import torch.nn as nn
import torchvision.models as tv_models


class FrequencyTransform(nn.Module):
    """Map RGB images to a log-magnitude frequency representation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) == 3:
            x = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        fft = torch.fft.fft2(x, dim=(-2, -1))
        fft_shift = torch.fft.fftshift(fft, dim=(-2, -1))
        magnitude = torch.log1p(torch.abs(fft_shift))
        return magnitude


class FrequencyBranch(nn.Module):
    """Lightweight CNN branch operating on frequency-domain inputs."""

    def __init__(self, feature_dim: int = 256):
        super().__init__()
        self.transform = FrequencyTransform()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.projection = nn.Linear(256, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.transform(x)
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.projection(x)


class HybridSpatialFrequencyModel(nn.Module):
    """Two-branch detector combining RGB spatial cues with frequency artifacts."""

    def __init__(
        self,
        num_classes: int = 1,
        spatial_feature_dim: int = 512,
        frequency_feature_dim: int = 256,
    ):
        super().__init__()
        backbone = tv_models.resnet50(weights=None)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, spatial_feature_dim)
        self.spatial_branch = backbone
        self.frequency_branch = FrequencyBranch(feature_dim=frequency_feature_dim)
        self.classifier = nn.Sequential(
            nn.Linear(spatial_feature_dim + frequency_feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial_features = self.spatial_branch(x)
        frequency_features = self.frequency_branch(x)
        fused = torch.cat([spatial_features, frequency_features], dim=1)
        return self.classifier(fused)


def build_hybrid_model(num_classes: int = 1) -> HybridSpatialFrequencyModel:
    return HybridSpatialFrequencyModel(num_classes=num_classes)
