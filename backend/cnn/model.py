"""Lightweight CNN fault classifier (Module 3)."""
import torch
import torch.nn as nn

from backend import config as C


class FaultCNN(nn.Module):
    def __init__(self, in_channels: int = 2, num_classes: int = len(C.CLASSES),
                 feature_dim: int = C.FEATURE_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 32 -> 16
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),            # 16 -> 8
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),    # -> 64x1x1
        )
        self.feature = nn.Sequential(nn.Flatten(), nn.Linear(64, feature_dim), nn.ReLU())
        self.classifier = nn.Linear(feature_dim, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature(self.conv(x))

    def forward_with_features(self, x: torch.Tensor):
        feats = self.extract_features(x)
        return feats, self.classifier(feats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(x))
