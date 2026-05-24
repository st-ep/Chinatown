"""Compact contrastive-regression model for viscosity videos."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: tuple[int, int, int]) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SmallVideoEncoder(nn.Module):
    """Small 3D CNN baseline for 16-32 frame clips."""

    def __init__(self, feature_dim: int = 256) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(
                3,
                32,
                kernel_size=(3, 7, 7),
                stride=(1, 2, 2),
                padding=(1, 3, 3),
                bias=False,
            ),
            nn.BatchNorm3d(32),
            nn.GELU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
        )
        self.blocks = nn.Sequential(
            ConvBlock3d(32, 64, stride=(2, 2, 2)),
            ConvBlock3d(64, 128, stride=(2, 2, 2)),
            ConvBlock3d(128, feature_dim, stride=(2, 2, 2)),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        x = self.stem(video)
        x = self.blocks(x)
        return self.pool(x).flatten(1)


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class PropertyEncoder(nn.Module):
    def __init__(self, input_dim: int = 1, hidden_dim: int = 128, output_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, properties: torch.Tensor) -> torch.Tensor:
        if properties.ndim == 1:
            properties = properties[:, None]
        return F.normalize(self.net(properties), dim=-1)


class RegressionHead(nn.Module):
    def __init__(self, input_dim: int = 128, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.net(embedding).squeeze(-1)


class ViscosityContrastiveModel(nn.Module):
    def __init__(self, feature_dim: int = 256, embedding_dim: int = 128) -> None:
        super().__init__()
        self.video_encoder = SmallVideoEncoder(feature_dim=feature_dim)
        self.video_projection = ProjectionHead(feature_dim, output_dim=embedding_dim)
        self.property_encoder = PropertyEncoder(output_dim=embedding_dim)
        self.regression_head = RegressionHead(input_dim=embedding_dim)

    def forward(self, video: torch.Tensor, log10_mu: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.video_encoder(video)
        video_embedding = self.video_projection(features)
        property_embedding = self.property_encoder(log10_mu)
        prediction = self.regression_head(video_embedding)
        return {
            "video_embedding": video_embedding,
            "property_embedding": property_embedding,
            "pred_log10_mu": prediction,
        }

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        return self.video_projection(self.video_encoder(video))

    def encode_property(self, log10_mu: torch.Tensor) -> torch.Tensor:
        return self.property_encoder(log10_mu)


def soft_contrastive_loss(
    video_embedding: torch.Tensor,
    property_embedding: torch.Tensor,
    targets: torch.Tensor,
    *,
    temperature: float = 0.07,
    sigma: float = 0.08,
) -> torch.Tensor:
    logits = video_embedding @ property_embedding.T / temperature
    distances = targets[:, None] - targets[None, :]
    soft_targets = torch.softmax(-(distances * distances) / (2.0 * sigma * sigma), dim=1)
    video_to_property = -(soft_targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
    property_to_video = -(soft_targets.T * F.log_softmax(logits.T, dim=1)).sum(dim=1).mean()
    return 0.5 * (video_to_property + property_to_video)


def regression_loss(prediction: torch.Tensor, target: torch.Tensor, *, beta: float = 0.05) -> torch.Tensor:
    return F.smooth_l1_loss(prediction, target, beta=beta)
