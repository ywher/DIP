"""Modernized ResNet-101 embedding encoder matching DIP's native model."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import resnet as resnet_models


class PyramidPooling(nn.Module):
    def __init__(self, in_channels: int, bins=(1, 2, 3, 6)):
        super().__init__()
        reduction = in_channels // len(bins)
        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(bin_size),
                    nn.Conv2d(in_channels, reduction, 1, bias=False),
                    nn.BatchNorm2d(reduction),
                    nn.ReLU(inplace=True),
                )
                for bin_size in bins
            ]
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        size = features.shape[-2:]
        pooled = [features]
        pooled.extend(
            F.interpolate(stage(features), size=size, mode="bilinear", align_corners=True)
            for stage in self.stages
        )
        return torch.cat(pooled, dim=1)


class NativeDIPEncoder(nn.Module):
    feature_dim = 1024

    def __init__(self, pretrained: str | None = None, freeze_stem: bool = True):
        super().__init__()
        backbone = resnet_models.resnet101(pretrained=False)
        if pretrained:
            self._load_backbone_checkpoint(backbone, pretrained)
        self.layer0 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu1,
            backbone.conv2,
            backbone.bn2,
            backbone.relu2,
            backbone.conv3,
            backbone.bn3,
            backbone.relu3,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        for name, module in self.layer3.named_modules():
            if "conv2" in name:
                module.dilation = module.padding = (2, 2)
                module.stride = (1, 1)
            elif "downsample.0" in name:
                module.stride = (1, 1)
        for name, module in self.layer4.named_modules():
            if "conv2" in name:
                module.dilation = module.padding = (4, 4)
                module.stride = (1, 1)
            elif "downsample.0" in name:
                module.stride = (1, 1)

        self.ppm = PyramidPooling(2048)
        self.down = nn.Sequential(
            nn.Conv2d(4096, self.feature_dim, 1, bias=False),
            nn.BatchNorm2d(self.feature_dim),
            nn.ReLU(inplace=True),
        )
        self.projector = nn.Sequential(
            nn.Conv2d(self.feature_dim, self.feature_dim, 1, bias=False),
            nn.BatchNorm2d(self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feature_dim, self.feature_dim, 1, bias=False),
            nn.BatchNorm2d(self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feature_dim, self.feature_dim, 1, bias=False),
        )
        if freeze_stem:
            for module in (self.layer0, self.layer1):
                module.eval()
                for parameter in module.parameters():
                    parameter.requires_grad = False

    @staticmethod
    def _load_backbone_checkpoint(backbone: nn.Module, path: str) -> None:
        path = Path(path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"ResNet-101 checkpoint not found: {path}")
        state = torch.load(path, map_location="cpu")
        state = state.get("state_dict", state.get("model", state))
        state = {key.removeprefix("module."): value for key, value in state.items()}
        missing, unexpected = backbone.load_state_dict(state, strict=False)
        loaded = len(state) - len(unexpected)
        if loaded == 0:
            raise RuntimeError(f"No ResNet parameters were loaded from {path}")

    def train(self, mode: bool = True):
        super().train(mode)
        return self

    def extract_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        features = self.layer0(images)
        features = self.layer1(features)
        features = self.layer2(features)
        features = self.layer3(features)
        features = self.layer4(features)
        return self.projector(self.down(self.ppm(features)))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.extract_embeddings(images)

    def checkpoint_state(self) -> dict:
        return self.state_dict()

    def load_checkpoint_state(self, state: dict) -> None:
        self.load_state_dict(state, strict=True)
