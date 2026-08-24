"""Prototype construction and nearest-prototype prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F


def resize_labels(labels: torch.Tensor, spatial_size: Sequence[int]) -> torch.Tensor:
    if labels.ndim == 2:
        labels = labels.unsqueeze(0)
    return F.interpolate(
        labels.unsqueeze(1).float(), size=tuple(spatial_size), mode="nearest"
    ).squeeze(1).long()


def masked_class_statistics(
    features: torch.Tensor,
    labels: torch.Tensor,
    class_ids: Sequence[int] | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-class feature sums and pixel counts."""

    labels = resize_labels(labels, features.shape[-2:])
    class_ids = torch.as_tensor(class_ids, device=features.device, dtype=torch.long)
    sums, counts = [], []
    for class_id in class_ids:
        mask = labels == class_id
        count = mask.sum()
        class_sum = (features * mask.unsqueeze(1)).sum(dim=(0, 2, 3))
        sums.append(class_sum)
        counts.append(count)
    return torch.stack(sums), torch.stack(counts).to(features.dtype)


def build_prototypes(
    features: torch.Tensor,
    labels: torch.Tensor,
    class_ids: Sequence[int] | torch.Tensor,
) -> torch.Tensor:
    sums, counts = masked_class_statistics(features, labels, class_ids)
    return sums / counts.unsqueeze(1).clamp_min(1.0)


def prototype_logits(
    features: torch.Tensor, prototypes: torch.Tensor, scale: float = 20.0
) -> torch.Tensor:
    features = F.normalize(features, dim=1)
    prototypes = F.normalize(prototypes, dim=1)
    return torch.einsum("bchw,kc->bkhw", features, prototypes) * float(scale)


@dataclass
class PrototypeBank:
    class_ids: torch.Tensor
    sums: torch.Tensor
    counts: torch.Tensor

    @classmethod
    def empty(cls, class_ids: Sequence[int], feature_dim: int, device: torch.device):
        ids = torch.as_tensor(class_ids, dtype=torch.long, device=device)
        return cls(
            class_ids=ids,
            sums=torch.zeros((len(ids), feature_dim), device=device),
            counts=torch.zeros(len(ids), device=device),
        )

    def update(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        sums, counts = masked_class_statistics(features, labels, self.class_ids)
        self.sums += sums
        self.counts += counts

    @property
    def prototypes(self) -> torch.Tensor:
        return self.sums / self.counts.unsqueeze(1).clamp_min(1.0)

    @property
    def valid(self) -> torch.Tensor:
        return self.counts > 0

    def state_dict(self) -> dict:
        return {
            "class_ids": self.class_ids.cpu(),
            "sums": self.sums.cpu(),
            "counts": self.counts.cpu(),
        }

    @classmethod
    def from_state_dict(cls, state: dict, device: torch.device):
        return cls(
            class_ids=state["class_ids"].to(device),
            sums=state["sums"].to(device),
            counts=state["counts"].to(device),
        )
