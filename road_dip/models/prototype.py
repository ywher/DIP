"""Prototype construction and nearest-prototype prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F


def masked_class_statistics(
    features: torch.Tensor,
    labels: torch.Tensor,
    class_ids: Sequence[int] | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return soft-mask feature sums and areas, matching DIP's Weighted_GAP."""

    if labels.ndim == 2:
        labels = labels.unsqueeze(0)
    class_ids = torch.as_tensor(class_ids, device=features.device, dtype=torch.long)
    masks = (labels[:, None] == class_ids[None, :, None, None]).to(features.dtype)
    masks = F.interpolate(
        masks,
        size=features.shape[-2:],
        mode="bilinear",
        align_corners=True,
    )
    sums = torch.einsum("bchw,bkhw->bkc", features, masks)
    areas = masks.sum(dim=(2, 3))
    return sums, areas


def build_prototypes(
    features: torch.Tensor,
    labels: torch.Tensor,
    class_ids: Sequence[int] | torch.Tensor,
) -> torch.Tensor:
    sums, areas = masked_class_statistics(features, labels, class_ids)
    sums = sums.sum(dim=0)
    areas = areas.sum(dim=0)
    return sums / areas.unsqueeze(1).clamp_min(5e-4)


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
    max_shots: int = 5

    @classmethod
    def empty(
        cls,
        class_ids: Sequence[int],
        feature_dim: int,
        device: torch.device,
        max_shots: int = 5,
    ):
        ids = torch.as_tensor(class_ids, dtype=torch.long, device=device)
        return cls(
            class_ids=ids,
            sums=torch.zeros((len(ids), feature_dim), device=device),
            counts=torch.zeros(len(ids), device=device),
            max_shots=int(max_shots),
        )

    def update(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        sums, areas = masked_class_statistics(features, labels, self.class_ids)
        means = sums / areas.unsqueeze(2).clamp_min(5e-4)
        for batch_index in range(features.shape[0]):
            available = (areas[batch_index] > 0) & (self.counts < self.max_shots)
            self.sums[available] += means[batch_index, available]
            self.counts[available] += 1

    @property
    def prototypes(self) -> torch.Tensor:
        return self.sums / self.counts.unsqueeze(1).clamp_min(1.0)

    @property
    def valid(self) -> torch.Tensor:
        return self.counts > 0

    @property
    def complete(self) -> torch.Tensor:
        return self.counts >= self.max_shots

    def state_dict(self) -> dict:
        return {
            "class_ids": self.class_ids.cpu(),
            "sums": self.sums.cpu(),
            "counts": self.counts.cpu(),
            "max_shots": self.max_shots,
        }

    @classmethod
    def from_state_dict(cls, state: dict, device: torch.device):
        return cls(
            class_ids=state["class_ids"].to(device),
            sums=state["sums"].to(device),
            counts=state["counts"].to(device),
            max_shots=int(state.get("max_shots", 5)),
        )
