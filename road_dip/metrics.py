"""Semantic segmentation metrics."""

from __future__ import annotations

import numpy as np
import torch


class ConfusionMatrix:
    def __init__(self, class_ids):
        self.class_ids = tuple(int(item) for item in class_ids)
        self.to_local = {class_id: index for index, class_id in enumerate(self.class_ids)}
        self.matrix = np.zeros((len(self.class_ids), len(self.class_ids)), dtype=np.int64)

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        prediction = prediction.detach().cpu().numpy().reshape(-1)
        target = target.detach().cpu().numpy().reshape(-1)
        pred_local = np.full(prediction.shape, -1, dtype=np.int64)
        target_local = np.full(target.shape, -1, dtype=np.int64)
        for canonical_id, local_id in self.to_local.items():
            pred_local[prediction == canonical_id] = local_id
            target_local[target == canonical_id] = local_id
        valid = (target_local >= 0) & (pred_local >= 0)
        encoded = target_local[valid] * len(self.class_ids) + pred_local[valid]
        self.matrix += np.bincount(
            encoded, minlength=len(self.class_ids) ** 2
        ).reshape(self.matrix.shape)

    def summary(self) -> dict:
        diagonal = np.diag(self.matrix).astype(np.float64)
        denominator = self.matrix.sum(1) + self.matrix.sum(0) - diagonal
        iou = np.divide(diagonal, denominator, out=np.full_like(diagonal, np.nan), where=denominator > 0)
        return {
            "miou": float(np.nanmean(iou) * 100.0),
            "iou": {str(class_id): float(value * 100.0) for class_id, value in zip(self.class_ids, iou)},
            "confusion": self.matrix.tolist(),
        }

