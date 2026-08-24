"""Model factory for native and VFM DIP profiles."""

from __future__ import annotations

import torch

from .native import NativeDIPEncoder
from .vfm import VFMHRDADIPEncoder, WarmupPolyLR, build_vfm_optimizer


def build_model(config: dict):
    model_cfg = config["model"]
    model_type = model_cfg["type"]
    if model_type == "native_r101":
        return NativeDIPEncoder(
            pretrained=model_cfg.get("pretrained"),
            freeze_stem=model_cfg.get("freeze_stem", True),
        )
    if model_type == "dinov3_base_rein_hrda":
        detail_crop = model_cfg.get("hr_crop_size", [512, 512])
        return VFMHRDADIPEncoder(
            pretrained=model_cfg["pretrained"],
            crop_size=detail_crop,
            init=model_cfg.get("init"),
        )
    raise ValueError(model_type)


def build_optimizer_scheduler(model, config: dict):
    train_cfg = config["train"]
    if config["model"]["type"] == "dinov3_base_rein_hrda":
        optimizer = build_vfm_optimizer(model, float(train_cfg.get("base_lr", 1e-4)))
        scheduler = WarmupPolyLR(
            optimizer,
            max_iters=int(train_cfg["max_iters"]),
            warmup_iters=int(train_cfg.get("warmup_iters", 1500)),
            power=float(train_cfg.get("power", 1.0)),
        )
    else:
        optimizer = torch.optim.SGD(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=float(train_cfg.get("base_lr", 1e-3)),
            momentum=float(train_cfg.get("momentum", 0.9)),
            weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        )
        scheduler = torch.optim.lr_scheduler.PolynomialLR(
            optimizer,
            total_iters=int(train_cfg["max_iters"]),
            power=float(train_cfg.get("power", 0.9)),
        )
    return optimizer, scheduler


__all__ = ["build_model", "build_optimizer_scheduler"]

