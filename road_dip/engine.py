"""Training, prototype extraction, and validation for road DIP."""

from __future__ import annotations

import json
import logging
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import (
    DIPPairDataset,
    EvalTransform,
    JointTrainTransform,
    LabeledTargetDataset,
    canonical_classes,
    seed_worker,
    single_item_collate,
    verify_records,
)
from .metrics import ConfusionMatrix
from .models import build_model, build_optimizer_scheduler
from .models.prototype import PrototypeBank, build_prototypes, prototype_logits


LOGGER = logging.getLogger("road_dip")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(output_dir / "run.log")):
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)


def output_directory(config: dict) -> Path:
    root = Path(config["experiment"].get("output_root", "outputs/road"))
    return (root / config["experiment"]["name"]).expanduser().resolve()


def _make_eval_dataset(config: dict, support: bool) -> LabeledTargetDataset:
    data_cfg = config["data"]
    prefix = "support" if support else "val"
    resize = data_cfg.get("prototype_resize" if support else "eval_resize")
    return LabeledTargetDataset(
        root=data_cfg[f"{prefix}_root"],
        manifest=data_cfg[f"{prefix}_list"],
        kind=data_cfg[f"{prefix}_kind"],
        label_space=data_cfg.get(f"{prefix}_label_space", "trainid"),
        transform=EvalTransform(resize),
    )


def build_prototype_bank(model, config: dict, device: torch.device) -> PrototypeBank:
    dataset = _make_eval_dataset(config, support=True)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(config["train"].get("workers", 4)),
        pin_memory=True,
    )
    classes = canonical_classes(config["experiment"]["task"])
    bank = PrototypeBank.empty(classes, model.feature_dim, device)
    model.eval()
    with torch.no_grad():
        for index, batch in enumerate(loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            features = model.extract_embeddings(images)
            bank.update(features, labels)
            if index % 20 == 0 or index == len(loader):
                LOGGER.info("Prototype extraction %d/%d", index, len(loader))
    missing = bank.class_ids[~bank.valid].tolist()
    if missing:
        LOGGER.warning("No support pixels for canonical classes: %s", missing)
    return bank


def evaluate(model, config: dict, device: torch.device, bank: PrototypeBank | None = None) -> dict:
    if bank is None:
        bank = build_prototype_bank(model, config, device)
    valid = bank.valid
    prototypes = bank.prototypes[valid]
    prototype_classes = bank.class_ids[valid]
    dataset = _make_eval_dataset(config, support=False)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(config["train"].get("workers", 4)),
        pin_memory=True,
    )
    metric = ConfusionMatrix(canonical_classes(config["experiment"]["task"]))
    model.eval()
    start = time.perf_counter()
    with torch.no_grad():
        for index, batch in enumerate(loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            embeddings = model.extract_embeddings(images)
            logits = prototype_logits(
                embeddings, prototypes, scale=config["model"].get("prototype_scale", 20.0)
            )
            logits = F.interpolate(
                logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
            )
            local_prediction = logits.argmax(dim=1)
            prediction = prototype_classes[local_prediction]
            metric.update(prediction, labels)
            if index % 50 == 0 or index == len(loader):
                elapsed = time.perf_counter() - start
                eta = elapsed / index * (len(loader) - index)
                LOGGER.info("Validation %d/%d | ETA %.1f min", index, len(loader), eta / 60.0)
    result = metric.summary()
    result["prototype_counts"] = {
        str(class_id): float(count)
        for class_id, count in zip(bank.class_ids.tolist(), bank.counts.tolist())
    }
    return result


def save_checkpoint(path: Path, model, optimizer, scheduler, iteration: int, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "iteration": iteration,
            "model": model.checkpoint_state(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": config,
        },
        path,
    )


def load_checkpoint(path: str | Path, model, optimizer=None, scheduler=None, device="cpu") -> int:
    checkpoint = torch.load(Path(path).expanduser(), map_location=device)
    model.load_checkpoint_state(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return int(checkpoint.get("iteration", 0))


def train(config: dict, resume: str | None = None) -> Path:
    output_dir = output_directory(config)
    setup_logging(output_dir)
    seed = int(config["experiment"].get("seed", 0))
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        LOGGER.warning("CUDA is unavailable; training will be impractically slow")

    train_cfg = config["train"]
    transform = JointTrainTransform(
        crop_size=train_cfg["crop_size"],
        scale_range=train_cfg.get("scale_range", [0.5, 2.0]),
        flip_probability=train_cfg.get("flip_probability", 0.5),
        color_jitter=train_cfg.get("color_jitter", 0.2),
        blur_probability=train_cfg.get("blur_probability", 0.5),
    )
    dataset = DIPPairDataset(
        config["data"], config["experiment"]["task"], train_cfg["max_iters"], transform
    )
    verify_records(dataset.source, limit=int(config["data"].get("verify_files", 20)))
    verify_records(dataset.support, limit=None)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(train_cfg.get("workers", 4)),
        pin_memory=True,
        collate_fn=single_item_collate,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=int(train_cfg.get("workers", 4)) > 0,
    )
    model = build_model(config).to(device)
    optimizer, scheduler = build_optimizer_scheduler(model, config)
    start_iteration = 0
    if resume:
        start_iteration = load_checkpoint(resume, model, optimizer, scheduler, device)
        LOGGER.info("Resumed %s at iteration %d", resume, start_iteration)

    use_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    support_weight = float(train_cfg.get("support_loss_weight", 0.2))
    grad_accum = int(train_cfg.get("gradient_accumulation", 1))
    checkpoint_period = int(train_cfg.get("checkpoint_period", 10000))
    eval_period = int(train_cfg.get("eval_period", 0))
    print_period = int(train_cfg.get("print_period", 50))
    max_iters = int(train_cfg["max_iters"])
    if start_iteration >= max_iters:
        LOGGER.info("Checkpoint already reached max_iters=%d", max_iters)
        return output_dir
    optimizer.zero_grad(set_to_none=True)
    start_time = time.perf_counter()
    running_loss = 0.0

    for iteration, batch in enumerate(loader, start=start_iteration + 1):
        model.train()
        source_image = batch["source_image"].unsqueeze(0).to(device, non_blocking=True)
        support_image = batch["support_image"].unsqueeze(0).to(device, non_blocking=True)
        source_label = batch["source_label"].unsqueeze(0).to(device, non_blocking=True)
        support_label = batch["support_label"].unsqueeze(0).to(device, non_blocking=True)
        images = torch.cat([source_image, support_image], dim=0)
        pair_classes = torch.arange(len(batch["shared_classes"]), device=device)

        with torch.cuda.amp.autocast(enabled=use_amp):
            embeddings = model.extract_embeddings(images)
            source_embedding, support_embedding = embeddings[:1], embeddings[1:]
            prototypes = build_prototypes(support_embedding, support_label, pair_classes)
            source_logits = prototype_logits(
                source_embedding, prototypes, config["model"].get("prototype_scale", 20.0)
            )
            support_logits = prototype_logits(
                support_embedding, prototypes, config["model"].get("prototype_scale", 20.0)
            )
            source_logits = F.interpolate(
                source_logits, source_label.shape[-2:], mode="bilinear", align_corners=False
            )
            support_logits = F.interpolate(
                support_logits, support_label.shape[-2:], mode="bilinear", align_corners=False
            )
            source_loss = F.cross_entropy(source_logits, source_label, ignore_index=255)
            support_loss = F.cross_entropy(support_logits, support_label, ignore_index=255)
            loss = (source_loss + support_weight * support_loss) / grad_accum

        scaler.scale(loss).backward()
        if iteration % grad_accum == 0:
            scale_before_step = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            # AMP can skip the optimizer step after detecting overflow.
            if scaler.get_scale() >= scale_before_step:
                scheduler.step()
        running_loss += float(loss.detach()) * grad_accum

        if iteration % print_period == 0 or iteration == 1:
            elapsed = time.perf_counter() - start_time
            speed = elapsed / max(1, iteration - start_iteration)
            eta = speed * (max_iters - iteration)
            LOGGER.info(
                "iter %d/%d | loss %.4f | shared %d | lr %.2e | ETA %.1f h",
                iteration,
                max_iters,
                running_loss / (1 if iteration == 1 else print_period),
                len(pair_classes),
                optimizer.param_groups[0]["lr"],
                eta / 3600.0,
            )
            running_loss = 0.0

        should_save = iteration % checkpoint_period == 0 or iteration == max_iters
        if should_save:
            checkpoint_path = output_dir / f"checkpoint_{iteration:06d}.pth"
            save_checkpoint(checkpoint_path, model, optimizer, scheduler, iteration, config)
            LOGGER.info("Saved %s", checkpoint_path)
        if eval_period and (iteration % eval_period == 0 or iteration == max_iters):
            result = evaluate(model, config, device)
            result["iteration"] = iteration
            with (output_dir / f"metrics_{iteration:06d}.json").open("w", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2)
            LOGGER.info("Validation mIoU %.2f", result["miou"])
        if iteration >= max_iters:
            break
    return output_dir


def evaluate_checkpoint(config: dict, checkpoint: str | Path) -> dict:
    output_dir = output_directory(config)
    setup_logging(output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    iteration = load_checkpoint(checkpoint, model, device=device)
    result = evaluate(model, config, device)
    result["iteration"] = iteration
    result["checkpoint"] = str(Path(checkpoint).expanduser().resolve())
    result_path = output_dir / f"eval_{iteration:06d}.json"
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    LOGGER.info("Checkpoint %d | mIoU %.2f | %s", iteration, result["miou"], result_path)
    return result
