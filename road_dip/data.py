"""Manifest-based datasets for the five road-scene DIP transfers."""

from __future__ import annotations

import math
import random
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torch.utils.data import Dataset


IGNORE_LABEL = 255
CITYSCAPES_CLASSES = tuple(range(19))
SYNTHIA_CITYSCAPES_CLASSES = (0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18)
SYNTHIA_RAW_TO_CITYSCAPES = {
    3: 0,
    4: 1,
    2: 2,
    21: 3,
    5: 4,
    7: 5,
    15: 6,
    9: 7,
    6: 8,
    1: 10,
    10: 11,
    17: 12,
    8: 13,
    19: 15,
    12: 17,
    11: 18,
}


@dataclass(frozen=True)
class Sample:
    image: Path
    label: Path
    name: str


def canonical_classes(task: str) -> tuple[int, ...]:
    if task == "synthia2cityscapes":
        return SYNTHIA_CITYSCAPES_CLASSES
    return CITYSCAPES_CLASSES


def _split_manifest_line(line: str) -> list[str]:
    line = line.strip()
    if not line or line.startswith("#"):
        return []
    if "," in line:
        return [item.strip() for item in line.split(",") if item.strip()]
    return shlex.split(line)


def _derive_pair(root: Path, token: str, kind: str) -> tuple[Path, Path]:
    if kind == "gta":
        image = root / "images" / token
        label_name = token.replace(".png", "_labelTrainIds.png")
        return image, root / "labels" / label_name
    if kind == "synthia":
        synthia_root = root / "RAND_CITYSCAPES"
        if not synthia_root.is_dir():
            synthia_root = root
        return (
            synthia_root / "RGB" / token,
            synthia_root / "GT" / "LABELS" / token,
        )
    if kind == "cityscapes_source":
        return (
            root / "leftImg8bit" / "train_all" / token,
            root / "gtFine" / "train_all" / token.replace(
                "_leftImg8bit", "_gtFine_labelTrainIds"
            ),
        )
    if kind in {"cityscapes", "cityscapes_val"}:
        image = root / token
        label_rel = token.replace("leftImg8bit", "gtFine").replace(
            "_leftImg8bit", "_gtFine_labelTrainIds"
        )
        return image, root / label_rel
    raise ValueError(
        f"Manifest line for kind={kind!r} must contain both image and label paths"
    )


def read_manifest(root: str | Path, manifest: str | Path, kind: str) -> list[Sample]:
    root = Path(root).expanduser().resolve()
    manifest = Path(manifest).expanduser().resolve()
    records: list[Sample] = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            fields = _split_manifest_line(line)
            if not fields:
                continue
            if len(fields) == 1:
                image, label = _derive_pair(root, fields[0], kind)
            elif len(fields) == 2:
                image, label = root / fields[0], root / fields[1]
            else:
                raise ValueError(f"{manifest}:{line_no}: expected one or two paths")
            records.append(Sample(image=image, label=label, name=image.stem))
    if not records:
        raise ValueError(f"Empty manifest: {manifest}")
    return records


def verify_records(records: Sequence[Sample], limit: int | None = None) -> None:
    checked = records if limit is None else records[:limit]
    missing = [str(path) for record in checked for path in (record.image, record.label) if not path.is_file()]
    if missing:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(f"Missing dataset files ({len(missing)}):\n{preview}")


def _map_label(label: np.ndarray, label_space: str) -> np.ndarray:
    if label_space == "trainid":
        return label.astype(np.uint8, copy=False)
    if label_space == "synthia_raw":
        mapped = np.full(label.shape, IGNORE_LABEL, dtype=np.uint8)
        for source_id, target_id in SYNTHIA_RAW_TO_CITYSCAPES.items():
            mapped[label == source_id] = target_id
        return mapped
    raise ValueError(f"Unsupported label space: {label_space}")


def load_image_label(sample: Sample, label_space: str) -> tuple[Image.Image, Image.Image]:
    image = Image.open(sample.image).convert("RGB")
    label = np.asarray(Image.open(sample.label), dtype=np.uint8)
    label = Image.fromarray(_map_label(label, label_space), mode="L")
    return image, label


class JointTrainTransform:
    def __init__(
        self,
        crop_size: Sequence[int],
        scale_range: Sequence[float] = (0.5, 2.0),
        flip_probability: float = 0.5,
        color_jitter: float = 0.2,
        blur_probability: float = 0.5,
    ):
        self.crop_h, self.crop_w = (int(crop_size[0]), int(crop_size[1]))
        self.scale_min, self.scale_max = map(float, scale_range)
        self.flip_probability = float(flip_probability)
        self.color_jitter = float(color_jitter)
        self.blur_probability = float(blur_probability)

    def __call__(self, image: Image.Image, label: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        scale = random.uniform(self.scale_min, self.scale_max)
        new_w = max(1, int(round(image.width * scale)))
        new_h = max(1, int(round(image.height * scale)))
        image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
        label = label.resize((new_w, new_h), Image.Resampling.NEAREST)

        pad_w = max(0, self.crop_w - new_w)
        pad_h = max(0, self.crop_h - new_h)
        if pad_w or pad_h:
            image = ImageOps.expand(image, border=(0, 0, pad_w, pad_h), fill=(0, 0, 0))
            label = ImageOps.expand(label, border=(0, 0, pad_w, pad_h), fill=IGNORE_LABEL)

        max_x = max(0, image.width - self.crop_w)
        max_y = max(0, image.height - self.crop_h)
        x0 = random.randint(0, max_x) if max_x else 0
        y0 = random.randint(0, max_y) if max_y else 0
        box = (x0, y0, x0 + self.crop_w, y0 + self.crop_h)
        image, label = image.crop(box), label.crop(box)

        if random.random() < self.flip_probability:
            image = ImageOps.mirror(image)
            label = ImageOps.mirror(label)

        if self.color_jitter > 0:
            strength = self.color_jitter
            image = ImageEnhance.Brightness(image).enhance(random.uniform(1 - strength, 1 + strength))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(1 - strength, 1 + strength))
            image = ImageEnhance.Color(image).enhance(random.uniform(1 - strength, 1 + strength))
        if random.random() < self.blur_probability:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 1.5)))
        return image_to_tensor(image), torch.from_numpy(np.asarray(label, dtype=np.int64).copy())


class EvalTransform:
    def __init__(self, resize: Sequence[int] | None = None):
        self.resize = tuple(map(int, resize)) if resize else None

    def __call__(self, image: Image.Image, label: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        if self.resize:
            height, width = self.resize
            image = image.resize((width, height), Image.Resampling.BILINEAR)
            label = label.resize((width, height), Image.Resampling.NEAREST)
        return image_to_tensor(image), torch.from_numpy(np.asarray(label, dtype=np.int64).copy())


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32)
    mean = np.asarray((123.675, 116.28, 103.53), dtype=np.float32)
    std = np.asarray((58.395, 57.12, 57.375), dtype=np.float32)
    array = (array - mean) / std
    return torch.from_numpy(array.transpose(2, 0, 1).copy())


def _present_classes(label: torch.Tensor, allowed: Iterable[int]) -> set[int]:
    present = set(int(item) for item in torch.unique(label).tolist())
    return present.intersection(allowed)


def remap_pair_labels(
    source_label: torch.Tensor,
    support_label: torch.Tensor,
    shared_classes: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    source_out = torch.full_like(source_label, IGNORE_LABEL)
    support_out = torch.full_like(support_label, IGNORE_LABEL)
    for local_id, canonical_id in enumerate(shared_classes):
        source_out[source_label == canonical_id] = local_id
        support_out[support_label == canonical_id] = local_id
    return source_out, support_out


class DIPPairDataset(Dataset):
    """Pair a labeled source query with a selected labeled target support."""

    def __init__(self, data_cfg: dict, task: str, max_iters: int, transform: JointTrainTransform):
        self.source = read_manifest(
            data_cfg["source_root"], data_cfg["source_list"], data_cfg["source_kind"]
        )
        self.support = read_manifest(
            data_cfg["support_root"], data_cfg["support_list"], data_cfg["support_kind"]
        )
        self.source_label_space = data_cfg.get("source_label_space", "trainid")
        self.support_label_space = data_cfg.get("support_label_space", "trainid")
        self.classes = canonical_classes(task)
        self.max_iters = int(max_iters)
        self.transform = transform
        self.max_pair_attempts = int(data_cfg.get("max_pair_attempts", 20))
        self.min_shared_classes = int(data_cfg.get("min_shared_classes", 2))

    def __len__(self) -> int:
        return self.max_iters

    def __getitem__(self, index: int) -> dict:
        del index
        source_record = random.choice(self.source)
        source_image, source_label = load_image_label(source_record, self.source_label_space)
        source_image, source_label = self.transform(source_image, source_label)
        source_classes = _present_classes(source_label, self.classes)

        best = None
        for _ in range(self.max_pair_attempts):
            support_record = random.choice(self.support)
            support_image, support_label = load_image_label(support_record, self.support_label_space)
            support_image, support_label = self.transform(support_image, support_label)
            shared = sorted(source_classes.intersection(_present_classes(support_label, self.classes)))
            if best is None or len(shared) > len(best[-1]):
                best = (support_record, support_image, support_label, shared)
            if len(shared) >= self.min_shared_classes:
                break

        if best is None or not best[-1]:
            raise RuntimeError(f"No shared class found for source sample {source_record.name}")
        support_record, support_image, support_label, shared = best
        source_pair, support_pair = remap_pair_labels(source_label, support_label, shared)
        return {
            "source_image": source_image,
            "source_label": source_pair,
            "support_image": support_image,
            "support_label": support_pair,
            "shared_classes": torch.tensor(shared, dtype=torch.long),
            "source_name": source_record.name,
            "support_name": support_record.name,
        }


class LabeledTargetDataset(Dataset):
    def __init__(self, root: str, manifest: str, kind: str, label_space: str, transform: EvalTransform):
        self.records = read_manifest(root, manifest, kind)
        self.label_space = label_space
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        image, label = load_image_label(record, self.label_space)
        image, label = self.transform(image, label)
        return {"image": image, "label": label, "name": record.name}


def single_item_collate(batch: list[dict]) -> dict:
    if len(batch) != 1:
        raise ValueError("DIP uses one source-support pair per step; set batch_size=1")
    return batch[0]


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
