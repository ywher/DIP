"""DINOv3-B + ReIN + HRDA embedding encoder for DIP."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .vfm_impl import HRDAEncoderDecoder, HRDAHead, ReinsDINOv3


def _backbone_config() -> dict:
    return {
        "dinov3_config": {
            "img_size": 512,
            "patch_size": 16,
            "pos_embed_rope_rescale_coords": 2.0,
            "pos_embed_rope_dtype": "fp32",
            "embed_dim": 768,
            "depth": 12,
            "num_heads": 12,
            "ffn_ratio": 4.0,
            "qkv_bias": True,
            "layerscale_init": 1e-5,
            "ffn_layer": "mlp",
            "ffn_bias": True,
            "proj_bias": True,
            "n_storage_tokens": 4,
            "mask_k_bias": True,
            "out_indices": [2, 5, 8, 11],
        },
        "reins_config": {
            "lora_dim": 16,
            "num_layers": 12,
            "non_adapter_layers": 0,
            "embed_dims": 768,
            "patch_size": 16,
            "token_length": 100,
            "link_token_to_query": False,
        },
    }


def _decoder_config(feature_dim: int, crop_size: tuple[int, int]) -> dict:
    return {
        "in_channels": [768, 768, 768, 768],
        "in_index": [0, 1, 2, 3],
        "channels": 256,
        "dropout_ratio": 0.1,
        "num_classes": feature_dim,
        "norm_cfg": {"type": "BN", "requires_grad": True},
        "align_corners": False,
        "loss_decode": {
            "type": "CrossEntropyLoss",
            "use_sigmoid": False,
            "loss_weight": 1.0,
        },
        "single_scale_head": "DAFormerHead",
        "interpolate": False,
        "decoder_params": {
            "embed_dims": 256,
            "embed_cfg": {"type": "mlp", "act_cfg": None, "norm_cfg": None},
            "embed_neck_cfg": {"type": "mlp", "act_cfg": None, "norm_cfg": None},
            "fusion_cfg": {
                "type": "aspp",
                "sep": True,
                "dilations": [1, 6, 12, 18],
                "pool": False,
                "act_cfg": {"type": "ReLU"},
                "norm_cfg": {"type": "BN", "requires_grad": True},
            },
        },
        "lr_loss_weight": 0,
        "hr_loss_weight": 0,
        "scales": [0.5, 1.0],
        "attention_embed_dim": 256,
        # A scalar gate fuses semantic embeddings before class prototypes exist.
        "attention_classwise": False,
        "enable_hr_crop": True,
        "hr_crop_size": list(crop_size),
        "hr_slide_inference": True,
        "hr_slide_overlapping": True,
        "hr_slide_batch_size": 4,
        "crop_coord_divisible": 8,
        "blur_hr_crop": False,
        "feature_scale": 0.5,
        "fixed_attention": None,
        "debug_output_attention": False,
    }


class VFMHRDADIPEncoder(nn.Module):
    """Use HRDA's fused output as a dense embedding, not class logits."""

    feature_dim = 256

    def __init__(self, pretrained: str, crop_size=(512, 512), init: str | None = None):
        super().__init__()
        pretrained_path = Path(pretrained).expanduser().resolve()
        if not pretrained_path.is_file():
            raise FileNotFoundError(f"DINOv3-B checkpoint not found: {pretrained_path}")
        backbone = ReinsDINOv3(
            backbone_config=_backbone_config(),
            pretrained={"dinov3": str(pretrained_path)},
        )
        decode_head = HRDAHead(_decoder_config(self.feature_dim, tuple(crop_size)))
        self.segmentor = HRDAEncoderDecoder(
            backbone=backbone,
            decode_head=decode_head,
            auxiliary_head=None,
            token_mask_ratio=None,
            train_cfg={},
            test_cfg={
                "mode": "slide",
                "stride": list(crop_size),
                "crop_size": [int(crop_size[0] * 2), int(crop_size[1] * 2)],
                "batched_slide": False,
            },
        )
        if init:
            checkpoint = torch.load(Path(init).expanduser(), map_location="cpu")
            self.load_checkpoint_state(checkpoint.get("model", checkpoint))

    def extract_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        if self.training:
            try:
                features, _ = self.segmentor._forward_train_features(images)
                output = self.segmentor.decode_head.forward_train(features)
                return output[0] if isinstance(output, (tuple, list)) else output
            finally:
                self.segmentor.decode_head.reset_crop()
        self.segmentor.decode_head.reset_crop()
        output = self.segmentor.encode_decode(images, upscale_pred=False)
        return output["seg_logits"]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.extract_embeddings(images)

    def checkpoint_state(self) -> dict:
        return {
            "adapter": self.segmentor.backbone.adapter.state_dict(),
            "decoder": self.segmentor.decode_head.state_dict(),
        }

    def load_checkpoint_state(self, state: dict) -> None:
        self.segmentor.backbone.adapter.load_state_dict(state["adapter"], strict=True)
        self.segmentor.decode_head.load_state_dict(state["decoder"], strict=True)


class WarmupPolyLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, max_iters: int, warmup_iters: int = 1500, power: float = 1.0):
        self.max_iters = int(max_iters)
        self.warmup_iters = int(warmup_iters)
        self.power = float(power)
        super().__init__(optimizer)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_iters:
            ratio = 1e-6 + (1.0 - 1e-6) * step / max(1, self.warmup_iters)
        else:
            progress = (step - self.warmup_iters) / max(1, self.max_iters - self.warmup_iters)
            ratio = max(0.0, 1.0 - progress) ** self.power
        return [base_lr * ratio for base_lr in self.base_lrs]


def build_vfm_optimizer(model: VFMHRDADIPEncoder, base_lr: float = 1e-4):
    groups = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        groups.append(
            {
                "params": [parameter],
                "lr": base_lr * (10.0 if "decode_head" in name else 1.0),
                "weight_decay": 0.0 if parameter.ndim == 1 else 0.05,
            }
        )
    return torch.optim.AdamW(groups, lr=base_lr, betas=(0.9, 0.999), weight_decay=0.05)

