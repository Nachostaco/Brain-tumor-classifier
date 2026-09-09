# TODO: build_model(config) -> backbone (timm) + tryb: full / lora / linear_probe
# LoRA: użyj biblioteki `peft` (LoraConfig, get_peft_model)
# TODO: count_trainable_params() do porównania metod

from __future__ import annotations

import timm
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model


def build_model(
    num_classes: int,
    backbone_name: str = "resnet18",
    method: str = "full",
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: list[str] | None = None,
    img_size: int = 224,
) -> nn.Module:
    create_kwargs = {"pretrained": True, "num_classes": num_classes}
    if "vit" in backbone_name or "dinov2" in backbone_name:
        create_kwargs["img_size"] = img_size
    backbone = timm.create_model(backbone_name, **create_kwargs)

    if method == "full":
        return backbone

    if method == "linear_probe":
        for name, param in backbone.named_parameters():
            if "head" not in name:
                param.requires_grad = False
        return backbone

    if method == "lora":
        target_modules = lora_target_modules or ["qkv", "proj"]
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
            modules_to_save=["head"],
        )
        return get_peft_model(backbone, lora_config)
    raise ValueError(f"Method is not valid: {method}")


def count_trainable_params(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


if __name__ == "__main__":
    backbone = build_model(
        backbone_name="vit_small_patch14_dinov2.lvd142m", method="lora", num_classes=4
    )
    for name, module in backbone.named_modules():
        if "qkv" in name or "proj" in name:
            print(name)
