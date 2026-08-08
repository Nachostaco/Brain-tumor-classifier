from __future__ import annotations

import timm
import torch
import torch.nn as nn
import os


def build_model(num_classes: int, backbone_name: str = "resnet18") -> nn.Module:
    model = timm.create_model(
        backbone_name,
        pretrained=True,
        num_classes=num_classes,
    )

    return model


def count_trainable_params(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


if __name__ == "__main__":
    from src.data.dataset import build_dataloaders

    NUM_CLASSES = 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(os.getcwd())

    train_loader, val_loader, test_loader, class_to_idx = build_dataloaders(
        train_dir="src/data/brisc2025/classification_task/train/",
        test_dir="src/data/brisc2025/classification_task/test/",
    )

    print(f"Class map: {class_to_idx}")

    model = build_model(NUM_CLASSES).to(device)

    trainable, total = count_trainable_params(model)
    print(f"Trainable params {trainable}/{total}")

    images, labels = next(iter(train_loader))
    images = images.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(images)

    print(f"Output shape: {outputs.shape}")
