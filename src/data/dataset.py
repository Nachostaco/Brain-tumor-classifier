from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch
import numpy as np


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(image_size: int, train: bool = True):
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (image_size, image_size), scale=(0.85, 1.0)
                ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=10),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_dataloaders(
    train_dir: str,
    test_dir: str,
    image_size: int = 224,
    batch_size: int = 32,
    val_fraction: float = 0.15,
    num_workers: int = 4,
):
    train_transform = build_transforms(image_size)
    test_transform = build_transforms(image_size, train=False)

    train_dataset = ImageFolder(train_dir, transform=train_transform)
    train_dataset_eval = ImageFolder(train_dir, transform=test_transform)
    test_dataset = ImageFolder(test_dir, transform=test_transform)

    n_val = int(len(train_dataset) * val_fraction)
    n_train = len(train_dataset) - n_val

    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = random_split(
        train_dataset, [n_train, n_val], generator=generator
    )

    val_subset = torch.utils.data.Subset(train_dataset_eval, val_subset.indices)

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader, train_dataset.class_to_idx
