# TODO: Dataset PyTorch ładujący obrazy + etykiety (CSV: image_path, label)
# TODO: transformacje (augmentacje train, resize+normalize dla val/test)
# TODO: build_dataloaders() -> train/val/test split
#

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt

IMG_SIZE = 224

TRAIN_DIR = "src/data/brisc2025/classification_task/train/"
TEST_DIR = "src/data/brisc2025/classification_task/test/"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

BATCH_SIZE = 32


train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)

train_dataset = ImageFolder(TRAIN_DIR, transform=train_transform)
test_dataset = ImageFolder(TEST_DIR, transform=test_transform)

val_fraction = 0.15
n_val = int(len(train_dataset) * val_fraction)
n_train = len(train_dataset) - n_val

train_subset, val_subset = random_split(
    train_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
)

train_dataset_eval_transform = ImageFolder(TRAIN_DIR, transform=test_transform)
val_subset = torch.utils.data.Subset(train_dataset_eval_transform, val_subset.indices)

train_loader = DataLoader(
    train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
)

images, labels = next(iter(train_loader))

print(f"Images shape: {images.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Unique labels in batch: {labels.unique()}")
print(f"Pixels values range: {images.min().item()}, {images.max().item()}")


def denormalize(img_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    img = img_tensor.clone().numpy().transpose(1, 2, 0)
    mean = np.array(mean)
    std = np.array(std)
    img = img * std + mean
    return np.clip(img, 0, 1)


N = 8
fig, axes = plt.subplots(2, N // 2, figsize=(N * 2, 6))

for i, ax in enumerate(axes.flat):
    img = denormalize(images[i])
    ax.imshow(img)
    ax.axis("off")

plt.suptitle("Example batch after augmentation")
plt.tight_layout()
plt.savefig("batch.jpg")
