from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import wandb

from src.data.dataset import build_dataloaders
from src.models.small_model import build_model, count_trainable_params

NUM_CLASSES = 4
BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-4
EARLY_STOPING_PATIENCE = 5
OUTPUT_DIR = Path("outputs")
TRAIN_DIR = "src/data/brisc2025/classification_task/train/"
TEST_DIR = "src/data/brisc2025/classification_task/test/"


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, n = 0.0, 0, 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in tqdm(loader, leave=False):
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            n += images.size(0)

    return total_loss / n, correct / n


def save_checkpoint(path, model, optimizer, epoch, best_val_acc, patience_counter):
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_acc": best_val_acc,
            "patience_counter": patience_counter,
        },
        path,
    )


def load_checkpoint(path, model, optimizer, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    return (
        checkpoint["epoch"],
        checkpoint["best_val_acc"],
        checkpoint["patience_counter"],
    )


def main():
    from_checkpoint = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    run = wandb.init(
        entity="aincen-politechnika-l-ska",
        project="Brain_detector",
        config={
            "backbone": "resnet18",
            "method": "full",
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
        },
    )

    OUTPUT_DIR.mkdir(exist_ok=True)

    train_loader, val_loader, test_loader, class_to_idx = build_dataloaders(
        train_dir=TRAIN_DIR, test_dir=TEST_DIR, batch_size=BATCH_SIZE
    )

    print(f"Class mapping: {class_to_idx}")

    model = build_model(NUM_CLASSES).to(device)
    trainable, total = count_trainable_params(model)
    print(f"Trainable params: {trainable}/{total}")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LR)

    start_epoch = 0
    best_val_acc = 0.0
    patience_counter = 0

    last_checkpoint_path = OUTPUT_DIR / "checkpoint_last.pt"
    best_checkpoint_path = OUTPUT_DIR / "best_model.pt"

    if last_checkpoint_path.exists() and from_checkpoint:
        start_epoch, best_val_acc, patience_counter = load_checkpoint(
            last_checkpoint_path, model, optimizer, device
        )
        start_epoch += 1
        print(f"Return to training since epoch: {start_epoch}")

    for epoch in range(start_epoch, EPOCHS):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False
        )

        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"train loss = {train_loss:.4f} train acc = {train_acc:.4f} | "
            f"val loss = {val_loss:.4f} val acc = {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f" New best model saved! val acc = {val_acc:.4f}")
        else:
            patience_counter += 1

        save_checkpoint(
            last_checkpoint_path,
            model,
            optimizer,
            epoch,
            best_val_acc,
            patience_counter,
        )

        if patience_counter >= EARLY_STOPING_PATIENCE:
            print(f"Early stopping, last update {patience_counter} epochs before")

    wandb.finish()
    print(f"Training complete: val accuracy = {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
