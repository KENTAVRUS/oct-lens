import os
import sys
import json
import time
import copy
import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "OCT2017_")
TRAIN_DIR = os.path.join(DATA_DIR, "train")

NUM_CLASSES = 4
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
IMG_SIZE = 224
NUM_WORKERS = 8
EARLY_STOP_PATIENCE = 5
VAL_PER_CLASS = 500
VAL_SEED = 42

CLASS_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]

MODEL_SAVE_PATH = os.path.join(BASE_DIR, "best_model.pth")
HISTORY_SAVE_PATH = os.path.join(BASE_DIR, "training_history.json")


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ColorJitter(brightness=0.25, contrast=0.25),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def create_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False

    for param in model.features[-4:].parameters():
        param.requires_grad = True

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )

    return model


def compute_balanced_weights(targets, num_classes: int, device) -> torch.Tensor:
    targets = np.asarray(targets)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes),
        y=targets,
    )
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"[INFO] Используется устройство: {device}")
    if device.type == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    else:
        print("[WARN] CUDA недоступна — обучение будет медленным (CPU).")

    print("[INFO] Загрузка датасетов...")

    full_aug = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
    full_noaug = datasets.ImageFolder(TRAIN_DIR, transform=val_transforms)
    all_targets = np.array(full_aug.targets)

    rng = np.random.default_rng(VAL_SEED)
    val_idx, train_idx = [], []
    for c in range(NUM_CLASSES):
        cls_idx = np.where(all_targets == c)[0]
        rng.shuffle(cls_idx)
        val_idx.extend(cls_idx[:VAL_PER_CLASS].tolist())
        train_idx.extend(cls_idx[VAL_PER_CLASS:].tolist())

    train_dataset = Subset(full_aug, train_idx)
    val_dataset = Subset(full_noaug, val_idx)
    train_targets = all_targets[train_idx]

    print(f"[INFO] Train: {len(train_dataset)} изображений")
    print(f"[INFO] Val:   {len(val_dataset)} изображений "
          f"({VAL_PER_CLASS}/класс, сбалансированно)")
    print(f"[INFO] Классы: {full_aug.classes}")

    counts = np.bincount(train_targets, minlength=NUM_CLASSES)
    for name, c in zip(CLASS_NAMES, counts):
        print(f"        {name}: {c}")

    loader_kwargs = dict(
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
        pin_memory=use_amp, persistent_workers=NUM_WORKERS > 0,
    )
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    model = create_model(NUM_CLASSES).to(device)

    class_weights = compute_balanced_weights(train_targets, NUM_CLASSES, device)
    print(f"[INFO] Веса классов (balanced): "
          f"{ {n: round(w, 3) for n, w in zip(CLASS_NAMES, class_weights.tolist())} }")
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=LABEL_SMOOTHING
    )

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(
        trainable_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [], "val_f1": [],
    }

    best_val_f1 = -1.0
    best_val_loss = float("inf")
    best_model_weights = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    print(f"\n{'='*60}")
    print(f" Начало обучения: {NUM_EPOCHS} эпох")
    print(f"{'='*60}\n")

    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()

        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels).item()
            total_samples += inputs.size(0)

            if (batch_idx + 1) % 100 == 0:
                batch_acc = running_corrects / total_samples * 100
                print(f"  Эпоха {epoch+1}/{NUM_EPOCHS} | "
                      f"Батч {batch_idx+1}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f} | Acc: {batch_acc:.1f}%")

        train_loss = running_loss / total_samples
        train_acc = running_corrects / total_samples

        model.eval()
        val_running_loss = 0.0
        val_total = 0
        val_preds, val_labels = [], []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels_dev = labels.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels_dev)

                _, preds = torch.max(outputs, 1)
                val_running_loss += loss.item() * inputs.size(0)
                val_total += inputs.size(0)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.numpy())

        val_loss = val_running_loss / val_total
        val_acc = np.mean(np.array(val_preds) == np.array(val_labels))
        val_f1 = f1_score(val_labels, val_preds, average="macro")

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        epoch_time = time.time() - epoch_start

        print(f"\n  Эпоха [{epoch+1}/{NUM_EPOCHS}] — {epoch_time:.0f}с")
        print(f"    Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"    Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc*100:.2f}% | "
              f"Val macro-F1: {val_f1:.4f}")

        improved = (val_f1 > best_val_f1) or \
                   (val_f1 == best_val_f1 and val_loss < best_val_loss)
        if improved:
            best_val_f1 = val_f1
            best_val_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            torch.save(best_model_weights, MODEL_SAVE_PATH)
            print(f"    ✓ Лучшая модель сохранена "
                  f"(Val F1: {val_f1:.4f}, Val Loss: {val_loss:.4f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"    · Без улучшения ({epochs_no_improve}/{EARLY_STOP_PATIENCE})")

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"\n[INFO] Early stopping: macro-F1 не растёт "
                  f"{EARLY_STOP_PATIENCE} эпох подряд.")
            break

        print()

    print(f"{'='*60}")
    print(f" Обучение завершено!")
    print(f" Лучший Val macro-F1: {best_val_f1:.4f}")
    print(f" Модель сохранена: {MODEL_SAVE_PATH}")
    print(f"{'='*60}")

    with open(HISTORY_SAVE_PATH, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[INFO] История обучения сохранена: {HISTORY_SAVE_PATH}")


if __name__ == "__main__":
    train_model()
