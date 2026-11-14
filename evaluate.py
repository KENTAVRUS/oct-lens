import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score, f1_score, accuracy_score,
)
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
from itertools import cycle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(BASE_DIR, "OCT2017_", "test")
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pth")
HISTORY_PATH = os.path.join(BASE_DIR, "training_history.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "evaluation_results")

NUM_CLASSES = 4
IMG_SIZE = 224
BATCH_SIZE = 64
NUM_WORKERS = 8
CLASS_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]

SEED = 42
SPLIT_NAME = "тест, 968 снимков"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

eval_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def create_model(num_classes=NUM_CLASSES):
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


@torch.no_grad()
def collect(model, loader, device):
    feat_extractor = nn.Sequential(
        model.features, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()
    ).to(device).eval()

    all_labels, all_probs, all_feats = [], [], []
    for x, y in loader:
        x = x.to(device)
        feats = feat_extractor(x)
        logits = model.classifier(feats)
        probs = torch.softmax(logits, dim=1)
        all_labels.append(y.numpy())
        all_probs.append(probs.cpu().numpy())
        all_feats.append(feats.cpu().numpy())
    return (np.concatenate(all_labels), np.concatenate(all_probs),
            np.concatenate(all_feats))


def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=ax, linewidths=0.5, square=True)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j + 0.5, i + 0.72, f"({cm[i][j]})",
                    ha="center", va="center", fontsize=8, color="gray")
    ax.set_xlabel("Предсказанный класс", fontsize=12)
    ax.set_ylabel("Истинный класс", fontsize=12)
    ax.set_title(f"Матрица ошибок ({SPLIT_NAME})", fontsize=14, pad=15)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[OK] Матрица ошибок: {save_path}")


def plot_classification_report(y_true, y_pred, save_path):
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
    metrics = ["precision", "recall", "f1-score"]
    data = []
    for cls in CLASS_NAMES:
        data.append([report[cls][m] for m in metrics] + [report[cls]["support"]])
    for avg in ["accuracy", "macro avg", "weighted avg"]:
        if avg == "accuracy":
            acc = report["accuracy"]
            data.append([acc, acc, acc, sum(report[c]["support"] for c in CLASS_NAMES)])
        else:
            data.append([report[avg][m] for m in metrics] + [report[avg]["support"]])

    row_labels = CLASS_NAMES + ["accuracy", "macro avg", "weighted avg"]
    col_labels = ["Precision", "Recall", "F1-Score", "Support"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")
    ax.set_title(f"Classification Report ({SPLIT_NAME})", fontsize=16, pad=20, fontweight="bold")

    cell_colors = []
    for row in data:
        crow = []
        for i, val in enumerate(row):
            if i < 3:
                g = int(val * 200); r = int((1 - val) * 200)
                crow.append(f"#{r:02x}{g:02x}50")
            else:
                crow.append("#f0f0f0")
        cell_colors.append(crow)
    cell_text = [[f"{r[0]:.4f}", f"{r[1]:.4f}", f"{r[2]:.4f}", f"{int(r[3])}"] for r in data]

    table = ax.table(cellText=cell_text, rowLabels=row_labels, colLabels=col_labels,
                     cellColours=cell_colors, loc="center", cellLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(11); table.scale(1.2, 1.8)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[OK] Classification Report: {save_path}")


def plot_roc_curve(y_true, y_probs, save_path):
    y_bin = label_binarize(y_true, classes=range(NUM_CLASSES))
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(NUM_CLASSES):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), y_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(NUM_CLASSES)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(NUM_CLASSES):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= NUM_CLASSES
    fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.figure(figsize=(10, 8))
    plt.plot(fpr["micro"], tpr["micro"], label=f"Micro-average (AUC = {roc_auc['micro']:.3f})",
             color="deeppink", linestyle=":", linewidth=4)
    plt.plot(fpr["macro"], tpr["macro"], label=f"Macro-average (AUC = {roc_auc['macro']:.3f})",
             color="navy", linestyle=":", linewidth=4)
    colors = cycle(["aqua", "darkorange", "cornflowerblue", "green"])
    for i, color in zip(range(NUM_CLASSES), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f"{CLASS_NAMES[i]} (AUC = {roc_auc[i]:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0, 1]); plt.ylim([0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(f"ROC-кривые ({SPLIT_NAME})", fontsize=14)
    plt.legend(loc="lower right"); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[OK] ROC-кривые: {save_path}")


def plot_loss_accuracy(save_path):
    if not os.path.exists(HISTORY_PATH):
        print(f"[WARN] Нет {HISTORY_PATH} — пропускаю кривые обучения.")
        return
    with open(HISTORY_PATH, "r", encoding="utf-8") as f:
        history = json.load(f)
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#e74c3c")
    ax1.plot(epochs, history["val_loss"], "s-", label="Val Loss", color="#3498db")
    ax1.set_xlabel("Эпоха"); ax1.set_ylabel("Loss")
    ax1.set_title("Изменение Loss по эпохам"); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], "o-", label="Train Accuracy", color="#e74c3c")
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], "s-", label="Val Accuracy", color="#3498db")
    ax2.set_xlabel("Эпоха"); ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Изменение Accuracy по эпохам"); ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[OK] Кривые обучения: {save_path}")


def plot_pr_curves(y_true, y_probs, save_path):
    y_bin = label_binarize(y_true, classes=range(NUM_CLASSES))
    plt.figure(figsize=(10, 8))
    prec, rec, _ = precision_recall_curve(y_bin.ravel(), y_probs.ravel())
    ap_micro = average_precision_score(y_bin, y_probs, average="micro")
    plt.plot(rec, prec, color="deeppink", linestyle=":", linewidth=4,
             label=f"Micro-average (AP = {ap_micro:.3f})")
    colors = cycle(["aqua", "darkorange", "cornflowerblue", "green"])
    aps = []
    for i, color in zip(range(NUM_CLASSES), colors):
        prec, rec, _ = precision_recall_curve(y_bin[:, i], y_probs[:, i])
        ap = average_precision_score(y_bin[:, i], y_probs[:, i])
        aps.append(ap)
        plt.plot(rec, prec, color=color, lw=2, label=f"{CLASS_NAMES[i]} (AP = {ap:.3f})")
    plt.xlabel("Recall (Полнота)", fontsize=12)
    plt.ylabel("Precision (Точность)", fontsize=12)
    plt.title(f"Precision-Recall кривые ({SPLIT_NAME}, macro-AP = {np.mean(aps):.3f})", fontsize=14)
    plt.xlim([0, 1]); plt.ylim([0, 1.05])
    plt.legend(loc="lower left"); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[OK] PR-кривые: {save_path}")


def plot_tsne(features, labels, save_path):
    n = len(features)
    perplexity = min(30, max(5, n // 100))
    print(f"[INFO] t-SNE на {n} точках (perplexity={perplexity})...")
    emb = TSNE(n_components=2, perplexity=perplexity, init="pca",
               learning_rate="auto", random_state=SEED).fit_transform(features)
    plt.figure(figsize=(10, 8))
    palette = ["#e74c3c", "#3498db", "#f39c12", "#2ecc71"]
    for i in range(NUM_CLASSES):
        m = labels == i
        plt.scatter(emb[m, 0], emb[m, 1], s=12, alpha=0.6, color=palette[i], label=CLASS_NAMES[i])
    plt.title(f"t-SNE проекция признаков ({SPLIT_NAME})", fontsize=14)
    plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2")
    plt.legend(markerscale=2); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[OK] t-SNE: {save_path}")


def plot_grad_cam(model, dataset, device, save_path, preds, probs):
    target_layer = model.features[-1]
    activations, gradients = {}, {}
    h1 = target_layer.register_forward_hook(lambda m, i, o: activations.__setitem__("v", o.detach()))
    h2 = target_layer.register_full_backward_hook(lambda m, gi, go: gradients.__setitem__("v", go[0].detach()))

    labels_arr = np.array(dataset.targets)
    examples = []
    for c in range(NUM_CLASSES):
        pos_c = np.where(labels_arr == c)[0]
        correct = [k for k in pos_c if preds[k] == c]
        best_k = max(correct, key=lambda k: probs[k, c]) if correct else pos_c[0]
        examples.append(best_k)

    fig, axes = plt.subplots(2, NUM_CLASSES, figsize=(4 * NUM_CLASSES, 8))
    model.eval()
    for col, idx in enumerate(examples):
        img_tensor, label = dataset[idx]
        x = img_tensor.unsqueeze(0).to(device)
        x.requires_grad_(True)
        model.zero_grad()
        logits = model(x)
        pred = logits.argmax(1).item()
        logits[0, pred].backward()

        grad, act = gradients["v"], activations["v"]
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * act).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        img = img_tensor.cpu().numpy().transpose(1, 2, 0)
        img = np.clip(img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN), 0, 1)

        axes[0, col].imshow(img)
        axes[0, col].set_title(f"{CLASS_NAMES[label]}\n(предсказано: {CLASS_NAMES[pred]})", fontsize=11)
        axes[0, col].axis("off")
        axes[1, col].imshow(img); axes[1, col].imshow(cam, cmap="jet", alpha=0.5)
        axes[1, col].set_title("Grad-CAM (зоны внимания)", fontsize=10)
        axes[1, col].axis("off")

    h1.remove(); h2.remove()
    plt.suptitle("Grad-CAM: на какие зоны ОКТ-снимка смотрит модель", fontsize=15)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[OK] Grad-CAM: {save_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Устройство: {device}")

    model = create_model(NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.to(device).eval()

    test_ds = datasets.ImageFolder(TEST_DIR, transform=eval_transforms)
    print(f"[INFO] Тест: {len(test_ds)} снимков ({len(test_ds) // NUM_CLASSES}/класс)")
    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    labels, probs, feats = collect(model, loader, device)
    preds = probs.argmax(1)

    acc = accuracy_score(labels, preds) * 100
    macro_f1 = f1_score(labels, preds, average="macro")
    print("\n" + "=" * 55)
    print(f" МЕТРИКИ НА ТЕСТЕ ({len(labels)} снимков)")
    print("=" * 55)
    print(f"Accuracy:  {acc:.2f}%")
    print(f"Macro-F1:  {macro_f1:.4f}")
    print(classification_report(labels, preds, target_names=CLASS_NAMES, digits=4))

    plot_confusion_matrix(labels, preds, os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    plot_classification_report(labels, preds, os.path.join(OUTPUT_DIR, "classification_report.png"))
    plot_roc_curve(labels, probs, os.path.join(OUTPUT_DIR, "roc_auc_curve.png"))
    plot_loss_accuracy(os.path.join(OUTPUT_DIR, "loss_accuracy.png"))
    plot_pr_curves(labels, probs, os.path.join(OUTPUT_DIR, "precision_recall.png"))
    plot_tsne(feats, labels, os.path.join(OUTPUT_DIR, "tsne_features.png"))
    plot_grad_cam(model, test_ds, device, os.path.join(OUTPUT_DIR, "grad_cam.png"), preds, probs)

    print(f"\n[OK] Все 7 графиков (на тесте) в: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
