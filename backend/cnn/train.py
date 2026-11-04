"""CNN training: train/val split, metrics, plots, checkpoint."""
import json
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from backend import config as C
from backend.cnn.model import FaultCNN
from backend.cnn.spectrogram import batch_spectrograms
from backend.data.dataset_generator import generate_dataset, load_dataset


def _plots(history: dict, cm: np.ndarray):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = range(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(ep, history["train_loss"], label="train"); ax[0].plot(ep, history["val_loss"], label="val")
    ax[0].set_title("CNN loss"); ax[0].set_xlabel("Epoch"); ax[0].legend()
    ax[1].plot(ep, history["train_acc"], label="train"); ax[1].plot(ep, history["val_acc"], label="val")
    ax[1].set_title("CNN accuracy"); ax[1].set_xlabel("Epoch"); ax[1].legend()
    fig.tight_layout(); fig.savefig(C.PLOTS_DIR / "cnn_training_curves.png", dpi=90); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4.4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(C.CLASSES)), C.CLASSES, rotation=30); ax.set_yticks(range(len(C.CLASSES)), C.CLASSES)
    for r in range(cm.shape[0]):
        for c in range(cm.shape[1]):
            ax.text(c, r, int(cm[r, c]), ha="center", va="center",
                    color="white" if cm[r, c] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("CNN confusion matrix (validation)")
    fig.tight_layout(); fig.savefig(C.PLOTS_DIR / "cnn_confusion_matrix.png", dpi=90); plt.close(fig)


def train_cnn(epochs: int = C.CNN_EPOCHS, regenerate: bool = False, seed: int = C.SEED) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    t0 = time.time()
    data = generate_dataset() if regenerate else load_dataset()
    X = batch_spectrograms(data["voltage"], data["current"])
    y = data["labels"]
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    Xtr, Xva = torch.tensor(Xtr), torch.tensor(Xva)
    ytr, yva = torch.tensor(ytr), torch.tensor(yva)

    model = FaultCNN()
    opt = torch.optim.Adam(model.parameters(), lr=C.CNN_LR)
    loss_fn = nn.CrossEntropyLoss()
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc, best_state, best_epoch = -1.0, None, 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        tot_loss, correct = 0.0, 0
        for b in range(0, len(Xtr), C.CNN_BATCH):
            idx = perm[b:b + C.CNN_BATCH]
            logits = model(Xtr[idx])
            loss = loss_fn(logits, ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot_loss += loss.item() * len(idx)
            correct += (logits.argmax(1) == ytr[idx]).sum().item()
        model.eval()
        with torch.no_grad():
            vlogits = model(Xva)
            vloss = loss_fn(vlogits, yva).item()
            vacc = (vlogits.argmax(1) == yva).float().mean().item()
        history["train_loss"].append(tot_loss / len(Xtr))
        history["train_acc"].append(correct / len(Xtr))
        history["val_loss"].append(vloss)
        history["val_acc"].append(vacc)
        if vacc >= best_acc:     # keep the best validation checkpoint
            best_acc, best_epoch = vacc, epoch + 1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"[CNN] epoch {epoch + 1:02d}/{epochs} loss={history['train_loss'][-1]:.4f} "
              f"train_acc={history['train_acc'][-1]:.3f} val_loss={vloss:.4f} val_acc={vacc:.3f}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds = model(Xva).argmax(1).numpy()
    cm = confusion_matrix(yva.numpy(), preds, labels=list(range(len(C.CLASSES))))
    report = classification_report(yva.numpy(), preds, target_names=C.CLASSES, output_dict=True, zero_division=0)

    torch.save({"state_dict": model.state_dict(), "classes": C.CLASSES, "feature_dim": C.FEATURE_DIM,
                "input_shape": [2, C.SPEC_SIZE, C.SPEC_SIZE]}, C.CNN_MODEL_PATH)
    _plots(history, cm)

    metrics = {
        "epochs": epochs,
        "train_samples": int(len(Xtr)),
        "val_samples": int(len(Xva)),
        "final_train_accuracy": history["train_acc"][-1],
        "final_val_accuracy": best_acc,
        "best_epoch": best_epoch,
        "history": history,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "classes": C.CLASSES,
        "training_time_s": round(time.time() - t0, 2),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    C.CNN_METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"[CNN] saved model -> {C.CNN_MODEL_PATH}  (val acc {metrics['final_val_accuracy']:.3f})")
    return metrics


if __name__ == "__main__":
    train_cnn()
