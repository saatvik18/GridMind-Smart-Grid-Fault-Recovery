"""CNN inference: predict_fault() and extract_cnn_features()."""
import threading

import numpy as np
import torch

from backend import config as C
from backend.cnn.model import FaultCNN
from backend.cnn.spectrogram import signal_to_spectrogram
from backend.data.dataset_generator import generate_signal

_model: FaultCNN | None = None
_lock = threading.Lock()


def load_cnn(force: bool = False) -> FaultCNN:
    global _model
    with _lock:
        if _model is None or force:
            if not C.CNN_MODEL_PATH.exists():
                raise FileNotFoundError("CNN model not trained yet. Run: python -m backend.cnn.train")
            ckpt = torch.load(C.CNN_MODEL_PATH, map_location="cpu")
            m = FaultCNN(feature_dim=ckpt.get("feature_dim", C.FEATURE_DIM))
            m.load_state_dict(ckpt["state_dict"])
            m.eval()
            _model = m
    return _model


def _tensor(spectrogram: np.ndarray) -> torch.Tensor:
    x = torch.as_tensor(np.asarray(spectrogram, dtype=np.float32))
    return x.unsqueeze(0) if x.dim() == 3 else x


def analyze(spectrogram: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Single forward pass -> (feature vector, class probabilities)."""
    model = load_cnn()
    with torch.no_grad():
        feats, logits = model.forward_with_features(_tensor(spectrogram))
        probs = torch.softmax(logits, dim=1)
    return feats[0].numpy(), probs[0].numpy()


def predict_fault(spectrogram: np.ndarray) -> dict:
    _, probs = analyze(spectrogram)
    idx = int(np.argmax(probs))
    return {
        "class_index": idx,
        "fault_class": C.CLASSES[idx],
        "confidence": float(probs[idx]),
        "probabilities": {c: float(p) for c, p in zip(C.CLASSES, probs)},
    }


def extract_cnn_features(spectrogram: np.ndarray) -> np.ndarray:
    """Return the intermediate FEATURE_DIM vector (before the classification layer)."""
    feats, _ = analyze(spectrogram)
    return feats


class FeatureBank:
    """Pool of CNN outputs on freshly generated (unseen) signals, used to drive DQN scenarios."""

    def __init__(self, per_class: int = 40, seed: int = C.SEED + 100):
        rng = np.random.default_rng(seed)
        self.bank: dict[str, list[dict]] = {}
        for cls in C.CLASSES:
            entries = []
            for _ in range(per_class):
                s = generate_signal(cls, rng)
                feats, probs = analyze(signal_to_spectrogram(s["voltage"], s["current"]))
                entries.append({"features": feats, "probs": probs, "predicted": C.CLASSES[int(np.argmax(probs))]})
            self.bank[cls] = entries

    def sample(self, fault_type: str, rng: np.random.Generator) -> dict:
        e = self.bank[fault_type][int(rng.integers(len(self.bank[fault_type])))]
        return {"features": e["features"].copy(), "probs": e["probs"].copy(), "predicted": e["predicted"]}
