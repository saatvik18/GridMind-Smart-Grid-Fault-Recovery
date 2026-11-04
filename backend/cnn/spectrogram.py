"""Spectrogram generation (Module 2): signals -> fixed (2, 32, 32) tensors."""
import base64
import io
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom
from scipy.signal import spectrogram

from backend import config as C

V_REF = C.V_RMS * np.sqrt(2)
I_REF = C.I_RMS * np.sqrt(2)
COMPRESS = 20.0   # log-compression gain for the CNN amplitude representation


def _power_spectrogram(x: np.ndarray, ref: float):
    """Power spectrogram of a per-unit signal (fixed reference keeps amplitude info)."""
    return spectrogram(np.asarray(x, dtype=np.float64) / ref, fs=C.FS,
                       nperseg=C.SPEC_NPERSEG, noverlap=C.SPEC_NOVERLAP,
                       scaling="spectrum", mode="psd")


def _raw_spectrogram(x: np.ndarray, ref: float):
    """Log-power spectrogram, used for display."""
    f, t, sxx = _power_spectrogram(x, ref)
    return f, t, np.log10(sxx + 1e-6)


def _to_fixed(sxx: np.ndarray) -> np.ndarray:
    # log-compressed amplitude: keeps sag depth / over-current magnitude visible to the CNN
    norm = np.log1p(COMPRESS * np.sqrt(sxx)) / np.log1p(COMPRESS)
    resized = zoom(norm, (C.SPEC_SIZE / norm.shape[0], C.SPEC_SIZE / norm.shape[1]), order=1)
    return resized[: C.SPEC_SIZE, : C.SPEC_SIZE].astype(np.float32)


def signal_to_spectrogram(voltage: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Return a (2, SPEC_SIZE, SPEC_SIZE) float32 array: [voltage, current] channels."""
    _, _, sv = _power_spectrogram(voltage, V_REF)
    _, _, si = _power_spectrogram(current, I_REF)
    return np.stack([_to_fixed(sv), _to_fixed(si)])


def batch_spectrograms(V: np.ndarray, I: np.ndarray) -> np.ndarray:
    return np.stack([signal_to_spectrogram(v, i) for v, i in zip(V, I)])


def _figure(voltage, current, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bg, fg, muted = "#fffcf4", "#2b231a", "#86765f"      # cream theme, matches frontend/style.css
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), facecolor=bg)
    for ax, sig, ref, name in ((axes[0], voltage, V_REF, "Voltage"), (axes[1], current, I_REF, "Current")):
        f, t, s = _raw_spectrogram(sig, ref)
        mesh = ax.pcolormesh(t * 1000, f, s, shading="gouraud", cmap="inferno", vmin=-6, vmax=1)
        ax.set_title(f"{name} spectrogram", color=fg, fontsize=10)
        ax.set_xlabel("Time (ms)", color=muted); ax.set_ylabel("Frequency (Hz)", color=muted)
        ax.tick_params(colors=muted); ax.set_facecolor(bg)
        for spine in ax.spines.values():
            spine.set_color("#e2d4b4")
        cb = fig.colorbar(mesh, ax=ax); cb.ax.tick_params(colors=muted)
        cb.set_label("log10 power (p.u.)", color=muted)
    fig.suptitle(title, color=fg, fontsize=11)
    fig.tight_layout()
    return fig, plt


def save_spectrogram_image(voltage, current, path: Path, title: str = "") -> str:
    fig, plt = _figure(voltage, current, f"Spectrogram - {title}")
    fig.savefig(path, dpi=90, facecolor=fig.get_facecolor()); plt.close(fig)
    return str(path)


def spectrogram_image_base64(voltage, current, title: str = "") -> str:
    fig, plt = _figure(voltage, current, title)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, facecolor=fig.get_facecolor()); plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
