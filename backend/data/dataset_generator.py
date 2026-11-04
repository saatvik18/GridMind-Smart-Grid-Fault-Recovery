"""Synthetic voltage / current / frequency signal generator (Module 1)."""
import json
import time

import numpy as np

from backend import config as C


def time_axis() -> np.ndarray:
    return np.arange(int(C.FS * C.DURATION)) / C.FS


def generate_signal(fault_type: str, rng: np.random.Generator | None = None) -> dict:
    """Generate one 0.2 s, 1 kHz waveform set for the given operating condition."""
    if fault_type not in C.CLASSES:
        raise ValueError(f"Unknown fault type '{fault_type}'. Choose from {C.CLASSES}")
    rng = rng or np.random.default_rng()
    t = time_axis()
    n = len(t)
    v_pk = C.V_RMS * np.sqrt(2)
    i_pk = C.I_RMS * np.sqrt(2)

    freq = C.F_NOMINAL + rng.normal(0, 0.01, n)
    v_env = np.full(n, rng.uniform(0.98, 1.02))
    i_env = np.full(n, rng.uniform(0.9, 1.1))
    pf_angle = rng.uniform(0.2, 0.5)
    v_noise, i_noise = 0.01, 0.02
    hv, hi = 0.0, rng.uniform(0.0, 0.02)
    i_dc = np.zeros(n)

    start = rng.uniform(0.03, 0.08)
    dur = rng.uniform(0.06, 0.11)
    win = (t >= start) & (t < start + dur)
    dt = np.clip(t - start, 0, None)

    if fault_type == "Voltage Sag":
        v_env[win] *= rng.uniform(0.4, 0.8)
        i_env[win] *= rng.uniform(0.85, 1.0)
    elif fault_type == "Overload":
        i_env *= rng.uniform(1.6, 2.5)               # sustained over-current
        v_env *= rng.uniform(0.92, 0.96)             # feeder voltage drop
        freq -= rng.uniform(0.1, 0.3)                # generation/load imbalance
        hi = rng.uniform(0.03, 0.08)
    elif fault_type == "Fault":
        v_env[win] *= rng.uniform(0.05, 0.35)        # severe voltage collapse
        i_env[win] *= rng.uniform(4.0, 8.0)          # short-circuit current
        tau = rng.uniform(0.01, 0.03)
        i_dc = np.where(win, rng.choice([-1, 1]) * i_pk * rng.uniform(1, 3) * np.exp(-dt / tau), 0.0)
        osc = rng.uniform(0.5, 1.5) * np.sin(2 * np.pi * rng.uniform(5, 15) * dt)
        freq = np.where(win, freq + osc, freq)
        hv = rng.uniform(0.1, 0.25)
        v_noise = 0.04

    phase = 2 * np.pi * np.cumsum(freq) / C.FS
    v = v_pk * v_env * (np.sin(phase) + hv * (0.6 * np.sin(3 * phase) + 0.4 * np.sin(5 * phase)))
    v += v_noise * v_pk * rng.standard_normal(n)
    if fault_type == "Fault":
        v += np.where(win, 0.08 * v_pk * rng.standard_normal(n), 0.0)   # arcing transients
    i = i_pk * i_env * (np.sin(phase - pf_angle) + hi * (0.7 * np.sin(3 * phase) + 0.3 * np.sin(5 * phase)))
    i += i_dc + i_noise * i_pk * rng.standard_normal(n)

    return {
        "t": t,
        "voltage": v.astype(np.float32),
        "current": i.astype(np.float32),
        "frequency": freq.astype(np.float32),
        "label": C.CLASSES.index(fault_type),
        "fault_type": fault_type,
        "event_window": [float(start), float(min(start + dur, C.DURATION))],
    }


def generate_dataset(samples_per_class: int = C.SAMPLES_PER_CLASS, seed: int = C.SEED,
                     save: bool = True) -> dict:
    rng = np.random.default_rng(seed)
    V, I, F, y = [], [], [], []
    for label, cls in enumerate(C.CLASSES):
        for _ in range(samples_per_class):
            s = generate_signal(cls, rng)
            V.append(s["voltage"]); I.append(s["current"]); F.append(s["frequency"]); y.append(label)
    data = {"voltage": np.stack(V), "current": np.stack(I), "frequency": np.stack(F),
            "labels": np.array(y, dtype=np.int64)}
    if save:
        np.savez_compressed(C.DATASET_PATH, **data)
        meta = {
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "classes": C.CLASSES,
            "samples_per_class": samples_per_class,
            "total_samples": int(len(y)),
            "sampling_frequency_hz": C.FS,
            "duration_s": C.DURATION,
            "samples_per_signal": int(C.FS * C.DURATION),
            "nominal_frequency_hz": C.F_NOMINAL,
            "nominal_voltage_rms": C.V_RMS,
            "nominal_current_rms": C.I_RMS,
            "seed": seed,
        }
        C.DATASET_META_PATH.write_text(json.dumps(meta, indent=2))
        save_sample_plots(seed)
    return data


def load_dataset() -> dict:
    if not C.DATASET_PATH.exists():
        return generate_dataset()
    d = np.load(C.DATASET_PATH)
    return {k: d[k] for k in d.files}


def save_sample_plots(seed: int = C.SEED) -> list[str]:
    """Save one representative waveform plot and spectrogram per class."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from backend.cnn.spectrogram import save_spectrogram_image

    rng = np.random.default_rng(seed + 1)
    paths = []
    for cls in C.CLASSES:
        s = generate_signal(cls, rng)
        fig, ax = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
        ax[0].plot(s["t"], s["voltage"], color="#1f77b4"); ax[0].set_ylabel("Voltage (V)")
        ax[1].plot(s["t"], s["current"], color="#d62728"); ax[1].set_ylabel("Current (A)")
        ax[2].plot(s["t"], s["frequency"], color="#2ca02c"); ax[2].set_ylabel("Freq (Hz)")
        ax[2].set_xlabel("Time (s)")
        fig.suptitle(f"Sample waveform - {cls}")
        fig.tight_layout()
        slug = cls.lower().replace(" ", "_")
        p = C.PLOTS_DIR / f"waveform_{slug}.png"
        fig.savefig(p, dpi=90); plt.close(fig)
        save_spectrogram_image(s["voltage"], s["current"], C.PLOTS_DIR / f"spectrogram_{slug}.png", cls)
        paths.append(str(p))
    return paths


if __name__ == "__main__":
    d = generate_dataset()
    print(f"Generated dataset: {d['labels'].shape[0]} samples, signal length {d['voltage'].shape[1]}")
    print(f"Saved to {C.DATASET_PATH}")
