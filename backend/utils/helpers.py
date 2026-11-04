"""Shared helpers: signal statistics and JSON-safe conversion."""
import json
import time
from pathlib import Path

import numpy as np

from backend import config as C


def cycle_rms(x: np.ndarray) -> np.ndarray:
    n = int(C.FS / C.F_NOMINAL)
    k = len(x) // n
    return np.sqrt(np.mean(np.asarray(x[: k * n], dtype=np.float64).reshape(k, n) ** 2, axis=1))


def signal_stats(signal: dict) -> dict:
    v_rms, i_rms = cycle_rms(signal["voltage"]), cycle_rms(signal["current"])
    f = np.asarray(signal["frequency"])
    v_min_pu, i_max_pu = float(v_rms.min() / C.V_RMS), float(i_rms.max() / C.I_RMS)
    f_dev = float(np.max(np.abs(f - C.F_NOMINAL)))
    v_status = "Normal" if v_min_pu >= 0.9 else "Sag" if v_min_pu >= 0.4 else "Severe collapse"
    i_status = "Normal" if i_max_pu <= 1.2 else "Overcurrent" if i_max_pu <= 3.0 else "Fault current"
    f_status = "Normal" if f_dev <= 0.1 else "Deviation" if f_dev <= 0.5 else "Severe deviation"
    return {
        "voltage_rms_min": float(v_rms.min()), "voltage_rms_max": float(v_rms.max()), "voltage_min_pu": v_min_pu,
        "current_rms_max": float(i_rms.max()), "current_max_pu": i_max_pu,
        "frequency_min": float(f.min()), "frequency_max": float(f.max()), "frequency_max_dev": f_dev,
        "voltage_status": f"{v_status} (min {v_rms.min():.1f} V RMS, {v_min_pu:.2f} pu)",
        "current_status": f"{i_status} (max {i_rms.max():.1f} A RMS, {i_max_pu:.2f} pu)",
        "frequency_status": f"{f_status} (range {f.min():.2f}-{f.max():.2f} Hz)",
    }


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def save_json(path: Path, data) -> str:
    path.write_text(json.dumps(to_jsonable(data), indent=2))
    return str(path)


def read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")
