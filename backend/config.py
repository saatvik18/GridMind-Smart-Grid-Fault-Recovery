"""Central configuration for the smart grid recovery project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
LOGS_DIR = RESULTS_DIR / "logs"
PLOTS_DIR = RESULTS_DIR / "plots"
REPORTS_DIR = RESULTS_DIR / "reports"
FRONTEND_DIR = ROOT / "frontend"

for _d in (MODELS_DIR, DATA_DIR, LOGS_DIR, PLOTS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42

# ---- Signal generation ----
FS = 1000             # Hz
DURATION = 0.2        # s
F_NOMINAL = 50.0      # Hz
V_RMS = 230.0         # V
I_RMS = 10.0          # A
CLASSES = ["Normal", "Voltage Sag", "Overload", "Fault"]
SAMPLES_PER_CLASS = 150   # 600 samples total
DATASET_PATH = DATA_DIR / "dataset.npz"
DATASET_META_PATH = DATA_DIR / "dataset_metadata.json"

# ---- Spectrogram ----
SPEC_NPERSEG = 32
SPEC_NOVERLAP = 28
SPEC_SIZE = 32        # CNN input is (2, 32, 32): voltage + current channels

# ---- CNN ----
CNN_EPOCHS = 40
CNN_BATCH = 32
CNN_LR = 3e-3
FEATURE_DIM = 32
CNN_MODEL_PATH = MODELS_DIR / "cnn_fault_detector.pth"
CNN_METRICS_PATH = RESULTS_DIR / "cnn_metrics.json"

# ---- DQN ----
DQN_EPISODES = 300
DQN_GAMMA = 0.95
DQN_LR = 5e-4
DQN_BATCH = 64
DQN_BUFFER = 10000
DQN_HIDDEN = 128
DQN_EPS_START = 1.0
DQN_EPS_END = 0.05
DQN_EPS_DECAY = 0.98      # multiplicative, per episode
DQN_TARGET_UPDATE = 10    # episodes
DQN_MAX_STEPS = 12
DQN_MODEL_PATH = MODELS_DIR / "dqn_agent.pth"
DQN_TRAINING_PATH = RESULTS_DIR / "dqn_training.json"
EVALUATION_PATH = RESULTS_DIR / "evaluation.json"
EVAL_EPISODES = 100

# ---- Ollama ----
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2"
OLLAMA_TIMEOUT = 300      # s (CPU inference measured at 110-180 s on this machine)
OLLAMA_KEEP_ALIVE = "30m" # keep llama3.2 loaded between reports

# ---- Server ----
HOST = "127.0.0.1"
PORT = 8765          # override with: python run.py --port <n>
