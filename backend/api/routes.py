"""REST API routes (Module 9)."""
import threading

import numpy as np
from fastapi import APIRouter, HTTPException

from backend import config as C
from backend.api.schemas import (ClassifyRequest, ClassifyResponse, EvaluateRequest, GenericResponse,
                                 HealthResponse, LLMRequest, LLMResponse, RestoreResponse, SimulateRequest,
                                 SimulateResponse, TrainCNNRequest, TrainDQNRequest)
from backend.cnn import predict as cnn_predict
from backend.cnn.spectrogram import signal_to_spectrogram, spectrogram_image_base64
from backend.data.dataset_generator import generate_signal
from backend.drl.dqn_agent import DQNAgent
from backend.drl.environment import SECTIONS, GridEnvironment
from backend.evaluation.baseline import RuleBasedAgent
from backend.evaluation.metrics import evaluate_agents, run_episode
from backend.llm.ollama_client import OllamaError, check_ollama, generate_explanation
from backend.utils.helpers import read_json, save_json, signal_stats, timestamp, to_jsonable

router = APIRouter(prefix="/api")


class AppState:
    def __init__(self):
        self.lock = threading.Lock()
        self.rng = np.random.default_rng()
        self.signal = None
        self.stats = None
        self.classification = None
        self.features = None
        self.scenario = None
        self.env = GridEnvironment()          # scenario loaded on first simulate
        self.env_ready = False
        self.agent = None
        self.bank = None
        self.last_restoration = None
        self.last_baseline = None

    def get_agent(self) -> DQNAgent:
        if self.agent is None:
            if not C.DQN_MODEL_PATH.exists():
                raise HTTPException(409, "DQN agent not trained. Call POST /api/train/dqn first.")
            self.agent = DQNAgent.from_checkpoint()
        return self.agent

    def get_bank(self):
        if self.bank is None:
            self.bank = cnn_predict.FeatureBank()
        return self.bank


S = AppState()


def _require_cnn():
    if not C.CNN_MODEL_PATH.exists():
        raise HTTPException(409, "CNN not trained. Call POST /api/train/cnn first.")


def _classify(signal: dict):
    spec = signal_to_spectrogram(signal["voltage"], signal["current"])
    feats, probs = cnn_predict.analyze(spec)
    idx = int(np.argmax(probs))
    cls = {"class_index": idx, "fault_class": C.CLASSES[idx], "confidence": float(probs[idx]),
           "probabilities": {c: float(p) for c, p in zip(C.CLASSES, probs)}}
    return cls, feats, probs


def _simulate(fault_type: str, section: int | None, seed: int | None):
    _require_cnn()
    rng = np.random.default_rng(seed) if seed is not None else S.rng
    signal = generate_signal(fault_type, rng)
    cls, feats, probs = _classify(signal)
    sec_idx = (section - 1) if section else int(rng.integers(1, len(SECTIONS)))
    env = GridEnvironment(feature_bank=None, seed=seed)
    scenario = env.make_scenario(fault_type, sec_idx, cnn={"features": feats, "probs": probs,
                                                          "predicted": cls["fault_class"]})
    env.reset(scenario)
    S.signal, S.stats, S.classification, S.features = signal, signal_stats(signal), cls, feats
    S.scenario, S.env, S.env_ready = scenario, env, True
    S.last_restoration = S.last_baseline = None
    return signal, cls, feats


def _ensure_scenario():
    if not S.env_ready:
        _simulate("Fault", None, None)


@router.get("/health", response_model=HealthResponse)
def health():
    return {"backend": "ok", "cnn_model": C.CNN_MODEL_PATH.exists(), "dqn_model": C.DQN_MODEL_PATH.exists(),
            "ollama": check_ollama()}


@router.post("/simulate", response_model=SimulateResponse)
def simulate(req: SimulateRequest):
    with S.lock:
        signal, cls, feats = _simulate(req.fault_type, req.section, req.seed)
        title = f"{req.fault_type} - CNN: {cls['fault_class']} ({cls['confidence']:.1%})"
        return {
            "fault_type": req.fault_type,
            "section": None if S.scenario["section"] is None else S.scenario["section"] + 1,
            "waveform": {"t": signal["t"].tolist(), "voltage": signal["voltage"].tolist(),
                         "current": signal["current"].tolist(), "frequency": signal["frequency"].tolist()},
            "stats": S.stats,
            "spectrogram_image": spectrogram_image_base64(signal["voltage"], signal["current"], title),
            "classification": cls,
            "feature_vector": feats.tolist(),
            "network": S.env.get_network_status(),
        }


@router.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest):
    _require_cnn()
    if req.fault_type:
        rng = np.random.default_rng(req.seed) if req.seed is not None else S.rng
        signal = generate_signal(req.fault_type, rng)
    elif S.signal is not None:
        signal = S.signal
    else:
        raise HTTPException(400, "No current waveform. Provide fault_type or call /api/simulate first.")
    cls, feats, _ = _classify(signal)
    return {"true_class": signal["fault_type"], "classification": cls, "feature_vector": feats.tolist(),
            "correct": cls["fault_class"] == signal["fault_type"]}


@router.post("/train/cnn", response_model=GenericResponse)
def train_cnn_endpoint(req: TrainCNNRequest):
    from backend.cnn.train import train_cnn
    with S.lock:
        metrics = train_cnn(epochs=req.epochs, regenerate=req.regenerate_dataset)
        cnn_predict.load_cnn(force=True)
        S.bank = None           # CNN changed -> feature bank must be rebuilt
        S.agent = None
        return {"data": to_jsonable(metrics)}


@router.post("/train/dqn", response_model=GenericResponse)
def train_dqn_endpoint(req: TrainDQNRequest):
    _require_cnn()
    from backend.drl.train import train_dqn
    with S.lock:
        agent, log = train_dqn(episodes=req.episodes, feature_bank=S.get_bank())
        S.agent = agent
        out = {"training": log}
        if req.evaluate:
            out["evaluation"] = evaluate_agents(agent, RuleBasedAgent(), S.get_bank())
        return {"data": to_jsonable(out)}


@router.post("/evaluate", response_model=GenericResponse)
def evaluate_endpoint(req: EvaluateRequest):
    _require_cnn()
    with S.lock:
        return {"data": to_jsonable(evaluate_agents(S.get_agent(), RuleBasedAgent(), S.get_bank(),
                                                    n_episodes=req.episodes))}


def _run(agent, name: str, record_q: bool):
    _ensure_scenario()
    initial = GridEnvironment(scenario=S.scenario)
    res = run_episode(agent, S.env, S.scenario, record_q=record_q)
    summary = {k: v for k, v in res.items() if k != "trajectory"}
    out = {"agent": name, "scenario": {"fault_type": S.scenario["fault_type"],
                                       "section": None if S.scenario["section"] is None else S.scenario["section"] + 1,
                                       "cnn_prediction": S.classification},
           "trajectory": to_jsonable(res["trajectory"]), "summary": to_jsonable(summary),
           "initial_network": initial.get_network_status()}
    out["log_path"] = save_json(C.LOGS_DIR / f"restoration_{name.lower().replace(' ', '_')}_{timestamp()}.json", out)
    return out


@router.post("/restore", response_model=RestoreResponse)
def restore():
    with S.lock:
        S.last_restoration = _run(S.get_agent(), "DQN", record_q=True)
        return S.last_restoration


@router.post("/baseline", response_model=RestoreResponse)
def baseline():
    with S.lock:
        S.last_baseline = _run(RuleBasedAgent(), "Rule-Based", record_q=False)
        return S.last_baseline


@router.post("/reset", response_model=GenericResponse)
def reset_grid():
    with S.lock:
        _ensure_scenario()
        S.env.reset(S.scenario)
        S.last_restoration = None
        return {"data": S.env.get_network_status()}


@router.post("/llm/explain", response_model=LLMResponse)
def llm_explain(req: LLMRequest):
    with S.lock:
        _ensure_scenario()
        restoration = S.last_restoration or _run(S.get_agent(), "DQN", record_q=True)
        S.last_restoration = restoration
        net = restoration["trajectory"][-1]["network"] if restoration["trajectory"] else S.env.get_network_status()
        ctx = {
            "fault_class": S.classification["fault_class"],
            "cnn_confidence": S.classification["confidence"],
            "probabilities": {k: round(v, 4) for k, v in S.classification["probabilities"].items()},
            "voltage_status": S.stats["voltage_status"],
            "current_status": S.stats["current_status"],
            "frequency_status": S.stats["frequency_status"],
            "affected_section": (f"S{S.scenario['section'] + 1} ({SECTIONS[S.scenario['section']]['name']})"
                                 if S.scenario["section"] is not None else "none"),
            "network_status": [f"S{s['id']} {s['name']}: {s['status']}" for s in net["sections"]],
            "dqn_actions": [f"step {st['step']}: {st['action_name']}" for st in restoration["trajectory"]],
            "safety_results": [f"step {st['step']}: {'APPROVED' if st['validation']['valid'] else 'REJECTED'} - "
                               f"{st['validation']['reason']}" for st in restoration["trajectory"]],
            "restoration_progress": restoration["summary"]["progress_pct"] / 100,
            "served_load": restoration["summary"]["served_load_pct"] / 100,
        }
        if req.include_baseline and S.last_baseline:
            ctx["baseline_summary"] = S.last_baseline["summary"]
    try:
        return generate_explanation(ctx)
    except OllamaError as e:
        raise HTTPException(503, str(e))


@router.get("/results", response_model=GenericResponse)
def results():
    plots = sorted(p.name for p in C.PLOTS_DIR.glob("*.png"))
    reports = sorted((p.name for p in C.REPORTS_DIR.glob("*.json")), reverse=True)
    return {"data": {"cnn": read_json(C.CNN_METRICS_PATH), "dqn_training": read_json(C.DQN_TRAINING_PATH),
                     "evaluation": read_json(C.EVALUATION_PATH), "plots": plots, "reports": reports[:10]}}


@router.get("/network", response_model=GenericResponse)
def network():
    with S.lock:
        _ensure_scenario()
        return {"data": S.env.get_network_status()}
