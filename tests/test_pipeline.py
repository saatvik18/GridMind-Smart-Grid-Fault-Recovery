"""End-to-end tests. Run from the project root:  python -m pytest -q tests
Ollama tests are skipped automatically if Ollama / llama3.2 is not available.
"""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend import config as C
from backend.cnn.predict import FeatureBank, extract_cnn_features, predict_fault
from backend.cnn.spectrogram import signal_to_spectrogram
from backend.data.dataset_generator import generate_dataset, generate_signal
from backend.drl.dqn_agent import DQNAgent
from backend.drl.environment import STATE_DIM, GridEnvironment
from backend.evaluation.baseline import RuleBasedAgent
from backend.evaluation.metrics import run_episode
from backend.llm.ollama_client import check_ollama, generate_explanation
from backend.safety.validator import validate_action

RNG = np.random.default_rng(0)


@pytest.fixture(scope="module")
def bank():
    return FeatureBank(per_class=8)


def test_dataset_generation():
    d = generate_dataset(samples_per_class=5, save=False)
    assert d["voltage"].shape == (20, int(C.FS * C.DURATION))
    assert sorted(set(d["labels"].tolist())) == [0, 1, 2, 3]
    normal, sag = generate_signal("Normal", RNG), generate_signal("Voltage Sag", RNG)
    assert np.abs(sag["voltage"]).min() <= np.abs(normal["voltage"]).max()


def test_spectrogram_shape():
    s = generate_signal("Fault", RNG)
    spec = signal_to_spectrogram(s["voltage"], s["current"])
    assert spec.shape == (2, C.SPEC_SIZE, C.SPEC_SIZE) and np.isfinite(spec).all()


def test_cnn_inference_and_features():
    correct = 0
    for cls in C.CLASSES:
        for _ in range(5):
            s = generate_signal(cls, RNG)
            spec = signal_to_spectrogram(s["voltage"], s["current"])
            p = predict_fault(spec)
            assert abs(sum(p["probabilities"].values()) - 1) < 1e-4
            correct += p["fault_class"] == cls
            f = extract_cnn_features(spec)
            assert f.shape == (C.FEATURE_DIM,)
    assert correct / 20 >= 0.7


def test_environment_reset_step(bank):
    env = GridEnvironment(feature_bank=bank, seed=1)
    sc = env.make_scenario("Fault", 2)
    st = env.reset(sc)
    assert st.shape == (STATE_DIM,)
    assert env.served_load_ratio() == 0.0          # feeder breaker tripped
    _, r, done, info = env.step(2)                 # reconnect before isolation -> rejected
    assert not info["validation"]["valid"] and r < 0 and len(env.invalid_actions) == 1
    _, r, _, info = env.step(1)
    assert info["validation"]["valid"] and env.isolated[2]
    while not env.restoration_complete():
        _, _, done, info = env.step(2)
        assert info["validation"]["valid"]
    assert env.restoration_progress() == pytest.approx(1.0)


def test_safety_validator(bank):
    env = GridEnvironment(feature_bank=bank, seed=2)
    env.reset(env.make_scenario("Voltage Sag", 3))
    v = validate_action(1, env.get_env_state(env.propose(1)))
    assert not v["valid"] and "isolation" in v["reason"].lower()
    st = env.get_env_state({"target": 9})
    assert not validate_action(1, st)["valid"]                       # invalid index
    env.reset(env.make_scenario("Overload", 2))
    st = env.get_env_state({"target": 2, "edge": (1, 2)})
    assert not validate_action(2, st)["valid"]                       # reconnect faulty section
    st = env.get_env_state({"target": 1, "edge": (0, 1)})
    assert not validate_action(2, st)["valid"]                       # already energised -> loop
    st = env.get_env_state(env.propose(1)); st["source_available"] = False
    assert not validate_action(1, st)["valid"]                       # source unavailable


def test_out_of_range_action_rejected_without_crash(bank):
    env = GridEnvironment(feature_bank=bank, seed=6)
    env.reset(env.make_scenario("Fault", 2))
    closed = sorted(env.closed)
    for a in (5, -1):
        _, r, _, info = env.step(a)
        assert not info["validation"]["valid"] and r < 0 and info["action_name"] == "UNKNOWN"
    assert sorted(env.closed) == closed and len(env.invalid_actions) == 2


def test_every_action_passes_safety_layer(bank, monkeypatch):
    import backend.drl.environment as E
    calls = []
    real = E.validate_action
    monkeypatch.setattr(E, "validate_action", lambda a, s: (calls.append(a), real(a, s))[1])
    env = GridEnvironment(feature_bank=bank, seed=7)
    res = run_episode(DQNAgent.from_checkpoint(), env, env.make_scenario("Overload", 2))
    assert len(calls) == res["steps"] >= 1


def test_dqn_action_and_train_step(bank):
    agent = DQNAgent(STATE_DIM, 3, batch_size=8)
    env = GridEnvironment(feature_bank=bank, seed=3)
    s = env.reset()
    assert agent.select_action(s) in (0, 1, 2)
    for _ in range(10):
        a = agent.select_action(s)
        s2, r, d, _ = env.step(a)
        agent.remember(s, a, r, s2, d)
        s = env.reset() if d else s2
    assert isinstance(agent.train_step(), float)


def test_trained_dqn_and_baseline(bank):
    agent = DQNAgent.from_checkpoint()
    env = GridEnvironment(feature_bank=bank, seed=4)
    for ft in C.CLASSES:
        sc = env.make_scenario(ft, 2)
        for a in (agent, RuleBasedAgent()):
            res = run_episode(a, env, sc)
            assert res["steps"] >= 1
    sc = env.make_scenario("Fault", 3)
    assert run_episode(RuleBasedAgent(), env, sc)["success"] or sc["predicted"] != "Fault"


ollama_ok = check_ollama()["model_available"]


@pytest.mark.skipif(not ollama_ok, reason="Ollama/llama3.2 not available")
def test_ollama_inference():
    r = generate_explanation({"fault_class": "Overload", "cnn_confidence": 0.99, "voltage_status": "Normal",
                              "current_status": "Overcurrent", "frequency_status": "Deviation",
                              "restoration_progress": 1.0, "served_load": 0.82}, save=False)
    assert r["report"]["fault_explanation"]


def test_api_endpoints():
    c = TestClient(__import__("backend.main", fromlist=["app"]).app)
    assert c.get("/api/health").json()["backend"] == "ok"
    r = c.post("/api/simulate", json={"fault_type": "Overload", "section": 3}).json()
    assert len(r["waveform"]["voltage"]) == 200 and r["spectrogram_image"].startswith("data:image/png")
    assert c.post("/api/classify", json={}).status_code == 200
    rr = c.post("/api/restore").json()
    assert all("validation" in s for s in rr["trajectory"]) and rr["trajectory"][0]["q_values"]
    assert c.post("/api/baseline").status_code == 200
    assert c.get("/api/network").json()["data"]["sections"][0]["type"] == "source"
    assert c.post("/api/reset").status_code == 200
    assert c.get("/api/results").json()["data"]["cnn"]["final_val_accuracy"] > 0.7
    assert c.get("/").status_code == 200
    assert c.post("/api/simulate", json={"fault_type": "Bogus"}).status_code == 422
