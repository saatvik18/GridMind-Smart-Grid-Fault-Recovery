# Final Audit Report: Smart Grid Fault Detection & Autonomous Restoration

**Audit date:** 2026-11-04
**Project location:** `S:\smart_grid_recovery`
**Environment:** Windows 11, Python 3.11.9, torch 2.10.0 (CPU), Ollama with `llama3.2:latest`

**Labels:** PASS = tested and working · FAIL = tested and failed · PARTIAL = implemented but not fully verified · NOT TESTED = not verified

The trained checkpoints were backed up before testing. After the tests, their MD5 hashes match the originals (`cnn_fault_detector.pth` da88dcc6…, `dqn_agent.pth` 0579437b…). The training endpoints ran with tiny settings (1 epoch, 10 episodes) and the original models and results were restored afterwards.

---

## 1. Project structure: PASS
All 26 required files and folders exist and are non-empty, plus `backend/config.py` and `backend/utils/helpers.py`.
- `python -m compileall backend run.py tests`: no syntax errors.
- Every project module imports cleanly.
- All dependencies are installed: numpy 2.4.3, scipy 1.17.1, matplotlib 3.10.8, scikit-learn 1.8.0, torch 2.10.0, fastapi 0.115.0, uvicorn 0.30.0, pydantic 2.9.0, requests 2.34.2, httpx 0.27.0, pytest 9.1.1.

## 2. CNN: PASS
| Check | Result |
|---|---|
| PyTorch model, 4 classes | PASS. Conv(2→16)-ReLU-MaxPool → Conv(16→32)-ReLU-MaxPool → Conv(32→64)-ReLU-AdaptiveAvgPool → Linear(64→32)-ReLU → Linear(32→4) |
| Dataset size | PASS. 600 samples, 150 per class |
| Spectrogram input | PASS. `scipy.signal.spectrogram` on voltage and current → (600, 2, 32, 32), all values finite |
| Train/validation split | PASS. Stratified 480 / 120, no shared indices, 0 identical waveforms across the splits (MD5 of each signal) |
| Checkpoint load | PASS. Stored validation accuracy (0.975) matches a fresh recomputation exactly |
| Feature extraction | PASS. 32-dim vector, identical to the penultimate-layer output |
| Fresh-signal prediction | PASS. 400 new signals (seed 2026): accuracy 0.975 |

**Validation set (120 samples): accuracy 97.5%**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Normal | 0.938 | 1.000 | 0.968 |
| Voltage Sag | 1.000 | 0.900 | 0.947 |
| Overload | 0.968 | 1.000 | 0.984 |
| Fault | 1.000 | 1.000 | 1.000 |

Confusion matrix (rows = true, columns = predicted; order Normal, Sag, Overload, Fault): `[[30,0,0,0],[2,27,1,0],[0,0,30,0],[0,0,0,30]]`

**Fresh signals (400): accuracy 97.5%.** Confusion matrix `[[100,0,0,0],[10,90,0,0],[0,0,100,0],[0,0,0,100]]`. The only errors are mild voltage sags classified as Normal.

## 3. DQN: PASS
| Check | Result |
|---|---|
| Q-network | PASS. Linear(54→128)-ReLU-Linear(128→128)-ReLU-Linear(128→3) |
| Experience replay | PASS. Buffer filled, random minibatches sampled |
| Target network | PASS. Separate instance; stays frozen during `train_step`, synced by `update_target()` |
| Epsilon-greedy | PASS. With ε=1.0, 300 draws gave all 3 actions (101 / 95 / 104) |
| Bellman loss | PASS. Target = r + γ·max Q_target(s′)·(1−done), Smooth-L1 loss; policy weights change after a step |
| Save/load | PASS. Loaded weights are bit-identical to the checkpoint; state_dim 54 matches the environment |
| Observation | PASS. obs[0:32] = log1p(CNN features) (raw max 184.6 → 5.22), then 4 CNN probabilities, 5 energised, 5 fault, 5 isolated, served load ratio, progress, step fraction |
| All 3 actions | PASS |
| Trained model used in evaluation | PASS. On 100 scenarios whose CNN outputs were never seen in training: trained DQN success 98%, avg reward 21.75; untrained DQN 51%, −22.10 |

**DQN vs rule-based on the same 100 scenarios (CNN outputs not used in training):**

| Metric | DQN | Rule-based |
|---|---|---|
| Successful restorations | 98 | 100 |
| Avg restoration steps | 1.75 | 1.53 |
| Avg served load | 87.6% | 88.0% |
| Invalid actions | 0 | 0 |
| Avg cumulative reward | 21.75 | 22.43 |

## 4. Safety layer and grid environment: PASS
| Test | Result |
|---|---|
| Isolate faulty section | PASS. Section isolated and de-energised |
| Reconnect healthy section | PASS. S5 reconnected via the tie switch, load restored |
| Reconnect before isolation | PASS. Rejected, penalty −5.5, suggested corrected action = isolate |
| Reconnect a faulty section (isolated or not) | PASS. Rejected |
| Unnecessary isolation of a healthy section | PASS. Rejected |
| Invalid indices (7, −1, source section) and action 5 | PASS. Rejected |
| Source unavailable | PASS. All switching blocked |
| Loop, dead feeder, non-existent switch | PASS. All rejected |
| State unchanged after a rejected action | PASS. Switches, isolation flags and observation identical; rejection logged in `invalid_actions` |
| Every DQN action passes the validator | PASS. Instrumented run: 2 steps, 2 validator calls |
| Radial invariant | PASS. 300 random-action episodes, 0 violations |

## 5. Ollama: PASS
- Server running; `llama3.2:latest` available (`GET /api/tags`).
- Live HTTP inference: 10 successful generations during the audit, every response parsed as JSON with all 6 required keys. Reports are saved to `results/reports/` as `.json` and `.md`.
- Error handling:
  - Unreachable server → `OllamaError` with "Ollama is not running…"; API returns 503.
  - Timeout → "Ollama request timed out after … s".
  - Missing model → "Model '…' not found. Run: ollama pull …".
- **The LLM cannot control the grid.** `ollama_client.py` imports only `requests`, config and helpers. Its output is shown to the user and saved; it is never passed to the agent, the environment or the validator.

## 6. REST API: PASS
| Endpoint | Status | Notes |
|---|---|---|
| GET /api/health | PASS (200) | backend, model files, Ollama status |
| POST /api/simulate | PASS (200 / 422) | 200 samples per waveform, 32-dim features, PNG spectrogram; 422 for an unknown fault type or section 9 |
| POST /api/classify | PASS (200 / 400) | 400 when no waveform exists yet |
| POST /api/train/cnn | PASS (200 / 422) | ran with epochs=1, then models restored; 422 for epochs=0 |
| POST /api/train/dqn | PASS (200 / 422) | ran with episodes=10 including evaluation, then restored; 422 for episodes=1 |
| POST /api/restore | PASS (200 / 409) | Q-values and validation result on every step; 409 when no DQN checkpoint exists |
| POST /api/baseline | PASS (200) | |
| POST /api/llm/explain | PASS (200 / 503) | real generation; 503 when Ollama is unreachable |
| GET /api/results | PASS (200) | |
| GET /api/network | PASS (200) | 5 sections, 5 switches |
| POST /api/evaluate, POST /api/reset | PASS (200) | |
| Static files: `/`, Chart.js, `/results/plots/*` | PASS (200) | |

## 7. Frontend: PASS
Tested in the browser against the live server.
- Loads with **0 console errors**. Every asset returns 200; Chart.js 4.4.1 is served locally.
- All 4 status lights green.
- All four simulate buttons give the correct CNN class and confidence. Voltage, current and frequency charts show 200 points each; the probability chart and spectrogram image render.
- Network canvas and section table update. Restoration animates (Q-values, safety badge, progress 100%, trajectory table), and the Overview cards update.
- Reset and rule-based baseline work (rule-based rows show "n/a (rule)" for Q-values). The evaluation button works.
- Charts render: CNN loss and accuracy (40 epochs), confusion matrix (16 cells), DQN reward (300 episodes), 4 comparison charts plus tables, and the spectrogram gallery (4 images).
- The **real AI report** generated from the button in 93 s, with all 6 sections shown.
- The loading spinner appears during the request. A simulated Ollama 503 shows the red error box and re-enables the button. A backend 422 shows an error toast.
- At phone width (375 px) none of the 8 views scrolls sideways (after fix 4 below).

## 8. End-to-end scenarios: PASS (4/4)
Each scenario ran through the full REST stack: waveform → spectrogram → CNN → features → DQN → safety → grid → Ollama.

| Scenario | Signal status | CNN (confidence) | DQN actions (all APPROVED) | Final grid | Served | Ollama |
|---|---|---|---|---|---|---|
| Normal | V 0.98 pu, I 1.00 pu, f 49.96–50.03 | Normal (92.5%) | MAINTAIN | all connected | 100% | 200, JSON, 84 s |
| Voltage Sag @S3 | V 0.63 pu | Voltage Sag (99.97%) | RECONNECT (S5 via tie) | all connected | 100% | 200, JSON, 104 s |
| Overload @S3 | I 2.10 pu, f 49.78–49.85 | Overload (100%) | ISOLATE S3 → RECONNECT S5 | S3 isolated | 64.3% | 200, JSON, 94 s |
| Fault @S4 | V 0.24 pu, I 4.85 pu, f 48.97–51.05 | Fault (100%) | ISOLATE S4 → RECONNECT S5 → RECONNECT S2 | S4 isolated | 75.0% | 200, JSON, 112 s |

The 32-dim feature vector was extracted in every scenario. Served load below 100% means the isolated section's load can't be restored, which is correct.

## 9. Automated tests: PASS
`python -m pytest -v tests` → **11 passed in 164 s**. This includes the live Ollama test and two regression tests added during this audit.

---

## Bugs found and fixed
1. **`env.step()` crashed on an out-of-range action** (`KeyError` in `ACTION_NAMES[action]`). The validator rejected it, but building the step info then crashed. Fixed in `backend/drl/environment.py` (`ACTION_NAMES.get`). Regression test added.
2. **Ollama timeout too tight.** Measured CPU latency was 84–177 s against a 180 s timeout. Raised to 300 s and added `keep_alive: 30m`, which brought latency down to about 84–112 s.
3. **Missing favicon** (`/favicon.ico` 404). Added an inline icon.
4. **Mobile layout.** Fixed-size network nodes covered the switches at narrow widths, and tables were cut off. Node size now scales with the canvas, and tables scroll horizontally on small screens.
5. **Port conflict** (reported by the user). Port 8000 was taken. The default is now **8765**, `python run.py --port N` overrides it, `run.py` exits with a clear message if the port is busy, and CORS accepts any localhost port.

## Remaining limitations (not bugs)
- Simplified synthetic signal and grid models; no power-flow simulation.
- The CNN misses about 10% of mild voltage sags (classified as Normal).
- The DQN matches but doesn't beat the rule-based baseline (98% vs 100% success). The simplified environment exposes per-section fault flags, so a rule policy is already near-optimal.
- Llama 3.2 output is advisory and sometimes imprecise. For example, in the sag scenario it suggested reconnecting S3 first, while the system actually reconnected S5 through the tie switch.
- Llama 3.2 on CPU takes about 1.5–3 minutes per report.
- API error toasts show raw validation JSON for 422 errors (works, but not user-friendly).

---

## Commands
Run from `S:\smart_grid_recovery`:

```bash
python run.py
```
Dashboard: **http://127.0.0.1:8765**. Use `python run.py --port 9000` if 8765 is busy.

```bash
python -m pytest -v tests
```
