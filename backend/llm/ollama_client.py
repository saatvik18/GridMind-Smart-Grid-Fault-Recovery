"""Local Ollama (llama3.2) integration (Module 8).

The LLM is advisory only: it explains faults and decisions. It never calls env.step();
the DQN selects actions and the safety validator approves them.
"""
import json
import time

import requests

from backend import config as C
from backend.utils.helpers import save_json, timestamp

REPORT_KEYS = ["fault_explanation", "possible_causes", "dqn_action_justification",
               "safety_considerations", "recommended_restoration_sequence", "grid_condition_summary"]


class OllamaError(RuntimeError):
    pass


def check_ollama() -> dict:
    """Verify the Ollama server is reachable and llama3.2 is installed."""
    try:
        r = requests.get(f"{C.OLLAMA_URL}/api/tags", timeout=5)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
    except requests.exceptions.ConnectionError:
        return {"available": False, "model_available": False, "models": [],
                "error": f"Ollama is not running at {C.OLLAMA_URL}. Start it with: ollama serve"}
    except Exception as e:  # noqa: BLE001
        return {"available": False, "model_available": False, "models": [], "error": f"Ollama check failed: {e}"}
    ok = any(m == C.OLLAMA_MODEL or m.startswith(C.OLLAMA_MODEL + ":") for m in models)
    return {"available": True, "model_available": ok, "models": models, "model": C.OLLAMA_MODEL,
            "error": None if ok else f"Model '{C.OLLAMA_MODEL}' not found. Run: ollama pull {C.OLLAMA_MODEL}"}


def build_prompt(ctx: dict) -> str:
    return f"""You are a senior power distribution engineer supervising an autonomous grid restoration agent.
You only ADVISE. A Deep Q-Network selects switching actions and an independent safety layer approves them.

Simulated 5-section radial feeder: S1 Main Source -> S2 Residential -> S3 Industrial -> S4 Commercial -> S5 Critical,
with a normally-open tie switch S1-S5 for alternate feed.

ACTUAL SIMULATION DATA:
- Detected fault class (CNN): {ctx.get('fault_class')}
- CNN confidence: {ctx.get('cnn_confidence', 0):.1%}
- CNN class probabilities: {json.dumps(ctx.get('probabilities', {}))}
- Voltage status: {ctx.get('voltage_status')}
- Current status: {ctx.get('current_status')}
- Frequency status: {ctx.get('frequency_status')}
- Affected section (simulation ground truth): {ctx.get('affected_section')}
- Network section status: {json.dumps(ctx.get('network_status', []))}
- DQN action sequence: {json.dumps(ctx.get('dqn_actions', []))}
- Safety validation results: {json.dumps(ctx.get('safety_results', []))}
- Restoration progress: {ctx.get('restoration_progress', 0):.1%}
- Served load: {ctx.get('served_load', 0):.1%}

Respond ONLY with a JSON object with exactly these keys:
"fault_explanation": string,
"possible_causes": array of strings,
"dqn_action_justification": string (why the DQN actions are appropriate or not),
"safety_considerations": array of strings,
"recommended_restoration_sequence": array of strings (ordered steps),
"grid_condition_summary": string
Be concise and technical. Base your answer strictly on the data above."""


def generate_explanation(ctx: dict, save: bool = True) -> dict:
    status = check_ollama()
    if not status["available"] or not status["model_available"]:
        raise OllamaError(status["error"])
    payload = {"model": C.OLLAMA_MODEL, "prompt": build_prompt(ctx), "stream": False, "format": "json", "keep_alive": C.OLLAMA_KEEP_ALIVE,
               "options": {"temperature": 0.3, "num_predict": 700}}
    t0 = time.time()
    try:
        r = requests.post(f"{C.OLLAMA_URL}/api/generate", json=payload, timeout=C.OLLAMA_TIMEOUT)
        r.raise_for_status()
    except requests.exceptions.Timeout:
        raise OllamaError(f"Ollama request timed out after {C.OLLAMA_TIMEOUT} s")
    except requests.exceptions.ConnectionError:
        raise OllamaError(f"Lost connection to Ollama at {C.OLLAMA_URL}")
    except requests.exceptions.HTTPError as e:
        raise OllamaError(f"Ollama returned HTTP error: {e} - {r.text[:300]}")
    body = r.json()
    raw = body.get("response", "")
    try:
        report = json.loads(raw)
        parse_ok = True
    except json.JSONDecodeError:
        report, parse_ok = {"grid_condition_summary": raw}, False
    for k in REPORT_KEYS:
        report.setdefault(k, [] if k in ("possible_causes", "safety_considerations",
                                         "recommended_restoration_sequence") else "")
    result = {"model": body.get("model", C.OLLAMA_MODEL), "report": report, "raw_response": raw,
              "json_parsed": parse_ok, "latency_s": round(time.time() - t0, 2),
              "eval_tokens": body.get("eval_count"), "context": ctx,
              "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    if save:
        ts = timestamp()
        result["report_path"] = save_json(C.REPORTS_DIR / f"ai_restoration_report_{ts}.json", result)
        (C.REPORTS_DIR / f"ai_restoration_report_{ts}.md").write_text(to_markdown(result), encoding="utf-8")
    return result


def _as_list(v):
    return v if isinstance(v, list) else [v] if v else []


def to_markdown(result: dict) -> str:
    rep, ctx = result["report"], result["context"]
    lines = [f"# AI Restoration Report ({result['model']})", f"_Generated {result['generated_at']}_", "",
             f"**Detected fault:** {ctx.get('fault_class')} ({ctx.get('cnn_confidence', 0):.1%} confidence)  ",
             f"**Restoration progress:** {ctx.get('restoration_progress', 0):.1%}", "",
             "## Fault explanation", str(rep["fault_explanation"]), "", "## Possible causes"]
    lines += [f"- {c}" for c in _as_list(rep["possible_causes"])]
    lines += ["", "## DQN action justification", str(rep["dqn_action_justification"]), "", "## Safety considerations"]
    lines += [f"- {c}" for c in _as_list(rep["safety_considerations"])]
    lines += ["", "## Recommended restoration sequence"]
    lines += [f"{i}. {c}" for i, c in enumerate(_as_list(rep["recommended_restoration_sequence"]), 1)]
    lines += ["", "## Grid condition summary", str(rep["grid_condition_summary"]), ""]
    return "\n".join(lines)
