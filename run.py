"""Single entry point.

    python run.py            # train anything missing, then serve dashboard at http://127.0.0.1:8765
    python run.py --train    # force: regenerate dataset, retrain CNN + DQN, re-evaluate
    python run.py --train-only
    python run.py --no-browser
    python run.py --port 9000  # use a different port
"""
import argparse
import socket
import sys
import threading
import webbrowser

import uvicorn

from backend import config as C


def train_pipeline(force: bool):
    from backend.cnn.predict import FeatureBank, load_cnn
    from backend.cnn.train import train_cnn
    from backend.data.dataset_generator import generate_dataset
    from backend.drl.train import train_dqn
    from backend.evaluation.baseline import RuleBasedAgent
    from backend.evaluation.metrics import evaluate_agents

    if force or not C.DATASET_PATH.exists():
        print("[1/4] Generating synthetic dataset..."); generate_dataset()
    if force or not C.CNN_MODEL_PATH.exists():
        print("[2/4] Training CNN..."); train_cnn()
    load_cnn(force=True)
    if force or not C.DQN_MODEL_PATH.exists() or not C.EVALUATION_PATH.exists():
        print("[3/4] Training DQN...")
        bank = FeatureBank()
        agent, _ = train_dqn(feature_bank=bank)
        print("[4/4] Evaluating DQN vs rule-based...")
        ev = evaluate_agents(agent, RuleBasedAgent(), bank)
        for k in ("dqn", "rule_based"):
            e = ev[k]
            print(f"   {k:10s} success={e['success_rate']:.2%} steps={e['avg_restoration_steps']:.2f} "
                  f"served={e['avg_served_load_pct']:.1f}% invalid={e['invalid_actions_total']} "
                  f"reward={e['avg_cumulative_reward']:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true", help="force full retraining")
    ap.add_argument("--train-only", action="store_true", help="train and exit")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--port", type=int, default=C.PORT, help=f"server port (default {C.PORT})")
    args = ap.parse_args()

    if not args.train_only and port_in_use(args.port):
        sys.exit(f"Port {args.port} is already in use. Stop the other server or run: python run.py --port <free port>")
    train_pipeline(force=args.train)
    if args.train_only:
        return
    from backend.llm.ollama_client import check_ollama
    st = check_ollama()
    print("Ollama:", "llama3.2 ready" if st["model_available"] else st["error"])
    url = f"http://{C.HOST}:{args.port}"
    print(f"Dashboard: {url}")
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run("backend.main:app", host=C.HOST, port=args.port, log_level="info")


def port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((C.HOST, port)) == 0


if __name__ == "__main__":
    main()
