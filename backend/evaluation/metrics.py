"""DQN vs rule-based evaluation on identical scenarios."""
import copy
import json
import time

import numpy as np

from backend import config as C
from backend.drl.environment import GridEnvironment


def run_episode(agent, env: GridEnvironment, scenario: dict, record_q: bool = False) -> dict:
    state = env.reset(copy.deepcopy(scenario))
    steps = []
    done = False
    while not done:
        q = agent.q_values(state).tolist() if record_q and hasattr(agent, "q_values") else None
        action = agent.select_action(state, greedy=True)
        state, reward, done, info = env.step(action)
        info["q_values"] = q
        steps.append(info)
    return {
        "steps": env.steps,
        "served_load_pct": 100.0 * env.served_load_ratio(),
        "progress_pct": 100.0 * env.restoration_progress(),
        "invalid_actions": len(env.invalid_actions),
        "cumulative_reward": env.cumulative_reward,
        "success": env.restoration_complete(),
        "trajectory": steps,
    }


def make_scenarios(env: GridEnvironment, n: int, seed: int) -> list[dict]:
    env.rng = np.random.default_rng(seed)
    return [env.random_scenario() for _ in range(n)]


def summarize(results: list[dict]) -> dict:
    total_actions = sum(r["steps"] for r in results)
    return {
        "episodes": len(results),
        "avg_restoration_steps": float(np.mean([r["steps"] for r in results])),
        "avg_served_load_pct": float(np.mean([r["served_load_pct"] for r in results])),
        "avg_progress_pct": float(np.mean([r["progress_pct"] for r in results])),
        "invalid_actions_total": int(sum(r["invalid_actions"] for r in results)),
        "invalid_action_rate": float(sum(r["invalid_actions"] for r in results) / max(total_actions, 1)),
        "avg_cumulative_reward": float(np.mean([r["cumulative_reward"] for r in results])),
        "successful_restorations": int(sum(r["success"] for r in results)),
        "success_rate": float(np.mean([r["success"] for r in results])),
    }


def evaluate_agents(dqn_agent, baseline_agent, feature_bank, n_episodes: int = C.EVAL_EPISODES,
                    seed: int = C.SEED + 7, save: bool = True) -> dict:
    env = GridEnvironment(feature_bank=feature_bank, seed=seed)
    scenarios = make_scenarios(env, n_episodes, seed)
    out = {"n_episodes": n_episodes, "seed": seed, "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    per_type = {}
    for name, agent in (("dqn", dqn_agent), ("rule_based", baseline_agent)):
        res = [run_episode(agent, env, sc) for sc in scenarios]
        out[name] = summarize(res)
        for sc, r in zip(scenarios, res):
            per_type.setdefault(sc["fault_type"], {}).setdefault(name, []).append(r)
    out["by_fault_type"] = {ft: {name: summarize(rs) for name, rs in d.items()} for ft, d in per_type.items()}
    if save:
        C.EVALUATION_PATH.write_text(json.dumps(out, indent=2))
        plot_comparison(out)
    return out


def plot_comparison(ev: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = [("avg_restoration_steps", "Avg restoration steps"), ("avg_served_load_pct", "Avg served load (%)"),
            ("invalid_actions_total", "Invalid actions (total)"), ("avg_cumulative_reward", "Avg cumulative reward"),
            ("successful_restorations", "Successful restorations")]
    fig, axes = plt.subplots(1, len(keys), figsize=(16, 3.6))
    for ax, (k, title) in zip(axes, keys):
        vals = [ev["dqn"][k], ev["rule_based"][k]]
        bars = ax.bar(["DQN", "Rule-based"], vals, color=["#00b4d8", "#f4a261"])
        ax.bar_label(bars, fmt="%.2f"); ax.set_title(title, fontsize=10)
    fig.suptitle(f"DQN vs rule-based on {ev['n_episodes']} identical scenarios")
    fig.tight_layout(); fig.savefig(C.PLOTS_DIR / "dqn_vs_rule_based.png", dpi=90); plt.close(fig)
