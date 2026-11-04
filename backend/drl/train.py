"""Episode-based DQN training loop."""
import json
import random
import time

import numpy as np
import torch

from backend import config as C
from backend.cnn.predict import FeatureBank
from backend.drl.dqn_agent import DQNAgent
from backend.drl.environment import N_ACTIONS, STATE_DIM, GridEnvironment


def _plot(rewards, losses):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(rewards, alpha=0.4, label="episode reward")
    w = 10
    if len(rewards) >= w:
        ax[0].plot(range(w - 1, len(rewards)), np.convolve(rewards, np.ones(w) / w, mode="valid"), label=f"{w}-ep mean")
    ax[0].set_title("DQN episode reward"); ax[0].set_xlabel("Episode"); ax[0].legend()
    ax[1].plot(losses, color="#d62728"); ax[1].set_title("DQN loss (per episode mean)"); ax[1].set_xlabel("Episode")
    fig.tight_layout(); fig.savefig(C.PLOTS_DIR / "dqn_rewards.png", dpi=90); plt.close(fig)


def train_dqn(episodes: int = C.DQN_EPISODES, feature_bank: FeatureBank | None = None,
              seed: int = C.SEED, progress_cb=None) -> tuple[DQNAgent, dict]:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    t0 = time.time()
    bank = feature_bank or FeatureBank()
    env = GridEnvironment(feature_bank=bank, seed=seed)
    agent = DQNAgent(STATE_DIM, N_ACTIONS)
    rewards, losses, epsilons, steps_hist, invalid_hist, success_hist = [], [], [], [], [], []
    # held-out greedy validation scenarios: keep the best policy checkpoint seen during training
    from backend.evaluation.metrics import make_scenarios, run_episode
    val_env = GridEnvironment(feature_bank=bank, seed=seed + 1)
    val_scenarios = make_scenarios(val_env, 40, seed + 1)
    best_score, best_state, val_hist = -1e9, None, []

    for ep in range(episodes):
        state = env.reset()
        done, ep_losses = False, []
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            loss = agent.train_step()
            if loss is not None:
                ep_losses.append(loss)
            state = next_state
        agent.decay_epsilon()
        if (ep + 1) % C.DQN_TARGET_UPDATE == 0:
            agent.update_target()
        rewards.append(env.cumulative_reward)
        losses.append(float(np.mean(ep_losses)) if ep_losses else 0.0)
        epsilons.append(agent.epsilon)
        steps_hist.append(env.steps)
        invalid_hist.append(len(env.invalid_actions))
        success_hist.append(bool(env.restoration_complete()))
        if (ep + 1) % 20 == 0 or ep == episodes - 1:
            print(f"[DQN] episode {ep + 1}/{episodes} reward(avg20)={np.mean(rewards[-20:]):.2f} "
                  f"eps={agent.epsilon:.3f} success(avg20)={np.mean(success_hist[-20:]):.2f}")
        if (ep + 1) % 10 == 0 and ep + 1 >= episodes // 3:
            res = [run_episode(agent, val_env, sc) for sc in val_scenarios]
            score = float(np.mean([r["cumulative_reward"] for r in res]))
            val_hist.append({"episode": ep + 1, "avg_reward": score})
            if score > best_score:
                best_score = score
                best_state = {k: v.clone() for k, v in agent.policy.state_dict().items()}
        if progress_cb:
            progress_cb(ep + 1, episodes)

    if best_state is not None:
        agent.policy.load_state_dict(best_state); agent.update_target()
    agent.save_model()
    _plot(rewards, losses)
    log = {"episodes": episodes, "rewards": rewards, "losses": losses, "epsilons": epsilons,
           "steps": steps_hist, "invalid_actions": invalid_hist, "success": success_hist,
           "final_avg_reward_20": float(np.mean(rewards[-20:])),
           "validation": val_hist, "best_validation_reward": best_score, "training_time_s": round(time.time() - t0, 2),
           "state_dim": STATE_DIM, "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    C.DQN_TRAINING_PATH.write_text(json.dumps(log, indent=2))
    print(f"[DQN] saved agent -> {C.DQN_MODEL_PATH} ({log['training_time_s']} s)")
    return agent, log


if __name__ == "__main__":
    from backend.evaluation.baseline import RuleBasedAgent
    from backend.evaluation.metrics import evaluate_agents

    bank = FeatureBank()
    agent, _ = train_dqn(feature_bank=bank)
    ev = evaluate_agents(agent, RuleBasedAgent(), bank)
    for k in ("dqn", "rule_based"):
        print(k, json.dumps(ev[k]))
