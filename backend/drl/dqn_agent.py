"""Deep Q-Network restoration agent (Module 6)."""
import random

import numpy as np
import torch
import torch.nn as nn

from backend import config as C
from backend.drl.replay_buffer import ReplayBuffer


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden: int = C.DQN_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_actions))

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, state_dim: int, n_actions: int = 3, gamma: float = C.DQN_GAMMA, lr: float = C.DQN_LR,
                 batch_size: int = C.DQN_BATCH, buffer_size: int = C.DQN_BUFFER,
                 epsilon: float = C.DQN_EPS_START, eps_end: float = C.DQN_EPS_END, eps_decay: float = C.DQN_EPS_DECAY):
        self.state_dim, self.n_actions = state_dim, n_actions
        self.gamma, self.batch_size = gamma, batch_size
        self.epsilon, self.eps_end, self.eps_decay = epsilon, eps_end, eps_decay
        self.policy = QNetwork(state_dim, n_actions)
        self.target = QNetwork(state_dim, n_actions)
        self.update_target()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.memory = ReplayBuffer(buffer_size)

    def q_values(self, state) -> np.ndarray:
        with torch.no_grad():
            return self.policy(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0))[0].numpy()

    def select_action(self, state, greedy: bool = False) -> int:
        """Epsilon-greedy action selection (greedy=True for evaluation/deployment)."""
        if not greedy and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        return int(np.argmax(self.q_values(state)))

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def train_step(self) -> float | None:
        if len(self.memory) < self.batch_size:
            return None
        s, a, r, s2, d = (torch.as_tensor(x) for x in self.memory.sample(self.batch_size))
        q = self.policy(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            # Bellman target: r + gamma * max_a' Q_target(s', a') for non-terminal transitions
            target = r + self.gamma * self.target(s2).max(1).values * (1 - d)
        loss = self.loss_fn(q, target)
        self.optimizer.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 5.0)
        self.optimizer.step()
        return float(loss.item())

    def update_target(self):
        self.target.load_state_dict(self.policy.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)

    def save_model(self, path=C.DQN_MODEL_PATH):
        torch.save({"policy": self.policy.state_dict(), "target": self.target.state_dict(),
                    "state_dim": self.state_dim, "n_actions": self.n_actions, "epsilon": self.epsilon}, path)

    def load_model(self, path=C.DQN_MODEL_PATH):
        ckpt = torch.load(path, map_location="cpu")
        self.policy.load_state_dict(ckpt["policy"]); self.target.load_state_dict(ckpt["target"])
        self.epsilon = ckpt.get("epsilon", self.eps_end)
        self.policy.eval()
        return self

    @classmethod
    def from_checkpoint(cls, path=C.DQN_MODEL_PATH):
        ckpt = torch.load(path, map_location="cpu")
        return cls(ckpt["state_dim"], ckpt["n_actions"]).load_model(path)
