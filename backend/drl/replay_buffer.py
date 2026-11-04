"""Experience replay buffer."""
import random
from collections import deque

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (np.stack(s), np.array(a, dtype=np.int64), np.array(r, dtype=np.float32),
                np.stack(s2), np.array(d, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)
