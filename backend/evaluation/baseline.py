"""Rule-based restoration baseline (Module 7)."""
import numpy as np

from backend import config as C
from backend.drl.environment import N

_P = C.FEATURE_DIM                    # offsets into the observation vector
_EN = _P + len(C.CLASSES)
_FA = _EN + N
_IS = _FA + N


class RuleBasedAgent:
    """Uses the CNN classification + network status, same observation as the DQN."""
    name = "Rule-Based"

    def select_action(self, state: np.ndarray, greedy: bool = True) -> int:
        probs = state[_P:_EN]
        energized, faulty, isolated = state[_EN:_FA] > 0.5, state[_FA:_IS] > 0.5, state[_IS:_IS + N] > 0.5
        predicted = C.CLASSES[int(np.argmax(probs))]
        fault_detected = predicted in ("Fault", "Overload")
        # Rule 1: fault detected and nothing isolated yet -> isolate faulty section
        if fault_detected and not isolated.any():
            return 1
        # Rule 2: healthy sections still disconnected -> reconnect
        if any(not energized[i] and not faulty[i] and not isolated[i] for i in range(1, N)):
            return 2
        # Rule 3: otherwise maintain
        return 0
