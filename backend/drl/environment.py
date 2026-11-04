"""Simplified 5-section radial distribution network environment (Module 4).

Topology (indices 0..4 = Sections 1..5):
    S1(source) - S2 - S3 - S4 - S5      main feeder (normally closed switches)
    S1 ............................ S5  tie switch (normally open, alternate feed)

Scenarios:
    Normal       : nothing trips.
    Voltage Sag  : undervoltage relay at section k trips k and everything downstream; no fault.
    Overload     : section k overloaded (flagged abnormal) -> k and downstream trip.
    Fault        : short circuit at section k -> feeder breaker trips all of S2..S5.
"""
from collections import deque

import numpy as np

from backend import config as C
from backend.safety.validator import ACTION_NAMES, validate_action

SECTIONS = [
    {"id": 1, "name": "Main Source", "type": "source", "demand": 0.0, "priority": 0},
    {"id": 2, "name": "Residential Load", "type": "residential", "demand": 30.0, "priority": 1},
    {"id": 3, "name": "Industrial Load", "type": "industrial", "demand": 50.0, "priority": 1},
    {"id": 4, "name": "Commercial Load", "type": "commercial", "demand": 35.0, "priority": 1},
    {"id": 5, "name": "Critical Load", "type": "critical", "demand": 25.0, "priority": 2},
]
N = len(SECTIONS)
MAIN_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4)]
TIE_EDGES = [(0, 4)]
ALL_EDGES = MAIN_EDGES + TIE_EDGES
TOTAL_DEMAND = sum(s["demand"] for s in SECTIONS)
STATE_DIM = C.FEATURE_DIM + len(C.CLASSES) + 3 * N + 3
N_ACTIONS = 3


class GridEnvironment:
    def __init__(self, feature_bank=None, max_steps: int = C.DQN_MAX_STEPS, seed: int | None = None,
                 scenario: dict | None = None):
        self.feature_bank = feature_bank
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.source_available = True
        self.scenario = None
        if feature_bank is not None or scenario is not None:
            self.reset(scenario)

    # ------------------------------------------------------------------ core API
    def reset(self, scenario: dict | None = None) -> np.ndarray:
        if scenario is None:
            scenario = self.random_scenario()
        self.scenario = scenario
        self.closed = set(MAIN_EDGES)
        self.faulty = [False] * N
        self.isolated = [False] * N
        self.steps = 0
        self.cumulative_reward = 0.0
        self.invalid_actions = []
        self.history = []
        k = scenario.get("section")
        ft = scenario["fault_type"]
        if ft == "Voltage Sag":
            self.closed.discard((k - 1, k))
        elif ft == "Overload":
            self.faulty[k] = True
            self.closed.discard((k - 1, k))
        elif ft == "Fault":
            self.faulty[k] = True
            self.closed.discard((0, 1))
        self.initial_served = self.served_load_ratio()
        return self.get_state()

    def random_scenario(self) -> dict:
        ft = C.CLASSES[int(self.rng.integers(len(C.CLASSES)))]
        return self.make_scenario(ft, int(self.rng.integers(1, N)))

    def make_scenario(self, fault_type: str, section: int | None = None, cnn: dict | None = None) -> dict:
        if section is None:
            section = int(self.rng.integers(1, N))
        if cnn is None:
            if self.feature_bank is None:
                raise RuntimeError("No CNN outputs supplied and no feature bank configured")
            cnn = self.feature_bank.sample(fault_type, self.rng)
        return {"fault_type": fault_type, "section": None if fault_type == "Normal" else section,
                "features": np.asarray(cnn["features"], dtype=np.float32),
                "probs": np.asarray(cnn["probs"], dtype=np.float32),
                "predicted": cnn.get("predicted")}

    def step(self, action: int):
        prev_served = self.served_load_ratio()
        proposed = self.propose(action)
        env_state = self.get_env_state(proposed)
        validation = validate_action(action, env_state)        # safety layer gates every action
        detail = "No change"
        if validation["valid"]:
            if action == 1:
                t = proposed["target"]
                self.isolated[t] = True
                self.closed = {e for e in self.closed if t not in e}
                detail = f"Isolated section {t + 1} ({SECTIONS[t]['name']})"
            elif action == 2:
                self.closed.add(tuple(proposed["edge"]))
                t = proposed["target"]
                detail = f"Reconnected section {t + 1} ({SECTIONS[t]['name']}) via S{proposed['edge'][0] + 1}-S{proposed['edge'][1] + 1}"
        else:
            self.invalid_actions.append({"step": self.steps + 1, "action": action, "reason": validation["reason"]})
            detail = f"Rejected: {validation['reason']}"
        self.steps += 1
        complete = self.restoration_complete()
        reward = self.calculate_reward(action, validation["valid"], prev_served, self.served_load_ratio(),
                                       complete, proposed.get("target"))
        self.cumulative_reward += reward
        done = complete or self.steps >= self.max_steps
        info = {"step": self.steps, "action": action, "action_name": ACTION_NAMES.get(action, "UNKNOWN"),
                "validation": validation, "detail": detail, "reward": reward,
                "cumulative_reward": self.cumulative_reward, "complete": complete,
                "served_load_ratio": self.served_load_ratio(), "progress": self.restoration_progress(),
                "network": self.get_network_status()}
        self.history.append(info)
        return self.get_state(), reward, done, info

    def calculate_reward(self, action, valid, prev_served, new_served, complete, target=None) -> float:
        r = -0.5                                    # time penalty per step
        if not valid:
            r -= 5.0
        elif action == 1:
            r += 5.0
        elif action == 2:
            r += 20.0 * (new_served - prev_served)
            if target is not None and SECTIONS[target]["priority"] == 2:
                r += 3.0                            # critical load restored
        elif action == 0 and not complete:
            r -= 1.0                                # idling while customers are out
        if complete and valid:
            r += 10.0
        return float(r)

    def get_state(self) -> np.ndarray:
        en = self.energized()
        return np.concatenate([
            np.log1p(self.scenario["features"]),   # CNN ReLU features are >= 0; log1p keeps scale comparable
            self.scenario["probs"],
            np.array(en, dtype=np.float32), np.array(self.faulty, dtype=np.float32),
            np.array(self.isolated, dtype=np.float32),
            np.array([self.served_load_ratio(), self.restoration_progress(), self.steps / self.max_steps],
                     dtype=np.float32),
        ]).astype(np.float32)

    def get_network_status(self) -> dict:
        en = self.energized()
        sections = []
        for i, s in enumerate(SECTIONS):
            status = ("faulty" if self.faulty[i] and not self.isolated[i] else
                      "isolated" if self.isolated[i] else
                      "connected" if en[i] else "disconnected")
            sections.append({**s, "index": i, "connected": en[i], "fault": self.faulty[i],
                             "isolated": self.isolated[i], "status": status,
                             "served_load": s["demand"] if en[i] else 0.0})
        edges = [{"from": u + 1, "to": v + 1, "closed": (u, v) in self.closed, "tie": (u, v) in TIE_EDGES}
                 for u, v in ALL_EDGES]
        return {"sections": sections, "edges": edges, "source_available": self.source_available,
                "served_load_ratio": self.served_load_ratio(), "progress": self.restoration_progress(),
                "complete": self.restoration_complete(), "steps": self.steps,
                "scenario": {"fault_type": self.scenario["fault_type"],
                             "section": None if self.scenario["section"] is None else self.scenario["section"] + 1}}

    # ------------------------------------------------------------------ helpers
    def energized(self) -> list[bool]:
        en = [False] * N
        if not self.source_available:
            return en
        en[0] = True
        q = deque([0])
        while q:
            u = q.popleft()
            for a, b in self.closed:
                v = b if a == u else a if b == u else None
                if v is not None and not en[v] and not self.isolated[v]:
                    en[v] = True; q.append(v)
        return en

    def served_load_ratio(self) -> float:
        en = self.energized()
        return float(sum(s["demand"] for i, s in enumerate(SECTIONS) if en[i] and not self.faulty[i]) / TOTAL_DEMAND)

    def restorable_demand(self) -> float:
        return sum(s["demand"] for i, s in enumerate(SECTIONS) if not self.faulty[i])

    def restoration_progress(self) -> float:
        en = self.energized()
        served = sum(s["demand"] for i, s in enumerate(SECTIONS) if en[i] and not self.faulty[i])
        return float(served / self.restorable_demand())

    def restoration_complete(self) -> bool:
        unisolated = any(self.faulty[i] and not self.isolated[i] for i in range(N))
        return not unisolated and self.restoration_progress() >= 0.999

    def eligible_reconnections(self) -> list[tuple[int, tuple]]:
        en = self.energized()
        out = []
        for i in range(1, N):
            if en[i] or self.faulty[i] or self.isolated[i]:
                continue
            for u, v in ALL_EDGES:
                if (u, v) in self.closed:
                    continue
                feeder = u if v == i else v if u == i else None
                if feeder is not None and en[feeder]:
                    out.append((i, (u, v)))
                    break
        out.sort(key=lambda x: (-SECTIONS[x[0]]["priority"], -SECTIONS[x[0]]["demand"]))
        return out

    def propose(self, action: int) -> dict:
        """Translate an abstract action into a concrete switching target (validated separately)."""
        if action == 1:
            cands = [i for i in range(N) if self.faulty[i] and not self.isolated[i]]
            return {"target": cands[0] if cands else None}
        if action == 2:
            # pick next de-energised healthy section with a neighbouring live feeder
            if any(self.faulty[i] and not self.isolated[i] for i in range(N)):
                en = self.energized()
                downs = [i for i in range(1, N) if not en[i] and not self.isolated[i]]
                for i in downs:  # a closing attempt toward the fault zone
                    for u, v in ALL_EDGES:
                        if (u, v) not in self.closed and i in (u, v) and en[u if v == i else v]:
                            return {"target": i, "edge": (u, v)}
                return {"target": None}
            el = self.eligible_reconnections()
            return {"target": el[0][0], "edge": el[0][1]} if el else {"target": None}
        return {}

    def get_env_state(self, proposed: dict | None = None) -> dict:
        return {"n_sections": N, "topology_edges": ALL_EDGES, "closed_edges": sorted(self.closed),
                "faulty": list(self.faulty), "isolated": list(self.isolated),
                "source_available": self.source_available, "proposed": proposed or {}}
