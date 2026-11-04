"""Independent safety-validation layer (Module 5).

Every action proposed by any agent (DQN or rule-based) is checked here before the
environment executes it. The validator recomputes energisation from the raw switch
states itself instead of trusting the environment's bookkeeping.
"""
from collections import deque

ACTION_NAMES = {0: "MAINTAIN CURRENT STATE", 1: "ISOLATE FAULTY SECTION", 2: "RECONNECT HEALTHY SECTION"}


def _energized(n: int, closed_edges, isolated, source_available: bool) -> list[bool]:
    en = [False] * n
    if not source_available:
        return en
    adj = {i: [] for i in range(n)}
    for u, v in closed_edges:
        adj[u].append(v); adj[v].append(u)
    en[0] = True
    q = deque([0])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if not en[v] and not isolated[v]:
                en[v] = True; q.append(v)
    return en


def _is_radial(n: int, closed_edges, isolated, source_available: bool) -> bool:
    """Energised part of the network must be a tree (no loops / parallel feeds)."""
    en = _energized(n, closed_edges, isolated, source_available)
    live_edges = [(u, v) for u, v in closed_edges if en[u] and en[v]]
    return len(live_edges) == sum(en) - 1 if any(en) else True


def validate_action(action: int, environment_state: dict) -> dict:
    """Validate a proposed action against the current network state.

    environment_state keys: n_sections, topology_edges, closed_edges, faulty, isolated,
    source_available, proposed (dict with 'target' and optional 'edge').
    Returns {valid, reason, corrected_action}.
    """
    s = environment_state
    n = s["n_sections"]
    faulty, isolated = s["faulty"], s["isolated"]
    closed = [tuple(e) for e in s["closed_edges"]]
    topology = {tuple(sorted(e)) for e in s["topology_edges"]}
    proposed = s.get("proposed") or {}
    en = _energized(n, closed, isolated, s["source_available"])
    unisolated_fault = any(faulty[i] and not isolated[i] for i in range(n))
    healthy_down = [i for i in range(1, n) if not en[i] and not faulty[i] and not isolated[i]]

    def result(valid, reason, corrected=None):
        return {"valid": valid, "reason": reason, "corrected_action": corrected,
                "action": action, "action_name": ACTION_NAMES.get(action, "UNKNOWN")}

    if action not in ACTION_NAMES:
        return result(False, f"Unknown action index {action}", 0)
    if action == 0:
        if unisolated_fault:
            return result(True, "Maintain accepted, but an energisable fault is still not isolated", 1)
        return result(True, "No switching change; network state preserved")

    if not s["source_available"]:
        return result(False, "Main source unavailable: switching operations blocked", 0)

    target = proposed.get("target")
    if target is None:
        if action == 1:
            return result(False, "Unnecessary isolation: no faulty section requires isolation",
                          2 if healthy_down else 0)
        if unisolated_fault:
            return result(False, "Reconnection blocked: faulty section not yet isolated", 1)
        return result(False, "No eligible healthy section to reconnect", 0)
    if not isinstance(target, int) or not (0 <= target < n):
        return result(False, f"Invalid section index {target}", 0)
    if target == 0:
        return result(False, "Main source section cannot be switched", 0)

    if action == 1:
        if isolated[target]:
            return result(False, f"Section {target + 1} is already isolated", 0)
        if not faulty[target]:
            return result(False, f"Unnecessary isolation of healthy section {target + 1}",
                          2 if healthy_down else 0)
        return result(True, f"Isolating faulty section {target + 1} removes the fault from the network")

    # action == 2: reconnect
    if unisolated_fault:
        return result(False, "Reconnection blocked: faulty section not yet isolated (would re-energise fault)", 1)
    if faulty[target]:
        return result(False, f"Faulty section {target + 1} cannot be reconnected", 0)
    if isolated[target]:
        return result(False, f"Section {target + 1} is isolated", 0)
    if en[target]:
        return result(False, f"Section {target + 1} is already energised (closing would create a loop)", 0)
    edge = proposed.get("edge")
    if edge is None or tuple(sorted(edge)) not in topology:
        return result(False, "Proposed switch does not exist in network topology", 0)
    u, v = edge
    feeder = u if v == target else v
    if not en[feeder]:
        return result(False, f"Feeding section {feeder + 1} is not energised", 0)
    after = closed + [tuple(edge)]
    if not _is_radial(n, after, isolated, s["source_available"]):
        return result(False, "Action would violate radial network constraint", 0)
    return result(True, f"Section {target + 1} reconnected via switch S{u + 1}-S{v + 1}; network stays radial")
