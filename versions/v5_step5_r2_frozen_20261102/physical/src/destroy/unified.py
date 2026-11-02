"""Seeded removal with exact customer budgets and explicit policy provenance."""
from dataclasses import dataclass
import hashlib
import json

import numpy as np

from src.core.protocol import DestroyType, DestroyedState, Route, Solution
from src.evaluation.vrptw_eval import evaluate_route, is_solution_feasible


POLICY_VERSION = "unified_destroy_v1"
RATIOS = (0.10, 0.20, 0.30)
RATIO_PROBABILITIES = (0.25, 0.50, 0.25)
TYPES = tuple(DestroyType)


@dataclass
class DestroyResult:
    state: DestroyedState
    destroy_type: DestroyType
    requested_ratio: float
    actual_ratio: float
    seed: int
    state_hash: str
    closed_routes: int
    policy_version: str = POLICY_VERSION

    def metadata(self):
        return dict(policy_version=self.policy_version, destroy_type=self.destroy_type.value,
                    requested_ratio=self.requested_ratio, actual_ratio=self.actual_ratio,
                    seed=self.seed, state_hash=self.state_hash, closed_routes=self.closed_routes,
                    removed_count=len(self.state.removed_customers))


def _pick(values, scores, count, rng):
    if count == 0:
        return []
    weights = np.maximum(np.asarray(scores, dtype=float), 0.0)
    weights += max(float(weights.max(initial=0)) * 0.05, 1e-9)
    return [int(c) for c in rng.choice(values, size=min(count, len(values)),
                                     replace=False, p=weights / weights.sum())]


def _time_scores(routes, instance, tardiness):
    scores = {}
    for route in routes:
        time, previous = 0.0, 0
        for c in route:
            arrival = time + float(instance.duration_matrix[previous, c])
            early, late = instance.customer_time_windows[c - 1]
            scores[c] = max(0.0, arrival - late) if tardiness else max(0.0, early - arrival)
            time = max(arrival, early) + float(instance.customer_service_times[c - 1])
            previous = c
    return scores


def _savings(routes, instance):
    scores = {}
    d = instance.distance_matrix
    for route in routes:
        for i, c in enumerate(route):
            prev = route[i - 1] if i else 0
            nxt = route[i + 1] if i + 1 < len(route) else 0
            scores[c] = max(0.0, float(d[prev, c] + d[c, nxt] - d[prev, nxt]))
    return scores


def _segment(route, count, rng, anchor=None):
    take = min(count, len(route))
    lo = 0 if anchor is None else max(0, anchor - take + 1)
    hi = len(route) - take if anchor is None else min(anchor, len(route) - take)
    start = int(rng.integers(lo, hi + 1))
    return route[start:start + take]


def destroy(solution, instance, destroy_type, ratio, *, seed):
    """Remove exactly round(N * ratio) customers (clamped to [1, N-1]).

    Input must be a complete feasible solution. Preference is deliberately not
    consumed: a frozen destroyed state can be reused for any repair preference.
    Route order and retained customer order are preserved; empty routes vanish.
    """
    kind = DestroyType(destroy_type)
    if instance.n_customers < 2 or not np.isfinite(ratio) or not 0 < ratio < 1:
        raise ValueError("Destroy requires N >= 2 and a finite ratio in (0, 1)")
    if not is_solution_feasible(solution, instance):
        raise ValueError("Destroy requires a complete feasible source solution")
    rng = np.random.default_rng(seed)
    seed = int(seed)
    routes = [list(r.customers) for r in solution.routes if r.customers]
    customers = [c for r in routes for c in r]
    count = max(1, min(instance.n_customers - 1, int(round(instance.n_customers * ratio))))

    if kind == DestroyType.ROUTE_REPACK:
        eligible = [i for i, r in enumerate(routes) if len(r) <= count]
        choices = eligible or list(range(len(routes)))
        rid = _pick(choices, [1 / len(routes[i]) for i in choices], 1, rng)[0]
        # Close a whole route only when it fits the budget. Otherwise remove a
        # segment; never silently enlarge the neighbourhood to force closure.
        removed = list(routes[rid]) if eligible else _segment(routes[rid], count, rng)
    elif kind == DestroyType.WORST_EDGE_RELATED:
        scores = _savings(routes, instance)
        anchor = _pick(customers, [scores[c] for c in customers], 1, rng)[0]
        rest = [c for c in customers if c != anchor]
        d = instance.distance_matrix
        scale = max(float(d.max()), 1.0)
        related = [1 / (0.01 + float(d[anchor, c] + d[c, anchor]) / (2 * scale)) for c in rest]
        removed = [anchor] + _pick(rest, related, count - 1, rng)
    elif kind == DestroyType.MAKESPAN_SEGMENT:
        durations = [evaluate_route(Route(r), instance)[1] for r in routes]
        rid = _pick(list(range(len(routes))), durations, 1, rng)[0]
        removed = _segment(routes[rid], count, rng)
    elif kind in (DestroyType.WAITING_SEGMENT, DestroyType.TARDINESS_SEGMENT):
        scores = _time_scores(routes, instance, kind == DestroyType.TARDINESS_SEGMENT)
        anchor = _pick(customers, [scores[c] for c in customers], 1, rng)[0]
        route = next(r for r in routes if anchor in r)
        removed = _segment(route, count, rng, route.index(anchor))
    elif kind == DestroyType.GENERIC_MIXED:
        scores = _savings(routes, instance)
        removed = _pick(customers, [scores[c] for c in customers], (count + 1) // 2, rng)
    else:
        removed = []

    remaining = [c for c in customers if c not in set(removed)]
    if len(removed) < count:
        removed += [int(c) for c in rng.choice(remaining, size=count - len(removed), replace=False)]
    removed = sorted(removed)
    removed_set = set(removed)
    retained = [[c for c in r if c not in removed_set] for r in routes]
    partial = Solution([Route(r) for r in retained if r])
    state = DestroyedState(partial, removed)
    # Include route order because insertion actions reference route indices.
    content = dict(instance_id=instance.instance_id,
                   routes=[r.customers for r in partial.routes], removed=removed)
    digest = hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return DestroyResult(state, kind, float(ratio), count / instance.n_customers,
                         seed, digest, sum(not r for r in retained))


def sample_destroy(solution, instance, *, seed):
    """Fixed preference-independent pool: uniform types, 25/50/25% ratios."""
    rng = np.random.default_rng(seed)
    kind = TYPES[int(rng.integers(len(TYPES)))]
    ratio = float(rng.choice(RATIOS, p=RATIO_PROBABILITIES))
    child_seed = int(rng.integers(0, 2**32))
    return destroy(solution, instance, kind, ratio, seed=child_seed)
