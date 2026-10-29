"""Read-only bridge to the existing five-objective problem and four repairers."""
import copy
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

from lib.vrp_helper_env import (build_edge_index_near, build_edge_index_routes,
                                find_ruins, vrp_env_sisr)

OBJECTIVES = ("n_vehicles", "total_distance", "makespan", "total_waiting", "total_tardiness")
REPAIRERS = ("greedy", "regret2", "regret3", "noisy_greedy")
NODE_FEATURES = (
    "coordinate_x_centered", "coordinate_y_centered", "demand_capacity",
    "window_start", "window_end", "service_time", "window_width",
    "prefix_load", "prefix_distance", "route_load", "route_distance",
    "arrival", "waiting", "tardiness", "hard_time_slack", "return_slack", "is_depot",
)
CONTEXT_FEATURES = tuple([f"preference_{i}" for i in range(5)]
                         + [f"objective_{n}" for n in OBJECTIVES]
                         + ["remaining_time_fraction", "stagnation_fraction", "last_accepted",
                            "last_reward", "archive_fraction", "max_delay_fraction",
                            "n_customers_scaled", "removal_cap_fraction"]
                         + [f"repair_{r}" for r in REPAIRERS])


def project_api(project_root):
    root = Path(project_root).resolve()
    if not (root / "src/evaluation/vrptw_eval.py").is_file():
        raise FileNotFoundError(f"Existing five-objective project not found: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.core.protocol import Route, Solution
    from src.data_generation.sharded import _instance_from_row
    from src.evaluation.vrptw_eval import evaluate_solution, is_solution_feasible
    from hemt_destroy_feature_labels_v1.repair import repair
    return SimpleNamespace(Route=Route, Solution=Solution, instance=_instance_from_row,
                           evaluate=evaluate_solution, feasible=is_solution_feasible, repair=repair)


class TrainPool:
    """Read only the TRAIN allowlist; DEV, holdout and formal45 cannot be selected."""
    def __init__(self, cfg):
        self.api = project_api(cfg.project_root)
        self.cfg = json.loads((Path(cfg.study_root) / "config.json").read_text(encoding="utf-8"))
        allowed = tuple(self.cfg["allowed_ids_by_split"]["train"])
        self.ids = tuple(cfg.train_instance_ids) or allowed
        if not self.ids or not set(self.ids) <= set(allowed):
            raise ValueError("Only the existing TRAIN allowlist may be used in this adapter")
        self.cache = {}
        self.input_content_ids = {}

    def load(self, iid):
        if iid not in self.ids:
            raise ValueError("IID is not in this run's TRAIN allowlist")
        if iid not in self.cache:
            import hashlib
            raw = Path(self.cfg["inputs"][iid]["path"]).read_bytes()
            actual_hash = hashlib.sha256(raw).hexdigest()
            if actual_hash != self.cfg["inputs"][iid]["sha256"]:
                raise ValueError(f"Input source changed since frozen configuration: {iid}")
            self.input_content_ids[iid] = actual_hash
            row = json.loads(raw.decode("utf-8"))
            if row.get("split") != "train" or row["instance_id"] != iid:
                raise ValueError("TRAIN row identity/split mismatch")
            instance = self.api.instance(row)
            source = self.api.Solution([self.api.Route(list(r)) for r in row["source_solution"]])
            if not self.api.feasible(source, instance):
                raise ValueError(f"Initial source is infeasible: {iid}")
            box = copy.deepcopy(self.cfg["boxes_by_instance"][iid])
            qs = tuple(int(q) for q in self.cfg["q_values_by_instance"][iid])
            if any(q < 1 or q > min(15, math.floor(.15 * instance.n_customers)) for q in qs):
                raise ValueError("Frozen q support violates removal cap")
            self.cache[iid] = (instance, row, box, qs)
        return self.cache[iid]


class Environment:
    def __init__(self, api, instance, row, box, q_values, preference, cfg):
        self.api, self.instance, self.cfg = api, instance, cfg
        self.box = copy.deepcopy(box)
        self.scales = np.asarray(box["scale"], dtype=np.float64)
        self.offset = np.asarray(box["offset"], dtype=np.float64)
        self.q_values = tuple(q_values)
        self.preference = np.asarray(preference, dtype=np.float64)
        if self.preference.shape != (5,) or (self.preference < 0).any() or not np.isfinite(self.preference).all() or not np.isclose(self.preference.sum(), 1.):
            raise ValueError("Invalid five-objective preference")
        self.routes = copy.deepcopy(row["source_solution"])
        self.objectives = self.evaluate(self.routes)
        self.archive = [{"routes": copy.deepcopy(self.routes), "objectives": self.objectives.tolist()}]
        self.steps = self.stagnation = 0
        self.last_accepted = False
        self.last_reward = 0.
        self.last_removed = []
        self.neighbors = [list(np.argsort(d, kind="stable")) for d in instance.distance_matrix]
        self.near = torch.as_tensor(build_edge_index_near(
            [instance.distance_matrix], min(cfg.near_neighbors, instance.n_customers))[0], dtype=torch.long)
        self.fixed_q_index = None
        if cfg.fixed_q is not None:
            if cfg.fixed_q not in self.q_values:
                raise ValueError(f"fixed_q must be in {self.q_values}")
            self.fixed_q_index = self.q_values.index(cfg.fixed_q)

    def solution(self, routes):
        return self.api.Solution([self.api.Route(list(r)) for r in routes])

    def evaluate(self, routes):
        sol = self.solution(routes)
        if not self.api.feasible(sol, self.instance):
            raise ValueError("Complete solution failed coverage/capacity/max-delay/depot-return checks")
        values = self.api.evaluate(sol, self.instance).to_array()
        if not np.isfinite(values).all():
            raise ValueError("Non-finite objectives")
        return values

    def score(self, values):
        weighted = self.preference * ((np.asarray(values) - self.offset) / self.scales)
        return float(weighted.max() + .05 * weighted.sum())

    def node_features(self):
        inst = self.instance
        n = inst.n_customers
        horizon = max(float(inst.depot_time_window[1]), 1.)
        data = np.zeros((n + 1, 6), dtype=float)
        data[1:, :2] = inst.customer_coords
        data[1:, 2] = inst.customer_demands
        data[0, 3:5] = inst.depot_time_window
        data[1:, 3:5] = inst.customer_time_windows
        data[1:, 5] = inst.customer_service_times
        # Reuse upstream's eleven-field dynamic route embedding.
        upstream_env = vrp_env_sisr(inst.vehicle_capacity, data, inst.distance_matrix,
                                    self.routes, 0., 1., 1.)
        base = upstream_env.get_embedding()
        # Coordinates and travel distances have different units from travel time.
        center = inst.customer_coords.mean(axis=0)
        coord_scale = max(float(np.sqrt(np.mean(np.sum((inst.customer_coords-center)**2, axis=1)))), 1e-9)
        base[1:, :2] = (inst.customer_coords - center) / coord_scale
        base[0, :2] = 0.  # Depot has no coordinates in the protocol; explicit flag below.
        distance_scale = max(float(np.max(inst.distance_matrix)), 1e-9)
        base[:, 8] *= horizon / distance_scale
        extra = np.zeros((n + 1, 6), dtype=float)
        extra[0, -1] = 1.
        for route in self.routes:
            prev, current_time = 0, 0.
            route_distance = 0.
            for c in route:
                arrival = current_time + inst.duration_matrix[prev, c]
                early, late = inst.customer_time_windows[c-1]
                start = max(arrival, early)
                extra[c, :4] = (arrival / horizon, max(early-arrival, 0.) / horizon,
                                 max(arrival-late, 0.) / horizon,
                                 (late + inst.max_delay - start) / horizon)
                current_time = start + inst.customer_service_times[c-1]
                route_distance += inst.distance_matrix[prev, c]
                prev = c
            route_distance += inst.distance_matrix[prev, 0]
            return_time = current_time + inst.duration_matrix[prev, 0]
            for c in route:
                extra[c, 4] = (inst.depot_time_window[1]-return_time) / horizon
                base[c, 10] = route_distance / distance_scale
        x = np.concatenate([base, extra], axis=1).astype(np.float32)
        if not np.isfinite(x).all():
            raise ValueError("Non-finite predecision features")
        return x

    def observation(self, hidden, repair_id, remaining_fraction):
        if repair_id not in REPAIRERS:
            raise ValueError("Unknown repairer")
        x = torch.from_numpy(self.node_features())
        previous = x[self.last_removed].clone() if self.last_removed and self.last_accepted else torch.zeros((1, x.shape[1]))
        values = list(self.preference) + list((self.objectives-self.offset)/self.scales)
        values += [max(0., min(1., remaining_fraction)), self.stagnation/self.cfg.episode_steps,
                   float(self.last_accepted), self.last_reward, len(self.archive)/(self.steps+1),
                   self.instance.max_delay/max(self.instance.depot_time_window[1], 1.),
                   self.instance.n_customers / 250., max(self.q_values) / self.instance.n_customers]
        values += [float(repair_id == r) for r in REPAIRERS]
        return {
            "x": x, "edge_index_n": self.near.clone(),
            "edge_index_r0": torch.as_tensor(build_edge_index_routes([self.routes])[0], dtype=torch.long),
            "edge_index_r1": torch.as_tensor(build_edge_index_routes([self.routes], inverse=True)[0], dtype=torch.long),
            "state_": hidden.detach().cpu().reshape(1, -1).clone(), "input_": previous,
            "context": torch.tensor(values, dtype=torch.float32),
            "q_values": self.q_values, "fixed_q_index": self.fixed_q_index,
        }

    def deletion(self, action):
        q = int(action["q"])
        anchors, coefficients = action["anchors"], action["coefficients"]
        if q not in self.q_values or len(anchors) != min(self.cfg.anchors, q) or len(anchors) != len(coefficients):
            raise ValueError("Action does not match legal q/anchor support")
        if len(set(anchors)) != len(anchors) or any(c < 1 or c > self.instance.n_customers for c in anchors):
            raise ValueError("Invalid/duplicate anchors")
        if any(not 0 < c < 1 for c in coefficients):
            raise ValueError("Coefficients must be strictly between zero and one")
        removed = []
        for i, (anchor, coefficient) in enumerate(zip(anchors, coefficients)):
            quota = q // len(anchors) + int(i < q % len(anchors))
            selected = find_ruins(quota, anchor, coefficient, self.neighbors, self.routes, removed)
            if len(selected) != quota:
                raise RuntimeError("Upstream DPR did not fill the specified removal quota")
            removed.extend(selected)
        if len(removed) != q or len(set(removed)) != q or 0 in removed:
            raise RuntimeError("DPR removal set violates exact q/coverage")
        return removed

    def step(self, action, repair_id, seed, deadline, *, removal_override=None):
        started = time.perf_counter()
        removed = self.deletion(action) if removal_override is None else list(removal_override)
        if (int(action["q"]) not in self.q_values or len(removed) != int(action["q"])
                or len(set(removed)) != len(removed)
                or any(isinstance(c, bool) or not isinstance(c, (int, np.integer))
                       or not 1 <= c <= self.instance.n_customers for c in removed)):
            raise ValueError("Invalid exact-q deletion set")
        before = self.objectives.copy()
        result = self.api.repair(self.instance, self.routes, removed, repair_id, self.preference,
                                 {"mins": self.offset, "ranges": self.scales}, seed,
                                 deadline=min(deadline, started + self.cfg.repair_seconds))
        # Deadline covers the complete action, including feature/model time in caller.
        feasible = bool(result["feasible"])
        timeout = bool(result["timeout"])
        candidate, accepted = None, False
        status = result["reason"]
        if feasible:
            candidate = self.evaluate(result["routes"])
            np.testing.assert_allclose(candidate, result["objectives"], atol=1e-7, rtol=0)
            timeout = timeout or time.perf_counter() > deadline
        reward = float(self.cfg.failure_reward)
        if feasible and not timeout:
            reward = self.score(before) - self.score(candidate)
            point = {"routes": copy.deepcopy(result["routes"]), "objectives": candidate.tolist()}
            # Exact Pareto union; no HV calculation in the online training loop.
            if not any(np.all(np.asarray(p["objectives"]) <= candidate) for p in self.archive):
                self.archive = [p for p in self.archive if not np.all(candidate <= np.asarray(p["objectives"]))] + [point]
            if self.score(candidate) <= self.score(before):
                self.routes = copy.deepcopy(result["routes"])
                self.objectives = candidate.copy()
                accepted = True
        self.stagnation = 0 if accepted and reward > 0 else self.stagnation + 1
        self.last_reward, self.last_accepted, self.last_removed = reward, accepted, removed
        self.steps += 1
        if timeout:
            status = "timeout"
        return {
            "instance_id": self.instance.instance_id, "repair": repair_id, "preference": self.preference.tolist(),
            "repair_seed": int(seed),
            "q": int(action["q"]), "anchors": list(action["anchors"]),
            "deletion_source": "upstream_DPR" if removal_override is None else "explicit_baseline_override",
            "coefficients": list(action["coefficients"]), "removed": removed,
            "before_objectives": before.tolist(),
            "candidate_objectives": None if candidate is None else candidate.tolist(),
            "candidate_routes": None if candidate is None else copy.deepcopy(result["routes"]),
            "current_routes": copy.deepcopy(self.routes),
            "current_objectives": self.objectives.tolist(), "accepted": accepted,
            "feasible": feasible, "timeout": timeout, "status": status, "reward": reward,
            "archive_size": len(self.archive), "repair_seconds": result["seconds"],
            "environment_seconds": time.perf_counter()-started,
        }
