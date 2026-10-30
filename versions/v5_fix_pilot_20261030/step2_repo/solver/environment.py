"""Physical environment, predecision ALNS state and common timed execution.

All modes call LocalSession.prepare/finish. Training interleaves independent CPU
lanes and charges each lane its own CPU work and a measured single-graph forward;
other lanes' serial solver work and optimizer work do not age its solver clock.
Evaluation uses a single lane plus an absolute outer wall-time cutoff.
"""
import copy
import time
from types import SimpleNamespace
import numpy as np
import torch
from alns.select import RouletteWheel
from alns.Outcome import Outcome
from .base_environment import Environment as OriginalEnvironment, REPAIRERS
from .config import CONTEXT, q_values

DESTROYERS = ("random", "related", "worst_distance", "route")


class Adaptive:
    def __init__(self, nq, heuristic=False):
        self.heuristic = heuristic
        self.nq = nq
        self.selector = RouletteWheel([5., 3., 0., 0.], .8, len(DESTROYERS)*nq if heuristic else 1, 4)
        self.count = np.zeros(4); self.seconds = np.zeros(4)
        self.success = np.zeros(4); self.improvement = np.zeros(4)
        self.utility_count = np.zeros(4)
        self.best = float("inf")
        self.recent_seconds = np.zeros(4)
        self.recent_success = np.zeros(4)
        self.recent_improvement = np.zeros(4)

    def choose(self, rng):
        return tuple(int(x) for x in self.selector(rng, None, None))

    def features(self):
        w = self.selector.repair_weights
        den = np.maximum(self.count, 1.)
        return np.concatenate([w/w.sum(), self.recent_seconds/300., self.recent_success,
                               self.recent_improvement, self.count/1000.]).tolist()

    def update(self, d, r, result, final_score):
        outcome = Outcome.REJECT
        if result["accepted"]:
            outcome = Outcome.BEST if final_score < self.best else Outcome.BETTER
        self.best = min(self.best, final_score)
        self.selector.update(None, d, r, outcome)
        rate = 1. if self.count[r] == 0 else .2
        for key, value in (("seconds", result["environment_seconds"]),
                           ("success", float(result["feasible"] and not result["timeout"])),
                           ("improvement", result["reward"])):
            arr = getattr(self, "recent_"+key)
            arr[r] += rate*(value-arr[r])
        self.count[r] += 1; self.seconds[r] += result["environment_seconds"]
        self.success[r] += float(result["feasible"] and not result["timeout"])
        self.improvement[r] += result["reward"]
        self.utility_count[r] += 1


class Environment(OriginalEnvironment):
    def __init__(self, pool, iid, cfg, weight, routes=None):
        inst, row, box, _ = pool.load(iid)
        from .repair import repair
        api = SimpleNamespace(**vars(pool.api)); api.repair = repair
        from .parallel_context import current_client
        client=current_client()
        if client is not None:api.repair=client.bound_repair(iid)
        super().__init__(api, inst, row, box, q_values(inst.n_customers), weight, cfg)
        if routes is not None:
            self.routes = copy.deepcopy(routes)
            self.objectives = self.evaluate(self.routes)
            self.archive = [dict(routes=copy.deepcopy(self.routes), objectives=self.objectives.tolist())]

    def observation(self, hidden, repair_id, adaptive, global_remaining, global_budget, local_remaining, local_budget):
        # Reuse established node/history/edge extraction, replace the obsolete
        # 150-step context completely rather than leaving mislabeled progress.
        obs = super().observation(hidden, repair_id, local_remaining/local_budget)
        values = list(self.preference)+list((self.objectives-self.offset)/self.scales)
        values += [max(0., global_remaining)/300., 1-max(0., global_remaining)/global_budget,
                   max(0., local_remaining)/300., max(0., local_remaining)/local_budget,
                   min(1., self.stagnation/10.), float(self.last_accepted), self.last_reward,
                   len(self.archive)/(self.steps+1), self.instance.max_delay/max(self.instance.depot_time_window[1], 1.),
                   self.instance.n_customers/250., max(self.q_values)/self.instance.n_customers]
        values += [float(repair_id == r) for r in REPAIRERS]
        values += adaptive.features()
        if len(values) != len(CONTEXT) or not np.isfinite(values).all():
            raise ValueError("Invalid predecision context")
        obs["context"] = torch.tensor(values, dtype=torch.float32)
        obs["routes"] = copy.deepcopy(self.routes)
        obs["search_state"] = dict(last_accepted=bool(self.last_accepted),stagnation=int(self.stagnation))
        return obs

    def deletion(self, action):
        removed=list(action["removed"])
        if action['q']==self.instance.n_customers:
            if removed!=list(range(1,self.instance.n_customers+1)):raise ValueError("Full removal must be canonical")
        return removed

    def step(self, action, repair_id, seed, deadline, *, removal_override=None):
        started = time.perf_counter()
        removed = self.deletion(action) if removal_override is None else list(removal_override)
        is_heur=removal_override is not None
        if (not isinstance(action["q"],(int,np.integer)) or isinstance(action["q"],bool)
                or not 1<=int(action["q"])<=self.instance.n_customers
                or (is_heur and action["q"] not in self.q_values)
                or len(removed)!=action["q"] or len(set(removed))!=len(removed)
                or any(not isinstance(c,(int,np.integer)) or isinstance(c,bool)
                       or not 1<=c<=self.instance.n_customers for c in removed)):
            raise ValueError("Invalid exact-q deletion set")
        before = self.objectives.copy()
        result = self.api.repair(self.instance, self.routes, removed, repair_id, self.preference,
            {"mins": self.offset, "ranges": self.scales}, seed, min(deadline, started+self.cfg.repair_seconds))
        candidate = None
        feasible, timeout = bool(result["feasible"]), bool(result["timeout"])
        accepted, reward, candidate_finished = False, 0., None
        fatal = ("source_infeasible", "invalid_removal_set", "removal_cap_exceeded", "nonfinite_candidate_score")
        if result["reason"] in fatal:
            raise ValueError(f"Scientific/legality failure: {result['reason']}")
        if feasible and (result.get("failed", False) or result["reason"] != "ok"
                         or result.get("routes") is None or result.get("objectives") is None):
            raise ValueError("Contradictory successful repair result")
        if not feasible and (result.get("routes") is not None or result.get("objectives") is not None):
            raise ValueError("Failed repair returned usable routes/objectives")
        if not feasible and not timeout and result["reason"] not in ("no_legal_insertion", "final_solution_infeasible"):
            raise ValueError(f"Unknown repair failure: {result['reason']}")
        if feasible:
            candidate = self.evaluate(result["routes"])
            np.testing.assert_allclose(candidate, result["objectives"], atol=1e-7, rtol=0)
            checked_at = time.perf_counter()
            timeout |= checked_at > deadline
            if not timeout:
                candidate_finished = checked_at
                candidate_score, before_score = self.score(candidate), self.score(before)
                point = dict(routes=copy.deepcopy(result["routes"]), objectives=candidate.tolist())
                if not any(np.all(np.asarray(p["objectives"]) <= candidate) for p in self.archive):
                    self.archive = [p for p in self.archive if not np.all(candidate <= np.asarray(p["objectives"]))]+[point]
                if candidate_score < before_score:
                    self.routes = copy.deepcopy(result["routes"]); self.objectives = candidate.copy()
                    accepted = True; reward = before_score-candidate_score

        self.last_removed, self.last_accepted, self.last_reward = removed, accepted, reward
        self.stagnation = 0 if accepted else self.stagnation+1
        self.steps += 1
        return dict(repair=repair_id, repair_seed=int(seed), repair_counters=result.get("counters", {}), q=action["q"], removed=removed, accepted=accepted, reward=reward,
            feasible=feasible, failed=not feasible and not timeout, timeout=timeout, reason="timeout" if timeout else result["reason"],
            candidate_objectives=None if candidate is None or timeout else candidate.tolist(),
            candidate_finished_perf=candidate_finished,
            before_objectives=before.tolist(), current_objectives=self.objectives.tolist(),
            repair_seconds=result["seconds"], environment_seconds=time.perf_counter()-started)


def classical_removal(env, d, rng):
    kind, qi = divmod(d, len(env.q_values))
    q = env.q_values[qi]; n = env.instance.n_customers
    if kind == 0:
        return rng.choice(np.arange(1, n+1), q, replace=False).tolist()
    if kind == 1:
        anchor = int(rng.integers(1, n+1))
        return [c for c in env.neighbors[anchor] if c != 0][:q]
    if kind == 2:
        costs = []
        dist = env.instance.distance_matrix
        for route in env.routes:
            seq = [0]+route+[0]
            costs += [(float(dist[seq[i-1], c]+dist[c, seq[i+1]]-dist[seq[i-1], seq[i+1]]), c)
                      for i, c in enumerate(seq[1:-1], 1)]
        return [c for _, c in sorted(costs, key=lambda x: (-x[0], x[1]))[:q]]
    if kind == 3:
        order = rng.permutation(len(env.routes))
        customers = [c for i in order for c in env.routes[i]]
        return customers[:q]
    raise ValueError("Unknown destroyer")


class LocalSession:
    def __init__(self, env, adaptive, seed, global_budget, global_remaining, local_budget, outer_deadline=None):
        self.env, self.adaptive = env, adaptive
        self.rng = np.random.default_rng(seed)
        self.global_budget, self.global_start = global_budget, global_remaining
        # Keep one segment from consuming sixteenth-direction coverage on N=50.
        # Same effective budget in collection, adaptation and deployment.
        self.budget = min(local_budget, global_budget/16., global_remaining)
        self.elapsed = 0.
        self.outer_deadline = outer_deadline
        self.solver_task_deadline = None
        self.hidden = torch.zeros((1, env.cfg.hidden_dim))
        self.pending = None
        self.records = []
        self.adaptive.best = min(self.adaptive.best, env.score(env.objectives))

    def ending(self):
        now = time.perf_counter()
        reason = ('local_budget' if self.budget-self.elapsed <= 0 else
                  'solver_budget' if self.solver_task_deadline is not None and now >= self.solver_task_deadline else
                  'external_resource' if self.outer_deadline is not None and now >= self.outer_deadline else
                  'continuing')
        terminated = reason in ('local_budget', 'solver_budget')
        truncated = reason == 'external_resource'
        return dict(terminated=terminated, truncated=truncated,
                    terminal=terminated or truncated, ending_reason=reason)

    def finalize(self):
        ending = self.ending()
        if self.records:
            self.records[-1].update(ending)
        return ending

    def remaining(self):
        left = self.budget-self.elapsed
        if self.solver_task_deadline is not None:
            left = min(left, self.solver_task_deadline-time.perf_counter())
        if self.outer_deadline is not None:
            left = min(left, self.outer_deadline-time.perf_counter())
        return max(0., left)

    def prepare(self, include_observation=True):
        if self.pending is not None:
            raise RuntimeError("Unfinished decision")
        started = time.perf_counter()
        d, r = self.adaptive.choose(self.rng)
        self.decision_global_used=1-max(0.,self.global_start-self.elapsed)/self.global_budget
        self.decision_stall=min(1.,self.env.stagnation/10.)
        self.decision_last_accepted=bool(self.env.last_accepted)
        self.decision_stagnation=int(self.env.stagnation)
        obs = self.env.observation(self.hidden, REPAIRERS[r], self.adaptive,
            max(0., self.global_start-self.elapsed), self.global_budget, self.remaining(), self.budget) if include_observation else None
        self.elapsed += time.perf_counter()-started
        self.pending = d, r
        return obs

    def finish(self, action, hidden, inference_seconds=0.):
        if self.pending is None:
            raise RuntimeError("No prepared decision")
        d, r = self.pending; self.pending = None
        self.elapsed += inference_seconds
        started = time.perf_counter()
        override = classical_removal(self.env, d, self.rng) if self.adaptive.heuristic else None
        if override is not None:
            action = dict(q=len(override), anchors=[], coefficients=[])
        deadline = started+max(0., self.budget-self.elapsed)
        if self.solver_task_deadline is not None:
            deadline = min(deadline, self.solver_task_deadline)
        if self.outer_deadline is not None:
            deadline = min(deadline, self.outer_deadline)
        result = self.env.step(action, REPAIRERS[r], int(self.rng.integers(0, 2**31-1)), deadline,
                               removal_override=override)
        self.adaptive.update(d, r, result, self.env.score(self.env.objectives))
        self.hidden = hidden.reshape(1, -1).clone()
        self.elapsed += time.perf_counter()-started
        result.update(global_used_fraction_before=self.decision_global_used,stall_fraction_before=self.decision_stall,
                      last_accepted_before=self.decision_last_accepted,stagnation_before=self.decision_stagnation,
                      local_elapsed=self.elapsed, local_budget=self.budget,
                      **self.ending(), inference_seconds=inference_seconds,
                      action=action)
        self.records.append(result)
        return result
