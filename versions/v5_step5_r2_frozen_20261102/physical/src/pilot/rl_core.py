"""Host-independent destroy/repair rollouts used by RepairNet RL."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.nn import functional as F

from src.core.protocol import DestroyType, PreferenceCategory, RepairAction, Route, Solution
from src.data_generation.sharded import _solution
from src.destroy import destroy
from src.evaluation.vrptw_eval import evaluate_solution, is_solution_feasible
from src.pilot.config import ROOT
from src.pilot.data import canonical, load_shard, read_json, score, select_teacher, sha
from src.pilot.engine import apply_action, complete_ok, enumerate_actions
from src.pilot.model import tensor_features
from src.preference.generation import generate_preference_weights


PREFERENCE_FRACTIONS = (
    (PreferenceCategory.EXTREME, 0.30),
    (PreferenceCategory.BIASED_MIX, 0.30),
    (PreferenceCategory.GENERAL_MIX, 0.25),
    (PreferenceCategory.NEAR_UNIFORM, 0.15),
)


@dataclass(frozen=True)
class SourceCandidate:
    routes: tuple[tuple[int, ...], ...]
    objectives: tuple[float, ...]

    def solution(self) -> Solution:
        return Solution([Route(list(route)) for route in self.routes])


@dataclass(frozen=True)
class RepairSpec:
    instance_id: str
    source_index: int
    destroy_type: str
    destroy_ratio: float
    destroy_seed: int
    preference_category: str
    preference: tuple[float, ...]


@dataclass
class RepairTask:
    spec: RepairSpec
    instance: object
    source: Solution
    source_objectives: np.ndarray
    partial_routes: list[list[int]]
    removed_customers: list[int]
    state_hash: str


@dataclass
class Rollout:
    actions: list[dict]
    final_routes: list[list[int]]
    objectives: list[float]
    completed: bool
    feasible: bool
    failure: str | None


class RLDataset:
    """Pilot reader that keeps the final test split closed during training."""

    def __init__(self, run, splits=("train", "dev")):
        self.run = Path(run)
        self.manifest = read_json(self.run / "pilot_manifest.json")
        for relative, expected in self.manifest["code_hashes"].items():
            if sha(ROOT / relative) != expected:
                raise ValueError(f"Code changed after prepare: {relative}")
        self.normalization = self.manifest["normalization"]
        self.items = {}
        self.instances = {}
        self.states = {}
        self.loaded_splits = set()
        for split in splits:
            self.load_split(split)

    def load_split(self, split):
        if split in self.loaded_splits:
            return
        ids = []
        for relative in self.manifest["splits"][split]:
            source_file = Path(self.manifest["source"]) / relative
            if sha(source_file) != self.manifest["source_hashes"][relative]:
                raise ValueError(f"Source changed after prepare: {source_file}")
            row, states, tasks = load_shard(source_file)
            self.instances[row["instance_id"]] = self._instance(row)
            self.states.update(states)
            for task in tasks:
                task = dict(task, instance_id=row["instance_id"], split=split)
                task["label"] = select_teacher(task, self.normalization)
                self.items[task["task_id"]] = task
                ids.append(task["task_id"])
        setattr(self, split, sorted(ids))
        self.loaded_splits.add(split)

    @staticmethod
    def _instance(row):
        from src.data_generation.sharded import _instance_from_row
        return _instance_from_row(row)

    def get(self, task_id):
        task = self.items[task_id]
        return self.instances[task["instance_id"]], self.states[task["state_id"]], task


def _quotas(total: int, fractions: Iterable[float]) -> list[int]:
    fractions = list(fractions)
    raw = np.asarray(fractions, dtype=float) * total
    result = np.floor(raw).astype(int)
    for index in np.argsort(-(raw - result))[: total - int(result.sum())]:
        result[index] += 1
    return result.tolist()


def ratio_counts(epoch: int, total: int) -> dict[float, int]:
    if epoch == 1:
        return {0.10: total, 0.20: 0, 0.30: 0}
    if epoch == 2:
        low, medium = _quotas(total, (0.50, 0.50))
        return {0.10: low, 0.20: medium, 0.30: 0}
    low, medium, high = _quotas(total, (0.25, 0.50, 0.25))
    return {0.10: low, 0.20: medium, 0.30: high}


def build_solution_pools(dataset, split: str) -> dict[str, list[SourceCandidate]]:
    """Collect generic feasible starting solutions without crossing the split."""
    pools: dict[str, dict[tuple, SourceCandidate]] = {}
    for task_id in getattr(dataset, split):
        instance, state, task = dataset.get(task_id)
        bucket = pools.setdefault(instance.instance_id, {})
        rows = [state["source_solution"]]
        rows.extend(
            candidate["final_solution"]
            for candidate in task["teacher_candidates"]
            if candidate["is_feasible"]
        )
        for routes in rows:
            key = canonical(routes)
            if key in bucket:
                continue
            solution = _solution(routes)
            if not is_solution_feasible(solution, instance):
                continue
            objectives = evaluate_solution(solution, instance).to_array()
            bucket[key] = SourceCandidate(
                tuple(tuple(route) for route in routes if route), tuple(float(x) for x in objectives)
            )
    result = {instance_id: list(bucket.values()) for instance_id, bucket in pools.items()}
    if not result or any(not values for values in result.values()):
        raise ValueError(f"No feasible source solutions for split {split}")
    return result


def make_specs(
    pools: dict[str, list[SourceCandidate]],
    *,
    seed: int,
    epoch: int,
    total: int,
    shared_fraction: float = 0.20,
) -> list[RepairSpec]:
    """Create a deterministic, stratified online-RL schedule."""
    if total < 1 or epoch < 1:
        raise ValueError("total and epoch must be positive")
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 17017]))

    kinds = list(DestroyType)
    kind_counts = _quotas(total, [1 / len(kinds)] * len(kinds))
    kind_rows = [kind for kind, count in zip(kinds, kind_counts) for _ in range(count)]
    ratio_rows = [ratio for ratio, count in ratio_counts(epoch, total).items() for _ in range(count)]
    category_counts = _quotas(total, [fraction for _, fraction in PREFERENCE_FRACTIONS])
    category_rows = [category for (category, _), count in zip(PREFERENCE_FRACTIONS, category_counts)
                     for _ in range(count)]
    rng.shuffle(kind_rows)
    rng.shuffle(ratio_rows)
    rng.shuffle(category_rows)

    instance_ids = sorted(pools)
    instance_rows = (instance_ids * ((total + len(instance_ids) - 1) // len(instance_ids)))[:total]
    rng.shuffle(instance_rows)
    sources = [int(rng.integers(len(pools[instance_id]))) for instance_id in instance_rows]
    destroy_seeds = rng.integers(0, 2**32, size=total, dtype=np.uint64).tolist()

    # Reuse 20% of states with a second preference. Pair only equal
    # destroy-type/ratio slots so all requested marginals remain exact.
    target_pairs = int(total * shared_fraction) // 2
    buckets: dict[tuple[str, float], list[int]] = {}
    for index, (kind, ratio) in enumerate(zip(kind_rows, ratio_rows)):
        buckets.setdefault((kind.value, ratio), []).append(index)
    pairs = 0
    for indices in buckets.values():
        rng.shuffle(indices)
        for left, right in zip(indices[::2], indices[1::2]):
            if pairs >= target_pairs:
                break
            instance_rows[right] = instance_rows[left]
            sources[right] = sources[left]
            destroy_seeds[right] = destroy_seeds[left]
            pairs += 1
        if pairs >= target_pairs:
            break

    specs = []
    for index in range(total):
        weights, actual = generate_preference_weights(category_rows[index], rng)
        specs.append(RepairSpec(
            instance_id=instance_rows[index],
            source_index=sources[index],
            destroy_type=kind_rows[index].value,
            destroy_ratio=float(ratio_rows[index]),
            destroy_seed=int(destroy_seeds[index]),
            preference_category=actual.value,
            preference=tuple(float(x) for x in weights),
        ))
    return specs


def materialize(spec: RepairSpec, dataset, pools) -> RepairTask:
    instance = dataset.instances[spec.instance_id]
    candidate = pools[spec.instance_id][spec.source_index]
    source = candidate.solution()
    destroyed = destroy(
        source,
        instance,
        spec.destroy_type,
        spec.destroy_ratio,
        seed=spec.destroy_seed,
    )
    return RepairTask(
        spec=spec,
        instance=instance,
        source=source,
        source_objectives=np.asarray(candidate.objectives, dtype=np.float64),
        partial_routes=[list(route.customers) for route in destroyed.state.partial_solution.routes],
        removed_customers=list(destroyed.state.removed_customers),
        state_hash=destroyed.state_hash,
    )


def rollout(model, task: RepairTask, device: str, *, greedy: bool, temperature: float = 1.0,
            generator: torch.Generator | None = None) -> Rollout:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    partial = _solution(task.partial_routes)
    remaining = list(task.removed_customers)
    actions_taken: list[dict] = []
    failure = None
    if hasattr(model, "reset_context"):
        model.reset_context()
    model.eval()
    preference = torch.as_tensor(task.spec.preference, dtype=torch.float32, device=device)
    with torch.no_grad():
        for _ in range(len(remaining)):
            options = enumerate_actions(task.instance, partial, remaining)
            if not options.mask.any():
                failure = "no_legal_action"
                break
            logits = model(tensor_features(task.instance, partial, remaining, options, device), preference)
            if greedy:
                index = int(logits.argmax().item())
            else:
                probabilities = torch.softmax(logits / temperature, dim=0)
                index = int(torch.multinomial(probabilities, 1, generator=generator).item())
            action = options.actions[index]
            actions_taken.append(asdict(action))
            partial, remaining = apply_action(partial, remaining, action)
    feasible = complete_ok(task.instance, partial, remaining)
    if not feasible and failure is None:
        failure = "final_infeasible"
    objectives = evaluate_solution(partial, task.instance).to_array().tolist()
    if hasattr(model, "reset_context"):
        model.reset_context()
    return Rollout(
        actions=actions_taken,
        final_routes=[list(route.customers) for route in partial.routes],
        objectives=[float(x) for x in objectives],
        completed=not remaining,
        feasible=feasible,
        failure=failure,
    )


def trajectory_policy_stats(model, task: RepairTask, rollout_result: Rollout, device: str,
                            temperature: float = 1.0):
    """Replay one sampled trajectory with gradients and bounded memory."""
    if not rollout_result.actions:
        raise ValueError("Cannot replay an empty trajectory")
    partial = _solution(task.partial_routes)
    remaining = list(task.removed_customers)
    preference = torch.as_tensor(task.spec.preference, dtype=torch.float32, device=device)
    log_probabilities, entropies = [], []
    if hasattr(model, "reset_context"):
        model.reset_context()
    model.train()
    for raw in rollout_result.actions:
        options = enumerate_actions(task.instance, partial, remaining)
        action = RepairAction(**raw)
        index = options.actions.index(action)
        if not options.mask[index]:
            raise ValueError("Sampled trajectory became illegal during replay")
        logits = model(tensor_features(task.instance, partial, remaining, options, device), preference)
        log_probs = F.log_softmax(logits / temperature, dim=0)
        probabilities = log_probs.exp()
        log_probabilities.append(log_probs[index])
        legal_count = int(options.mask.sum())
        entropy = -(probabilities[options.mask] * log_probs[options.mask]).sum()
        if legal_count > 1:
            entropy = entropy / np.log(legal_count)
        else:
            entropy = entropy * 0.0
        entropies.append(entropy)
        partial, remaining = apply_action(partial, remaining, action)
    if remaining:
        raise ValueError("Sampled trajectory is incomplete")
    result = torch.stack(log_probabilities).mean(), torch.stack(entropies).mean()
    return result


def teacher_bc_loss(model, instance, state, task, device: str):
    partial = _solution(state["partial_solution"])
    remaining = list(state["removed_customers"])
    preference = torch.as_tensor(task["preference_weights"], dtype=torch.float32, device=device)
    losses = []
    if hasattr(model, "reset_context"):
        model.reset_context()
    model.train()
    for raw in task["label"]["actions"]:
        options = enumerate_actions(instance, partial, remaining)
        action = RepairAction(**raw)
        index = options.actions.index(action)
        if not options.mask[index]:
            raise ValueError(f"Masked BC action in {task['task_id']}")
        logits = model(tensor_features(instance, partial, remaining, options, device), preference)
        losses.append(F.cross_entropy(logits[None], torch.tensor([index], device=device)))
        partial, remaining = apply_action(partial, remaining, action)
    if not losses or remaining:
        raise ValueError(f"Invalid BC trajectory in {task['task_id']}")
    return torch.stack(losses).mean()


def leave_one_out_advantages(rewards, tolerance: float = 1e-6):
    rewards = np.asarray(rewards, dtype=np.float64)
    if rewards.ndim != 1 or len(rewards) < 2:
        raise ValueError("At least two rewards are required")
    if float(np.ptp(rewards)) <= tolerance:
        return np.zeros_like(rewards), False
    baseline = (rewards.sum() - rewards) / (len(rewards) - 1)
    return rewards - baseline, True


def scalar_score(objectives, preference, normalization) -> float:
    return score(objectives, preference, normalization)
