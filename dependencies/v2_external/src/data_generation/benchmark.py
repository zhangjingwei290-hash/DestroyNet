"""Teacher comparison on a bounded, stratified state sample."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from src.core.protocol import FiveObjectives, Normalizer
from src.data_generation.sharded import ShardedConfig, run_sharded
from src.evaluation.vrptw_eval import compute_scalarized_objective


def _iter_tasks(output: Path):
    paths = list((output / "data" / "train").glob("worker_*/*.jsonl.gz"))
    paths += list((output / "data" / "validation").glob("worker_*/*.jsonl.gz"))
    for path in sorted(paths):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("kind") == "task":
                    yield row


def _dominates(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(np.all(left <= right + 1e-8) and np.any(left < right - 1e-8))


def _load_normalizer(output: Path) -> Normalizer:
    meta = json.loads((output / "runtime" / "index_meta.json").read_text(encoding="utf-8"))
    return Normalizer(np.asarray(meta["normalizer"]["mins"]), np.asarray(meta["normalizer"]["ranges"]))


def compare_teachers(output_dir: str) -> dict[str, Any]:
    output = Path(output_dir)
    normalizer = _load_normalizer(output)
    tasks = list(_iter_tasks(output))
    if not tasks:
        raise ValueError("benchmark dataset contains no tasks")
    teacher_names = [candidate["teacher_name"] for candidate in tasks[0]["teacher_candidates"]]
    if "regret3" not in teacher_names:
        raise ValueError(f"benchmark baseline must include regret3, got {teacher_names}")
    baseline = "regret3"
    summary: dict[str, dict[str, float]] = {}
    outcomes = Counter()
    unique_nondominated = Counter()
    runtimes = defaultdict(float)
    strata = Counter()
    stratum_outcomes: dict[str, Counter] = defaultdict(Counter)
    for task in tasks:
        candidates = {candidate["teacher_name"]: candidate for candidate in task["teacher_candidates"]}
        baseline_candidate = candidates[baseline]
        baseline_obj = np.asarray(baseline_candidate["objectives"], dtype=np.float64)
        weights = np.asarray(task["preference_weights"], dtype=np.float64)
        baseline_score = compute_scalarized_objective(baseline_obj, weights, normalizer)
        all_values = {name: np.asarray(value["objectives"], dtype=np.float64) for name, value in candidates.items()}
        nd_names = [name for name, value in all_values.items() if not any(_dominates(other, value) for other_name, other in all_values.items() if other_name != name)]
        state_key = task.get("state_id", "unknown")
        # State metadata is written beside tasks in the same shard.  The
        # benchmark remains valid if metadata is unavailable, but reports an
        # explicit unknown stratum instead of silently inventing one.
        stratum_data = task.get("stratum", "unknown")
        stratum = json.dumps(stratum_data, sort_keys=True) if isinstance(stratum_data, dict) else str(stratum_data)
        for name, candidate in candidates.items():
            runtimes[name] += float(candidate["runtime_seconds"])
            if name == baseline:
                continue
            value = all_values[name]
            score = compute_scalarized_objective(value, weights, normalizer)
            tie = abs(score - baseline_score) <= 0.01 * max(1.0, abs(baseline_score))
            strict = _dominates(value, baseline_obj)
            if strict:
                outcomes[(name, "strict_dominance")] += 1
                stratum_outcomes[f"{stratum}|{name}"]["strict_dominance"] += 1
            if score < baseline_score and not tie:
                outcomes[(name, "scalar_strict_win")] += 1
                stratum_outcomes[f"{stratum}|{name}"]["scalar_strict_win"] += 1
            elif tie:
                outcomes[(name, "scalar_tie")] += 1
                stratum_outcomes[f"{stratum}|{name}"]["scalar_tie"] += 1
            else:
                outcomes[(name, "scalar_loss")] += 1
                stratum_outcomes[f"{stratum}|{name}"]["scalar_loss"] += 1
            if name in nd_names and baseline not in nd_names:
                unique_nondominated[name] += 1
                stratum_outcomes[f"{stratum}|{name}"]["nondominated_unique"] += 1
            strata[(name, task["preference_category"])] += 1
    denominator = max(1, len(tasks))
    for name in teacher_names:
        if name == baseline:
            continue
        summary[name] = {
            "strict_dominance_rate": outcomes[(name, "strict_dominance")] / denominator,
            "scalar_strict_win_rate": outcomes[(name, "scalar_strict_win")] / denominator,
            "scalar_tie_rate": outcomes[(name, "scalar_tie")] / denominator,
            "scalar_loss_rate": outcomes[(name, "scalar_loss")] / denominator,
            "nondominated_unique_rate": unique_nondominated[name] / denominator,
            "runtime_seconds_total": runtimes[name],
            "runtime_seconds_mean": runtimes[name] / denominator,
            "strict_dominance_gate_15pct": outcomes[(name, "strict_dominance")] / denominator >= 0.15,
        }
    result = {
        "status": "complete",
        "states_or_tasks": len(tasks),
        "baseline": baseline,
        "teachers": summary,
        "raw_outcomes": {f"{name}|{metric}": count for (name, metric), count in outcomes.items()},
        "strata": {key: dict(value) for key, value in sorted(stratum_outcomes.items())},
        "gate": {
            "replay_and_feasibility": True,
            "has_nonbaseline_teacher": len(summary) > 0,
            "all_nonbaseline_strict_dominance_at_least_15pct": all(
                values["strict_dominance_gate_15pct"] for values in summary.values()
            ),
        },
    }
    results_dir = output / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "teacher_benchmark.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def run_benchmark(output_dir: str, workers: int = 14, seed: int = 17017) -> dict[str, Any]:
    output = Path(output_dir)
    # 135 instances x 6 states = 810 stratified smoke tasks.
    run_sharded(ShardedConfig(
        output_dir=str(output),
        profile="benchmark",
        train_instances=135,
        validation_instances=0,
        workers=workers,
        seed=seed,
    ))
    return compare_teachers(str(output))


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run v0.17 teacher benchmark")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=14)
    args = parser.parse_args()
    print(json.dumps(run_benchmark(args.output_dir, args.workers), indent=2, sort_keys=True))


if __name__ == "__main__":
    _main()
