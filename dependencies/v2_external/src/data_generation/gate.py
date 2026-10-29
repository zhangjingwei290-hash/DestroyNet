"""Integrity gate for sharded v0.17 data."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from src.core.protocol import (
    ActionTrajectory,
    DestroyType,
    DestroyedState,
    PreferenceCategory,
    PreferenceTaskRecord,
    RepairAction,
    StateRecord,
    TeacherCandidate,
    Normalizer,
)
from src.core.replay import validate_trajectory_replay
from src.core.task_protocol import DEFAULT_PROTOCOL, TaskProtocol
from src.data_generation.sharded import (
    _hash_file,
    _instance_from_row,
    _objectives,
    _route,
    _solution,
)
from src.evaluation.vrptw_eval import compute_scalarized_objective, evaluate_solution, is_solution_feasible


def _rows(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _candidate(row: dict[str, Any]) -> TeacherCandidate:
    trajectory = ActionTrajectory([
        RepairAction(
            customer_id=int(action["customer_id"]),
            route_index=int(action["route_index"]),
            insert_position=int(action["insert_position"]),
            opens_new_route=bool(action["opens_new_route"]),
        )
        for action in row["actions"]
    ])
    return TeacherCandidate(
        teacher_name=row["teacher_name"],
        trajectory=trajectory,
        final_solution=_solution(row["final_solution"]),
        objectives=_objectives(row["objectives"]),
        is_feasible=bool(row["is_feasible"]),
        runtime_seconds=float(row["runtime_seconds"]),
    )


def _task(row: dict[str, Any]) -> PreferenceTaskRecord:
    return PreferenceTaskRecord(
        task_id=row["task_id"],
        state_id=row["state_id"],
        preference_category=PreferenceCategory(row["preference_category"]),
        preference_weights=np.asarray(row["preference_weights"], dtype=np.float64),
        teacher_candidates=[_candidate(candidate) for candidate in row["teacher_candidates"]],
        selected_teacher=row.get("selected_teacher"),
        selected_score=row.get("selected_score"),
    )


def validate_shards(output_dir: str, profile: str = "trainval", protocol: TaskProtocol = DEFAULT_PROTOCOL) -> dict[str, Any]:
    output = Path(output_dir)
    data = output / "data"
    runtime = output / "runtime"
    results_dir = output / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    index_meta = json.loads((runtime / "index_meta.json").read_text(encoding="utf-8"))
    protocol_normalizer = Normalizer(
        np.asarray(index_meta["normalizer"]["mins"], dtype=np.float64),
        np.asarray(index_meta["normalizer"]["ranges"], dtype=np.float64),
    )

    index_path = data / "instance_index.jsonl.gz"
    if not index_path.exists():
        raise FileNotFoundError(index_path)
    instances: dict[str, Any] = {}
    instance_split: dict[str, str] = {}
    instance_meta: dict[str, dict[str, Any]] = {}
    for row in _rows(index_path):
        instance_id = row["instance_id"]
        if instance_id in instances:
            errors.append(f"duplicate instance: {instance_id}")
        instances[instance_id] = _instance_from_row(row)
        instance_split[instance_id] = row["split"]
        instance_meta[instance_id] = row

    states: dict[str, StateRecord] = {}
    tasks: dict[str, PreferenceTaskRecord] = {}
    file_hashes: dict[str, str] = {str(index_path.relative_to(output)): _hash_file(index_path)}
    # Train and validation are physically separated so a split can be copied
    # or validated independently without relying on filename conventions.
    shard_paths = sorted(
        list((data / "train").glob("worker_*/*.jsonl.gz"))
        + list((data / "validation").glob("worker_*/*.jsonl.gz"))
    )
    for shard in shard_paths:
        file_hashes[str(shard.relative_to(output))] = _hash_file(shard)
        for row in _rows(shard):
            kind = row.get("kind")
            if kind == "instance":
                continue
            if kind == "state":
                if row["state_id"] in states:
                    errors.append(f"duplicate state: {row['state_id']}")
                    continue
                instance = instances.get(row["instance_id"])
                if instance is None:
                    errors.append(f"state references missing instance: {row['instance_id']}")
                    continue
                state = StateRecord(
                    state_id=row["state_id"],
                    instance=instance,
                    source_solution=_solution(row["source_solution"]),
                    source_objectives=_objectives(row["source_objectives"]),
                    destroyed_state=DestroyedState(_solution(row["partial_solution"]), list(row["removed_customers"])),
                    destroy_type=DestroyType(row["destroy_type"]),
                    destroy_ratio=float(row["destroy_ratio"]),
                    random_seed=int(row["random_seed"]),
                )
                states[state.state_id] = state
            elif kind == "task":
                if row["task_id"] in tasks:
                    errors.append(f"duplicate task: {row['task_id']}")
                    continue
                try:
                    tasks[row["task_id"]] = _task(row)
                except Exception as error:
                    errors.append(f"invalid task {row.get('task_id')}: {error!r}")
            else:
                errors.append(f"unknown record kind in {shard}: {kind!r}")

    states_per_instance = Counter(state.instance.instance_id for state in states.values())
    if profile == "trainval":
        expected_instances = protocol.n_train_instances + protocol.n_val_instances
        expected_tasks = 18000
        if len(instances) != expected_instances:
            errors.append(f"instance count {len(instances)} != {expected_instances}")
        if len(tasks) != expected_tasks:
            errors.append(f"task count {len(tasks)} != {expected_tasks}")
    elif profile in ("smoke", "benchmark"):
        expected_tasks = len(instances) * 6
        if len(tasks) != expected_tasks:
            errors.append(f"{profile} task count {len(tasks)} != {expected_tasks}")
    else:
        errors.append(f"unknown profile {profile!r}")

    if any(count != 6 for count in states_per_instance.values()):
        errors.append("each instance must have exactly six destroyed states")
    if len(states) != len(instances) * 6:
        errors.append(f"state count {len(states)} != {len(instances) * 6}")

    destroy_counts = Counter(state.destroy_type.value for state in states.values())
    required_destroy_types = {"route_repack", "worst_edge_related", "makespan_segment", "waiting_segment", "tardiness_segment", "generic_random", "generic_mixed"}
    if profile == "trainval" and not required_destroy_types.issubset(destroy_counts):
        errors.append(f"missing destroy types: {sorted(required_destroy_types - set(destroy_counts))}")

    split_task_counts = Counter()
    category_counts = defaultdict(Counter)
    teacher_runtime = defaultdict(float)
    teacher_counts = Counter()
    for task in tasks.values():
        state = states.get(task.state_id)
        if state is None:
            errors.append(f"task references missing state: {task.task_id}")
            continue
        split = instance_split.get(state.instance.instance_id)
        split_task_counts[split] += 1
        category_counts[split][task.preference_category.value] += 1
        for candidate in task.teacher_candidates:
            teacher_counts[candidate.teacher_name] += 1
            teacher_runtime[candidate.teacher_name] += candidate.runtime_seconds
            try:
                if not candidate.is_feasible or not is_solution_feasible(candidate.final_solution, state.instance):
                    errors.append(f"infeasible candidate: {task.task_id}/{candidate.teacher_name}")
                if not validate_trajectory_replay(state.destroyed_state, candidate.trajectory, candidate.final_solution):
                    errors.append(f"replay mismatch: {task.task_id}/{candidate.teacher_name}")
                if evaluate_solution(candidate.final_solution, state.instance) != candidate.objectives:
                    errors.append(f"objective mismatch: {task.task_id}/{candidate.teacher_name}")
            except Exception as error:
                errors.append(f"candidate validation error {task.task_id}/{candidate.teacher_name}: {error!r}")
        if profile == "trainval":
            feasible = [candidate for candidate in task.teacher_candidates if candidate.is_feasible]
            scores = {
                candidate.teacher_name: compute_scalarized_objective(
                    candidate.objectives.to_array(), task.preference_weights, protocol_normalizer
                )
                for candidate in feasible
            }
            expected = min(feasible, key=lambda candidate: (scores[candidate.teacher_name], candidate.teacher_name))
            if task.selected_teacher != expected.teacher_name:
                errors.append(f"selected teacher mismatch: {task.task_id}")
            if task.selected_score is None or not np.isclose(task.selected_score, scores[expected.teacher_name], atol=1e-8):
                errors.append(f"selected score mismatch: {task.task_id}")

    if profile == "trainval":
        if split_task_counts["train"] != 15000:
            errors.append(f"train task count {split_task_counts['train']} != 15000")
        if split_task_counts["validation"] != 3000:
            errors.append(f"validation task count {split_task_counts['validation']} != 3000")
        expected_train_categories = {"extreme": 4500, "biased_mix": 4500, "general_mix": 3750, "near_uniform": 2250}
        expected_val_categories = {"extreme": 900, "biased_mix": 900, "general_mix": 750, "near_uniform": 450}
        if dict(category_counts["train"]) != expected_train_categories:
            errors.append(f"train preference quotas mismatch: {dict(category_counts['train'])}")
        if dict(category_counts["validation"]) != expected_val_categories:
            errors.append(f"validation preference quotas mismatch: {dict(category_counts['validation'])}")

    manifest = {
        "schema_version": protocol.schema_version,
        "protocol_hash": protocol.compute_protocol_hash(),
        "profile": profile,
        "files": file_hashes,
        "counts": {"instances": len(instances), "states": len(states), "tasks": len(tasks), "by_split": dict(split_task_counts)},
        "destroy_counts": dict(destroy_counts),
        "preference_categories": {split: dict(counts) for split, counts in category_counts.items()},
        "teacher_counts": dict(teacher_counts),
        "teacher_runtime_seconds": dict(teacher_runtime),
    }
    manifest_payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    manifest["manifest_sha256"] = __import__("hashlib").sha256(manifest_payload).hexdigest()
    (results_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    gate = {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "replay_valid": not any("replay" in error or "candidate validation" in error for error in errors),
        "counts": manifest["counts"],
        "destroy_counts": manifest["destroy_counts"],
        "preference_categories": manifest["preference_categories"],
        "teacher_counts": manifest["teacher_counts"],
        "teacher_runtime_seconds": manifest["teacher_runtime_seconds"],
    }
    (results_dir / "data_gate.json").write_text(json.dumps(gate, indent=2, sort_keys=True), encoding="utf-8")
    return gate


def _main() -> None:
    parser = argparse.ArgumentParser(description="Validate v0.17 sharded data")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile", default="trainval", choices=("smoke", "benchmark", "trainval"))
    args = parser.parse_args()
    result = validate_shards(args.output_dir, args.profile)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    _main()
