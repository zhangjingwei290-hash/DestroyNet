"""Resumable, process-parallel v0.17 data generation.

The coordinator creates a deterministic instance/preference index first.  Each
worker then owns a disjoint instance range and writes only its own gzip JSONL
shards.  No worker mutates a shared dataset file.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.core.protocol import (
    ActionTrajectory,
    DestroyType,
    DestroyedState,
    FiveObjectives,
    Normalizer,
    PreferenceCategory,
    PreferenceTaskRecord,
    RepairAction,
    Route,
    Solution,
    StateRecord,
    TeacherCandidate,
    VRPTWInstance,
)
from src.core.replay import validate_trajectory_replay
from src.core.task_protocol import DEFAULT_PROTOCOL, TaskProtocol
from src.data_generation.pipeline import (
    GenerationConfig,
    _make_base_solution,
    _normalizer,
    _shape_time_windows,
    _teacher,
    destroy_solution,
)
from src.evaluation.vrptw_eval import compute_scalarized_objective, evaluate_solution, is_solution_feasible
from src.preference.generation import (
    generate_main_task_preferences,
    generate_preference_weights,
    generate_shared_state_preferences,
)


WORKER_COUNT = 14
MAX_INFLIGHT_TASKS = 28
FLUSH_EVERY_RECORDS = 100
MIN_AVAILABLE_MEMORY_GB = 4.0
MAX_GENERATION_MEMORY_GB = 20.0


@dataclass(frozen=True)
class ShardedConfig:
    output_dir: str
    seed: int = 17017
    workers: int = WORKER_COUNT
    max_inflight_tasks: int = MAX_INFLIGHT_TASKS
    flush_every_records: int = FLUSH_EVERY_RECORDS
    train_instances: int = 2000
    validation_instances: int = 400
    sizes: tuple[int, ...] = (50, 150, 250)
    instance_types: tuple[str, ...] = ("R", "C", "RC")
    teacher_names: tuple[str, ...] = (
        "regret3",
        "objective_greedy",
        "bounded_beam",
    )
    max_insertion_candidates_per_customer: int = 12
    beam_width: int = 4
    max_beam_expansions: int = 500
    profile: str = "trainval"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    encoded = json.dumps(payload, default=_json_default, sort_keys=True, indent=2)
    temp.write_text(encoded, encoding="utf-8")
    for attempt in range(5):
        try:
            os.replace(temp, path)
            return
        except PermissionError:
            if attempt == 4:
                path.write_text(encoded, encoding="utf-8")
                temp.unlink(missing_ok=True)
                return
            time.sleep(0.2 * (attempt + 1))


def _write_jsonl_gz(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with gzip.open(temp, "wt", encoding="utf-8", compresslevel=6) as handle:
        for row in rows:
            handle.write(json.dumps(row, default=_json_default, sort_keys=True))
            handle.write("\n")
    os.replace(temp, path)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resource_snapshot() -> dict[str, Any]:
    try:
        import psutil
        memory = psutil.virtual_memory()
        return {
            "cpu_percent": psutil.cpu_percent(interval=0.05),
            "available_memory_gb": memory.available / (1024 ** 3),
            "used_memory_gb": memory.used / (1024 ** 3),
            "total_memory_gb": memory.total / (1024 ** 3),
        }
    except ImportError:
        return {"psutil_available": False}


def _worker_snapshot(started: float, completed_instances: int, completed_records: int, idle_seconds: float = 0.0) -> dict[str, Any]:
    """Return worker-local progress and resource counters without shared writes."""
    payload: dict[str, Any] = {
        "completed_instances": completed_instances,
        "completed_records": completed_records,
        "elapsed_seconds": time.monotonic() - started,
        "records_per_minute": completed_records / max((time.monotonic() - started) / 60.0, 1e-9),
        "idle_seconds": idle_seconds,
        "io_wait_seconds": 0.0,
    }
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory = process.memory_info()
        cpu_times = process.cpu_times()
        io = process.io_counters()
        payload.update({
            "pid": os.getpid(),
            "cpu_percent": process.cpu_percent(interval=None),
            "rss_gb": memory.rss / (1024 ** 3),
            "cpu_user_seconds": cpu_times.user,
            "cpu_system_seconds": cpu_times.system,
            "io_read_bytes": io.read_bytes,
            "io_write_bytes": io.write_bytes,
        })
    except (ImportError, OSError):
        payload["psutil_available"] = False
    return payload


def _part_counts(path: Path) -> tuple[int, int]:
    instances = 0
    records = 0
    for row in _iter_jsonl_gz(path):
        if row.get("kind") == "instance":
            instances += 1
        elif row.get("kind") == "task":
            records += 1
    return instances, records


def _obj(objective: FiveObjectives) -> list[float]:
    return objective.to_array().tolist()


def _route(solution: Solution) -> list[list[int]]:
    return [list(route.customers) for route in solution.routes]


def _instance_row(instance: VRPTWInstance, instance_type: str, split: str, source: Solution) -> dict[str, Any]:
    return {
        "kind": "instance",
        "instance_id": instance.instance_id,
        "split": split,
        "instance_type": instance_type,
        "n_customers": instance.n_customers,
        "vehicle_capacity": instance.vehicle_capacity,
        "max_delay": instance.max_delay,
        "depot_time_window": list(instance.depot_time_window),
        "customer_coords": instance.customer_coords.tolist(),
        "customer_demands": instance.customer_demands.tolist(),
        "customer_time_windows": instance.customer_time_windows.tolist(),
        "customer_service_times": instance.customer_service_times.tolist(),
        "distance_matrix": instance.distance_matrix.tolist(),
        "duration_matrix": instance.duration_matrix.tolist(),
        "source_solution": _route(source),
        "source_objectives": _obj(evaluate_solution(source, instance)),
    }


def _instance_from_row(row: dict[str, Any]) -> VRPTWInstance:
    return VRPTWInstance(
        instance_id=row["instance_id"],
        n_customers=int(row["n_customers"]),
        vehicle_capacity=float(row["vehicle_capacity"]),
        max_delay=float(row["max_delay"]),
        depot_time_window=tuple(row["depot_time_window"]),
        customer_coords=np.asarray(row["customer_coords"], dtype=np.float64),
        customer_demands=np.asarray(row["customer_demands"], dtype=np.float64),
        customer_time_windows=np.asarray(row["customer_time_windows"], dtype=np.float64),
        customer_service_times=np.asarray(row["customer_service_times"], dtype=np.float64),
        distance_matrix=np.asarray(row["distance_matrix"], dtype=np.float64),
        duration_matrix=np.asarray(row["duration_matrix"], dtype=np.float64),
    )


def _solution(rows: list[list[int]]) -> Solution:
    return Solution([Route(list(route)) for route in rows])


def _objectives(values: list[float]) -> FiveObjectives:
    return FiveObjectives.from_array(np.asarray(values, dtype=np.float64))


def _state_row(state: StateRecord) -> dict[str, Any]:
    return {
        "kind": "state",
        "state_id": state.state_id,
        "instance_id": state.instance.instance_id,
        "source_solution": _route(state.source_solution),
        "source_objectives": _obj(state.source_objectives),
        "partial_solution": _route(state.destroyed_state.partial_solution),
        "removed_customers": list(state.destroyed_state.removed_customers),
        "destroy_type": state.destroy_type.value,
        "destroy_ratio": state.destroy_ratio,
        "random_seed": state.random_seed,
    }


def _task_row(task: PreferenceTaskRecord, state: StateRecord, instance_type: str) -> dict[str, Any]:
    candidates = []
    for candidate in task.teacher_candidates:
        candidates.append({
            "teacher_name": candidate.teacher_name,
            "actions": [
                {
                    "customer_id": action.customer_id,
                    "route_index": action.route_index,
                    "insert_position": action.insert_position,
                    "opens_new_route": action.opens_new_route,
                }
                for action in candidate.trajectory.actions
            ],
            "final_solution": _route(candidate.final_solution),
            "objectives": _obj(candidate.objectives),
            "is_feasible": candidate.is_feasible,
            "runtime_seconds": candidate.runtime_seconds,
        })
    return {
        "kind": "task",
        "task_id": task.task_id,
        "state_id": task.state_id,
        "stratum": {
            "instance_type": instance_type,
            "n_customers": state.instance.n_customers,
            "destroy_type": state.destroy_type.value,
            "destroy_ratio": state.destroy_ratio,
        },
        "preference_category": task.preference_category.value,
        "preference_weights": task.preference_weights.tolist(),
        "teacher_candidates": candidates,
        "selected_teacher": task.selected_teacher,
        "selected_score": task.selected_score,
    }


def _typed_coordinates(rng: np.random.Generator, n: int, kind: str) -> np.ndarray:
    if kind == "R":
        return rng.uniform(0.0, 100.0, size=(n, 2)).astype(np.float64)
    centers = rng.uniform(10.0, 90.0, size=(4, 2))
    cluster_ids = rng.integers(0, len(centers), size=n)
    clustered = centers[cluster_ids] + rng.normal(0.0, 7.0, size=(n, 2))
    clustered = np.clip(clustered, 0.0, 100.0)
    if kind == "C":
        return clustered.astype(np.float64)
    random_points = rng.uniform(0.0, 100.0, size=(n, 2))
    mask = rng.random(n) < 0.5
    return np.where(mask[:, None], clustered, random_points).astype(np.float64)


def _make_typed_instance(instance_id: str, n: int, kind: str, rng: np.random.Generator) -> VRPTWInstance:
    from src.data_generation.pipeline import _distance_matrix

    coords = _typed_coordinates(rng, n, kind)
    demands = rng.integers(1, 5, size=n).astype(np.float64)
    distance = _distance_matrix(coords)
    return VRPTWInstance(
        instance_id=instance_id,
        n_customers=n,
        vehicle_capacity=max(20.0, float(np.sum(demands) / 3.0)),
        max_delay=1800.0,
        depot_time_window=(0.0, 10000.0),
        customer_coords=coords,
        customer_demands=demands,
        customer_time_windows=np.column_stack((np.zeros(n), np.full(n, 9000.0))),
        customer_service_times=np.full(n, 2.0),
        distance_matrix=distance,
        duration_matrix=distance.copy(),
    )


def _instance_specs(config: ShardedConfig) -> list[dict[str, Any]]:
    specs = []
    total = config.train_instances + config.validation_instances
    for index in range(total):
        split = "train" if index < config.train_instances else "validation"
        local_index = index if split == "train" else index - config.train_instances
        kind = config.instance_types[local_index % len(config.instance_types)]
        size = config.sizes[(local_index // len(config.instance_types)) % len(config.sizes)]
        specs.append({
            "global_index": index,
            "split": split,
            "local_index": local_index,
            "instance_id": f"{split}_instance_{local_index:04d}",
            "instance_type": kind,
            "n_customers": size,
        })
    return specs


def _state_layout(instance_index: int, seed: int) -> list[tuple[DestroyType, float, int]]:
    return [
        (DestroyType.ROUTE_REPACK, 0.10, seed + 0),
        (DestroyType.WORST_EDGE_RELATED, 0.20, seed + 1),
        (DestroyType.MAKESPAN_SEGMENT, 0.20, seed + 2),
        (DestroyType.WAITING_SEGMENT, 0.30, seed + 3),
        (DestroyType.TARDINESS_SEGMENT, 0.20, seed + 4),
        (DestroyType.GENERIC_RANDOM if instance_index % 2 == 0 else DestroyType.GENERIC_MIXED, 0.30, seed + 5),
    ]


def _preference_plan(config: ShardedConfig, protocol: TaskProtocol, specs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    rng = np.random.default_rng(config.seed + 100003)
    if config.profile in ("smoke", "benchmark"):
        result: dict[str, dict[str, Any]] = {}
        categories = (
            PreferenceCategory.EXTREME,
            PreferenceCategory.BIASED_MIX,
            PreferenceCategory.GENERAL_MIX,
            PreferenceCategory.NEAR_UNIFORM,
            PreferenceCategory.EXTREME,
            PreferenceCategory.GENERAL_MIX,
        )
        for spec in specs:
            main = []
            for state_number in range(6):
                category = categories[(spec["global_index"] + state_number) % len(categories)]
                if category == PreferenceCategory.EXTREME:
                    weights = np.zeros(5, dtype=np.float64)
                    weights[(spec["global_index"] + state_number) % 5] = 1.0
                else:
                    weights, _ = generate_preference_weights(category, rng)
                main.append([weights.tolist(), category.value])
            result[spec["instance_id"]] = {"main": main, "shared": {}}
        return result
    train_main = generate_main_task_preferences(protocol, "train", rng)
    val_main = generate_main_task_preferences(protocol, "val", rng)
    result: dict[str, dict[str, Any]] = {}
    main_indices = {"train": 0, "validation": 0}
    shared_counts = {"train": protocol.n_shared_states, "validation": protocol.n_shared_val_states}
    selected: dict[str, set[tuple[int, int]]] = {}
    for split, count in shared_counts.items():
        split_specs = [s for s in specs if s["split"] == split]
        all_states = [
            (spec["global_index"], state_number)
            for spec in split_specs
            for state_number in range(6)
        ]
        stride = max(1, len(all_states) // count)
        selected[split] = set(all_states[::stride][:count])
    for spec in specs:
        split = spec["split"]
        main_values = train_main if split == "train" else val_main
        main = []
        shared: dict[str, list[list[Any]]] = {}
        for state_number in range(6):
            main_value = main_values[main_indices[split]]
            main_indices[split] += 1
            main.append([main_value[0].tolist(), main_value[1].value])
            if (spec["global_index"], state_number) in selected[split]:
                shared[str(state_number)] = [
                    [weights.tolist(), category.value]
                    for weights, category in generate_shared_state_preferences(rng)
                ]
        result[spec["instance_id"]] = {
            "main": main,
            "shared": shared,
        }
    return result


def build_index(config: ShardedConfig, protocol: TaskProtocol) -> dict[str, Any]:
    output = Path(config.output_dir)
    runtime = output / "runtime"
    data = output / "data"
    runtime.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    specs = _instance_specs(config)
    prefs = _preference_plan(config, protocol, specs)
    index_path = data / "instance_index.jsonl.gz"
    source_values = []
    expected_records = 0
    index_temp = index_path.with_name(f"{index_path.name}.{os.getpid()}.tmp")
    with gzip.open(index_temp, "wt", encoding="utf-8", compresslevel=6) as handle:
        for spec in specs:
            rng = np.random.default_rng(config.seed + spec["global_index"] * 1009)
            instance = _make_typed_instance(spec["instance_id"], spec["n_customers"], spec["instance_type"], rng)
            source = _make_base_solution(instance, rng)
            _shape_time_windows(instance, source, rng)
            if not is_solution_feasible(source, instance):
                raise RuntimeError(f"infeasible source: {spec['instance_id']}")
            source_values.append(evaluate_solution(source, instance))
            row = _instance_row(instance, spec["instance_type"], spec["split"], source)
            row.update({"global_index": spec["global_index"], "preferences": prefs[spec["instance_id"]]})
            expected_records += 6 + sum(len(value) for value in row["preferences"].get("shared", {}).values())
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")
    os.replace(index_temp, index_path)
    normalizer = _normalizer(source_values)
    _write_json(runtime / "index_meta.json", {
        "protocol_hash": protocol.compute_protocol_hash(),
        "schema_version": protocol.schema_version,
        "normalizer": {"mins": normalizer.mins.tolist(), "ranges": normalizer.ranges.tolist()},
        "instance_count": len(specs),
        "expected_records": expected_records,
        "expected_tasks": expected_records,
        "expected_states": len(specs) * 6,
        "index_sha256": _hash_file(index_path),
    })
    return {
        "index_path": str(index_path),
        "normalizer": normalizer,
        "instance_count": len(specs),
        "expected_records": expected_records,
        "expected_tasks": expected_records,
        "expected_states": len(specs) * 6,
    }


def _task_from_payload(row: dict[str, Any], state: StateRecord, weights: list[float], category: str, config: ShardedConfig, normalizer: Normalizer, index: int, task_index: int) -> PreferenceTaskRecord:
    candidates = []
    destroyed = state.destroyed_state
    for teacher_name in config.teacher_names:
        started = time.monotonic()
        final, trajectory = _teacher(
            teacher_name,
            state.instance,
            destroyed,
            np.asarray(weights, dtype=np.float64),
            normalizer,
            GenerationConfig(
                output_dir="",
                teacher_names=config.teacher_names,
                beam_width=config.beam_width,
                max_beam_expansions=config.max_beam_expansions,
                max_insertion_candidates_per_customer=config.max_insertion_candidates_per_customer,
            ),
        )
        runtime = time.monotonic() - started
        objectives = evaluate_solution(final, state.instance)
        feasible = is_solution_feasible(final, state.instance)
        if not feasible or not validate_trajectory_replay(destroyed, trajectory, final):
            raise RuntimeError(f"invalid {teacher_name} result for {state.state_id}")
        candidates.append(TeacherCandidate(teacher_name, trajectory, final, objectives, feasible, runtime))
    feasible_candidates = [candidate for candidate in candidates if candidate.is_feasible]
    if not feasible_candidates:
        raise RuntimeError(f"no feasible teacher result for {state.state_id}")
    scores = {
        candidate.teacher_name: compute_scalarized_objective(
            candidate.objectives.to_array(), np.asarray(weights, dtype=np.float64), normalizer
        )
        for candidate in feasible_candidates
    }
    selected = min(
        feasible_candidates,
        key=lambda candidate: (scores[candidate.teacher_name], candidate.teacher_name),
    )
    task_id = f"{state.state_id}_task_{task_index:02d}"
    return PreferenceTaskRecord(
        task_id=task_id,
        state_id=state.state_id,
        preference_category=PreferenceCategory(category),
        preference_weights=np.asarray(weights, dtype=np.float64),
        teacher_candidates=candidates,
        selected_teacher=selected.teacher_name,
        selected_score=float(scores[selected.teacher_name]),
    )


def _iter_jsonl_gz(path: Path) -> Iterable[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _worker(worker_id: int, config: ShardedConfig, protocol: TaskProtocol, index_path: str, normalizer_values: dict[str, list[float]]) -> dict[str, Any]:
    os.environ.update({name: "1" for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
    output = Path(config.output_dir)
    status_path = output / "runtime" / "workers" / f"worker_{worker_id:02d}.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    normalizer = Normalizer(np.asarray(normalizer_values["mins"]), np.asarray(normalizer_values["ranges"]))
    completed_instances = 0
    completed_records = 0
    failed_states = 0
    started = time.monotonic()
    for row in _iter_jsonl_gz(Path(index_path)):
        if int(row["global_index"]) % config.workers != worker_id:
            continue
        worker_dir = output / "data" / row["split"] / f"worker_{worker_id:02d}"
        worker_dir.mkdir(parents=True, exist_ok=True)
        part = worker_dir / f"instance_{int(row['global_index']):05d}.jsonl.gz"
        if part.exists():
            existing_instances, existing_records = _part_counts(part)
            completed_instances += existing_instances
            completed_records += existing_records
            continue
        try:
            instance = _instance_from_row(row)
            source = _solution(row["source_solution"])
            source_objectives = _objectives(row["source_objectives"])
            lines = [_instance_row(instance, row.get("instance_type", "unknown"), row["split"], source)]
            for state_number, (destroy_type, ratio, state_seed) in enumerate(_state_layout(int(row["global_index"]), config.seed + int(row["global_index"]) * 100)):
                destroyed = destroy_solution(source, instance, destroy_type, ratio, np.random.default_rng(state_seed))
                state_id = f"{instance.instance_id}_state_{state_number:02d}"
                state = StateRecord(state_id, instance, source, source_objectives, destroyed, destroy_type, ratio, state_seed)
                lines.append(_state_row(state))
                preferences = [(row["preferences"]["main"][state_number], 0)]
                preferences.extend((value, index + 1) for index, value in enumerate(row["preferences"].get("shared", {}).get(str(state_number), [])))
                for (weights, category), task_index in preferences:
                    task = _task_from_payload(row, state, weights, category, config, normalizer, int(row["global_index"]), task_index)
                    lines.append(_task_row(task, state, row.get("instance_type", "unknown")))
                    completed_records += 1
            _write_jsonl_gz(part, lines)
            completed_instances += 1
        except Exception as error:
            failed_states += 1
            _write_json(status_path, {"worker_id": worker_id, "state": "error", "error": repr(error), "failed_states": failed_states, **_worker_snapshot(started, completed_instances, completed_records)})
            raise
        _write_json(status_path, {"worker_id": worker_id, "state": "running", "failed_states": failed_states, **_worker_snapshot(started, completed_instances, completed_records)})
    payload = {"worker_id": worker_id, "state": "complete", "failed_states": failed_states, **_worker_snapshot(started, completed_instances, completed_records)}
    _write_json(status_path, payload)
    return payload


def _split_rows(rows: list[dict[str, Any]], workers: int) -> list[list[dict[str, Any]]]:
    result = [[] for _ in range(workers)]
    for index, row in enumerate(rows):
        result[index % workers].append(row)
    return result


def run_sharded(config: ShardedConfig, protocol: TaskProtocol = DEFAULT_PROTOCOL) -> dict[str, Any]:
    if config.workers < 1:
        raise ValueError("workers must be positive")
    if config.profile not in ("trainval", "smoke", "benchmark"):
        raise ValueError("supported profiles are trainval, smoke, and benchmark")
    if config.max_inflight_tasks < config.workers:
        raise ValueError("max_inflight_tasks must be at least workers")
    if config.flush_every_records <= 0:
        raise ValueError("flush_every_records must be positive")
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    resources = _resource_snapshot()
    effective_workers = config.workers
    if resources.get("available_memory_gb", float("inf")) < MIN_AVAILABLE_MEMORY_GB:
        effective_workers = min(effective_workers, 10)
    _write_json(output / "runtime" / "resource_start.json", {"requested_workers": config.workers, "effective_workers": effective_workers, **resources})
    index_result = build_index(config, protocol)
    normalizer = index_result["normalizer"]
    expected_instances = config.train_instances + config.validation_instances
    expected_records = index_result["expected_tasks"]
    expected_states = index_result["expected_states"]
    if effective_workers != config.workers:
        config = ShardedConfig(**{**config.__dict__, "workers": effective_workers})
    _write_json(output / "runtime" / "generation_status.json", {"state": "running", "workers": config.workers, "requested_workers": WORKER_COUNT, "max_inflight_tasks": config.max_inflight_tasks, "flush_every_records": config.flush_every_records, "expected_instances": expected_instances, "expected_states": expected_states, "expected_records": expected_records, "completed_instances": 0, "completed_states": 0, "completed_records": 0, "failed_states": 0, "resource": _resource_snapshot()})
    worker_args = [(worker_id, config, protocol, index_result["index_path"], {"mins": normalizer.mins.tolist(), "ranges": normalizer.ranges.tolist()}) for worker_id in range(config.workers)]
    results = []
    futures = set()
    last_status_write = 0.0
    with ProcessPoolExecutor(max_workers=config.workers) as executor:
        futures = {executor.submit(_worker, *args) for args in worker_args}
        while futures:
            done, futures = wait(futures, timeout=5.0, return_when=FIRST_COMPLETED)
            for future in done:
                results.append(future.result())
            now = time.monotonic()
            if done or now - last_status_write >= 30.0:
                worker_results = []
                for worker_id in range(config.workers):
                    status_path = output / "runtime" / "workers" / f"worker_{worker_id:02d}.json"
                    if status_path.exists():
                        try:
                            worker_results.append(json.loads(status_path.read_text(encoding="utf-8")))
                        except json.JSONDecodeError:
                            pass
                current_resources = _resource_snapshot()
                _write_json(output / "runtime" / "generation_status.json", {"state": "running", "workers": config.workers, "requested_workers": WORKER_COUNT, "max_inflight_tasks": config.max_inflight_tasks, "flush_every_records": config.flush_every_records, "expected_instances": expected_instances, "expected_states": expected_states, "expected_records": expected_records, "completed_instances": sum(item.get("completed_instances", 0) for item in worker_results), "completed_states": sum(item.get("completed_instances", 0) for item in worker_results) * 6, "completed_records": sum(item.get("completed_records", 0) for item in worker_results), "failed_states": sum(item.get("failed_states", 0) for item in worker_results), "worker_results": worker_results, "resource": current_resources, "resource_warning": current_resources.get("used_memory_gb", 0) > MAX_GENERATION_MEMORY_GB or current_resources.get("available_memory_gb", float("inf")) < MIN_AVAILABLE_MEMORY_GB})
                last_status_write = now
    status = {"state": "complete", "workers": config.workers, "requested_workers": WORKER_COUNT, "max_inflight_tasks": config.max_inflight_tasks, "flush_every_records": config.flush_every_records, "expected_instances": expected_instances, "expected_states": expected_states, "expected_records": expected_records, "completed_instances": expected_instances, "completed_states": expected_states, "completed_records": sum(item["completed_records"] for item in results), "failed_states": sum(item["failed_states"] for item in results), "worker_results": results, "resource": _resource_snapshot()}
    _write_json(output / "runtime" / "generation_status.json", status)
    return status


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run resumable v0.17 sharded generation")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=WORKER_COUNT)
    parser.add_argument("--train-instances", type=int, default=2000)
    parser.add_argument("--validation-instances", type=int, default=400)
    parser.add_argument("--seed", type=int, default=17017)
    parser.add_argument("--profile", choices=("smoke", "benchmark", "trainval"), default="trainval")
    args = parser.parse_args()
    run_sharded(ShardedConfig(output_dir=args.output_dir, profile=args.profile, workers=args.workers, train_instances=args.train_instances, validation_instances=args.validation_instances, seed=args.seed))


if __name__ == "__main__":
    _main()
