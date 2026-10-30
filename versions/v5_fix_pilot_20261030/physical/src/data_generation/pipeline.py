"""Clean, protocol-driven Stage 2 data generation for v0.17.

This module is intentionally independent from the v0.13-v0.16 generators and
teachers.  The first version provides deterministic CPU teachers for a complete
smoke path: instance -> feasible source -> destroy -> repair trajectory ->
replay -> manifest.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

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
from src.core.replay import replay_trajectory
from src.core.task_protocol import DEFAULT_PROTOCOL, TaskProtocol
from src.evaluation.vrptw_eval import (
    compute_scalarized_objective,
    evaluate_route,
    evaluate_solution,
    is_solution_feasible,
)
from src.manifest.binding import create_manifest
from src.preference.generation import generate_preference_weights, generate_shared_state_preferences


@dataclass(frozen=True)
class GenerationConfig:
    """Runtime smoke configuration; scientific counts remain in TaskProtocol."""

    output_dir: str
    seed: int = 17017
    train_instances: int = 2
    val_instances: int = 1
    customers: tuple[int, ...] = (12,)
    states_per_instance: int = 2
    preferences_per_state: int = 3
    teacher_names: tuple[str, ...] = (
        "regret3",
        "objective_greedy",
        "bounded_beam",
    )
    destroy_ratios: tuple[float, ...] = (0.10, 0.20, 0.30)
    beam_width: int = 4
    max_beam_expansions: int = 500
    max_insertion_candidates_per_customer: int | None = 24


@dataclass(frozen=True)
class _RepairState:
    solution: Solution
    remaining: tuple[int, ...]
    objectives: FiveObjectives
    route_metrics: tuple[tuple[float, float, float, float], ...]


def _state_from_destroyed(
    destroyed: DestroyedState,
    instance: VRPTWInstance,
) -> _RepairState:
    route_metrics = tuple(evaluate_route(route, instance) for route in destroyed.partial_solution.routes)
    return _RepairState(
        solution=destroyed.partial_solution,
        remaining=tuple(destroyed.removed_customers),
        objectives=evaluate_solution(destroyed.partial_solution, instance),
        route_metrics=route_metrics,
    )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _distance_matrix(coords: np.ndarray) -> np.ndarray:
    points = np.vstack((np.zeros((1, 2), dtype=np.float64), coords))
    delta = points[:, None, :] - points[None, :, :]
    return np.sqrt(np.sum(delta * delta, axis=2))


def _make_instance(instance_id: str, n_customers: int, rng: np.random.Generator) -> VRPTWInstance:
    coords = rng.uniform(0.0, 100.0, size=(n_customers, 2)).astype(np.float64)
    demands = rng.integers(1, 5, size=n_customers).astype(np.float64)
    windows = np.column_stack(
        (np.zeros(n_customers, dtype=np.float64), np.full(n_customers, 9000.0))
    )
    service = np.full(n_customers, 2.0, dtype=np.float64)
    distance = _distance_matrix(coords)
    return VRPTWInstance(
        instance_id=instance_id,
        n_customers=n_customers,
        vehicle_capacity=max(20.0, float(np.sum(demands) / 3.0)),
        max_delay=1800.0,
        depot_time_window=(0.0, 10000.0),
        customer_coords=coords,
        customer_demands=demands,
        customer_time_windows=windows,
        customer_service_times=service,
        distance_matrix=distance,
        duration_matrix=distance.copy(),
    )


def _make_base_solution(instance: VRPTWInstance, rng: np.random.Generator) -> Solution:
    """Construct a feasible capacity-respecting source solution."""
    order = [int(value) for value in rng.permutation(np.arange(1, instance.n_customers + 1))]
    routes: list[Route] = []
    current: list[int] = []
    load = 0.0
    for customer in order:
        demand = float(instance.customer_demands[customer - 1])
        if current and load + demand > instance.vehicle_capacity:
            routes.append(Route(current))
            current, load = [], 0.0
        current.append(customer)
        load += demand
    if current:
        routes.append(Route(current))
    solution = Solution(routes)
    if not is_solution_feasible(solution, instance):
        raise RuntimeError(f"base solution is infeasible: {instance.instance_id}")
    return solution


def _shape_time_windows(
    instance: VRPTWInstance,
    solution: Solution,
    rng: np.random.Generator,
) -> None:
    """Create non-degenerate windows around the known feasible source solution.

    The source route order is scheduled without waiting first.  Windows are
    then centered around those arrivals, with a mixture of early and late
    releases.  This preserves a feasible source while making waiting and
    tardiness meaningful after destroy/repair changes the order.
    """
    arrivals = np.zeros(instance.n_customers, dtype=np.float64)
    for route in solution.routes:
        current_time = 0.0
        previous = 0
        for customer in route.customers:
            current_time += float(instance.duration_matrix[previous, customer])
            arrivals[customer - 1] = current_time
            current_time += float(instance.customer_service_times[customer - 1])
            previous = customer

    windows = np.zeros((instance.n_customers, 2), dtype=np.float64)
    for index, arrival in enumerate(arrivals):
        if rng.random() < 0.35:
            early = arrival + rng.uniform(20.0, 180.0)
        else:
            early = max(0.0, arrival - rng.uniform(20.0, 240.0))
        late = max(early + 120.0, arrival + rng.uniform(240.0, 900.0))
        windows[index] = (early, late)
    instance.customer_time_windows = windows


def _rank_removals(
    solution: Solution,
    instance: VRPTWInstance,
    destroy_type: DestroyType,
    rng: np.random.Generator,
) -> list[int]:
    customers = [customer for route in solution.routes for customer in route.customers]
    if destroy_type == DestroyType.ROUTE_REPACK:
        route = max(solution.routes, key=lambda item: len(item.customers))
        return list(route.customers) + [c for c in customers if c not in route.customers]
    if destroy_type == DestroyType.WORST_EDGE_RELATED:
        scored: list[tuple[float, int]] = []
        for route in solution.routes:
            previous = 0
            for customer in route.customers:
                scored.append((float(instance.distance_matrix[previous, customer]), customer))
                previous = customer
            if route.customers:
                scored.append((float(instance.distance_matrix[previous, 0]), route.customers[-1]))
        return [customer for _, customer in sorted(scored, reverse=True)]
    if destroy_type == DestroyType.MAKESPAN_SEGMENT:
        route_scores = [evaluate_route(route, instance)[1] for route in solution.routes]
        route = solution.routes[int(np.argmax(route_scores))]
        return list(route.customers) + [c for c in customers if c not in route.customers]
    if destroy_type == DestroyType.WAITING_SEGMENT:
        return sorted(customers, key=lambda c: float(instance.customer_time_windows[c - 1, 0]), reverse=True)
    if destroy_type == DestroyType.TARDINESS_SEGMENT:
        return sorted(customers, key=lambda c: float(instance.customer_time_windows[c - 1, 1]))
    if destroy_type == DestroyType.GENERIC_MIXED:
        shuffled = [int(value) for value in rng.permutation(customers)]
        return shuffled[::2] + shuffled[1::2]
    return [int(value) for value in rng.permutation(customers)]


def destroy_solution(
    solution: Solution,
    instance: VRPTWInstance,
    destroy_type: DestroyType,
    ratio: float,
    rng: np.random.Generator,
) -> DestroyedState:
    """Remove customers while retaining a complete, validated destroyed state."""
    if not 0.0 < ratio < 1.0:
        raise ValueError(f"destroy ratio must be in (0, 1), got {ratio}")
    count = max(1, min(instance.n_customers - 1, int(round(instance.n_customers * ratio))))
    removed = list(dict.fromkeys(_rank_removals(solution, instance, destroy_type, rng)))[:count]
    removed_set = set(removed)
    partial = Solution([
        Route([customer for customer in route.customers if customer not in removed_set])
        for route in solution.routes
    ])
    return DestroyedState(partial_solution=partial, removed_customers=removed)


def _cheap_insertion_cost(
    route: Route,
    customer: int,
    position: int,
    instance: VRPTWInstance,
) -> float:
    previous = 0 if position == 0 else route.customers[position - 1]
    following = 0 if position == len(route.customers) else route.customers[position]
    return float(
        instance.distance_matrix[previous, customer]
        + instance.distance_matrix[customer, following]
        - instance.distance_matrix[previous, following]
    )


def _actions(
    state: _RepairState,
    instance: VRPTWInstance | None = None,
    max_insertion_candidates_per_customer: int | None = None,
) -> Iterable[RepairAction]:
    for customer in state.remaining:
        candidates: list[tuple[float, RepairAction]] = []
        for route_index, route in enumerate(state.solution.routes):
            for position in range(len(route.customers) + 1):
                action = RepairAction(customer, route_index, position, False)
                cost = 0.0 if instance is None else _cheap_insertion_cost(route, customer, position, instance)
                candidates.append((cost, action))
        if max_insertion_candidates_per_customer is not None:
            if max_insertion_candidates_per_customer <= 0:
                raise ValueError("max_insertion_candidates_per_customer must be positive")
            candidates.sort(key=lambda item: (item[0], item[1].route_index, item[1].insert_position))
            candidates = candidates[:max_insertion_candidates_per_customer]
        for _, action in candidates:
            yield action
        yield RepairAction(customer, -1, 0, True)


def _apply_action(
    state: _RepairState,
    action: RepairAction,
    instance: VRPTWInstance,
) -> _RepairState:
    if action.customer_id not in state.remaining:
        raise ValueError(f"customer {action.customer_id} is not remaining")
    routes = [list(route.customers) for route in state.solution.routes]
    if action.opens_new_route:
        routes.append([action.customer_id])
    else:
        if not 0 <= action.route_index < len(routes):
            raise ValueError("invalid route index")
        if not 0 <= action.insert_position <= len(routes[action.route_index]):
            raise ValueError("invalid insert position")
        routes[action.route_index].insert(action.insert_position, action.customer_id)
    remaining = tuple(customer for customer in state.remaining if customer != action.customer_id)
    solution = Solution([Route(route) for route in routes])
    touched_route = solution.routes[-1] if action.opens_new_route else solution.routes[action.route_index]
    touched_metrics = evaluate_route(touched_route, instance)
    objectives = _incremental_objectives(state, action, touched_metrics, instance)
    route_metrics = list(state.route_metrics)
    if action.opens_new_route:
        route_metrics.append(touched_metrics)
    else:
        route_metrics[action.route_index] = touched_metrics
    return _RepairState(solution, remaining, objectives, tuple(route_metrics))


def _route_legal(route: Route, instance: VRPTWInstance) -> bool:
    load = sum(float(instance.customer_demands[c - 1]) for c in route.customers)
    if load > instance.vehicle_capacity + 1e-8:
        return False
    current_time = 0.0
    previous = 0
    for customer in route.customers:
        current_time += float(instance.duration_matrix[previous, customer])
        early, late = instance.customer_time_windows[customer - 1]
        if current_time < early:
            current_time = float(early)
        elif current_time - late > instance.max_delay + 1e-8:
            return False
        current_time += float(instance.customer_service_times[customer - 1])
        previous = customer
    current_time += float(instance.duration_matrix[previous, 0])
    return current_time <= instance.depot_time_window[1] + 1e-8


def _incremental_objectives(
    state: _RepairState,
    action: RepairAction,
    candidate_metrics: tuple[float, float, float, float],
    instance: VRPTWInstance,
) -> FiveObjectives:
    """Update objective totals after evaluating only the touched route."""
    new_distance, new_duration, new_waiting, new_tardiness = candidate_metrics
    if action.opens_new_route:
        return FiveObjectives(
            n_vehicles=state.objectives.n_vehicles + 1,
            total_distance=state.objectives.total_distance + new_distance,
            makespan=max(state.objectives.makespan, new_duration),
            total_waiting=state.objectives.total_waiting + new_waiting,
            total_tardiness=state.objectives.total_tardiness + new_tardiness,
        )

    old_distance, old_duration, old_waiting, old_tardiness = state.route_metrics[action.route_index]
    old_route_was_empty = not state.solution.routes[action.route_index].customers
    other_makespan = max(
        (metrics[1] for index, metrics in enumerate(state.route_metrics)
         if index != action.route_index),
        default=0.0,
    )
    return FiveObjectives(
        n_vehicles=state.objectives.n_vehicles + (1 if old_route_was_empty else 0),
        total_distance=state.objectives.total_distance - old_distance + new_distance,
        makespan=max(other_makespan, new_duration),
        total_waiting=state.objectives.total_waiting - old_waiting + new_waiting,
        total_tardiness=state.objectives.total_tardiness - old_tardiness + new_tardiness,
    )


def _candidate_score(
    state: _RepairState,
    action: RepairAction,
    instance: VRPTWInstance,
    weights: np.ndarray,
    normalizer: Normalizer,
) -> tuple[float, int, int, int]:
    candidate = _apply_action(state, action, instance)
    touched_route = candidate.solution.routes[-1] if action.opens_new_route else candidate.solution.routes[action.route_index]
    if not _route_legal(touched_route, instance):
        return (1e12, action.customer_id, action.route_index, action.insert_position)
    value = compute_scalarized_objective(
        candidate.objectives.to_array(), weights, normalizer
    )
    return (float(value), action.customer_id, action.route_index, action.insert_position)


def _greedy_repair(
    instance: VRPTWInstance,
    destroyed: DestroyedState,
    weights: np.ndarray,
    normalizer: Normalizer,
    reverse: bool,
    max_insertion_candidates_per_customer: int | None,
) -> tuple[Solution, ActionTrajectory]:
    state = _state_from_destroyed(destroyed, instance)
    history: list[RepairAction] = []
    while state.remaining:
        actions = list(_actions(state, instance, max_insertion_candidates_per_customer))
        if reverse:
            actions.reverse()
        action = min(actions, key=lambda item: _candidate_score(state, item, instance, weights, normalizer))
        state = _apply_action(state, action, instance)
        history.append(action)
    if not is_solution_feasible(state.solution, instance):
        raise RuntimeError("greedy repair could not produce a feasible solution")
    return state.solution, ActionTrajectory(history)


def _objective_priority(
    state: _RepairState,
    customer: int,
    instance: VRPTWInstance,
    weights: np.ndarray,
) -> float:
    """Cheap customer priority used by the objective-oriented teacher.

    Customers with high demand and tight/latest windows are considered first
    when the corresponding objective has weight.  The priority only chooses
    the insertion order; every selected action is still evaluated with the
    full scalarized objective and feasibility check.
    """
    demand = float(instance.customer_demands[customer - 1])
    early, late = instance.customer_time_windows[customer - 1]
    window_width = max(1.0, float(late - early))
    distance_from_depot = float(instance.distance_matrix[0, customer])
    vehicle_pressure = weights[0] * demand
    distance_pressure = weights[1] * distance_from_depot
    makespan_pressure = weights[2] * (1.0 / window_width)
    waiting_pressure = weights[3] * float(early)
    tardiness_pressure = weights[4] * (1.0 / max(1.0, float(late)))
    return vehicle_pressure + distance_pressure + makespan_pressure + waiting_pressure + tardiness_pressure


def _objective_greedy_repair(
    instance: VRPTWInstance,
    destroyed: DestroyedState,
    weights: np.ndarray,
    normalizer: Normalizer,
    max_insertion_candidates_per_customer: int | None,
) -> tuple[Solution, ActionTrajectory]:
    """Repair in preference-conditioned priority order, then scalar-greedily."""
    state = _state_from_destroyed(destroyed, instance)
    history: list[RepairAction] = []
    while state.remaining:
        customer = max(
            state.remaining,
            key=lambda value: (_objective_priority(state, value, instance, weights), -value),
        )
        candidates = [
            action for action in _actions(
                _RepairState(state.solution, (customer,), state.objectives, state.route_metrics),
                instance,
                max_insertion_candidates_per_customer,
            )
            if action.customer_id == customer
        ]
        action = min(
            candidates,
            key=lambda item: _candidate_score(state, item, instance, weights, normalizer),
        )
        state = _apply_action(state, action, instance)
        history.append(action)
    if not is_solution_feasible(state.solution, instance):
        raise RuntimeError("objective greedy repair could not produce a feasible solution")
    return state.solution, ActionTrajectory(history)


def _regret3_repair(
    instance: VRPTWInstance,
    destroyed: DestroyedState,
    weights: np.ndarray,
    normalizer: Normalizer,
    max_insertion_candidates_per_customer: int | None,
) -> tuple[Solution, ActionTrajectory]:
    """Regret-3 insertion teacher with the same scalarized evaluation contract."""
    state = _state_from_destroyed(destroyed, instance)
    history: list[RepairAction] = []
    while state.remaining:
        choices: list[tuple[float, float, int, RepairAction]] = []
        for customer in state.remaining:
            scored: list[tuple[float, RepairAction]] = []
            for action in _actions(
                _RepairState(state.solution, (customer,), state.objectives, state.route_metrics),
                instance,
                max_insertion_candidates_per_customer,
            ):
                score = _candidate_score(state, action, instance, weights, normalizer)[0]
                if score < 1e12:
                    scored.append((score, action))
            if not scored:
                raise RuntimeError(f"no feasible insertion for customer {customer}")
            scored.sort(key=lambda item: (item[0], item[1].route_index, item[1].insert_position))
            best_score, best_action = scored[0]
            second = scored[1][0] if len(scored) > 1 else best_score
            third = scored[2][0] if len(scored) > 2 else second
            regret = (second - best_score) + (third - best_score)
            choices.append((regret, -best_score, -customer, best_action))
        _, _, _, action = max(choices, key=lambda item: item[:3])
        state = _apply_action(state, action, instance)
        history.append(action)
    if not is_solution_feasible(state.solution, instance):
        raise RuntimeError("regret-3 repair could not produce a feasible solution")
    return state.solution, ActionTrajectory(history)


def _beam_repair(
    instance: VRPTWInstance,
    destroyed: DestroyedState,
    weights: np.ndarray,
    normalizer: Normalizer,
    width: int,
    max_expansions: int,
    max_insertion_candidates_per_customer: int | None,
) -> tuple[Solution, ActionTrajectory]:
    beam: list[tuple[_RepairState, list[RepairAction]]] = [
        (_state_from_destroyed(destroyed, instance), [])
    ]
    expansions = 0
    while beam and expansions < max_expansions:
        completed: list[tuple[float, _RepairState, list[RepairAction]]] = []
        children: list[tuple[float, _RepairState, list[RepairAction]]] = []
        for state, history in beam:
            if not state.remaining:
                if is_solution_feasible(state.solution, instance):
                    score = compute_scalarized_objective(state.objectives.to_array(), weights, normalizer)
                    completed.append((float(score), state, history))
                continue
            for action in _actions(state, instance, max_insertion_candidates_per_customer):
                expansions += 1
                next_state = _apply_action(state, action, instance)
                score = _candidate_score(state, action, instance, weights, normalizer)[0]
                if score < 1e12:
                    children.append((score, next_state, history + [action]))
                if expansions >= max_expansions:
                    break
            if expansions >= max_expansions:
                break
        if completed:
            _, best_state, history = min(completed, key=lambda item: item[0])
            return best_state.solution, ActionTrajectory(history)
        children.sort(key=lambda item: item[0])
        beam = [(state, history) for _, state, history in children[:max(1, width)]]
    return _greedy_repair(
        instance,
        destroyed,
        weights,
        normalizer,
        reverse=False,
        max_insertion_candidates_per_customer=max_insertion_candidates_per_customer,
    )


def _teacher(
    name: str,
    instance: VRPTWInstance,
    destroyed: DestroyedState,
    weights: np.ndarray,
    normalizer: Normalizer,
    config: GenerationConfig,
) -> tuple[Solution, ActionTrajectory]:
    if name == "regret3":
        return _regret3_repair(
            instance, destroyed, weights, normalizer,
            config.max_insertion_candidates_per_customer,
        )
    if name == "objective_greedy":
        return _objective_greedy_repair(
            instance, destroyed, weights, normalizer,
            config.max_insertion_candidates_per_customer,
        )
    if name == "bounded_beam":
        return _beam_repair(
            instance, destroyed, weights, normalizer, config.beam_width,
            config.max_beam_expansions, config.max_insertion_candidates_per_customer,
        )
    raise ValueError(f"unknown teacher: {name}")


def _normalizer(objectives: Sequence[FiveObjectives]) -> Normalizer:
    values = np.stack([objective.to_array() for objective in objectives])
    minimum = values.min(axis=0)
    return Normalizer(mins=minimum, ranges=np.maximum(values.max(axis=0) - minimum, 1.0))


def _preferences(config: GenerationConfig, rng: np.random.Generator):
    if config.preferences_per_state == 10:
        return generate_shared_state_preferences(rng)
    categories = list(PreferenceCategory)
    return [generate_preference_weights(categories[i % len(categories)], rng) for i in range(config.preferences_per_state)]


def generate_dataset(config: GenerationConfig, protocol: TaskProtocol = DEFAULT_PROTOCOL) -> dict[str, object]:
    """Generate a small dataset and validate every trajectory through the manifest."""
    if config.train_instances <= 0 or config.val_instances <= 0:
        raise ValueError("train_instances and val_instances must be positive")
    if not config.customers or any(value < 3 for value in config.customers):
        raise ValueError("customers must contain values >= 3")
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.seed)
    total_expected = (config.train_instances + config.val_instances) * config.states_per_instance * config.preferences_per_state
    _write_json(output / "generation_status.json", {"status": "running", "completed_records": 0, "expected_records": total_expected, "failed_states": 0})

    instances: dict[str, VRPTWInstance] = {}
    states: dict[str, StateRecord] = {}
    tasks: dict[str, PreferenceTaskRecord] = {}
    splits: dict[str, list[str]] = {"train": [], "validation": []}
    base_objectives: list[FiveObjectives] = []
    state_inputs: list[tuple[str, str, VRPTWInstance, Solution, DestroyedState, DestroyType, float, int]] = []
    completed = 0
    try:
        for split, count in (("train", config.train_instances), ("validation", config.val_instances)):
            for index in range(count):
                instance_id = f"{split}_instance_{index:04d}"
                instance = _make_instance(instance_id, config.customers[index % len(config.customers)], rng)
                solution = _make_base_solution(instance, rng)
                _shape_time_windows(instance, solution, rng)
                if not is_solution_feasible(solution, instance):
                    raise RuntimeError(f"shaped source solution is infeasible: {instance_id}")
                source_objectives = evaluate_solution(solution, instance)
                instances[instance_id] = instance
                base_objectives.append(source_objectives)
                for state_index in range(config.states_per_instance):
                    ratio = float(config.destroy_ratios[state_index % len(config.destroy_ratios)])
                    destroy_types = tuple(DestroyType)
                    destroy_type = destroy_types[(index + state_index) % len(destroy_types)]
                    destroyed = destroy_solution(solution, instance, destroy_type, ratio, rng)
                    state_id = f"{instance_id}_state_{state_index:03d}"
                    state_inputs.append((split, state_id, instance, solution, destroyed, destroy_type, ratio, config.seed + index * 100 + state_index))

        normalizer = _normalizer(base_objectives)
        for split, state_id, instance, solution, destroyed, destroy_type, ratio, state_seed in state_inputs:
            source_objectives = evaluate_solution(solution, instance)
            state = StateRecord(state_id, instance, solution, source_objectives, destroyed, destroy_type, ratio, state_seed)
            states[state_id] = state
            for task_index, (weights, category) in enumerate(_preferences(config, rng)):
                candidates: list[TeacherCandidate] = []
                for teacher_name in config.teacher_names:
                    started = time.monotonic()
                    final_solution, trajectory = _teacher(teacher_name, instance, destroyed, weights, normalizer, config)
                    runtime = time.monotonic() - started
                    if replay_trajectory(destroyed, trajectory).routes != final_solution.routes:
                        raise RuntimeError(f"trajectory replay mismatch for {teacher_name}")
                    objectives = evaluate_solution(final_solution, instance)
                    feasible = is_solution_feasible(final_solution, instance)
                    if not feasible:
                        raise RuntimeError(f"teacher {teacher_name} returned an infeasible solution")
                    candidates.append(TeacherCandidate(teacher_name, trajectory, final_solution, objectives, feasible, runtime))
                task_id = f"{state_id}_task_{task_index:03d}"
                tasks[task_id] = PreferenceTaskRecord(task_id, state_id, category, weights, candidates)
                splits[split].append(task_id)
                completed += 1
                _write_json(output / "generation_status.json", {"status": "running", "completed_records": completed, "expected_records": total_expected, "failed_states": 0})

        manifest = create_manifest(protocol, normalizer, instances, states, tasks, splits)
        manifest.save(output / "manifest.json")
        with (output / "dataset.pkl").open("wb") as handle:
            pickle.dump({"protocol": protocol, "normalizer": normalizer, "instances": instances, "states": states, "tasks": tasks, "splits": splits}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        _write_json(output / "generation_status.json", {"status": "completed", "completed_records": completed, "expected_records": total_expected, "failed_states": 0, "manifest_hash": manifest.manifest_hash})
        return {"manifest": manifest, "completed_records": completed}
    except Exception as error:
        _write_json(output / "generation_status.json", {"status": "failed", "completed_records": completed, "expected_records": total_expected, "failed_states": 1, "error": repr(error)})
        raise


def _main() -> None:
    parser = argparse.ArgumentParser(description="Generate v0.17 Stage 2 smoke data")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=17017)
    parser.add_argument("--train-instances", type=int, default=2)
    parser.add_argument("--val-instances", type=int, default=1)
    parser.add_argument("--customers", type=int, nargs="+", default=[12])
    parser.add_argument("--states-per-instance", type=int, default=2)
    parser.add_argument("--preferences-per-state", type=int, default=3)
    args = parser.parse_args()
    generate_dataset(GenerationConfig(output_dir=args.output_dir, seed=args.seed, train_instances=args.train_instances, val_instances=args.val_instances, customers=tuple(args.customers), states_per_instance=args.states_per_instance, preferences_per_state=args.preferences_per_state))


if __name__ == "__main__":
    _main()
