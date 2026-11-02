"""VRPTW solution evaluation logic."""

import numpy as np
from typing import List, Tuple
from ..core.protocol import VRPTWInstance, Solution, Route, FiveObjectives
from ..core.task_protocol import DEFAULT_PROTOCOL


def evaluate_route(
    route: Route,
    instance: VRPTWInstance,
) -> Tuple[float, float, float, float]:
    """
    Evaluate a single route.

    Args:
        route: Route to evaluate
        instance: VRPTW instance

    Returns:
        Tuple of (distance, duration, total_waiting, total_tardiness)
    """
    if len(route.customers) == 0:
        return 0.0, 0.0, 0.0, 0.0

    # Track metrics
    total_distance = 0.0
    total_load = 0.0
    current_time = 0.0
    total_waiting = 0.0
    total_tardiness = 0.0

    # Start from depot (index 0)
    prev = 0

    for customer in route.customers:
        # Add travel distance and duration
        travel_dist = instance.distance_matrix[prev, customer]
        travel_time = instance.duration_matrix[prev, customer]

        total_distance += travel_dist
        current_time += travel_time

        # Check arrival time vs time window
        early, late = instance.customer_time_windows[customer - 1]  # customer IDs are 1-indexed
        arrival_time = current_time

        # Calculate waiting if arrived early
        if arrival_time < early:
            waiting = early - arrival_time
            total_waiting += waiting
            current_time = early
        else:
            # Calculate tardiness if late
            if arrival_time > late:
                tardiness = arrival_time - late
                total_tardiness += tardiness
            current_time = arrival_time

        # Add service time
        service_time = instance.customer_service_times[customer - 1]
        current_time += service_time

        # Add demand
        demand = instance.customer_demands[customer - 1]
        total_load += demand

        prev = customer

    # Return to depot
    total_distance += instance.distance_matrix[prev, 0]
    return_time = instance.duration_matrix[prev, 0]
    final_time = current_time + return_time

    return total_distance, final_time, total_waiting, total_tardiness


def evaluate_solution(
    solution: Solution,
    instance: VRPTWInstance,
) -> FiveObjectives:
    """
    Evaluate a complete solution on all five objectives.

    Args:
        solution: Solution to evaluate
        instance: VRPTW instance

    Returns:
        FiveObjectives: (n_vehicles, total_distance, makespan, total_waiting, total_tardiness)
    """
    if len(solution.routes) == 0:
        return FiveObjectives(
            n_vehicles=0,
            total_distance=0.0,
            makespan=0.0,
            total_waiting=0.0,
            total_tardiness=0.0
        )

    total_distance = 0.0
    makespan = 0.0
    total_waiting = 0.0
    total_tardiness = 0.0
    n_vehicles = 0  # Count only non-empty routes

    for route in solution.routes:
        # Skip empty routes
        if len(route.customers) == 0:
            continue

        n_vehicles += 1
        dist, duration, waiting, tardiness = evaluate_route(route, instance)
        total_distance += dist
        makespan = max(makespan, duration)  # Parallel routes, max duration
        total_waiting += waiting
        total_tardiness += tardiness

    return FiveObjectives(
        n_vehicles=n_vehicles,
        total_distance=total_distance,
        makespan=makespan,
        total_waiting=total_waiting,
        total_tardiness=total_tardiness
    )


def is_solution_feasible(
    solution: Solution,
    instance: VRPTWInstance,
    tolerance: float = 1e-6
) -> bool:
    """
    Check if a solution is feasible (meets hard constraints).

    Hard constraints:
    - All customers visited exactly once
    - No capacity violations
    - Customer tardiness must not exceed max_delay
    - Depot return time must not exceed depot close time

    Args:
        solution: Solution to check
        instance: VRPTW instance
        tolerance: Numerical tolerance for constraint violations

    Returns:
        True if feasible, False otherwise
    """
    # Check all customers visited exactly once
    all_customers = solution.get_all_customers()
    expected_customers = set(range(1, instance.n_customers + 1))

    if set(all_customers) != expected_customers:
        return False

    if len(all_customers) != len(expected_customers):
        return False

    # Check each route for hard constraints
    for route in solution.routes:
        # Check capacity
        total_load = sum(instance.customer_demands[c - 1] for c in route.customers)
        if total_load > instance.vehicle_capacity + tolerance:
            return False

        # Check time windows and tardiness
        current_time = 0.0
        prev = 0

        for customer in route.customers:
            # Travel to customer
            travel_time = instance.duration_matrix[prev, customer]
            current_time += travel_time

            # Check customer time window
            early, late = instance.customer_time_windows[customer - 1]
            arrival_time = current_time

            # Wait if early
            if arrival_time < early:
                current_time = early
            else:
                # Check if tardiness exceeds max_delay
                tardiness = arrival_time - late
                if tardiness > instance.max_delay + tolerance:
                    return False
                current_time = arrival_time

            # Add service time
            service_time = instance.customer_service_times[customer - 1]
            current_time += service_time
            prev = customer

        # Return to depot and check depot close time
        return_time = instance.duration_matrix[prev, 0]
        final_time = current_time + return_time
        depot_close = instance.depot_time_window[1]

        if final_time > depot_close + tolerance:
            return False

    return True


def compute_scalarized_objective(
    objectives: np.ndarray,
    weights: np.ndarray,
    normalizer: 'Normalizer',
    method: str = "augmented_tchebycheff",
    rho: float = DEFAULT_PROTOCOL.tchebycheff_rho,
) -> float:
    """
    Compute scalarized objective value.

    Args:
        objectives: Array of shape (5,) with raw objective values
        weights: Array of shape (5,) with preference weights
        normalizer: Normalizer for objectives
        method: Scalarization method ("augmented_tchebycheff" or "weighted_sum")
        rho: Augmented Tchebycheff coefficient

    Returns:
        Scalar objective value
    """
    if objectives.shape != (5,):
        raise ValueError(f"Expected objectives shape (5,), got {objectives.shape}")
    if weights.shape != (5,):
        raise ValueError(f"Expected weights shape (5,), got {weights.shape}")

    # Normalize objectives
    norm_obj = normalizer.normalize(objectives)

    if method == "weighted_sum":
        return np.dot(weights, norm_obj)

    elif method == "augmented_tchebycheff":
        # max_i (w_i * norm_obj_i) + rho * sum_i (w_i * norm_obj_i)
        tcheby = np.max(weights * norm_obj)
        weighted_sum = np.sum(weights * norm_obj)
        return tcheby + rho * weighted_sum

    else:
        raise ValueError(f"Unknown scalarization method: {method}")


def is_pareto_dominated(
    candidate: np.ndarray,
    other: np.ndarray,
    tolerance: float = 1e-6
) -> bool:
    """
    Check if candidate is strictly dominated by other (minimization).

    Args:
        candidate: Objective vector to check
        other: Other objective vector
        tolerance: Numerical tolerance

    Returns:
        True if candidate is strictly dominated by other
    """
    if candidate.shape != other.shape:
        raise ValueError("Objectives must have same shape")

    # other must be <= candidate in all objectives
    all_leq = np.all(other <= candidate + tolerance)

    # other must be strictly < candidate in at least one objective
    any_less = np.any(other < candidate - tolerance)

    return all_leq and any_less
