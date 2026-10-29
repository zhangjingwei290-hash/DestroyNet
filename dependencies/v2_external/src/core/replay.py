"""Trajectory replay and validation."""

from typing import Set
from .protocol import (
    DestroyedState,
    RepairAction,
    ActionTrajectory,
    Solution,
    Route,
)


def replay_trajectory(
    destroyed_state: DestroyedState,
    trajectory: ActionTrajectory,
) -> Solution:
    """
    Replay a repair trajectory step-by-step.

    Args:
        destroyed_state: Initial state with partial solution and removed customers
        trajectory: Sequence of repair actions

    Returns:
        Final solution after applying all actions

    Raises:
        ValueError: If trajectory is invalid
    """
    # Start with partial solution
    current_routes = [Route(customers=list(r.customers)) for r in destroyed_state.partial_solution.routes]
    removed_set = set(destroyed_state.removed_customers)
    inserted_customers: Set[int] = set()

    for action in trajectory.actions:
        customer_id = action.customer_id

        # Validate customer is in removed set
        if customer_id not in removed_set:
            raise ValueError(
                f"Customer {customer_id} not in removed set {removed_set}"
            )

        # Validate customer not already inserted
        if customer_id in inserted_customers:
            raise ValueError(
                f"Customer {customer_id} inserted multiple times"
            )

        # Validate route index
        if action.opens_new_route:
            if action.route_index != -1:
                raise ValueError(
                    f"opens_new_route=True requires route_index=-1, got {action.route_index}"
                )
            # Create new route
            if action.insert_position != 0:
                raise ValueError(
                    f"New route must have insert_position=0, got {action.insert_position}"
                )
            current_routes.append(Route(customers=[customer_id]))
        else:
            # Insert into existing route
            if action.route_index < 0 or action.route_index >= len(current_routes):
                raise ValueError(
                    f"Invalid route_index {action.route_index}, "
                    f"must be in [0, {len(current_routes)})"
                )

            route = current_routes[action.route_index]
            if action.insert_position < 0 or action.insert_position > len(route.customers):
                raise ValueError(
                    f"Invalid insert_position {action.insert_position} for route with "
                    f"{len(route.customers)} customers, must be in [0, {len(route.customers)}]"
                )

            # Insert customer
            new_customers = list(route.customers)
            new_customers.insert(action.insert_position, customer_id)
            current_routes[action.route_index] = Route(customers=new_customers)

        inserted_customers.add(customer_id)

    # Verify all removed customers were inserted
    if inserted_customers != removed_set:
        missing = removed_set - inserted_customers
        raise ValueError(
            f"Not all removed customers were inserted. Missing: {missing}"
        )

    return Solution(routes=current_routes)


def validate_trajectory_replay(
    destroyed_state: DestroyedState,
    trajectory: ActionTrajectory,
    expected_solution: Solution,
) -> bool:
    """
    Validate that replaying trajectory produces expected solution.

    Args:
        destroyed_state: Initial state
        trajectory: Repair actions
        expected_solution: Expected final solution

    Returns:
        True if replay matches expected solution

    Raises:
        ValueError: If trajectory is invalid or result doesn't match
    """
    replayed_solution = replay_trajectory(destroyed_state, trajectory)

    # Compare solutions
    if len(replayed_solution.routes) != len(expected_solution.routes):
        raise ValueError(
            f"Route count mismatch: replayed has {len(replayed_solution.routes)}, "
            f"expected has {len(expected_solution.routes)}"
        )

    # Sort routes by first customer for comparison
    def route_key(route: Route) -> tuple:
        return (len(route.customers), route.customers[0] if route.customers else -1)

    replayed_sorted = sorted(replayed_solution.routes, key=route_key)
    expected_sorted = sorted(expected_solution.routes, key=route_key)

    for i, (replayed_route, expected_route) in enumerate(zip(replayed_sorted, expected_sorted)):
        if replayed_route.customers != expected_route.customers:
            raise ValueError(
                f"Route {i} mismatch: replayed={replayed_route.customers}, "
                f"expected={expected_route.customers}"
            )

    return True
