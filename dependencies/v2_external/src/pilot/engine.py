"""Full insertion action space and route-derived features."""
from dataclasses import dataclass
import numpy as np
from src.core.protocol import RepairAction, Route, Solution
from src.evaluation.vrptw_eval import is_solution_feasible


def route_trace(instance, customers):
    t, prev, load = 0., 0, 0.
    trace, ok = [], True
    for c in customers:
        arrival = t + instance.duration_matrix[prev, c]
        early, late = instance.customer_time_windows[c-1]
        waiting, tardy = max(0., early-arrival), max(0., arrival-late)
        t = arrival + waiting + instance.customer_service_times[c-1]
        load += instance.customer_demands[c-1]
        ok &= tardy <= instance.max_delay + 1e-6
        trace.append((arrival, waiting, tardy))
        prev = c
    ok &= load <= instance.vehicle_capacity + 1e-6
    ok &= t + instance.duration_matrix[prev, 0] <= instance.depot_time_window[1] + 1e-6
    return trace, bool(ok), load


def insertion_context(instance, customers):
    """Precompute exact prefix departures and suffix arrival limits."""
    prefix_departures = np.zeros(len(customers) + 1, dtype=np.float64)
    current_time, previous = 0.0, 0
    for index, customer in enumerate(customers):
        arrival = current_time + instance.duration_matrix[previous, customer]
        early = instance.customer_time_windows[customer - 1, 0]
        current_time = max(arrival, early) + instance.customer_service_times[customer - 1]
        prefix_departures[index + 1] = current_time
        previous = customer

    latest_arrivals = np.empty(len(customers), dtype=np.float64)
    suffix_limit = float(instance.depot_time_window[1])
    next_customer = 0
    for index in range(len(customers) - 1, -1, -1):
        customer = customers[index]
        _, late = instance.customer_time_windows[customer - 1]
        latest_arrival = min(
            float(late + instance.max_delay),
            suffix_limit
            - float(instance.duration_matrix[customer, next_customer])
            - float(instance.customer_service_times[customer - 1]),
        )
        latest_arrivals[index] = latest_arrival
        suffix_limit = latest_arrival
        next_customer = customer
    return prefix_departures, latest_arrivals


def insertion_legal(instance, customers, context, customer, position):
    prefix_departures, latest_arrivals = context
    previous = customers[position - 1] if position else 0
    arrival = prefix_departures[position] + instance.duration_matrix[previous, customer]
    early, late = instance.customer_time_windows[customer - 1]
    if arrival > late + instance.max_delay + 1e-6:
        return False
    departure = max(arrival, early) + instance.customer_service_times[customer - 1]
    if position < len(customers):
        following = customers[position]
        return departure + instance.duration_matrix[customer, following] <= latest_arrivals[position] + 1e-6
    return departure + instance.duration_matrix[customer, 0] <= instance.depot_time_window[1] + 1e-6


@dataclass
class Actions:
    actions: list
    mask: np.ndarray
    indices: np.ndarray
    features: np.ndarray


def enumerate_actions(instance, partial, remaining):
    actions, valid, indices, features = [], [], [], []
    dscale = max(float(instance.distance_matrix.max()), 1.)
    route_loads = [
        float(sum(instance.customer_demands[x-1] for x in route.customers))
        for route in partial.routes
    ]
    route_contexts = [
        insertion_context(instance, route.customers)
        for route in partial.routes
    ]
    for c in sorted(remaining):
        for rid in range(len(partial.routes) + 1):
            new = rid == len(partial.routes)
            customers = [] if new else partial.routes[rid].customers
            load = 0.0 if new else route_loads[rid]
            context = insertion_context(instance, customers) if new else route_contexts[rid]
            for pos in range(len(customers) + 1):
                prev = customers[pos-1] if pos else 0
                nxt = customers[pos] if pos < len(customers) else 0
                capacity_ok = load + instance.customer_demands[c - 1] <= instance.vehicle_capacity + 1e-6
                ok = capacity_ok and insertion_legal(instance, customers, context, c, pos)
                actions.append(RepairAction(c, -1 if new else rid, pos, new))
                indices.append((c, prev, nxt, rid))
                valid.append(ok)
                delta = instance.distance_matrix[prev, c] + instance.distance_matrix[c, nxt] - instance.distance_matrix[prev, nxt]
                features.append((delta / dscale, float(new), (instance.vehicle_capacity-load-instance.customer_demands[c-1])/instance.vehicle_capacity,
                                 pos/max(len(customers), 1)))
    return Actions(actions, np.asarray(valid, dtype=bool), np.asarray(indices, dtype=np.int64).reshape(-1, 4),
                   np.asarray(features, dtype=np.float32).reshape(-1, 4))


def apply_action(partial, remaining, action):
    if action.customer_id not in remaining:
        raise ValueError('Action must insert a remaining customer')
    routes = [list(r.customers) for r in partial.routes]
    if action.opens_new_route:
        if action.route_index != -1 or action.insert_position != 0:
            raise ValueError('Invalid new route action')
        routes.append([action.customer_id])
    else:
        if not 0 <= action.route_index < len(routes) or not 0 <= action.insert_position <= len(routes[action.route_index]):
            raise ValueError('Invalid insertion position')
        routes[action.route_index].insert(action.insert_position, action.customer_id)
    return Solution([Route(r) for r in routes]), [c for c in remaining if c != action.customer_id]


def encode_state(instance, partial, remaining):
    n = instance.n_customers
    x = np.zeros((n+1, 16), np.float32)
    time_scale = max(instance.depot_time_window[1], 1.)
    dist_scale = max(float(instance.distance_matrix.max()), 1.)
    coord_scale = max(float(np.abs(instance.customer_coords).max()), 1.)
    x[1:, :2] = instance.customer_coords / coord_scale
    x[1:, 2] = instance.customer_demands / instance.vehicle_capacity
    x[1:, 3:5] = instance.customer_time_windows / time_scale
    x[1:, 5] = instance.customer_service_times / time_scale
    x[:, 6] = instance.distance_matrix[0] / dist_scale
    x[:, 7] = instance.duration_matrix[0] / time_scale
    x[0, 15] = 1
    neighbors = np.zeros((n+1, 2), np.int64)
    membership = np.zeros((len(partial.routes)+1, n+1), np.float32)
    for rid, route in enumerate(partial.routes):
        trace, _, load = route_trace(instance, route.customers)
        for p, c in enumerate(route.customers):
            x[c, 8:12] = [1., p/max(len(route.customers), 1), load/instance.vehicle_capacity, len(route.customers)/n]
            x[c, 12:15] = np.asarray(trace[p])/time_scale
            neighbors[c] = [route.customers[p-1] if p else 0, route.customers[p+1] if p+1 < len(route.customers) else 0]
            membership[rid, c] = 1/max(len(route.customers), 1)
    return x, neighbors, membership


def complete_ok(instance, partial, remaining):
    return not remaining and is_solution_feasible(partial, instance)
