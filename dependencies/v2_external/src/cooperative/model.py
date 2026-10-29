"""Customer-level removal policy and zero-initialized repair heritage adapter."""
from __future__ import annotations

import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.core.protocol import Route, Solution
from src.pilot.engine import encode_state, route_trace


def origin_features(instance, source):
    """Route statistics/proxies, NOT true complete-candidate objective deltas.

    Columns: inverse route size, detour, critical-route travel/service,
    wait, tardiness, soft slack, original return time, downstream tardiness.
    Time quantities use the depot horizon; distance uses max arc distance.
    """
    if sorted(source.get_all_customers()) != list(range(1, instance.n_customers + 1)):
        raise ValueError("Destroy requires complete unique customer coverage")
    horizon = max(float(instance.depot_time_window[1]), 1.)
    distance_scale = max(float(instance.distance_matrix.max()), 1.)
    rows = np.zeros((instance.n_customers + 1, 8), dtype=np.float32)
    traces, finishes = [], []
    for route in source.routes:
        trace, _, _ = route_trace(instance, route.customers)
        traces.append(trace)
        if trace:
            last = route.customers[-1]
            finishes.append(trace[-1][0] + trace[-1][1] +
                            instance.customer_service_times[last - 1] + instance.duration_matrix[last, 0])
        else:
            finishes.append(0.)
    longest = max(finishes, default=0.)
    for route, trace, finish in zip(source.routes, traces, finishes):
        downstream = np.cumsum([point[2] for point in trace][::-1])[::-1]
        for position, customer in enumerate(route.customers):
            previous = route.customers[position - 1] if position else 0
            following = route.customers[position + 1] if position + 1 < len(trace) else 0
            detour = (instance.distance_matrix[previous, customer] +
                      instance.distance_matrix[customer, following] - instance.distance_matrix[previous, following])
            critical_cost = max(0., instance.duration_matrix[previous, customer] +
                                instance.duration_matrix[customer, following] - instance.duration_matrix[previous, following])
            critical_cost += instance.customer_service_times[customer - 1]
            arrival, waiting, tardiness = trace[position]
            rows[customer] = (1. / len(trace), detour / distance_scale,
                              critical_cost / horizon if np.isclose(finish, longest) else 0.,
                              waiting / horizon, tardiness / horizon,
                              (instance.customer_time_windows[customer - 1, 1] + instance.max_delay - arrival) / horizon,
                              finish / horizon, downstream[position] / horizon)
    return rows


def remove_customers(source, removed):
    removed = [int(c) for c in removed]
    if not removed or len(set(removed)) != len(removed) or not set(removed) <= set(source.get_all_customers()):
        raise ValueError("Removal must contain distinct existing customers, not the depot")
    excluded = set(removed)
    routes = [[c for c in route.customers if c not in excluded] for route in source.routes]
    return Solution([Route(route) for route in routes if route])


def validate_preference(preference, device):
    weights = torch.as_tensor(preference, dtype=torch.float32, device=device)
    if weights.shape != (5,) or not torch.isfinite(weights).all() or (weights < 0).any() or not torch.isclose(weights.sum(), weights.new_tensor(1.), atol=1e-5):
        raise ValueError("Preference must be a finite nonnegative five-dimensional simplex vector")
    return weights


class DestroyNet(nn.Module):
    """One encoder pass, then q cheap selected-set-aware pointer steps.

    Parameters are independent of RepairNet after initialization. No heuristic
    operator is chosen. Depot and already selected customers are hard masked.
    """
    def __init__(self, repair_base):
        super().__init__()
        self.nodes = copy.deepcopy(repair_base.nodes)
        self.adjacency = copy.deepcopy(repair_base.adjacency)
        self.encoder = copy.deepcopy(repair_base.encoder)
        self.preference = copy.deepcopy(repair_base.preference)
        self.physical = nn.Linear(8, 128)
        self.film = nn.Linear(128, 256)
        self.query = nn.Sequential(nn.Linear(128 * 3 + 2, 128), nn.Tanh())
        self.keys = nn.Linear(128, 128, bias=False)
        self.prior_strength = nn.Parameter(torch.tensor(.05))
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.requires_grad_(True)  # repair_base can be frozen

    def encode(self, instance, source, preference, device):
        weights = validate_preference(preference, device)
        x, neighbors, _ = encode_state(instance, source, [])
        x = torch.as_tensor(x, device=device)
        neighbors = torch.as_tensor(neighbors, device=device)
        physical = torch.as_tensor(origin_features(instance, source), device=device)
        p = self.preference(weights)
        h = self.nodes(x)
        h = h + self.adjacency(torch.cat((h[neighbors[:, 0]], h[neighbors[:, 1]]), -1)) + p
        h = self.encoder(h[None]).squeeze(0)
        scale, shift = self.film(p).chunk(2)
        h = h * (1. + .1 * torch.tanh(scale)) + .1 * shift + self.physical(physical)
        # Only a dimensionally scaled bias; not a claim of an exact gradient.
        prior = physical[:, :5] @ weights
        return h, p, prior

    def select(self, instance, source, preference, q, device, *, greedy=False, generator=None, replay=None):
        if not 1 <= q <= min(15, instance.n_customers):
            raise ValueError("q must be in [1, min(15, N)]")
        if replay is not None and len(replay) != q:
            raise ValueError("Replay length differs from q")
        h, p, prior = self.encode(instance, source, preference, device)
        keys = self.keys(h)
        chosen, log_probabilities, entropies = [], [], []
        for step in range(q):
            selected = h[chosen].mean(0) if chosen else torch.zeros_like(p)
            progress = h.new_tensor((q / instance.n_customers, step / q))
            query = self.query(torch.cat((h[1:].mean(0), selected, p, progress)))
            logits = (keys @ query) / np.sqrt(h.shape[-1]) + self.prior_strength * prior
            mask = torch.ones(len(h), dtype=torch.bool, device=device)
            mask[0] = False
            mask[chosen] = False
            logits = logits.masked_fill(~mask, -torch.inf)
            log_probs = F.log_softmax(logits, dim=0)
            if replay is not None:
                customer = int(replay[step])
                if customer < 1 or customer >= len(h) or not mask[customer]:
                    raise ValueError("Illegal removal replay")
            elif greedy:
                customer = int(logits.argmax().item())
            else:
                customer = int(torch.multinomial(log_probs.exp(), 1, generator=generator).item())
            chosen.append(customer)
            log_probabilities.append(log_probs[customer])
            entropy = -(log_probs[mask].exp() * log_probs[mask]).sum()
            entropies.append(entropy / max(np.log(int(mask.sum())), 1.))
        # Sum is the true ordered-without-replacement trajectory log probability.
        return chosen, torch.stack(log_probabilities).sum(), torch.stack(entropies).mean()


class HeritageRepair(nn.Module):
    """Keep existing RepairNet; add an initially zero action-logit correction.

    Original topology accompanies destroyed customers through reconstruction;
    set_source must run before every independent repair/replay.
    """
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.heritage = nn.Sequential(nn.Linear(8 + 5 + 4, 64), nn.ReLU(), nn.Linear(64, 1))
        nn.init.zeros_(self.heritage[-1].weight)
        nn.init.zeros_(self.heritage[-1].bias)
        self._origin = None

    @property
    def chunk_size(self):
        return self.base.chunk_size

    @chunk_size.setter
    def chunk_size(self, value):
        self.base.chunk_size = value

    def set_source(self, instance, source, device):
        self._origin = torch.as_tensor(origin_features(instance, source), device=device)

    def forward(self, features, weights):
        if self._origin is None:
            raise RuntimeError("set_source is required before repairing")
        indices, extras = features[3:5]
        context = torch.cat((self._origin[indices[:, 0]], weights.expand(len(indices), -1), extras), -1)
        return self.base(features, weights) + self.heritage(context).squeeze(-1)
