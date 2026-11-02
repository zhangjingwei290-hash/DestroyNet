import time
from dataclasses import asdict
import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from src.data_generation.sharded import _solution
from src.evaluation.vrptw_eval import evaluate_solution
from src.pilot.engine import encode_state, enumerate_actions, apply_action, complete_ok


class RepairNet(nn.Module):
    def __init__(self, conditioned=True, chunk_size=128):
        super().__init__()
        self.conditioned = conditioned
        self.chunk_size = chunk_size
        self.nodes = nn.Linear(16, 128)
        self.adjacency = nn.Linear(256, 128)
        self.preference = nn.Sequential(nn.Linear(5, 64), nn.ReLU(), nn.Linear(64, 128))
        layer = nn.TransformerEncoderLayer(128, 4, 256, dropout=0., batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.empty_route = nn.Parameter(torch.zeros(128))
        self.head = nn.Sequential(nn.Linear(128*6+4, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, features, weights):
        x, neighbors, membership, indices, extras, mask = features
        w = weights if self.conditioned else torch.zeros_like(weights)
        p = self.preference(w)
        h = self.nodes(x)
        h = h + self.adjacency(torch.cat([h[neighbors[:, 0]], h[neighbors[:, 1]]], -1)) + p
        h = self.encoder(h.unsqueeze(0)).squeeze(0)
        routes = membership @ h
        routes = routes + (membership.sum(-1) == 0).float().unsqueeze(-1)*self.empty_route
        global_h = h.mean(0)
        logits = []
        for start in range(0, len(indices), self.chunk_size):
            ids = indices[start:start+self.chunk_size]
            v = torch.cat([h[ids[:, 0]], h[ids[:, 1]], h[ids[:, 2]], routes[ids[:, 3]],
                           global_h.expand(len(ids), -1), p.expand(len(ids), -1), extras[start:start+len(ids)]], -1)
            out = checkpoint(self.head, v, use_reentrant=False) if self.training else self.head(v)
            logits.append(out.squeeze(-1))
        return torch.cat(logits).masked_fill(~mask, -torch.inf)


def tensor_features(instance, partial, remaining, actions, device):
    x, neighbors, membership = encode_state(instance, partial, remaining)
    return tuple(torch.as_tensor(v, device=device) for v in (x, neighbors, membership, actions.indices, actions.features, actions.mask))


def repair(model, instance, state, preference, device='cuda', profile=False):
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    started = time.perf_counter()
    partial = _solution(state['partial_solution'])
    remaining = list(state['removed_customers'])
    trajectory, failure = [], None
    phase_seconds = dict(enumerate=0.0, features=0.0, model=0.0, apply=0.0)
    model.eval()
    if hasattr(model, 'reset_context'):
        model.reset_context()
    preference_tensor = torch.as_tensor(preference, dtype=torch.float32, device=device)
    with torch.inference_mode():
        for _ in range(len(remaining)):
            phase_started = time.perf_counter()
            actions = enumerate_actions(instance, partial, remaining)
            phase_seconds['enumerate'] += time.perf_counter() - phase_started
            if not actions.mask.any():
                failure = 'no_legal_action'
                break
            phase_started = time.perf_counter()
            f = tensor_features(instance, partial, remaining, actions, device)
            if profile and device.startswith('cuda'):
                torch.cuda.synchronize()
            phase_seconds['features'] += time.perf_counter() - phase_started
            phase_started = time.perf_counter()
            logits = model(f, preference_tensor)
            action = actions.actions[int(logits.argmax().item())]
            phase_seconds['model'] += time.perf_counter() - phase_started
            phase_started = time.perf_counter()
            partial, remaining = apply_action(partial, remaining, action)
            trajectory.append(asdict(action))
            phase_seconds['apply'] += time.perf_counter() - phase_started
    feasible = complete_ok(instance, partial, remaining)
    if not feasible and failure is None:
        failure = 'final_infeasible'
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    return dict(completed=not remaining, feasible=feasible, failure=failure, actions=trajectory,
                final_solution=[r.customers for r in partial.routes],
                objectives=evaluate_solution(partial, instance).to_array().tolist(),
                runtime_seconds=time.perf_counter()-started,
                phase_seconds=phase_seconds if profile else None)
