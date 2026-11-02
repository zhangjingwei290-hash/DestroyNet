"""Solver-neutral, single-candidate local-search adapter with explicit FE ledger."""
from pathlib import Path
import time
import torch

from frozen_repairnet import load_frozen_repairnet
from src.cooperative.model import DestroyNet, HeritageRepair, remove_customers, validate_preference
from src.pilot.engine import enumerate_actions, apply_action, complete_ok
from src.pilot.model import tensor_features


class CooperativeOperator:
    def __init__(self, checkpoint, device="cuda"):
        payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
        if payload["binding"]["protocol"] != "preference_cooperative_operator_v1":
            raise ValueError("Not a cooperative checkpoint")
        base = load_frozen_repairnet(device)
        self.destroy = DestroyNet(base).to(device)
        self.repair = HeritageRepair(base).to(device)
        self.destroy.load_state_dict(payload["destroy"], strict=True)
        self.repair.load_state_dict(payload["repair"], strict=True)
        self.destroy.eval().requires_grad_(False)
        self.repair.eval().requires_grad_(False)
        self.repair.chunk_size = 4096
        self.device = device

    def propose(self, instance, source, preference, q, ledger):
        """Return one proposal, NEVER silently best-of-K or change acceptance.

        ledger uses existing complete-vector FE protocol. Prefix checks and
        insertion-distance features remain separately counted candidate work.
        The host retains its original acceptance/selection/archive rules.
        """
        if len(ledger.full) >= ledger.budget:
            raise StopIteration("Complete-vector FE budget exhausted")
        weights = validate_preference(preference, self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            removed, _, _ = self.destroy.select(instance, source, weights, q, self.device, greedy=True)
            partial = remove_customers(source, removed)
            remaining = list(removed)
            self.repair.set_source(instance, source, self.device)
            destroy_seconds = time.perf_counter() - started
            while remaining:
                actions = enumerate_actions(instance, partial, remaining)
                ledger.partial_checks += len(actions.actions)
                ledger.model_scores += len(actions.actions)
                if not actions.mask.any():
                    return dict(candidate=None, removed=removed, completed=False,
                                failure="no_legal_action", destroy_seconds=destroy_seconds)
                logits = self.repair(tensor_features(instance, partial, remaining, actions, self.device), weights)
                action = actions.actions[int(logits.argmax().item())]
                partial, remaining = apply_action(partial, remaining, action)
        objectives = ledger.evaluate(partial, complete=True)  # rejected/infeasible full proposals still charged
        feasible = complete_ok(instance, partial, remaining)
        return dict(candidate=partial if feasible else None, removed=removed, completed=True,
                    feasible=feasible, objectives=objectives.tolist(), destroy_seconds=destroy_seconds,
                    total_seconds=time.perf_counter() - started)
