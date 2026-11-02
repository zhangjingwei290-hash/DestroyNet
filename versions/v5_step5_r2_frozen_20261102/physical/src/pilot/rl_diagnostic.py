"""Paired BC/RL decoding diagnostics on the closed development split."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from src.pilot.architecture_models import build_model
from src.pilot.data import canonical, save_json, sha
from src.pilot.experiment import cuda_check, seed_all
from src.pilot.rl_core import (
    RLDataset,
    build_solution_pools,
    make_specs,
    materialize,
    rollout,
    scalar_score,
)


@dataclass(frozen=True)
class DiagnosticConfig:
    groups: int = 210
    sample_budgets: tuple[int, ...] = (4, 16)
    spec_seed: int = 17074
    sample_seed: int = 17116
    bootstrap_seed: int = 17117
    bootstrap_replicates: int = 2000
    tie_absolute: float = 1e-6
    tie_relative: float = 1e-5

    def __post_init__(self):
        if self.groups < 1 or not self.sample_budgets:
            raise ValueError("Diagnostic groups and sample budgets must be positive")
        if any(value < 1 for value in self.sample_budgets):
            raise ValueError("Sample budgets must be positive")


def _checkpoint_model(path: Path, architecture: str, dataset, device: str, *, rl: bool):
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    binding = payload.get("binding", {})
    expected_manifest = sha(dataset.run / "pilot_manifest.json")
    actual_manifest = binding.get(
        "dataset_manifest_sha256", binding.get("manifest_hash")
    )
    if actual_manifest != expected_manifest:
        raise ValueError(f"Checkpoint dataset mismatch: {path}")
    if binding.get("architecture") != architecture:
        raise ValueError(f"Checkpoint architecture mismatch: {path}")
    if rl:
        if binding.get("algorithm") != "preference_group_reinforce_v1":
            raise ValueError(f"Expected an RL checkpoint: {path}")
        progress = payload.get("progress", {})
        if not progress.get("best"):
            raise ValueError(f"RL checkpoint has no development selection record: {path}")
    elif binding.get("algorithm"):
        raise ValueError(f"Expected a BC checkpoint: {path}")
    model = build_model(architecture).to(device)
    model.load_state_dict(payload["model"])
    if hasattr(model, "chunk_size"):
        model.chunk_size = 4096
    model.eval()
    return model


def _route_hash(routes) -> str:
    encoded = json.dumps(canonical(routes), separators=(",", ":"))
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()


def _outcome(task, result, normalization, runtime_seconds):
    feasible = result.completed and result.feasible
    final_score = (
        scalar_score(result.objectives, task.spec.preference, normalization)
        if feasible else None
    )
    source_score = scalar_score(
        task.source_objectives, task.spec.preference, normalization
    )
    return {
        "feasible": feasible,
        "failure": result.failure,
        "score": final_score,
        "improvement": None if final_score is None else source_score - final_score,
        "objectives": result.objectives,
        "route_hash": _route_hash(result.final_routes),
        "runtime_seconds": runtime_seconds,
    }


def _choose(candidates, budget):
    subset = candidates[:budget]
    feasible = [row for row in subset if row["feasible"]]
    chosen = min(feasible, key=lambda row: row["score"]) if feasible else subset[0]
    return {
        "budget": budget,
        "feasible": bool(feasible),
        "failures": len(subset) - len(feasible),
        "score": chosen["score"] if feasible else None,
        "improvement": chosen["improvement"] if feasible else None,
        "objectives": chosen["objectives"],
        "route_hash": chosen["route_hash"],
        "unique_routes": len({row["route_hash"] for row in feasible}),
        "runtime_seconds": float(sum(row["runtime_seconds"] for row in subset)),
    }


def _evaluate_model(model, method, dataset, pools, specs, config, device, output):
    generator = torch.Generator(device=device)
    generator.manual_seed(config.sample_seed)
    max_budget = max(config.sample_budgets)
    rows = []

    warm = materialize(specs[0], dataset, pools)
    for _ in range(2):
        rollout(model, warm, device, greedy=True)

    for index, spec in enumerate(specs):
        task = materialize(spec, dataset, pools)
        started = time.perf_counter()
        greedy_result = rollout(model, task, device, greedy=True)
        greedy = _outcome(
            task, greedy_result, dataset.normalization, time.perf_counter() - started
        )
        sampled = []
        for _ in range(max_budget):
            started = time.perf_counter()
            result = rollout(model, task, device, greedy=False, generator=generator)
            sampled.append(
                _outcome(
                    task, result, dataset.normalization,
                    time.perf_counter() - started,
                )
            )
        rows.append({
            "index": index,
            "method": method,
            "spec": asdict(spec),
            "state_hash": task.state_hash,
            "source_score": scalar_score(
                task.source_objectives, spec.preference, dataset.normalization
            ),
            "greedy": _choose([greedy], 1),
            "sampled": {
                str(budget): _choose(sampled, budget)
                for budget in sorted(set(config.sample_budgets))
            },
            "sample_scores": [row["score"] for row in sampled],
            "sample_route_hashes": [row["route_hash"] for row in sampled],
        })
        if (index + 1) % 30 == 0 or index + 1 == len(specs):
            print(f"{method}: {index + 1}/{len(specs)} states", flush=True)

    temporary = Path(output).with_suffix(".partial")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    temporary.replace(output)
    return rows


def _metric(rows, key):
    decoded = [row["greedy"] if key == "greedy" else row["sampled"][key] for row in rows]
    valid = [row for row in decoded if row["feasible"]]
    improvements = np.asarray([row["improvement"] for row in valid], dtype=float)
    return {
        "states": len(rows),
        "failures": len(rows) - len(valid),
        "feasible_rate": len(valid) / len(rows),
        "mean_improvement": float(improvements.mean()) if len(improvements) else None,
        "median_improvement": float(np.median(improvements)) if len(improvements) else None,
        "p10_improvement": float(np.percentile(improvements, 10)) if len(improvements) else None,
        "improvement_rate": float(np.mean(improvements > 1e-6)) if len(improvements) else None,
        "mean_unique_routes": float(np.mean([row["unique_routes"] for row in decoded])),
        "mean_runtime_seconds": float(np.mean([row["runtime_seconds"] for row in decoded])),
    }


def _paired(bc_rows, rl_rows, key, config):
    instance_values = defaultdict(list)
    wins = ties = losses = invalid = 0
    for bc, rl in zip(bc_rows, rl_rows):
        if bc["state_hash"] != rl["state_hash"] or bc["spec"] != rl["spec"]:
            raise ValueError("BC and RL diagnostic rows are not paired")
        bc_decoded = bc["greedy"] if key == "greedy" else bc["sampled"][key]
        rl_decoded = rl["greedy"] if key == "greedy" else rl["sampled"][key]
        if not bc_decoded["feasible"] or not rl_decoded["feasible"]:
            invalid += 1
            continue
        gain = bc_decoded["score"] - rl_decoded["score"]
        tolerance = config.tie_absolute + config.tie_relative * abs(bc_decoded["score"])
        if gain > tolerance:
            wins += 1
        elif gain < -tolerance:
            losses += 1
        else:
            ties += 1
        instance_values[bc["spec"]["instance_id"]].append(gain)

    per_instance = np.asarray(
        [np.mean(values) for _, values in sorted(instance_values.items())], dtype=float
    )
    rng = np.random.default_rng(config.bootstrap_seed)
    bootstrap = np.asarray([
        rng.choice(per_instance, len(per_instance), replace=True).mean()
        for _ in range(config.bootstrap_replicates)
    ]) if len(per_instance) else np.empty(0)
    return {
        "paired_states": wins + ties + losses,
        "invalid_pairs": invalid,
        "rl_wins": wins,
        "ties": ties,
        "rl_losses": losses,
        "mean_rl_score_gain": float(per_instance.mean()) if len(per_instance) else None,
        "bootstrap95": (
            np.percentile(bootstrap, [2.5, 97.5]).tolist() if len(bootstrap) else None
        ),
        "instances": len(per_instance),
    }


def run_diagnostic(dataset_run, output, bc_checkpoint, rl_checkpoint, architecture,
                   config: DiagnosticConfig, device="cuda"):
    cuda_check()
    seed_all(config.sample_seed)
    dataset = RLDataset(dataset_run, splits=("dev",))
    pools = build_solution_pools(dataset, "dev")
    specs = make_specs(
        pools, seed=config.spec_seed, epoch=3, total=config.groups, shared_fraction=0.0
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    binding = {
        "dataset_manifest_sha256": sha(Path(dataset_run) / "pilot_manifest.json"),
        "bc_checkpoint_sha256": sha(bc_checkpoint),
        "rl_checkpoint_sha256": sha(rl_checkpoint),
        "architecture": architecture,
        "config": asdict(config),
        "primary_metric": "paired greedy scalar score; lower is better",
        "split": "dev",
    }
    save_json(output / "resolved_config.json", binding)

    methods = {}
    for method, checkpoint, is_rl in (
        ("bc", bc_checkpoint, False),
        ("rl", rl_checkpoint, True),
    ):
        model = _checkpoint_model(checkpoint, architecture, dataset, device, rl=is_rl)
        methods[method] = _evaluate_model(
            model, method, dataset, pools, specs, config, device, output / f"{method}.jsonl"
        )
        del model
        torch.cuda.empty_cache()

    keys = ["greedy", *(str(value) for value in sorted(set(config.sample_budgets)))]
    summary = {
        "resolved_config": binding,
        "methods": {
            method: {key: _metric(rows, key) for key in keys}
            for method, rows in methods.items()
        },
        "paired_rl_vs_bc": {
            key: _paired(methods["bc"], methods["rl"], key, config)
            for key in keys
        },
    }
    save_json(output / "summary.json", summary)
    save_json(output / "complete.json", {"status": "complete"})
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary
