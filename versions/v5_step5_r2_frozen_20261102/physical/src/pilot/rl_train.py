"""Resumable preference-conditioned policy-gradient training for RepairNet."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from src.pilot.architecture_models import build_model
from src.pilot.data import canonical, save_json, sha
from src.pilot.experiment import cuda_check, seed_all
from src.pilot.rl_core import (
    build_solution_pools,
    leave_one_out_advantages,
    make_specs,
    materialize,
    rollout,
    scalar_score,
    teacher_bc_loss,
    trajectory_policy_stats,
    RLDataset,
)


@dataclass(frozen=True)
class RLConfig:
    groups_per_epoch: int = 1400
    group_size: int = 4
    groups_per_update: int = 8
    max_epochs: int = 5
    patience: int = 4
    dev_groups: int = 210
    test_groups: int = 420
    learning_rate: float = 3e-5
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    bc_weight: float = 0.05
    entropy_weight: float = 0.0
    temperature: float = 1.0
    advantage_rms_floor: float = 1e-3
    signal_tolerance: float = 1e-6
    failure_reward: float = -1.0
    shared_fraction: float = 0.20

    def __post_init__(self):
        if self.group_size < 2 or self.groups_per_update < 1:
            raise ValueError("RL requires at least two rollouts and one group per update")
        if self.groups_per_epoch < self.groups_per_update:
            raise ValueError("groups_per_epoch is smaller than one update")
        if self.max_epochs < 1 or self.patience < 1:
            raise ValueError("max_epochs and patience must be positive")
        if not 0 <= self.shared_fraction < 1:
            raise ValueError("shared_fraction must be in [0, 1)")


def _binding(dataset, checkpoint: Path, architecture: str, seed: int, config: RLConfig):
    return dict(
        algorithm="preference_group_reinforce_v1",
        architecture=architecture,
        seed=seed,
        dataset_manifest_sha256=sha(dataset.run / "pilot_manifest.json"),
        bc_checkpoint_sha256=sha(checkpoint),
        config=asdict(config),
        implementation={
            name: sha(Path(__file__).parent / name)
            for name in ("rl_core.py", "rl_train.py")
        },
    )


def load_bc_model(checkpoint: Path, architecture: str, dataset, device: str):
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    binding = payload.get("binding", {})
    if binding:
        if binding.get("architecture") != architecture:
            raise ValueError("BC checkpoint architecture mismatch")
        if binding.get("manifest_hash") != sha(dataset.run / "pilot_manifest.json"):
            raise ValueError("BC checkpoint dataset mismatch")
    elif payload.get("conditioned") is not True:
        raise ValueError("RL requires a preference-conditioned BC checkpoint")
    model = build_model(architecture).to(device)
    model.load_state_dict(payload["model"])
    return model


def _atomic_checkpoint(path: Path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def save_checkpoint(path, model, optimizer, progress, binding):
    _atomic_checkpoint(Path(path), dict(
        binding=binding,
        model=model.state_dict(),
        optimizer=optimizer.state_dict(),
        progress=progress,
        torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all(),
        numpy_rng=np.random.get_state(),
        python_rng=random.getstate(),
    ))


def restore_checkpoint(path, model, optimizer, binding):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["binding"] != binding:
        raise ValueError("RL checkpoint configuration, data, code, or BC source changed")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    torch.set_rng_state(payload["torch_rng"])
    torch.cuda.set_rng_state_all(payload["cuda_rng"])
    np.random.set_state(payload["numpy_rng"])
    random.setstate(payload["python_rng"])
    return payload["progress"]


def _reward(task, result, normalization, failure_reward):
    preference = task.spec.preference
    source_score = scalar_score(task.source_objectives, preference, normalization)
    if not result.completed or not result.feasible:
        return failure_reward, source_score, None
    final_score = scalar_score(result.objectives, preference, normalization)
    return source_score - final_score, source_score, final_score


def _group_rewards(task, outcomes, normalization, failure_reward, tolerance):
    """Score a rollout group while keeping every failure below feasible outcomes."""
    rewards = [
        _reward(task, outcome, normalization, failure_reward)[0]
        if outcome.completed and outcome.feasible else None
        for outcome in outcomes
    ]
    feasible_rewards = [reward for reward in rewards if reward is not None]
    if feasible_rewards:
        failure_value = min(failure_reward, min(feasible_rewards) - max(tolerance, 1e-6))
    else:
        failure_value = failure_reward
    return np.asarray([
        failure_value if reward is None else reward
        for reward in rewards
    ], dtype=np.float64)


def _sample_groups(model, tasks, config, normalization, device):
    sampled = []
    for task in tasks:
        outcomes = [
            rollout(model, task, device, greedy=False, temperature=config.temperature)
            for _ in range(config.group_size)
        ]
        rewards = _group_rewards(
            task, outcomes, normalization, config.failure_reward, config.signal_tolerance
        )
        advantages, active = leave_one_out_advantages(rewards, config.signal_tolerance)
        sampled.append((task, outcomes, rewards, advantages, active))
    return sampled


def _committed_updates(path: Path, max_update: int):
    """Drop a log row written after the most recent durable checkpoint."""
    path = Path(path)
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    committed = [row for row in rows if row["update"] <= max_update]
    if len(committed) != len(rows):
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            for row in committed:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        temporary.replace(path)
    return committed


def _write_epoch_summary(path: Path, summary):
    """Atomically append an epoch, replacing a row replayed after interruption."""
    path = Path(path)
    rows = []
    if path.exists():
        rows = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
    rows = [row for row in rows if row["epoch"] != summary["epoch"]]
    rows.append(summary)
    rows.sort(key=lambda row: row["epoch"])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    temporary.replace(path)


def _backward_update(model, sampled, bc_item, config, device):
    active_advantages = np.concatenate([row[3] for row in sampled if row[4]]) \
        if any(row[4] for row in sampled) else np.empty(0)
    advantage_rms = float(np.sqrt(np.mean(active_advantages ** 2))) if len(active_advantages) else 0.0
    advantage_scale = max(advantage_rms, config.advantage_rms_floor)
    policy_values, entropy_values = [], []
    active_groups = sum(int(row[4]) for row in sampled)
    denominator = max(active_groups * config.group_size, 1)

    for task, outcomes, _, advantages, active in sampled:
        if not active:
            continue
        for outcome, advantage in zip(outcomes, advantages):
            if not outcome.actions:
                continue
            mean_log_probability, mean_entropy = trajectory_policy_stats(
                model, task, outcome, device, config.temperature
            )
            normalized_advantage = float(advantage / advantage_scale)
            policy_loss = -normalized_advantage * mean_log_probability
            loss = (policy_loss - config.entropy_weight * mean_entropy) / denominator
            loss.backward()
            policy_values.append(float(policy_loss.detach().cpu()))
            entropy_values.append(float(mean_entropy.detach().cpu()))

    instance, state, task = bc_item
    bc_loss = teacher_bc_loss(model, instance, state, task, device)
    (config.bc_weight * bc_loss).backward()
    return dict(
        policy_loss=float(np.mean(policy_values)) if policy_values else 0.0,
        bc_loss=float(bc_loss.detach().cpu()),
        entropy=float(np.mean(entropy_values)) if entropy_values else 0.0,
        active_groups=active_groups,
        advantage_rms=advantage_rms,
        advantage_scale=advantage_scale,
    )


def train_update(model, optimizer, tasks, bc_item, config, normalization, device):
    sampling_started = time.perf_counter()
    sampled = _sample_groups(model, tasks, config, normalization, device)
    sampling_seconds = time.perf_counter() - sampling_started
    rewards = np.concatenate([row[2] for row in sampled])
    feasible = [outcome.feasible and outcome.completed for _, outcomes, *_ in sampled for outcome in outcomes]

    replay_started = time.perf_counter()
    while True:
        optimizer.zero_grad(set_to_none=True)
        try:
            backward_metrics = _backward_update(model, sampled, bc_item, config, device)
            break
        except torch.cuda.OutOfMemoryError:
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            if not hasattr(model, "chunk_size") or model.chunk_size <= 1:
                raise
            model.chunk_size = max(1, model.chunk_size // 2)
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip))
    if not np.isfinite(gradient_norm):
        raise ValueError("Non-finite RL gradient norm")
    optimizer.step()
    return dict(
        **backward_metrics,
        mean_reward=float(rewards.mean()),
        best_group_reward=float(np.mean([row[2].max() for row in sampled])),
        improved_rollout_rate=float(np.mean(rewards > 0)),
        feasible_rate=float(np.mean(feasible)),
        gradient_norm=gradient_norm,
        sampling_seconds=sampling_seconds,
        replay_seconds=time.perf_counter() - replay_started,
        chunk_size=getattr(model, "chunk_size", None),
    )


def evaluate_specs(model, dataset, pools, specs, normalization, device, *, samples, seed, path):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    rows = []
    started = time.perf_counter()
    for spec in specs:
        task = materialize(spec, dataset, pools)
        outcomes = [
            rollout(model, task, device, greedy=samples == 1, generator=generator)
            for _ in range(samples)
        ]
        feasible = [outcome for outcome in outcomes if outcome.completed and outcome.feasible]
        source_score = scalar_score(task.source_objectives, spec.preference, normalization)
        if feasible:
            chosen = min(feasible, key=lambda result: scalar_score(result.objectives, spec.preference, normalization))
            final_score = scalar_score(chosen.objectives, spec.preference, normalization)
            improvement = source_score - final_score
        else:
            chosen, final_score, improvement = outcomes[0], None, None
        rows.append(dict(
            instance_id=spec.instance_id,
            state_hash=task.state_hash,
            destroy_type=spec.destroy_type,
            destroy_ratio=spec.destroy_ratio,
            preference_category=spec.preference_category,
            preference=list(spec.preference),
            source_score=source_score,
            final_score=final_score,
            improvement=improvement,
            feasible=chosen.feasible and chosen.completed,
            failure=chosen.failure,
            samples=samples,
            unique_sampled_routes=len({canonical(result.final_routes) for result in feasible}),
            objectives=chosen.objectives,
            final_routes=chosen.final_routes,
        ))
    with Path(path).open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    valid = [row for row in rows if row["feasible"]]
    improvements = [row["improvement"] for row in valid]
    return dict(
        groups=len(rows),
        failures=len(rows) - len(valid),
        feasible_rate=len(valid) / len(rows),
        mean_improvement=float(np.mean(improvements)) if improvements else None,
        median_improvement=float(np.median(improvements)) if improvements else None,
        p10_improvement=float(np.percentile(improvements, 10)) if improvements else None,
        improvement_rate=float(np.mean(np.asarray(improvements) > 1e-6)) if improvements else None,
        mean_unique_sampled_routes=float(np.mean([row["unique_sampled_routes"] for row in rows])),
        elapsed_seconds=time.perf_counter() - started,
    )


def _rank(metrics):
    mean = metrics["mean_improvement"]
    return metrics["failures"], -(mean if mean is not None else -float("inf"))


class RLRun:
    def __init__(self, dataset_run, output_root, run_name, bc_checkpoint, architecture, seed,
                 config: RLConfig, device="cuda"):
        self.dataset = RLDataset(dataset_run)
        self.output = Path(output_root) / run_name / f"{architecture}_{seed}"
        self.output.mkdir(parents=True, exist_ok=True)
        self.bc_checkpoint = Path(bc_checkpoint)
        self.architecture = architecture
        self.seed = seed
        self.config = config
        self.device = device
        # Test data stays unread until the explicit evaluate_test stage.
        self.pools = {
            split: build_solution_pools(self.dataset, split)
            for split in ("train", "dev")
        }
        self.binding = _binding(self.dataset, self.bc_checkpoint, architecture, seed, config)

    def _pool(self, split):
        if split not in self.pools:
            self.dataset.load_split(split)
            self.pools[split] = build_solution_pools(self.dataset, split)
        return self.pools[split]

    def _new_model(self):
        return load_bc_model(
            self.bc_checkpoint, self.architecture, self.dataset, self.device
        )

    def _fixed_specs(self, split):
        total = self.config.dev_groups if split == "dev" else self.config.test_groups
        split_seed = 17074 if split == "dev" else 17075
        return make_specs(
            self._pool(split), seed=split_seed, epoch=3, total=total, shared_fraction=0.0
        )

    def smoke(self):
        cuda_check()
        seed_all(self.seed)
        model = self._new_model()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay
        )
        specs = make_specs(self.pools["train"], seed=self.seed, epoch=1, total=2, shared_fraction=0.0)
        tasks = [materialize(spec, self.dataset, self.pools["train"]) for spec in specs]
        metrics = train_update(
            model, optimizer, tasks, self.dataset.get(self.dataset.train[0]),
            self.config, self.dataset.normalization, self.device
        )
        report = dict(status="passed", bc_checkpoint=str(self.bc_checkpoint.resolve()),
                      architecture=self.architecture, seed=self.seed, metrics=metrics)
        save_json(self.output / "smoke.json", report)
        print("RL smoke passed", report, flush=True)

    def train(self):
        cuda_check()
        seed_all(self.seed)
        save_json(self.output / "resolved_config.json", self.binding)
        model = self._new_model()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay
        )
        progress = dict(epoch=0, cursor=0, update=0, best=None, stale=0, status="running")
        last_path = self.output / "last.pt"
        if last_path.exists():
            progress = restore_checkpoint(last_path, model, optimizer, self.binding)
            if progress["status"] in ("complete", "early_stopped"):
                print("RL run already finished", progress, flush=True)
                return
            committed_updates = _committed_updates(
                self.output / "updates.jsonl", progress["update"]
            )
        else:
            committed_updates = []
            initial_specs = self._fixed_specs("dev")
            initial_metrics = evaluate_specs(
                model, self.dataset, self.pools["dev"], initial_specs,
                self.dataset.normalization, self.device, samples=1, seed=17076,
                path=self.output / "dev_initial.jsonl"
            )
            progress["best"] = initial_metrics
            save_json(self.output / "dev_initial_metrics.json", initial_metrics)
            save_checkpoint(self.output / "best.pt", model, optimizer, progress, self.binding)
            save_checkpoint(last_path, model, optimizer, progress, self.binding)

        while progress["epoch"] < self.config.max_epochs and progress["stale"] < self.config.patience:
            epoch_number = progress["epoch"] + 1
            specs = make_specs(
                self.pools["train"], seed=self.seed, epoch=epoch_number,
                total=self.config.groups_per_epoch, shared_fraction=self.config.shared_fraction
            )
            epoch_records = [
                row for row in committed_updates if row["epoch"] == epoch_number
            ]
            while progress["cursor"] < len(specs):
                stop = min(progress["cursor"] + self.config.groups_per_update, len(specs))
                tasks = [
                    materialize(spec, self.dataset, self.pools["train"])
                    for spec in specs[progress["cursor"]:stop]
                ]
                bc_id = random.choice(self.dataset.train)
                metrics = train_update(
                    model, optimizer, tasks, self.dataset.get(bc_id), self.config,
                    self.dataset.normalization, self.device
                )
                progress["cursor"] = stop
                progress["update"] += 1
                record = dict(epoch=epoch_number, update=progress["update"],
                              groups_done=stop, groups_total=len(specs), **metrics)
                epoch_records.append(record)
                committed_updates.append(record)
                with (self.output / "updates.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                save_checkpoint(last_path, model, optimizer, progress, self.binding)
                print("rl", record, flush=True)

            dev_metrics = evaluate_specs(
                model, self.dataset, self.pools["dev"], self._fixed_specs("dev"),
                self.dataset.normalization, self.device, samples=1, seed=17076,
                path=self.output / f"dev_epoch_{epoch_number:02d}.jsonl"
            )
            progress["epoch"] = epoch_number
            progress["cursor"] = 0
            improved = _rank(dev_metrics) < _rank(progress["best"])
            if improved:
                progress["best"] = dev_metrics
                progress["stale"] = 0
                save_checkpoint(self.output / "best.pt", model, optimizer, progress, self.binding)
            else:
                progress["stale"] += 1
            epoch_summary = dict(
                epoch=epoch_number,
                updates=len(epoch_records),
                train_mean_reward=float(np.mean([row["mean_reward"] for row in epoch_records])),
                train_feasible_rate=float(np.mean([row["feasible_rate"] for row in epoch_records])),
                development=dev_metrics,
                improved=improved,
                stale=progress["stale"],
            )
            _write_epoch_summary(self.output / "epochs.jsonl", epoch_summary)
            save_json(self.output / "progress.json", epoch_summary)
            save_checkpoint(last_path, model, optimizer, progress, self.binding)
            print("RL epoch complete", epoch_summary, flush=True)

        progress["status"] = "early_stopped" if progress["stale"] >= self.config.patience else "complete"
        save_checkpoint(last_path, model, optimizer, progress, self.binding)
        save_json(self.output / "complete.json", progress)
        print("RL training finished", progress, flush=True)

    def evaluate(self, split="dev"):
        cuda_check()
        seed_all(self.seed)
        if split not in ("dev", "test"):
            raise ValueError("split must be dev or test")
        checkpoint = self.output / "best.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        model = build_model(self.architecture).to(self.device)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload["binding"] != self.binding:
            raise ValueError("Best checkpoint binding mismatch")
        model.load_state_dict(payload["model"])
        specs = self._fixed_specs(split)
        summaries = {}
        for samples in (1, 4):
            path = self.output / f"{split}_best_of_{samples}.jsonl"
            summaries[f"best_of_{samples}"] = evaluate_specs(
                model, self.dataset, self._pool(split), specs, self.dataset.normalization,
                self.device, samples=samples, seed=17077 + samples, path=path
            )
        save_json(self.output / f"{split}_evaluation.json", summaries)
        print(f"RL {split} evaluation", summaries, flush=True)

    def status(self):
        progress = self.output / "complete.json"
        if not progress.exists():
            progress = self.output / "progress.json"
        print("Output:", self.output.resolve(), flush=True)
        print(progress.read_text(encoding="utf-8") if progress.exists() else "No training progress yet", flush=True)
