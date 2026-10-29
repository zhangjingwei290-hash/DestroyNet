"""Staged, resumable operator training. Formal benchmark instances stay closed."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import copy
import json
import random
import time
from contextlib import contextmanager

import numpy as np
import torch
from torch.nn import functional as F

from frozen_repairnet import CHECKPOINT, load_frozen_repairnet
from src.core.protocol import DestroyType, RepairAction
from src.data_generation.sharded import _solution
from src.pilot.engine import enumerate_actions, apply_action
from src.pilot.model import tensor_features
from src.destroy import destroy
from src.pilot.config import ROOT
from src.pilot.data import read_json, save_json, sha
from src.pilot.experiment import seed_all
from src.pilot.rl_core import (RLDataset, RepairTask, build_solution_pools, make_specs,
                               rollout, teacher_bc_loss,
                               leave_one_out_advantages, scalar_score)
from src.pilot.rl_train import _atomic_checkpoint
from src.cooperative.model import DestroyNet, HeritageRepair, remove_customers

DATA = ROOT / "baseline_frozen/artifacts/repairnet_pilot/pilot_50_v2"
SOURCE = ROOT / "baseline_frozen/data/destroy-repair-v017/destroy-repair-v017/artifacts/trainval_selected"
KINDS = (DestroyType.ROUTE_REPACK, DestroyType.WORST_EDGE_RELATED,
         DestroyType.MAKESPAN_SEGMENT, DestroyType.WAITING_SEGMENT, DestroyType.TARDINESS_SEGMENT)


@contextmanager
def exclusive_run(output):
    """OS-held lock automatically releases on crash; prevent two IDE launches."""
    handle = (output / ".run.lock").open("a+b")
    if handle.tell() == 0:
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    acquired = False
    try:
        import os
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError("This RUN_NAME is already running; do not launch a second copy") from error
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        acquired = True
        yield
    finally:
        if acquired:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


@dataclass(frozen=True)
class Config:
    seed: int = 17
    destroy_epochs: int = 4
    joint_epochs: int = 4
    groups_per_epoch: int = 400
    groups_per_update: int = 4
    group_size: int = 4
    dev_groups: int = 90
    destroy_lr: float = 1e-4
    repair_lr: float = 3e-6
    heritage_lr: float = 3e-5
    weight_decay: float = 1e-4
    gradient_clip: float = 1.
    entropy_weight: float = .005
    bc_weight: float = .05
    patience: int = 3
    invocation_hours: float = 4.

    def __post_init__(self):
        if min(self.destroy_epochs, self.joint_epochs, self.groups_per_epoch,
               self.groups_per_update, self.dev_groups, self.patience) < 1 or self.group_size < 2:
            raise ValueError("Counts must be positive; at least two samples per group")
        if self.groups_per_update > self.groups_per_epoch or self.invocation_hours <= 0:
            raise ValueError("Invalid update size or runtime guard")
        if min(self.destroy_lr, self.repair_lr, self.heritage_lr, self.gradient_clip) <= 0:
            raise ValueError("Learning rates and gradient clip must be positive")


class Dataset(RLDataset):
    """Relocate frozen source paths in memory; never rewrite the old manifest."""
    def __init__(self):
        self.run = DATA
        self.manifest = read_json(DATA / "pilot_manifest.json")
        for relative, expected in self.manifest["code_hashes"].items():
            if sha(ROOT / relative) != expected:
                raise ValueError(f"Inherited code changed: {relative}")
        self.manifest["source"] = str(SOURCE)
        self.normalization = self.manifest["normalization"]
        self.items, self.instances, self.states, self.loaded_splits = {}, {}, {}, set()
        for split in ("train", "dev"):
            self.load_split(split)


def audit_splits(dataset):
    """Exact physical fingerprints, including checks against exported benchmarks.

    Test shards are NOT opened. Their IDs/hashes are metadata-only exclusion
    checks. Historical BC teacher/normalization provenance remains a limitation.
    """
    from dataclasses import asdict as instance_dict
    from benchmark_report.data import fingerprint
    ids = dataset.manifest["instance_ids"]
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        if set(ids[left]) & set(ids[right]):
            raise ValueError("Instance IDs cross the frozen split")
        hashes = dataset.manifest["source_hashes"]
        if {hashes[p] for p in dataset.manifest["splits"][left]} & {hashes[p] for p in dataset.manifest["splits"][right]}:
            raise ValueError("Identical shards cross the frozen split")
    seen = {}
    split_ids = {split: {dataset.items[t]["instance_id"] for t in getattr(dataset, split)} for split in ("train", "dev")}
    for split, members in split_ids.items():
        for instance_id in members:
            digest = fingerprint(instance_dict(dataset.instances[instance_id]))
            if digest in seen and seen[digest] != split:
                raise ValueError("Physical train/dev duplicate")
            seen[digest] = split
    checked = set()
    # Locally exported original benchmark data, read for exclusion only. Never
    # feed benchmark solutions/objectives into training or checkpoint selection.
    for manifest_path in (ROOT / "artifacts").glob("benchmark_report*/data/benchmark_manifest.json"):
        manifest = read_json(manifest_path)
        for entry in manifest["instances"]:
            digest = entry["problem_sha256"]
            checked.add(digest)
            if digest in seen:
                raise ValueError(f"Training/dev overlaps benchmark: {entry['instance_id']}")
    if not checked:
        raise ValueError("No benchmark exclusion manifest found; audit before training")
    return dict(train_instances=len(split_ids["train"]), dev_instances=len(split_ids["dev"]),
                loaded_splits=sorted(dataset.loaded_splits), benchmark_fingerprints_checked=len(checked),
                exact_overlap=0, near_duplicate_audit=False,
                sizes=sorted({i.n_customers for i in dataset.instances.values()}),
                limitation="Inherited BC weights/teachers were not retrained; first pilot uses N50 only. No N250 efficacy claim.")


def train_normalization(pools):
    values = np.asarray([candidate.objectives for pool in pools.values() for candidate in pool])
    return dict(mins=values.min(0).tolist(), ranges=np.maximum(values.max(0) - values.min(0), 1.).tolist(),
                source="train_solution_pool_only", clipped=False)


def q_for(spec, instance):
    return min(15, instance.n_customers, max(1, round(instance.n_customers * spec.destroy_ratio)))


def task_for(spec, dataset, pools, removed):
    candidate = pools[spec.instance_id][spec.source_index]
    source = candidate.solution()
    partial = remove_customers(source, removed)
    return RepairTask(spec, dataset.instances[spec.instance_id], source, np.asarray(candidate.objectives),
                      [list(r.customers) for r in partial.routes], list(removed), "online_learned_removal")


def hand_removed(spec, dataset, pools):
    instance = dataset.instances[spec.instance_id]
    source = pools[spec.instance_id][spec.source_index].solution()
    kind = KINDS[int(np.argmax(spec.preference))]
    return destroy(source, instance, kind, q_for(spec, instance) / instance.n_customers,
                   seed=spec.destroy_seed).state.removed_customers


def repair_rollout(model, task, device, *, greedy, generator=None):
    model.set_source(task.instance, task.source, device)
    return rollout(model, task, device, greedy=greedy, generator=generator)


def repair_trajectory_logprob(model, task, result, device):
    """Replay complete trajectories OR failed decision prefixes with gradients.

    A legal insertion may cause a later dead end, especially with nonmetric
    matrices. Its negative advantage must not disappear just because repair
    failed. An empty prefix contains no sampled decision and cannot be replayed.
    """
    if not result.actions:
        raise ValueError("Cannot replay an empty repair prefix")
    model.set_source(task.instance, task.source, device)
    partial = _solution(task.partial_routes)
    remaining = list(task.removed_customers)
    weights = torch.as_tensor(task.spec.preference, dtype=torch.float32, device=device)
    model.train()  # all inherited stochastic dropout is zero
    probabilities = []
    for raw in result.actions:
        options = enumerate_actions(task.instance, partial, remaining)
        action = RepairAction(**raw)
        index = options.actions.index(action)
        if not options.mask[index]:
            raise ValueError("Repair prefix became illegal during replay")
        logits = model(tensor_features(task.instance, partial, remaining, options, device), weights)
        probabilities.append(F.log_softmax(logits, dim=0)[index])
        partial, remaining = apply_action(partial, remaining, action)
    if bool(not remaining) != result.completed:
        raise ValueError("Repair replay completion mismatch")
    return torch.stack(probabilities).sum()


def rewards_for(task, results, normalization):
    score = scalar_score(task.source_objectives, task.spec.preference, normalization)
    rewards = [score - scalar_score(r.objectives, task.spec.preference, normalization)
               if r.completed and r.feasible else None for r in results]
    valid = [r for r in rewards if r is not None]
    failure = min(-1., min(valid) - 1.) if valid else -1.
    return np.asarray([failure if r is None else r for r in rewards])


def advantages(rewards):
    adv, active = leave_one_out_advantages(rewards)
    return adv / max(float(np.sqrt(np.mean(adv ** 2))), 1e-3), active


def train_update(destroyer, repairer, destroy_optimizer, repair_optimizer, specs, dataset, pools, norm, config, device, phase, update):
    destroyer.eval()
    sampled, values, counts = [], [], dict(complete_evaluations=0, partial_objective_evaluations=0, repair_steps=0)
    started = time.perf_counter()
    with torch.no_grad():
        for spec in specs:
            source = pools[spec.instance_id][spec.source_index].solution()
            instance = dataset.instances[spec.instance_id]
            rows = []
            for _ in range(config.group_size):
                removed, _, _ = destroyer.select(instance, source, spec.preference, q_for(spec, instance), device)
                task = task_for(spec, dataset, pools, removed)
                result = repair_rollout(repairer, task, device, greedy=True)
                rows.append((removed, task, result))
                counts["complete_evaluations" if result.completed else "partial_objective_evaluations"] += 1
                counts["repair_steps"] += len(result.actions)
            rewards = rewards_for(rows[0][1], [r[2] for r in rows], norm)
            adv, active = advantages(rewards)
            sampled.append((spec, source, instance, rows, adv, active))
            values.extend(rewards.tolist())
    destroy_optimizer.zero_grad(set_to_none=True)
    # No stochastic layers: sampling and replay use the same policy distribution.
    destroyer.eval()
    for spec, source, instance, rows, adv, active in sampled:
        for (removed, _, _), advantage in zip(rows, adv):
            _, log_probability, entropy = destroyer.select(instance, source, spec.preference, len(removed), device, replay=removed)
            loss = (-float(advantage) * log_probability if active else log_probability * 0.)
            loss = (loss - config.entropy_weight * entropy) / (len(specs) * config.group_size)
            loss.backward()
    destroy_gradient = float(torch.nn.utils.clip_grad_norm_(destroyer.parameters(), config.gradient_clip))
    if not np.isfinite(destroy_gradient):
        raise ValueError("Nonfinite destroy gradient")
    destroy_optimizer.step()

    repair_gradient, bc_value = 0., None
    if phase == "joint":
        repair_optimizer.zero_grad(set_to_none=True)
        for spec in specs:
            source = pools[spec.instance_id][spec.source_index].solution()
            instance = dataset.instances[spec.instance_id]
            with torch.no_grad():
                removed, _, _ = destroyer.select(instance, source, spec.preference, q_for(spec, instance), device, greedy=True)
            task = task_for(spec, dataset, pools, removed)
            results = [repair_rollout(repairer, task, device, greedy=False) for _ in range(config.group_size)]
            adv, active = advantages(rewards_for(task, results, norm))
            for result, advantage in zip(results, adv):
                counts["complete_evaluations" if result.completed else "partial_objective_evaluations"] += 1
                counts["repair_steps"] += len(result.actions)
                if not active or not result.actions:
                    continue
                log_probability = repair_trajectory_logprob(repairer, task, result, device)
                loss = -float(advantage) * log_probability / (len(specs) * config.group_size)
                loss.backward()
        # Retain original behavior via old train-only teacher trajectories.
        instance, state, task = dataset.get(dataset.train[update % len(dataset.train)])
        repairer.set_source(instance, _solution(state["source_solution"]), device)
        bc_loss = teacher_bc_loss(repairer, instance, state, task, device)
        (config.bc_weight * bc_loss).backward()
        bc_value = float(bc_loss.detach().cpu())
        repair_gradient = float(torch.nn.utils.clip_grad_norm_(repairer.parameters(), config.gradient_clip))
        if not np.isfinite(repair_gradient):
            raise ValueError("Nonfinite repair gradient")
        repair_optimizer.step()
    return dict(mean_destroy_reward=float(np.mean(values)), active_groups=sum(row[-1] for row in sampled),
                destroy_gradient=destroy_gradient, repair_gradient=repair_gradient, bc_loss=bc_value,
                seconds=time.perf_counter() - started, **counts)


def rank(metrics):
    return metrics["failures"], -metrics["accepted_gain"], -metrics["raw_gain"]


def evaluate(destroyer, repairer, frozen, specs, dataset, pools, norm, device):
    """Paired single-candidate operator diagnosis, NOT multi-run EA/HV claims."""
    rows = []
    destroyer.eval()
    with torch.no_grad():
        for spec in specs:
            source = pools[spec.instance_id][spec.source_index].solution()
            instance = dataset.instances[spec.instance_id]
            learned, _, _ = destroyer.select(instance, source, spec.preference, q_for(spec, instance), device, greedy=True)
            for method, removed, model in (("hand_frozen", hand_removed(spec, dataset, pools), frozen),
                                           ("learned_frozen", learned, frozen), ("learned_current", learned, repairer)):
                task = task_for(spec, dataset, pools, removed)
                started = time.perf_counter()
                result = repair_rollout(model, task, device, greedy=True)
                reward = float(rewards_for(task, [result], norm)[0])
                rows.append(dict(method=method, instance_id=spec.instance_id, source_index=spec.source_index,
                                 q=len(removed), preference=list(spec.preference), removed=removed,
                                 feasible=bool(result.completed and result.feasible), raw_gain=reward,
                                 accepted_gain=max(0., reward) if result.completed and result.feasible else 0.,
                                 complete_evaluations=int(result.completed),
                                 partial_objective_evaluations=int(not result.completed),
                                 repair_seconds=time.perf_counter() - started, objectives=result.objectives))
    summaries = {}
    for method in ("hand_frozen", "learned_frozen", "learned_current"):
        selected = [r for r in rows if r["method"] == method]
        summaries[method] = dict(groups=len(selected), failures=sum(not r["feasible"] for r in selected),
                                accepted_gain=float(np.mean([r["accepted_gain"] for r in selected])),
                                raw_gain=float(np.mean([r["raw_gain"] for r in selected])),
                                improvement_rate=float(np.mean([r["accepted_gain"] > 1e-6 for r in selected])),
                                complete_evaluations=sum(r["complete_evaluations"] for r in selected))
    return summaries, rows


class Run:
    def __init__(self, run_name, config=Config(), device="cuda"):
        if not run_name or Path(run_name).name != run_name or run_name in (".", ".."):
            raise ValueError("RUN_NAME must be one directory name")
        self.config, self.device = config, device
        self.output = ROOT / "artifacts/cooperative_destroy_repair" / run_name
        self.output.mkdir(parents=True, exist_ok=True)
        self.dataset = Dataset()
        self.audit = audit_splits(self.dataset)
        self.pools = {split: build_solution_pools(self.dataset, split) for split in ("train", "dev")}
        self.norm = train_normalization(self.pools["train"])
        self.binding = dict(protocol="preference_cooperative_operator_v1", config=asdict(config), device=device,
                            repair_sha256=sha(ROOT / CHECKPOINT), dataset_sha256=sha(DATA / "pilot_manifest.json"),
                            code={p.name: sha(p) for p in Path(__file__).parent.glob("*.py")},
                            normalization=self.norm, audit=self.audit)
        path = self.output / "binding.json"
        # Same lock as train(), including the first metadata write. Different
        # configurations can never race on binding.json.tmp before training.
        with exclusive_run(self.output):
            if path.exists() and read_json(path) != self.binding:
                raise ValueError("Configuration/data/code changed; choose a new RUN_NAME. Old results are preserved.")
            if not path.exists():
                save_json(path, self.binding)
        self.dev_specs = make_specs(self.pools["dev"], seed=99171, epoch=3, total=config.dev_groups, shared_fraction=0.)

    def models(self):
        seed_all(self.config.seed)
        base = load_frozen_repairnet(self.device)
        base.chunk_size = 4096
        frozen = HeritageRepair(base).to(self.device).requires_grad_(False)
        repairer = HeritageRepair(copy.deepcopy(base)).to(self.device)
        destroyer = DestroyNet(base).to(self.device)
        return destroyer, repairer, frozen

    def optimizers(self, destroyer, repairer):
        cfg = self.config
        return (torch.optim.AdamW(destroyer.parameters(), lr=cfg.destroy_lr, weight_decay=cfg.weight_decay),
                torch.optim.AdamW([dict(params=repairer.base.parameters(), lr=cfg.repair_lr),
                                   dict(params=repairer.heritage.parameters(), lr=cfg.heritage_lr)], weight_decay=cfg.weight_decay))

    def save(self, name, destroyer, repairer, optimizers, progress):
        _atomic_checkpoint(self.output / name, dict(binding=self.binding, destroy=destroyer.state_dict(),
            repair=repairer.state_dict(), optimizers=[o.state_dict() for o in optimizers], progress=copy.deepcopy(progress),
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
            numpy_rng=np.random.get_state(), python_rng=random.getstate()))

    def restore(self, name, destroyer, repairer, optimizers, *, rng=True):
        payload = torch.load(self.output / name, map_location="cpu", weights_only=False)
        if payload["binding"] != self.binding:
            raise ValueError("Checkpoint binding mismatch")
        destroyer.load_state_dict(payload["destroy"])
        repairer.load_state_dict(payload["repair"])
        for optimizer, state in zip(optimizers, payload["optimizers"]):
            optimizer.load_state_dict(state)
        if rng:
            torch.set_rng_state(payload["torch_rng"])
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
            np.random.set_state(payload["numpy_rng"])
            random.setstate(payload["python_rng"])
        return payload["progress"]

    def smoke(self):
        d, r, frozen = self.models()
        opts = self.optimizers(d, r)
        specs = make_specs(self.pools["train"], seed=17, epoch=1, total=2, shared_fraction=0.)
        original = {k: v.clone() for k, v in r.state_dict().items()}
        r.requires_grad_(False)
        warm = train_update(d, r, *opts, specs, self.dataset, self.pools["train"], self.norm, self.config, self.device, "destroy", 0)
        if any(not torch.equal(v, r.state_dict()[k]) for k, v in original.items()):
            raise AssertionError("Frozen repair changed during destroy-only training")
        r.requires_grad_(True)
        joint = train_update(d, r, *opts, specs, self.dataset, self.pools["train"], self.norm, self.config, self.device, "joint", 1)
        if not any(not torch.equal(v, r.state_dict()[k]) for k, v in original.items()):
            raise AssertionError("Joint repair update did not change parameters")
        report = dict(status="passed", destroy_update=warm, joint_update=joint, audit=self.audit)
        save_json(self.output / "smoke.json", report)
        print(json.dumps(report, ensure_ascii=False), flush=True)

    def train(self):
        with exclusive_run(self.output):
            self._train()

    def _train(self):
        if not torch.cuda.is_available() and self.device.startswith("cuda"):
            raise RuntimeError("CUDA unavailable; no silent CPU training fallback")
        d, r, frozen = self.models()
        opts = self.optimizers(d, r)
        progress = dict(status="running", phase="destroy", epoch=0, cursor=0, update=0, stale=0,
                        best=None, train_complete_evaluations=0, train_partial_objective_evaluations=0)
        if (self.output / "last.pt").exists():
            progress = self.restore("last.pt", d, r, opts)
            if progress["status"] in ("complete", "warm_only_no_joint_gate", "early_stopped"):
                print("Already finished:", progress, flush=True)
                return
        else:
            metrics, rows = evaluate(d, r, frozen, self.dev_specs, self.dataset, self.pools["dev"], self.norm, self.device)
            save_json(self.output / "dev_initial.json", dict(metrics=metrics, rows=rows))
            progress["best"] = metrics["learned_current"]
            self.save("best_destroy.pt", d, r, opts, progress)
            self.save("best.pt", d, r, opts, progress)
            self.save("last.pt", d, r, opts, progress)
        progress["status"] = "running"
        invocation_start = time.perf_counter()
        r.requires_grad_(progress["phase"] == "joint")
        print(f"Training {progress['phase']}; output={self.output}; test split CLOSED", flush=True)
        while True:
            phase = progress["phase"]
            limit = self.config.destroy_epochs if phase == "destroy" else self.config.joint_epochs
            if progress["epoch"] >= limit or progress["stale"] >= self.config.patience:
                if phase == "joint":
                    progress["status"] = "complete" if progress["epoch"] >= limit else "early_stopped"
                    break
                # Joint training starts from the BEST development checkpoint, not the last epoch.
                best_progress = self.restore("best_destroy.pt", d, r, opts, rng=False)
                initial = read_json(self.output / "dev_initial.json")["metrics"]["hand_frozen"]
                warm = best_progress["best"]
                gate = (warm["failures"] <= initial["failures"] and warm["accepted_gain"] > initial["accepted_gain"] + 1e-6
                        and warm["raw_gain"] >= initial["raw_gain"])
                save_json(self.output / "joint_gate.json", dict(passed=gate, warm=warm, reference=initial))
                if not gate:
                    progress["status"] = "warm_only_no_joint_gate"
                    break
                progress.update(phase="joint", epoch=0, cursor=0, stale=0, best=warm)
                r.requires_grad_(True)
                self.save("last.pt", d, r, opts, progress)
                continue
            epoch = progress["epoch"] + 1
            specs = make_specs(self.pools["train"], seed=self.config.seed, epoch=epoch + (4 if phase == "joint" else 0),
                               total=self.config.groups_per_epoch, shared_fraction=.2)
            while progress["cursor"] < len(specs):
                if time.perf_counter() - invocation_start >= self.config.invocation_hours * 3600:
                    progress["status"] = "runtime_guard_resume_allowed"
                    self.save("last.pt", d, r, opts, progress)
                    save_json(self.output / "progress.json", progress)
                    print("Runtime guard: checkpoint saved; rerun same entry to resume", flush=True)
                    return
                stop = min(progress["cursor"] + self.config.groups_per_update, len(specs))
                metrics = train_update(d, r, *opts, specs[progress["cursor"]:stop], self.dataset,
                                       self.pools["train"], self.norm, self.config, self.device, phase, progress["update"])
                progress["cursor"], progress["update"] = stop, progress["update"] + 1
                progress["train_complete_evaluations"] += metrics["complete_evaluations"]
                progress["train_partial_objective_evaluations"] += metrics["partial_objective_evaluations"]
                progress["last_update_seconds"] = metrics["seconds"]
                self.save("last.pt", d, r, opts, progress)
                save_json(self.output / "progress.json", progress)
                with (self.output / "updates.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(dict(phase=phase, epoch=epoch, update=progress["update"], cursor=stop, **metrics), allow_nan=False) + "\n")
                print(f"{phase} epoch={epoch}/{limit} groups={stop}/{len(specs)} reward={metrics['mean_destroy_reward']:.5f} "
                      f"update={metrics['seconds']:.1f}s fullFE={progress['train_complete_evaluations']}", flush=True)
            metrics, rows = evaluate(d, r, frozen, self.dev_specs, self.dataset, self.pools["dev"], self.norm, self.device)
            progress["epoch"], progress["cursor"] = epoch, 0
            current = metrics["learned_current"]
            improved = rank(current) < rank(progress["best"])
            progress["stale"] = 0 if improved else progress["stale"] + 1
            if improved:
                progress["best"] = current
                self.save("best.pt", d, r, opts, progress)
                if phase == "destroy":
                    self.save("best_destroy.pt", d, r, opts, progress)
            save_json(self.output / f"dev_{phase}_{epoch:02d}.json", dict(metrics=metrics, rows=rows, selected=improved))
            self.save("last.pt", d, r, opts, progress)
            save_json(self.output / "progress.json", progress)
            print("Development operator diagnostic:", metrics, flush=True)
        self.save("last.pt", d, r, opts, progress)
        save_json(self.output / "progress.json", progress)
        print("Training ended:", progress, flush=True)

    def evaluate_best(self):
        d, r, frozen = self.models()
        self.restore("best.pt", d, r, self.optimizers(d, r), rng=False)
        metrics, rows = evaluate(d, r, frozen, self.dev_specs, self.dataset, self.pools["dev"], self.norm, self.device)
        save_json(self.output / "dev_best.json", dict(metrics=metrics, rows=rows))
        print(json.dumps(metrics, indent=2), flush=True)
