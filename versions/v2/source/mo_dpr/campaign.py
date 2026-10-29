"""Thin campaign wrapper for shared v2 training. No automatic stage3.

One worker, CPU one thread. Raw objectives and the declared repair/score/cap do
not change. Validation is a development holdout inside TRAIN54, not fresh test.
"""
from collections import Counter, defaultdict
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback

import numpy as np
import psutil
import torch

from .config import Config
from .environment import Environment, TrainPool, REPAIRERS, OBJECTIVES, NODE_FEATURES, CONTEXT_FEATURES
from .model import JointPolicy, PPOTrainer, SCHEMA
from .execution import (observation as aligned_observation, sample_action,
                        sample_repair, episode_terminal)


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name+".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, allow_nan=False, indent=2)
        for attempt in range(10):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(.05*(attempt+1))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def append(path, value):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+"\n")


def atomic_checkpoint(path, value):
    temporary = path.with_name(path.name+f".{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def split_ids(pool):
    cells = defaultdict(list)
    for iid in pool.ids:
        # The input IID schema is fixed; only size/distribution are grouping keys.
        tokens = iid.split("_")
        cells[(int(tokens[2]), tokens[3])].append(iid)
    if len(cells) != 9:
        raise ValueError("Expected 3 scales x 3 distributions in TRAIN54")
    fit, validation, benchmark = [], [], []
    for _, ids in sorted(cells.items()):
        ids = sorted(ids)
        validation.append(ids[-1])
        fit.extend(ids[:-1])
        benchmark.append(ids[0])
    if set(fit) & set(validation):
        raise AssertionError("Development split overlaps")
    return fit, validation, benchmark


def make_environment(pool, iid, cfg, preference):
    instance, row, box, qs = pool.load(iid)
    return Environment(pool.api, instance, row, box, qs, preference, cfg)


def stamp(root, stage, **kwargs):
    proc = psutil.Process()
    save(root/"health.json", dict(stage=stage, updated_at=time.time(), pid=proc.pid,
                                  created=proc.create_time(), **kwargs))


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def evaluate_one(pool, iid, cfg, policy, arm, preference, seed, seconds):
    from .evaluation import evaluate_episode
    return evaluate_episode(pool, iid, cfg, policy, arm, preference, seed, seconds)


def aggregate_validation(rows):
    """Legacy helper: guarded single-episode metrics; new reports label scope."""
    from .validation import aggregate
    return aggregate(rows, full_episodes=False)


def validate(root, pool, plan, cfg, policy, actions, checkpoint_path, budget_seconds=None):
    from .validation import validate as shared_validate
    return shared_validate(root, pool, plan, cfg, policy, actions, checkpoint_path, budget_seconds)


# Shared serialization: campaign has no independent environment/training path.
from .training import export_environment, restore_environment


def pilot(root, pool, plan, cfg):
    """Campaign owns configuration; collection/optimization belongs to runner."""
    from .runner import run
    if plan.get("version") != "dpr_stage12_v2":
        raise ValueError("Legacy stage12 plans cannot silently resume under the new protocol")
    expected_fit, expected_dev, _ = split_ids(pool)
    if plan["fit_ids"] != expected_fit or plan["validation_ids"] != expected_dev:
        raise ValueError("Campaign must use the frozen 45/9 manifest")
    return run(cfg, output_dir=root, validation_plan=plan)


def main(root_path, quick=False, train_steps=None, train_seconds=None):
    """Explicit new budgets required; never reopen the completed v1 study."""
    from .runner import run
    root=Path(root_path).resolve()
    if (root/"complete.json").exists():
        print("Campaign already complete; no restart.",flush=True)
        return root
    if (root/"plan.json").exists():
        plan=json.loads((root/"plan.json").read_text(encoding="utf-8"))
        if plan.get("version")!="dpr_stage12_v2":
            raise ValueError("Old study is immutable; new version requires a new output directory")
        cfg=Config(**plan["config"])
        if train_steps is not None and train_steps!=cfg.train_steps or train_seconds is not None and train_seconds!=cfg.train_seconds:
            raise ValueError("Cannot extend/change a saved campaign budget")
    else:
        if quick:
            cfg=Config(mode="smoke",smoke_steps=8,smoke_seconds=120)
            return run(cfg,output_dir=root)
        cfg=Config(mode="train",train_steps=train_steps,train_seconds=train_seconds)
        cfg.validate()  # no implicit 100k/12h run
        pool=TrainPool(cfg)
        fit,dev,benchmark_ids=split_ids(pool)
        plan=dict(version="dpr_stage12_v2",config=cfg.to_dict(),fit_ids=fit,validation_ids=dev,
            benchmark_ids=benchmark_ids,validate_every=25000,validation_seconds=min(7200.,cfg.train_seconds),
            validation_search_seconds=10.,validation_seeds=[73,109],
            validation_preferences=[[.2]*5,[.6,.1,.1,.1,.1]],
            scope="approved alignment implementation; budgets explicitly supplied at invocation",
            validation_boundary="first completed 600-transition rollout at/after threshold")
        save(root/"plan.json",plan)
    root.mkdir(parents=True,exist_ok=True)
    proc=psutil.Process()
    lock=root/"worker.lock"
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,str(proc.pid).encode())
    os.close(fd)
    save(root/"process.json",dict(pid=proc.pid,created=proc.create_time(),command=proc.cmdline(),
        device="cpu",threads=1,at=time.time()))
    try:
        return pilot(root,TrainPool(cfg),plan,cfg)
    finally:
        lock.unlink(missing_ok=True)
