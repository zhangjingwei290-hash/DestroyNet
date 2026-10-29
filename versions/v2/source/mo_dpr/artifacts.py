"""Versioned checkpoint/cache identities, computed at artifact boundaries only."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from functools import lru_cache

import torch

from .data import fingerprint
from .environment import NODE_FEATURES, CONTEXT_FEATURES, OBJECTIVES, REPAIRERS
from .execution import contract


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name+".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, allow_nan=False, indent=2)
        replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def replace(source, target):
    for attempt in range(10):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(.05*(attempt+1))


def save_torch(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name+".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, temporary)
        replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def file_id(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024*1024), b""):
            h.update(part)
    return h.hexdigest()


def model_id(weights):
    h = hashlib.sha256()
    for key, value in sorted(weights.items()):
        v = value.detach().cpu().contiguous()
        h.update(key.encode())
        h.update(str((v.dtype, tuple(v.shape))).encode())
        h.update(v.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def runtime_contract(cfg):
    return dict(version="dpr_runtime_v2", execution=contract(cfg),
        architecture={k:getattr(cfg,k) for k in ("hidden_dim", "critic_dim", "anchors", "near_neighbors")},
        node_features=list(NODE_FEATURES), context_features=list(CONTEXT_FEATURES),
        feature_units_normalization="mo_dpr_environment_v1_physical_affine_fixed_input_box",
        objectives=list(OBJECTIVES), constraints="exactly_once_capacity_max_delay_depot_close_v1",
        q_support="frozen_input_q_levels; cap=min(15,floor(.15N)); five_levels_max",
        fixed_q=cfg.fixed_q, repairers=list(REPAIRERS), repair_seconds=cfg.repair_seconds,
        reward="before_minus_candidate_augmented_tchebycheff_rho=.05_fixed_physical_box",
        failure_reward=cfg.failure_reward, acceptance="nonworsening_preference_scalar",
        archive="raw_exact_five_objective_pareto_union", recurrence="stored_hidden_one_step_truncated",
        hidden_reset="zeros_at_each_episode; no_weight_updates_during_collection",
        semantic_sources=semantic_sources(cfg.project_root))


@lru_cache(maxsize=4)
def semantic_sources(project_root):
    repo = Path(__file__).resolve().parents[1]
    files = [repo/p for p in ("mo_dpr/environment.py", "mo_dpr/execution.py", "mo_dpr/model.py",
                              "lib/ppo_1.py", "lib/vrp_helper_env.py")]
    files += [Path(project_root)/p for p in ("src/evaluation/vrptw_eval.py", "src/pilot/engine.py",
                                           "hemt_destroy_feature_labels_v1/repair.py")]
    return {str(p):file_id(p) for p in files}


def training_contract(cfg, split, selected_ids):
    diagnostic=cfg.mode=="smoke"
    collection=dict(contract(cfg))
    if diagnostic:
        collection["repair_selection"]="round_robin_four_repairers_diagnostic_only"
    return dict(version="dpr_smoke_diagnostic_v2" if diagnostic else "dpr_full_trajectory_ppo_v2",
        runtime=runtime_contract(cfg),collection_contract=collection,
        split_manifest=split, selected_fit_ids=list(selected_ids),
        episode_order="fixed_manifest_round_robin", preference="fixed_override_else_uniform_dirichlet_alternation",
        fixed_preference=None if cfg.preference is None else list(cfg.preference),
        rollout_episodes=cfg.rollout_episodes, rollout_transitions=cfg.episode_steps*cfg.rollout_episodes,
        minibatch="one_episode_or_explicit_smoke_fragment_chronological_order",
        tail="explicit_diagnostic_update_if_budget" if diagnostic else "saved_unoptimized_with_bootstrap",
        optimizer={k:getattr(cfg,k) for k in ("learning_rate", "gamma", "clip_epsilon", "ppo_epochs", "entropy_weight")},
        value_coefficient=.5, gradient_clip_norm=1., advantage="standardize_full_rollout_no_GAE",
        approximation="PPO replay uses stored behavior hidden; not full BPTT")


def checkpoint_payload(policy, trainer, cfg, split, selected_ids, **extra):
    weights = policy.state_dict()
    return dict(schema="mo_dpr_checkpoint_v2", runtime_contract=runtime_contract(cfg),
        training_contract=training_contract(cfg,split,selected_ids), model_id=model_id(weights),
        model=weights, optimizer=trainer.optimizer.state_dict(), **extra)


def load_checkpoint(path, cfg, policy, sidecar=None):
    payload = torch.load(path, map_location=cfg.device, weights_only=True)
    if payload.get("schema") == "mo_dpr_checkpoint_v2":
        declared = payload["runtime_contract"]
        if payload["model_id"] != model_id(payload["model"]):
            raise ValueError("Checkpoint model content ID mismatch")
    else:
        if not sidecar:
            raise ValueError("Legacy checkpoint needs an explicitly verified checkpoint_manifest sidecar")
        info = json.loads(Path(sidecar).read_text(encoding="utf-8"))
        if info.get("version") != "dpr_verified_legacy_v2" or info["checkpoint_sha256"] != file_id(path):
            raise ValueError("Legacy manifest/checkpoint binding mismatch")
        if info["plan_sha256"] != payload.get("plan_sha256"):
            raise ValueError("Legacy manifest/plan mismatch")
        declared = info["runtime_contract"]
    if declared != runtime_contract(cfg):
        changed = [k for k in set(declared) | set(runtime_contract(cfg))
                   if declared.get(k) != runtime_contract(cfg).get(k)]
        raise ValueError(f"Checkpoint execution contract mismatch: {changed}")
    policy.load_state_dict(payload["model"])
    return payload
