"""Versioned scientific contract for the GPU EMNH-to-DPR migration."""
from dataclasses import dataclass, asdict
from itertools import combinations
from pathlib import Path
import math
import numpy as np

RATIOS = (.01, .02, .05, .10, .20, .40, .60, .80, 1.)
REPAIRERS = ("greedy", "regret2", "regret3", "noisy_greedy")
CONTEXT = tuple([f"weight_{i}" for i in range(5)] + [f"objective_{i}" for i in range(5)]
    + ["global_remaining_seconds/300", "global_used_fraction", "local_remaining_seconds/300",
       "local_remaining_fraction", "stagnation_fraction", "last_accepted", "last_reward",
       "archive_fraction", "max_delay/horizon", "N/250", "max_q/N"]
    + [f"repair_is_{r}" for r in REPAIRERS]
    + [f"repair_{metric}_{r}" for metric in ("probability", "recent_seconds/300", "recent_success_rate",
                                            "recent_improvement", "count/1000") for r in REPAIRERS])


def directions():
    result = list(np.eye(5))
    for i, j in combinations(range(5), 2):
        v = np.zeros(5); v[[i, j]] = .5; result.append(v)
    return np.asarray(result + [np.full(5, .2)])


def q_values(n):
    if n < 1:
        raise ValueError("N must be positive")
    return tuple(sorted({max(1, min(n, math.ceil(n*r))) for r in RATIOS}))


@dataclass
class Config:
    version: str = "moead_emnh_dpr_gpu_v2_closed_loop"
    project_root: str = r"D:\WorkSpace\versions\version0.7"
    study_root: str = r"D:\WorkSpace\versions\version0.7\artifacts\hemt_feature_stage_v3_20261013"
    split_manifest: str = str(Path(__file__).resolve().parents[1]/"mo_dpr/train_development_v2.json")
    device: str = "cuda"
    seed: int = 20261020
    hidden_dim: int = 128
    critic_dim: int = 256
    anchors: int = 2
    near_neighbors: int = 10
    fixed_q: int | None = None
    episode_steps: int = 150  # inherited feature helper only; new progress is WALL TIME
    train_instance_ids: tuple = ()
    learning_rate: float = 1e-4
    entropy_weight: float = .001
    clip_epsilon: float = .2
    ppo_epochs: int = 2
    gamma: float = 1.
    task_count: int = 5
    replicas: int = 4
    inner_updates: int = 4
    microbatch: int = 8
    meta_seconds: float = 10800.
    adaptation_seconds: float = 3600.
    repair_seconds: float = 10.
    local_seconds: float = 10.
    cpu_threads: int = 1
    binding_id: str | None = None
    legacy_ppo_seconds: float = 335.0391342

    def validate(self):
        if self.device != "cuda":
            raise ValueError("This new experiment requires CUDA; no silent CPU fallback")
        if self.task_count != 5 or self.gamma != 1 or self.anchors != 2 or self.fixed_q is not None:
            raise ValueError("Scientific contract changed")
        if not 0 < self.meta_seconds <= 10800 or not 0 <= self.adaptation_seconds <= 3600:
            raise ValueError("First-run cap: 3h meta + 1h adaptation per method/seed")
        if not 0 <= self.legacy_ppo_seconds <= self.meta_seconds:
            raise ValueError("Invalid prior budget debit")
        if self.replicas < 2 or min(self.inner_updates, self.microbatch, self.cpu_threads) < 1:
            raise ValueError("Invalid batch dimensions")
        if min(self.local_seconds, self.repair_seconds, self.learning_rate) <= 0:
            raise ValueError("Invalid time/rate")

    def to_dict(self):
        return asdict(self)


def search_budget(n):
    return {50: 60., 150: 180., 250: 300.}[int(n)]


def runtime_contract(cfg):
    return dict(training_seed=cfg.seed, cpu_threads=cfg.cpu_threads, training_caps=dict(meta_seconds=cfg.meta_seconds, adaptation_seconds=cfg.adaptation_seconds,
            prior_ppo_debit=cfg.legacy_ppo_seconds),
        training_batch=dict(tasks=cfg.task_count, replicas=cfg.replicas, inner_updates=cfg.inner_updates,
                            microbatch=cfg.microbatch, learning_rate=cfg.learning_rate,
                            entropy=cfg.entropy_weight, clip=cfg.clip_epsilon, ppo_epochs=cfg.ppo_epochs),
        version=cfg.version, context=list(CONTEXT), ratios=list(RATIOS),
        node_schema="mo_dpr.environment.NODE_FEATURES_17_physical",
        decode="stochastic_q_then_distinct_anchors_then_Beta; same_in_all_entrypoints",
        hidden="zero_each_new_local_search; own_history; stored_one_step_detached_replay",
        objectives="five_raw_physical; fixed_affine_augmented_Tchebycheff_rho=.05",
        repairers=list(REPAIRERS), repair_selection="alns.RouletteWheel; per_actual_direction_persistent; EWMA_rate=.2_first_sample_exact",
        state_source="neural_MOEAD_full_feedback; auxiliary_same_start_replicas_do_not_drive_population",
        meta_tasks="five_continuous_cyclic_preferences_replace_five_axis_slots; eleven_background_directions",
        reward="accepted_before_minus_after; failures/rejections/timeouts_zero; objectives_null_on_failure",
        acceptance="strict_scalar_improvement; equal_scores_rejected",
        timing="global_solver_walltime; local=min(10s,global_budget/16,remaining); gamma=1; late_candidates_excluded",
        termination="local_budget_is_terminal; external_resource_cutoff_is_truncation",
        population_directions=directions().tolist(), repair_seconds=cfg.repair_seconds,
        local_seconds=cfg.local_seconds, architecture=dict(hidden=cfg.hidden_dim, critic=cfg.critic_dim,
        anchors=cfg.anchors, near=cfg.near_neighbors), archive="exact_5D_nondominated_union")
