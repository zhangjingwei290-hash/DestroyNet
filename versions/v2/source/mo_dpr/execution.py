"""Shared train/inference/validation contract, authorized 2026-10-16.

Search horizon is measured in actions as in upstream DPR. A wall-time guard can
truncate an episode but must not change the meaning of the progress input.
Diagnostics may explicitly compare alternative decoding outside this contract.
"""
from contextlib import contextmanager

import torch

from .environment import REPAIRERS


VERSION = "dpr_aligned_sampling_steps_v1"


def contract(cfg):
    return {
        "version": VERSION,
        "decoding": "stochastic_categorical_and_beta",
        "progress": "remaining_search_actions / episode_actions",
        "episode_actions": cfg.episode_steps,
        "initial_hidden": "zeros",
        "initial_previous_action_features": "zeros",
        "history_update": "previous_hidden_and_accepted_removed_customer_features",
        "repair_selection": "external_seeded_uniform_with_identity_in_observation",
        "wall_time": "external_truncation_guard_not_progress_or_natural_terminal",
        "schema": "mo_dpr_v1; same_node_and_context_order; changed_progress_semantics_in_execution",
    }


def remaining_actions_fraction(steps, horizon):
    if horizon < 1 or not 0 <= steps <= horizon:
        raise ValueError("Search steps must stay within the declared episode horizon")
    return 1. - steps/horizon


def observation(env, hidden, repair):
    return env.observation(hidden, repair,
                           remaining_actions_fraction(env.steps, env.cfg.episode_steps))


def sample_action(policy, obs):
    # Explicit rather than relying on a default that could diverge across callers.
    return policy.act(obs, greedy=False)


def sample_repair(rng):
    return str(rng.choice(REPAIRERS))


def episode_terminal(env):
    return env.steps >= env.cfg.episode_steps


@contextmanager
def isolated_policy_rng(seed, device):
    """Evaluation is reproducible and never consumes the training policy RNG."""
    device = torch.device(device)
    devices = [] if device.type != "cuda" else [device.index if device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        yield
