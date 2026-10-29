"""One aligned DPR episode; no optimizer and no implicit greedy decoding."""
import time

import numpy as np
import torch

from .environment import Environment
from .execution import (contract, episode_terminal, isolated_policy_rng, observation,
                         sample_action, sample_repair)


ARMS = ("learned", "random_dpr", "fixed_q_dpr", "related_fixed_q")


def baseline_action(env, arm, rng):
    q = int(rng.choice(env.q_values)) if arm == "random_dpr" else env.q_values[len(env.q_values)//2]
    anchors = [int(i) for i in rng.choice(np.arange(1, env.instance.n_customers+1),
                                         size=min(env.cfg.anchors, q), replace=False)]
    return dict(q=q, anchors=anchors, coefficients=[float(rng.beta(2., 2.)) for _ in anchors])


def archive_hv(env):
    import moocore
    values = np.asarray([p["objectives"] for p in env.archive], dtype=float)
    return float(moocore.hypervolume((values-env.offset)/env.scales,
                                    ref=np.asarray(env.box["reference"])))


def evaluate_episode(pool, iid, cfg, policy, arm, preference, seed, seconds,
                     max_actions=None):
    if arm not in ARMS or seconds <= 0:
        raise ValueError("Unknown evaluation arm or nonpositive time guard")
    target = cfg.episode_steps if max_actions is None else int(max_actions)
    if not 1 <= target <= cfg.episode_steps:
        raise ValueError("A single evaluation episode cannot exceed the training horizon")
    started = time.perf_counter()
    deadline = started+seconds
    inst, row, box, qs = pool.load(iid)
    env = Environment(pool.api, inst, row, box, qs, preference, cfg)
    initial_score = env.score(env.objectives)
    initial_objectives = env.objectives.tolist()
    hidden = torch.zeros(1, cfg.hidden_dim)
    # Separate exogenous repair RNG and baseline action RNG. Every comparison
    # arm sees the same repair order AND repair seed at the same action index.
    repair_rng = np.random.default_rng(seed)
    action_rng = np.random.default_rng(seed+100000)
    feature_seconds = inference_seconds = repair_seconds = 0.
    records = []
    with isolated_policy_rng(seed, policy.device):
        while len(records) < target and time.perf_counter() < deadline:
            repair = sample_repair(repair_rng)
            repair_seed = int(repair_rng.integers(0, 2**31-1))
            override = None
            before = time.perf_counter()
            remaining = 1-env.steps/cfg.episode_steps
            if arm == "learned":
                obs = observation(env, hidden, repair)
                feature_seconds += time.perf_counter()-before
                before_model = time.perf_counter()
                action, _, _, hidden = sample_action(policy, obs)
                inference_seconds += time.perf_counter()-before_model
            else:
                action = baseline_action(env, arm, action_rng)
                if arm == "related_fixed_q":
                    from hemt_feature_stage_v3.trajectory import deletion
                    override = deletion(env.routes, action["q"], "related", env.instance, seed+len(records))["removed_customers"]
            if time.perf_counter() >= deadline:
                break
            record = env.step(action, repair, repair_seed, deadline, removal_override=override)
            record.update(action_number=len(records)+1, remaining_actions_fraction=remaining)
            records.append(record)
            repair_seconds += record["repair_seconds"]
    actual_seconds = time.perf_counter()-started
    audit_start = time.perf_counter()
    from hemt_destroy_feature_labels_v1.audit import checked_objectives
    for p in env.archive:
        checked_objectives(p["routes"], env.instance, claimed=p["objectives"])
    terminal = episode_terminal(env)
    completed = len(records) == target
    result = dict(
        execution_contract=contract(cfg), iid=iid, arm=arm, preference=list(preference), search_seed=seed,
        search_budget_seconds=seconds, actual_search_seconds=actual_seconds, target_actions=target,
        steps=len(records), successes=sum(r["feasible"] and not r["timeout"] for r in records),
        episode_terminal=terminal, truncated=not terminal,
        stop_reason="episode_horizon" if terminal else ("diagnostic_action_cap" if completed else "time_guard"),
        requested_actions_completed=completed,
        initial_objectives=initial_objectives, final_objectives=env.objectives.tolist(),
        initial_score=initial_score, final_score=env.score(env.objectives),
        scalar_improvement=initial_score-env.score(env.objectives),
        final_hv=archive_hv(env), archive=env.archive, records=records,
        feature_seconds=feature_seconds, inference_seconds=inference_seconds, repair_seconds=repair_seconds,
    )
    result["offline_audit_hv_seconds"] = time.perf_counter()-audit_start
    return result
