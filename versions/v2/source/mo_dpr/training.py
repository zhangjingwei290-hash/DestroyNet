"""The single collection/update loop used by PyCharm, CLI and campaign.

Four complete episodes under frozen weights, then whole-episode minibatches.
Stopped partial rollouts are saved (with bootstrap), never counted as optimized.
"""
import copy
import time

import numpy as np
import torch

from .data import preference_for_episode
from .environment import Environment, REPAIRERS
from .execution import episode_terminal, observation, sample_action, sample_repair


def make_environment(pool, iid, cfg, preference):
    instance, row, box, qs = pool.load(iid)
    return Environment(pool.api, instance, row, box, qs, preference, cfg)


def export_environment(env):
    return {"iid": env.instance.instance_id, "preference": env.preference.tolist(), "routes": env.routes,
            "objectives": env.objectives.tolist(), "archive": env.archive, "steps": env.steps,
            "stagnation": env.stagnation, "last_accepted": env.last_accepted,
            "last_reward": env.last_reward, "last_removed": env.last_removed}


def restore_environment(pool, cfg, saved):
    env = make_environment(pool, saved["iid"], cfg, saved["preference"])
    for name in ("routes", "archive", "steps", "stagnation", "last_accepted", "last_reward", "last_removed"):
        setattr(env, name, copy.deepcopy(saved[name]))
    env.objectives = env.evaluate(env.routes)
    np.testing.assert_allclose(env.objectives, saved["objectives"], atol=1e-7, rtol=0)
    return env


class TrainingSession:
    def __init__(self, cfg, pool, policy, trainer, ids, saved=None):
        self.cfg, self.pool, self.policy, self.trainer, self.ids = cfg, pool, policy, trainer, tuple(ids)
        self.rng = np.random.default_rng(cfg.seed)
        self.actions = self.optimized_actions = self.episodes = self.completed_episodes = self.updates = 0
        self.optimizer_steps = self.successes = 0
        self.elapsed_seconds = 0.
        self.optimization_time_guard = False
        self.env = None
        self.hidden = torch.zeros(1, cfg.hidden_dim)
        self.pending_repair = None
        self.pending_seed = None
        self.episodes_buffer, self.current_episode = [], []
        if saved is not None:
            for name in ("actions", "optimized_actions", "episodes", "completed_episodes", "updates",
                         "optimizer_steps", "successes", "elapsed_seconds", "hidden", "pending_repair",
                         "pending_seed", "episodes_buffer", "current_episode"):
                setattr(self, name, saved[name])
            self.rng.bit_generator.state = saved["numpy_rng"]
            torch.set_rng_state(saved["torch_rng"])
            if "cuda_rng" in saved:
                torch.cuda.set_rng_state_all(saved["cuda_rng"])
            if saved["environment"] is not None:
                self.env = restore_environment(pool, cfg, saved["environment"])

    def state(self):
        obj = {name:getattr(self,name) for name in (
            "actions", "optimized_actions", "episodes", "completed_episodes", "updates", "optimizer_steps",
            "successes", "elapsed_seconds", "hidden", "pending_repair", "pending_seed",
            "episodes_buffer", "current_episode")}
        obj.update(numpy_rng=self.rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                   environment=None if self.env is None else export_environment(self.env))
        if self.cfg.device.startswith("cuda"):
            obj["cuda_rng"] = torch.cuda.get_rng_state_all()
        return obj

    def progress(self):
        return dict(actions=self.actions, optimized_actions=self.optimized_actions,
            pending_actions=self.actions-self.optimized_actions, episodes=self.episodes,
            completed_episodes=self.completed_episodes, updates=self.updates, optimizer_steps=self.optimizer_steps,
            successful_repairs=self.successes, elapsed_seconds=self.elapsed_seconds,
            partial_episode_steps=len(self.current_episode), pending_complete_episodes=len(self.episodes_buffer))

    def run(self, on_record=lambda r: None, on_update=lambda m: None, on_snapshot=lambda s: None,
            final_reserve_seconds=0.):
        cfg = self.cfg
        smoke = cfg.mode == "smoke"
        limit = cfg.smoke_steps if smoke else cfg.train_steps
        time_limit = cfg.smoke_seconds if smoke else cfg.train_seconds
        start, consumed = time.perf_counter(), self.elapsed_seconds
        deadline = start + max(0., time_limit-consumed-final_reserve_seconds)

        def tick():
            self.elapsed_seconds = consumed + time.perf_counter()-start

        def update(episodes):
            # A snapshot before optimization can recover frozen weights + buffer
            # after a failed partial optimizer pass. No sampling during update.
            tick()
            on_snapshot(self)
            before = time.perf_counter()
            old_weights = copy.deepcopy(self.policy.state_dict())
            old_optimizer = copy.deepcopy(self.trainer.optimizer.state_dict())
            try:
                metrics = self.trainer.update_episodes(episodes, deadline=deadline)
            except TimeoutError:
                self.policy.load_state_dict(old_weights)
                self.trainer.optimizer.load_state_dict(old_optimizer)
                self.trainer.optimizer.zero_grad()
                self.optimization_time_guard = True
                tick()
                return False
            self.optimized_actions += sum(len(ep) for ep in episodes)
            self.updates += 1
            self.optimizer_steps += metrics["optimizer_steps"]
            self.episodes_buffer, self.current_episode = [], []
            tick()
            metrics.update(actions=self.actions, optimized_actions=self.optimized_actions,
                           update_seconds=time.perf_counter()-before)
            on_snapshot(self)
            on_update(metrics)  # validation/checkpoint only at a completed rollout
            tick()
            return True

        while time.perf_counter() < deadline:
            if len(self.episodes_buffer) == cfg.rollout_episodes:
                if not update(self.episodes_buffer):
                    break
                continue
            if self.actions >= limit:
                break
            if self.env is None or episode_terminal(self.env):
                # Fixed manifest order cycles across the nine strata. No entry-
                # specific random instance selection or preference distribution.
                iid = self.ids[self.episodes % len(self.ids)]
                pref = preference_for_episode(cfg, self.episodes, self.rng)
                self.env = make_environment(self.pool, iid, cfg, pref)
                self.hidden = torch.zeros(1, cfg.hidden_dim)
                self.pending_repair = self.pending_seed = None
                self.episodes += 1
            before = time.perf_counter()
            if self.pending_repair is None:
                self.pending_repair = REPAIRERS[self.actions % 4] if smoke else sample_repair(self.rng)
                self.pending_seed = int(self.rng.integers(0, 2**31-1))
            obs = observation(self.env, self.hidden, self.pending_repair)
            features = time.perf_counter()-before
            t_model = time.perf_counter()
            action, lp, value, h = sample_action(self.policy, obs)
            inference = time.perf_counter()-t_model
            if time.perf_counter() >= deadline:
                break
            rec = self.env.step(action, self.pending_repair, self.pending_seed, deadline)
            self.hidden = h
            terminal = episode_terminal(self.env)
            self.pending_repair = self.pending_seed = None
            next_value = 0.
            if not terminal:
                self.pending_repair = REPAIRERS[(self.actions+1) % 4] if smoke else sample_repair(self.rng)
                self.pending_seed = int(self.rng.integers(0, 2**31-1))
                with torch.no_grad():
                    next_value = float(self.policy.encode(observation(self.env, h, self.pending_repair))[-1])
            self.current_episode.append(dict(observation=obs, action=action, logprob=lp, value=value,
                reward=rec["reward"], terminal=terminal, next_value=next_value,
                diagnostics=dict(q=action["q"],repair=rec.get("repair"),
                    feasible=rec["feasible"],timeout=rec["timeout"],accepted=rec.get("accepted"),
                    feature_seconds=features,inference_seconds=inference,
                    repair_seconds=rec.get("repair_seconds"),collection_seconds=time.perf_counter()-before)))
            self.actions += 1
            self.successes += int(rec["feasible"] and not rec["timeout"])
            rec.update(action_number=self.actions, step=self.actions, episode=self.episodes,
                episode_terminal=terminal, remaining_actions_fraction=1-(self.env.steps-1)/cfg.episode_steps,
                feature_seconds=features, inference_seconds=inference,
                collection_seconds=time.perf_counter()-before, old_logprob=lp, old_value=value)
            on_record(rec)
            if terminal:
                self.completed_episodes += 1
                self.episodes_buffer.append(self.current_episode)
                self.current_episode = []
            tick()
            if self.actions % 1000 == 0:
                on_snapshot(self)  # includes pending rollout; no discarded tail
        tick()
        if smoke and self.actions and time.perf_counter() < deadline:
            # Explicit eight-action diagnostic, not a complete-rollout training
            # result. Bootstrap at this artificial cut; do not change horizon.
            diagnostic = self.episodes_buffer + ([self.current_episode] if self.current_episode else [])
            if diagnostic:
                update(diagnostic)
        tick()
        on_snapshot(self)
        return dict(**self.progress(), stop_reason="action_budget" if self.actions >= limit else "time_guard",
                    partial_rollout_saved=self.actions > self.optimized_actions,
                    diagnostic_partial_update=smoke, optimization_time_guard=self.optimization_time_guard,
                    last_episode_terminal=False if self.env is None else episode_terminal(self.env),
                    last_episode_steps=0 if self.env is None else self.env.steps,
                    last_episode_result=None if self.env is None else export_environment(self.env),
                    quality_validated=False)
