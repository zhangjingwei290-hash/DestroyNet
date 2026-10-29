"""Extend the upstream HRGCN actor with context and a conditional q head.

The spatial/forward/reverse message passing, recurrent memory, anchor head and
Beta heads come from lib.ppo_1.Net_Actor. New decisions use tensor log-probs all
the way through PPO. No solver evaluation is performed by this module.
"""
from types import SimpleNamespace

import torch
from torch import nn
from torch.distributions import Beta, Categorical
from torch.nn import functional as F

from lib.ppo_1 import Net_Actor, Net_Generic
from .environment import CONTEXT_FEATURES, NODE_FEATURES


SCHEMA = "mo_dpr_v1"


class JointPolicy(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = Net_Actor(cfg.learning_rate, len(NODE_FEATURES), cfg.hidden_dim)
        self.context_encoder = nn.Sequential(nn.Linear(len(CONTEXT_FEATURES), cfg.hidden_dim),
                                              nn.Tanh())
        self.q_head = nn.Sequential(nn.Linear(2 * cfg.hidden_dim, cfg.hidden_dim), nn.Tanh(),
                                     nn.Linear(cfg.hidden_dim, 5))
        self.location_condition = nn.Linear(cfg.hidden_dim + 2, cfg.hidden_dim)
        self.critic = Net_Generic(cfg.learning_rate, 2 * cfg.hidden_dim, cfg.critic_dim, 1)

    @property
    def device(self):
        return next(self.parameters()).device

    def encode(self, observation):
        fields = {name: observation[name].to(self.device) for name in (
            "x", "edge_index_n", "edge_index_r0", "edge_index_r1", "state_", "input_")}
        h, nodes = self.encoder.encode(SimpleNamespace(**fields), observation["x"].shape[0])
        nodes = nodes[0, 1:]  # exclude depot from both action heads
        context = self.context_encoder(observation["context"].to(self.device))
        global_state = torch.cat([nodes.mean(dim=0), context])
        count = len(observation["q_values"])
        if not 1 <= count <= 5:
            raise ValueError("Expected one to five legal q values")
        q_logits = self.q_head(global_state)[:count]
        if observation["fixed_q_index"] is not None:
            allowed = torch.arange(count, device=self.device) == observation["fixed_q_index"]
            q_logits = q_logits.masked_fill(~allowed, -torch.inf)
        return h, nodes, context, Categorical(logits=q_logits), self.critic(global_state).squeeze(-1)

    def location_distributions(self, nodes, context, q, q_values):
        q_features = nodes.new_tensor([q / nodes.shape[0], q / max(q_values)])
        condition = torch.tanh(self.location_condition(torch.cat([context, q_features])))
        # A linear head on (node + shared_context) would only add a common
        # logit shift: anchor softmax would ignore q/context. Nonlinearity makes
        # the context change node-relative scores, not just their offset.
        conditioned = torch.tanh(nodes + condition.unsqueeze(0))
        logits = self.encoder.lin_prob(conditioned).squeeze(-1)
        alpha = self.encoder.sisr_activation(self.encoder.lin_sisr_mean(conditioned).squeeze(-1)) + 2
        beta = self.encoder.sisr_activation(self.encoder.lin_sisr_std(conditioned).squeeze(-1).abs()) + 2
        return logits, alpha, beta

    def action_terms(self, observation, action=None, greedy=False):
        h, nodes, context, q_dist, value = self.encode(observation)
        q_index = (int(torch.argmax(q_dist.logits)) if greedy else int(q_dist.sample())) if action is None else int(action["q_index"])
        q = int(observation["q_values"][q_index])
        if action is not None and int(action["q"]) != q:
            raise ValueError("Stored q index/value mismatch")
        qi = torch.tensor(q_index, device=self.device)
        logprob, entropy = q_dist.log_prob(qi), q_dist.entropy()
        logits, alpha, beta = self.location_distributions(nodes, context, q, observation["q_values"])
        active = min(self.cfg.anchors, q)
        if action is not None and (len(action["anchors"]) != active or len(action["coefficients"]) != active):
            raise ValueError("Stored anchor count mismatch")
        used = torch.zeros(nodes.shape[0], dtype=torch.bool, device=self.device)
        anchors, coefficients = [], []
        for i in range(active):
            anchor_dist = Categorical(logits=logits.masked_fill(used, -torch.inf))
            if action is None:
                index = int(torch.argmax(anchor_dist.logits)) if greedy else int(anchor_dist.sample())
            else:
                index = int(action["anchors"][i]) - 1
            if not 0 <= index < nodes.shape[0] or used[index]:
                raise ValueError("Stored anchor is invalid or repeated")
            coefficient_dist = Beta(alpha[index], beta[index])
            if action is None:
                coefficient = float(coefficient_dist.mean if greedy else coefficient_dist.sample())
                coefficient = min(max(coefficient, 1e-6), 1-1e-6)
            else:
                coefficient = float(action["coefficients"][i])
            logprob = logprob + anchor_dist.log_prob(torch.tensor(index, device=self.device))
            logprob = logprob + coefficient_dist.log_prob(nodes.new_tensor(coefficient))
            entropy = entropy + anchor_dist.entropy() + coefficient_dist.entropy()
            # New tensor: avoid autograd version conflicts with masked_fill's mask.
            used = used.clone()
            used[index] = True
            anchors.append(index + 1)
            coefficients.append(coefficient)
        result = {"q_index": q_index, "q": q, "anchors": anchors, "coefficients": coefficients}
        return result, logprob, value, entropy, h.detach().cpu()

    @torch.no_grad()
    def act(self, observation, greedy=False):
        action, logprob, value, _, hidden = self.action_terms(observation, greedy=greedy)
        return action, float(logprob), float(value), hidden


class PPOTrainer:
    """Upstream PPO clipped objective, extended to the conditional joint action.

    Rollouts keep predecision recurrent states as upstream does (one-step
    truncated recurrence, not full BPTT). Critic targets use unnormalized returns;
    only policy advantages are standardized. Time-limit truncation bootstraps.
    """
    def __init__(self, policy, cfg):
        self.policy, self.cfg = policy, cfg
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.learning_rate, betas=(.9, .999))

    def update(self, transitions):
        """Compatibility path for small adapter/benchmark diagnostics."""
        return self.update_episodes([transitions])

    def update_episodes(self, episodes, deadline=None):
        """Two passes over whole-episode minibatches; advantages use the rollout.

        Time order within a minibatch is preserved. Hidden states remain stored
        one-step states, as in upstream; this is NOT sequence BPTT.
        """
        if not episodes or any(not ep for ep in episodes):
            raise ValueError("Empty PPO episode")
        transitions = [t for ep in episodes for t in ep]
        if not transitions:
            raise ValueError("Empty PPO rollout")
        returns = []
        for ep in episodes:
            tail, running = [], float(ep[-1]["next_value"])
            for t in reversed(ep):
                if t["terminal"]:
                    running = 0.
                running = float(t["reward"]) + self.cfg.gamma * running
                tail.insert(0, running)
            returns.extend(tail)
        target = torch.tensor(returns, dtype=torch.float32, device=self.policy.device)
        old_values = target.new_tensor([t["value"] for t in transitions])
        advantages = target - old_values
        if len(transitions) > 1 and float(advantages.std(unbiased=False)) > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        old_logprob = target.new_tensor([t["logprob"] for t in transitions])
        loss_value = initial_ratio_error = grad_norm = None
        last_policy_loss = last_value_loss = last_entropy = None
        optimizer_steps = 0
        replay_kl = replay_clipped = replay_samples = 0
        head_gradients = {}
        for epoch in range(self.cfg.ppo_epochs):
            offset = 0
            for episode in episodes:
                self.optimizer.zero_grad()
                sums = dict(loss=0., policy=0., value=0., entropy=0., ratio_error=0.)
                for local_i, t in enumerate(episode):
                    if deadline is not None:
                        import time
                        if time.perf_counter() >= deadline:
                            raise TimeoutError("PPO time guard; caller must restore pre-update state")
                    i = offset + local_i
                    _, logprob, value, entropy, _ = self.policy.action_terms(t["observation"], t["action"])
                    ratio = torch.exp(logprob-old_logprob[i])
                    surrogate = torch.minimum(ratio*advantages[i], ratio.clamp(
                        1-self.cfg.clip_epsilon, 1+self.cfg.clip_epsilon)*advantages[i])
                    value_loss = F.mse_loss(value, target[i])
                    loss = -surrogate + .5*value_loss - self.cfg.entropy_weight*entropy
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite PPO loss")
                    (loss/len(episode)).backward()
                    sums["loss"] += float(loss.detach())/len(episode)
                    sums["policy"] += float(-surrogate.detach())/len(episode)
                    sums["value"] += float(value_loss.detach())/len(episode)
                    sums["entropy"] += float(entropy.detach())/len(episode)
                    sums["ratio_error"] = max(sums["ratio_error"], float((ratio.detach()-1).abs()))
                    log_ratio = (logprob-old_logprob[i]).detach()
                    replay_kl += float((ratio.detach()-1)-log_ratio)
                    replay_clipped += int(float((ratio.detach()-1).abs()) > self.cfg.clip_epsilon)
                    replay_samples += 1
                if epoch == 0 and offset == 0:
                    initial_ratio_error = sums["ratio_error"]
                gradients = [p.grad for p in self.policy.parameters() if p.grad is not None]
                if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                    raise FloatingPointError("Missing or non-finite PPO gradients")
                grad_norm = float(nn.utils.clip_grad_norm_(self.policy.parameters(), 1.))
                for name,p in self.policy.named_parameters():
                    if p.grad is not None and any(name.startswith(prefix) for prefix in (
                            "q_head.", "encoder.lin_prob.", "encoder.lin_sisr_mean.", "encoder.lin_sisr_std.")):
                        head_gradients.setdefault(name,[]).append(float(p.grad.detach().norm()))
                self.optimizer.step()
                optimizer_steps += 1
                offset += len(episode)
                loss_value = sums["loss"]
                last_policy_loss, last_value_loss, last_entropy = sums["policy"], sums["value"], sums["entropy"]
        import statistics
        rewards = [float(t["reward"]) for t in transitions]
        diagnostics = [t["diagnostics"] for t in transitions if "diagnostics" in t]
        collection_metrics = {"actions":len(diagnostics)}
        for key in ("feature_seconds", "inference_seconds", "repair_seconds", "collection_seconds"):
            values = [d[key] for d in diagnostics if d.get(key) is not None]
            if values:
                collection_metrics[key] = dict(total=sum(values),mean=statistics.mean(values))
        target_variance = float(target.var(unbiased=False))
        return {"loss": loss_value, "gradient_norm": grad_norm, "initial_ratio_error": initial_ratio_error,
                "policy_loss": last_policy_loss, "value_loss": last_value_loss, "entropy": last_entropy,
                "transitions": len(transitions), "ppo_epochs": self.cfg.ppo_epochs,
                "episode_minibatch_lengths": [len(ep) for ep in episodes],
                "optimizer_steps": optimizer_steps,
                "metric_scope": "losses_last_minibatch; replay_statistics_all_passes; head_gradients_after_global_clip",
                "approx_kl": replay_kl/replay_samples, "clip_fraction":replay_clipped/replay_samples,
                "head_gradient_mean_l2": {n:statistics.mean(v) for n,v in head_gradients.items()},
                "old_value_explained_variance": (1-float((target-old_values).var(unbiased=False))/target_variance) if target_variance>1e-12 else None,
                "reward_statistics":dict(mean=statistics.mean(rewards),std=statistics.pstdev(rewards),
                    minimum=min(rewards),maximum=max(rewards),nonzero_fraction=sum(abs(r)>1e-12 for r in rewards)/len(rewards)),
                "collection_metrics":collection_metrics}
