"""DPR HRGCN/GRU body, conditional q/location task heads, real CUDA graph batches.

EMNH migration: copy the inner shared body; interpolate only the mean task HEADS.
This is not the original POMO/Transformer architecture or its start-node baseline.
"""
import copy
from types import SimpleNamespace
import torch
from torch import nn
from torch.distributions import Categorical, Beta
from torch.nn import functional as F
from lib.ppo_1 import Net_Actor, Net_Generic
from mo_dpr.environment import NODE_FEATURES
from .config import CONTEXT


class Head(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.q = nn.Linear(width, 9)
        self.anchor = nn.Linear(width, 1)
        self.alpha = nn.Linear(width, 1)
        self.beta = nn.Linear(width, 1)


class Policy(nn.Module):
    def __init__(self, cfg, tasks=1, critic=False):
        super().__init__()
        self.cfg = cfg
        self.encoder = Net_Actor(cfg.learning_rate, len(NODE_FEATURES), cfg.hidden_dim)
        # The original final three linears become task heads. Keep the original body.
        for name in ("lin_prob", "lin_sisr_mean", "lin_sisr_std"):
            delattr(self.encoder, name)
        h = cfg.hidden_dim
        self.context = nn.Sequential(nn.Linear(len(CONTEXT), h), nn.Tanh())
        self.q_body = nn.Sequential(nn.Linear(h*2, h), nn.Tanh())
        self.location = nn.Linear(h+2, h)
        self.heads = nn.ModuleList([Head(h) for _ in range(tasks)])
        self.critics = nn.ModuleList([Net_Generic(cfg.learning_rate, h*2, cfg.critic_dim, 1)
                                      for _ in range(tasks)] if critic else [])

    @property
    def device(self):
        return next(self.parameters()).device

    def encode_batch(self, observations):
        """Batch equally sized graphs; variable N is grouped by the caller.

        Average previous removed features BEFORE batching: padding variable q and
        then averaging would silently change the original recurrent input.
        """
        n = len(observations[0]["x"])
        if any(len(o["x"]) != n for o in observations):
            raise ValueError("Group observations by graph size before batching")
        fields = {"x": torch.cat([o["x"] for o in observations]).to(self.device),
                  "state_": torch.cat([o["state_"] for o in observations]).to(self.device),
                  "input_": torch.stack([o["input_"].mean(0) for o in observations]).unsqueeze(1).to(self.device)}
        for name in ("edge_index_n", "edge_index_r0", "edge_index_r1"):
            fields[name] = torch.cat([o[name]+i*n for i, o in enumerate(observations)], 1).to(self.device)
        hidden, nodes = self.encoder.encode(SimpleNamespace(**fields), n)
        nodes = nodes[:, 1:]
        context = self.context(torch.stack([o["context"] for o in observations]).to(self.device))
        global_state = torch.cat([nodes.mean(1), context], -1)
        return hidden, nodes, context, global_state

    def terms(self, observations, task_ids=None, actions=None):
        ids = [0]*len(observations) if task_ids is None else list(task_ids)
        if len(ids) != len(observations):
            raise ValueError("Task count mismatch")
        hidden, nodes, context, global_state = self.encode_batch(observations)
        q_hidden = self.q_body(global_state)
        logits = torch.stack([self.heads[t].q(q_hidden[i]) for i, t in enumerate(ids)])
        legal = torch.tensor([len(o["q_values"]) for o in observations], device=self.device)
        q_dist = Categorical(logits=logits.masked_fill(torch.arange(9, device=self.device)[None] >= legal[:, None], -torch.inf))
        qi = q_dist.sample() if actions is None else torch.tensor([a["q_index"] for a in actions], device=self.device)
        qs = [o["q_values"][int(i)] for o, i in zip(observations, qi.detach().cpu())]
        if actions is not None and any(a["q"] != q for a, q in zip(actions, qs)):
            raise ValueError("Stored q/index mismatch")
        q_features = nodes.new_tensor([[q/len(nodes[i]), q/max(o["q_values"])]
                                       for i, (q, o) in enumerate(zip(qs, observations))])
        loc = torch.tanh(nodes + torch.tanh(self.location(torch.cat([context, q_features], -1)))[:, None])
        anchor_logits = torch.stack([self.heads[t].anchor(loc[i]).squeeze(-1) for i, t in enumerate(ids)])
        alpha = torch.stack([F.elu(self.heads[t].alpha(loc[i]).squeeze(-1))+2 for i, t in enumerate(ids)])
        beta = torch.stack([F.elu(self.heads[t].beta(loc[i]).squeeze(-1).abs())+2 for i, t in enumerate(ids)])
        logprob, entropy = q_dist.log_prob(qi), q_dist.entropy()
        used = torch.zeros_like(anchor_logits, dtype=torch.bool)
        output = [dict(q_index=int(i), q=int(q), anchors=[], coefficients=[]) for i, q in zip(qi.detach().cpu(), qs)]
        batch_indices = torch.arange(len(observations), device=self.device)
        for slot in range(self.cfg.anchors):
            active = torch.tensor([slot < min(self.cfg.anchors, q) for q in qs], device=self.device)
            dist = Categorical(logits=anchor_logits.masked_fill(used, -torch.inf))
            index = dist.sample() if actions is None else torch.tensor([
                a["anchors"][slot]-1 if bool(active[i]) else 0 for i, a in enumerate(actions)], device=self.device)
            # Inactive slots get a valid dummy index, excluded from likelihood.
            if actions is not None:
                index = torch.where(active, index, (~used).long().argmax(-1))
                if bool(((index < 0) | (index >= nodes.shape[1])).any()) or bool((used[batch_indices, index] & active).any()):
                    raise ValueError("Invalid/repeated stored anchor")
            coefficient_dist = Beta(alpha[batch_indices, index], beta[batch_indices, index])
            c = coefficient_dist.sample() if actions is None else nodes.new_tensor([
                a["coefficients"][slot] if bool(active[i]) else .5 for i, a in enumerate(actions)])
            if bool(((c <= 0) | (c >= 1)).any()):
                raise FloatingPointError("Beta sample outside open interval; never silently clip likelihood")
            logprob = logprob + torch.where(active, dist.log_prob(index)+coefficient_dist.log_prob(c), 0.)
            entropy = entropy + torch.where(active, dist.entropy()+coefficient_dist.entropy(), 0.)
            used = used.clone(); used[batch_indices, index] = True
            for i, (a, v) in enumerate(zip(index.detach().cpu(), c.detach().cpu())):
                if bool(active[i]):
                    output[i]["anchors"].append(int(a)+1); output[i]["coefficients"].append(float(v))
        values = torch.stack([self.critics[t](global_state[i]).squeeze(-1) for i, t in enumerate(ids)]) if self.critics else torch.zeros_like(logprob)
        return output, logprob, values, entropy, hidden.detach().cpu()

    @torch.no_grad()
    def act(self, observations, task_ids=None):
        a, lp, value, _, h = self.terms(observations, task_ids)
        return a, lp.cpu().tolist(), value.cpu().tolist(), h

    def actor_state(self):
        return {k: v.detach().clone() for k, v in self.state_dict().items() if not k.startswith("critics.")}

    def start_inner(self, meta):
        state = meta.actor_state()
        merged = self.state_dict()
        for k in merged:
            if k.startswith("critics."):
                continue
            if k.startswith("heads."):
                k_source = "heads.0." + k.split(".", 2)[2]
                merged[k] = state[k_source]
            else:
                merged[k] = state[k]
        self.load_state_dict(merged)

    @torch.no_grad()
    def meta_update(self, inner, epsilon):
        if not 0 <= epsilon <= 1 or len(self.heads) != 1:
            raise ValueError("Invalid meta interpolation")
        source = inner.state_dict()
        target = self.state_dict()
        for name in target:
            if name.startswith("critics."):
                continue
            if name.startswith("heads.0."):
                suffix = name.split(".", 2)[2]
                mean = torch.stack([source[f"heads.{t}.{suffix}"] for t in range(len(inner.heads))]).mean(0)
                target[name].lerp_(mean, epsilon)
            else:
                target[name].copy_(source[name])
        self.load_state_dict(target)

    def deploy(self, task=0):
        result = Policy(self.cfg).to(self.device)
        state = self.actor_state()
        target = result.state_dict()
        for key in target:
            target[key] = state[(f"heads.{task}."+key.split(".", 2)[2]) if key.startswith("heads.") else key]
        result.load_state_dict(target)
        return result
