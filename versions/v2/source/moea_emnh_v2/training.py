"""Finite GPU meta-training and preference adaptation with on-policy trajectories."""
import copy
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .config import directions, search_budget
from .environment import Environment, Adaptive, LocalSession
from .model import Policy
from .outer import Search
from mo_dpr.artifacts import save_json, save_torch


def require_resource_cutoff(deadline, margin):
    if time.perf_counter() < deadline-margin:
        raise RuntimeError("Phase transition before external resource deadline")


def adaptation_allocation(cap, debit, index):
    if not 0 <= debit <= cap or not 0 <= index < 16:
        raise ValueError("Invalid remaining adaptation budget allocation")
    return debit+(cap-debit)*(index+1)/16


def cyclic_tasks(rng):
    w = rng.dirichlet(np.ones(5))
    return np.stack([np.roll(w, i) for i in range(5)])


def collect_closed(pool, split, cfg, policy, meta, weights, state, rng, deadline, notify):
    from .outer import run_session, TaskPolicy, search_state, restore_search
    wanted = len(weights)
    if wanted == 5:
        refs = directions().copy(); refs[:5] = np.asarray(weights)
        target = list(range(5))
    else:
        refs = directions()
        target = [int(np.argmin(np.linalg.norm(refs-np.asarray(weights[0]), axis=1)))]
    mapping = {k:t for t,k in enumerate(target)}
    policies = [TaskPolicy(policy, mapping[k]) if k in mapping else TaskPolicy(meta) for k in range(16)]
    episodes = {}
    provenance = []
    search = state.pop("object", None)
    if search is None and state.get("search") is not None:
        search = restore_search(state.pop("search"), pool, cfg, policies, deadline)
    if search is not None:
        search.policies = policies; search.external_deadline = deadline
        # Optimizer/checkpoint/other sampling work is paused out of solver time.
        elapsed = state.pop("frozen_elapsed")
        search.started = time.perf_counter()-elapsed; search.deadline = search.started+search.budget
    def fork(k, primary):
        if k not in mapping or mapping[k] in episodes:
            run_session(primary, policies[k])
            return primary, 0., []
        task = mapping[k]
        # All four start from this actual offspring, same actual direction stats.
        original = copy.deepcopy(primary.adaptive)
        routes = copy.deepcopy(primary.env.routes)
        group = []
        run_session(primary, policies[k], task, group)
        batch = [group]
        aux_started = time.perf_counter()
        for _ in range(cfg.replicas-1):
            env = Environment(pool, search.iid,
                              cfg, refs[k], routes)
            clone = LocalSession(env, copy.deepcopy(original), int(rng.integers(0, 2**31-1)),
                primary.global_budget, primary.global_start, cfg.local_seconds, deadline)
            ep = []; run_session(clone, policies[k], task, ep); batch.append(ep)
        paused = time.perf_counter()-aux_started
        valid_group = all(ep and ep[-1]["terminal"] for ep in batch)
        if valid_group:
            episodes[task] = batch
        else:
            state.setdefault("unoptimized_boundary_groups", []).append(batch)
        provenance.append(dict(iid=search.iid, direction=k, weight=refs[k].tolist(),
            source="neural_MOEAD_actual_offspring", primary_committed=True,
            source_routes=routes, source_adaptive_counts=original.count.tolist(),
            source_recent_improvement=original.recent_improvement.tolist(),
            global_remaining=primary.global_start, global_budget=primary.global_budget,
            optimized_group=valid_group,
            primary_routes=copy.deepcopy(primary.env.routes),
            primary_objectives=primary.env.objectives.tolist(), auxiliary_rollouts=cfg.replicas-1))
        return primary, paused, batch
    last_beat = time.perf_counter()
    while len(episodes) < wanted and time.perf_counter() < deadline:
        if search is None or search.remaining() <= 0:
            if search is not None:
                state["completed_searches"] = state.get("completed_searches", 0)+1
            iid = split["fit_ids"][state["next_iid"] % len(split["fit_ids"])]
            state["next_iid"] += 1
            search = Search(pool, iid, cfg, int(rng.integers(0, 2**31-1)), policies,
                            resource_deadline=deadline, ref_dirs=refs)
        if not search.step(fork):
            if time.perf_counter() >= deadline:
                break
            search = None
        if time.perf_counter()-last_beat > 15:
            notify(dict(phase="collection", iid=search.iid if search else None,
                completed_tasks=len(episodes), target_tasks=wanted,
                actions=sum(len(ep) for group in episodes.values() for ep in group),
                state_source="neural_MOEAD_closed_loop"))
            last_beat = time.perf_counter()
    state["object"] = search
    state["frozen_elapsed"] = time.perf_counter()-search.started if search else 0.
    flat = [ep for t in range(wanted) for ep in episodes.get(t, [])]
    if len(episodes) != wanted:
        flat += [ep for group in state.get("unoptimized_boundary_groups", []) for ep in group]
    complete = len(flat) == wanted*cfg.replicas and all(ep and ep[-1]["terminal"] for ep in flat)
    return flat, complete, provenance


def freeze_collection(state):
    from .outer import search_state
    result = {k:v for k,v in state.items() if k != "object"}
    if state.get("object") is not None:
        result["search"] = search_state(state["object"])
        result["search"]["elapsed"] = state["frozen_elapsed"]
    return result


def returns(ep):
    running = ep[-1]["next_value"]
    result = []
    for t in reversed(ep):
        if t["terminal"]:
            running = 0.
        running = t["reward"]+running
        result.append(running)
    return list(reversed(result))


def update(policy, optimizer, episodes, method, cfg, deadline=None):
    if method not in ("ppo", "reinforce") or not episodes or any(not ep for ep in episodes):
        raise ValueError("Invalid update")
    if method == "reinforce" and any(not ep[-1]["terminal"] for ep in episodes):
        raise ValueError("REINFORCE requires complete local-budget trajectories")
    # Leave-one-out baseline from independent SAME-START replicas of each task.
    # Original POMO's different starting-node baseline is inapplicable to LNS.
    task_episodes = {}
    for ep in episodes:
        task_episodes.setdefault(ep[0]["task"], []).append(ep)
    baselines = {}
    for task, group in task_episodes.items():
        totals = [sum(t["reward"] for t in ep) for ep in group]
        if method == "reinforce" and len(group) < 2:
            raise ValueError("Independent rollout baseline requires >=2 replicas")
        for j, ep in enumerate(group):
            baselines[id(ep)] = (sum(totals)-totals[j])/(len(group)-1) if len(group)>1 else 0.
    samples = []
    for ep in episodes:
        task = ep[0]["task"]
        weight = 1./len(task_episodes)/len(task_episodes[task])
        for t, target in zip(ep, returns(ep)):
            samples.append((t, target, weight, baselines[id(ep)]))
    # Group graph sizes for actual batching without padding graphs or history.
    buckets = {}
    for s in samples:
        buckets.setdefault(len(s[0]["observation"]["x"]), []).append(s)
    epochs = cfg.ppo_epochs if method == "ppo" else 1
    metrics = dict(steps=0, transitions=len(samples), trajectories=len(episodes),
                   complete_trajectories=sum(ep[-1]["terminal"] for ep in episodes),
                   loss=0., actor_loss=0., value_loss=0., entropy=0., initial_ratio_error=0.,
                   baseline="same_start_leave_one_out" if method == "reinforce" else "task_critic_MC_gamma1")
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        for bucket in buckets.values():
            for offset in range(0, len(bucket), cfg.microbatch):
                if deadline is not None and time.perf_counter() >= deadline:
                    optimizer.zero_grad(set_to_none=True)
                    metrics["resource_truncated_update"] = True
                    return metrics
                micro = bucket[offset:offset+cfg.microbatch]
                ts = [x[0] for x in micro]
                _, lp, value, entropy, _ = policy.terms([t["observation"] for t in ts],
                    [t["task"] for t in ts], [t["action"] for t in ts])
                targets = lp.new_tensor([x[1] for x in micro])
                importance = lp.new_tensor([x[2] for x in micro])
                old = lp.new_tensor([t["logprob"] for t in ts])
                if method == "ppo":
                    adv = targets-lp.new_tensor([t["value"] for t in ts])
                    ratio = (lp-old).exp()
                    actor = -torch.minimum(ratio*adv, ratio.clamp(1-cfg.clip_epsilon, 1+cfg.clip_epsilon)*adv)
                    value_loss = .5*(value-targets).square()
                    if epoch == 0:
                        metrics["initial_ratio_error"] = max(metrics["initial_ratio_error"], float((ratio.detach()-1).abs().max()))
                else:
                    actor = -lp*(targets-lp.new_tensor([x[3] for x in micro]))
                    value_loss = torch.zeros_like(actor)
                loss = ((actor+value_loss-cfg.entropy_weight*entropy)*importance).sum()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                loss.backward()
                for key, v in (("loss", actor+value_loss-cfg.entropy_weight*entropy),
                               ("actor_loss", actor), ("value_loss", value_loss), ("entropy", entropy)):
                    metrics[key] += float((v.detach()*importance).sum())/epochs
        gradients = [p.grad for p in policy.parameters() if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise FloatingPointError("Missing/non-finite gradients")
        metrics["gradient_norm"] = float(nn.utils.clip_grad_norm_(policy.parameters(), 1.))
        metrics["head_gradients"] = {n:float(p.grad.norm()) for n, p in policy.named_parameters()
                                      if n.startswith("heads.") and p.grad is not None}
        if deadline is not None and time.perf_counter() >= deadline:
            optimizer.zero_grad(set_to_none=True)
            metrics["resource_truncated_update"] = True
            return metrics
        optimizer.step(); metrics["steps"] += 1
    failure_reasons = {}
    for transition, _, _, _ in samples:
        diag = transition["diagnostics"]
        if diag.get("failed"):
            key = diag.get("repair", "unknown")+":"+diag.get("reason", "unknown")
            failure_reasons[key] = failure_reasons.get(key, 0)+1
    metrics["failed_repairs"] = failure_reasons
    metrics["reward_sum"] = sum(t[0]["reward"] for t in samples)
    metrics["positive_reward_fraction"] = sum(t[0]["reward"] > 0 for t in samples)/len(samples)
    metrics["timeout_fraction"] = sum(t[0]["diagnostics"].get("timeout", False) for t in samples)/len(samples)
    return metrics


def training_gates(history, adapted, split):
    covered = sorted({iid for h in history if h.get("steps", 0) > 0 for iid in h.get("iids", [])})
    missing = sorted(set(split["fit_ids"])-set(covered))
    not_adapted = [d["direction"] for d in adapted if not d["adapted"]]
    phases = {i:{"early":0, "middle":0, "late":0} for i in split["fit_ids"]}
    for h in history:
        if h.get("steps", 0) <= 0: continue
        for p in h.get("provenance", []):
            if not p.get("optimized_group", True): continue
            used = 1-p["global_remaining"]/p["global_budget"]
            key = "early" if used < 1/3 else "middle" if used < 2/3 else "late"
            phases[p["iid"]][key] += 1
    return dict(state_phase_counts_by_iid=phases,
        all_three_phases_seen=all(sum(v[k] for v in phases.values()) > 0 for k in ("early", "middle", "late")),
        optimized_meta_iids=covered, missing_fit_iids=missing,
        fit_coverage_passed=not missing, unadapted_directions=not_adapted,
        adaptation_coverage_passed=len(adapted) == 16 and not not_adapted,
        training_coverage_passed=not missing and len(adapted) == 16 and not not_adapted,
        meta_learning_benefit_proven=False)


def train_method(pool, split, cfg, method, root, notify, spent_callback=None):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    if (root/"training_report.json").exists():
        from .runner import load_directions
        return load_directions(root, cfg, cfg.binding_id), __import__('json').loads((root/"training_report.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed); torch.cuda.manual_seed_all(cfg.seed)
    meta = Policy(cfg).to(cfg.device)
    inner = Policy(cfg, tasks=cfg.task_count, critic=method == "ppo").to(cfg.device)
    optimizer = torch.optim.Adam(inner.parameters(), lr=cfg.learning_rate)
    history = []; round_i = actions = next_u = next_iid = 0
    state = {}; weights = None; stage = "meta"; adapted = []
    meta_spent = float(cfg.legacy_ppo_seconds if method == "ppo" else 0.)
    adapt_spent = 0.; saved = None
    adaptation_restart_debit = 0.; recovery_events = []
    checkpoint = root/"resume.pt"
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=cfg.device, weights_only=False)
        if saved["schema"] != cfg.version or saved["binding_id"] != cfg.binding_id or saved["method"] != method:
            raise ValueError("Training checkpoint binding mismatch")
        meta.load_state_dict(saved["meta"]); inner.load_state_dict(saved["inner"])
        optimizer.load_state_dict(saved["optimizer"])
        history, round_i, actions = saved["history"], saved["round_i"], saved["actions"]
        next_u, next_iid, state, weights = saved["next_u"], saved["next_iid"], saved["state"], saved["weights"]
        stage, adapted, meta_spent, adapt_spent = saved["stage"], saved["adapted"], saved["meta_spent"], saved["adapt_spent"]
        adaptation_restart_debit = saved.get("adaptation_restart_debit", 0.)
        recovery_events = saved.get("recovery_events", [])
        rng.bit_generator.state = saved["rng_numpy"]
        torch.set_rng_state(saved["rng_torch"].cpu())
        torch.cuda.set_rng_state_all([v.cpu() for v in saved["rng_cuda"]])
    budget_file = root.parent/"budget.json"
    if budget_file.exists():
        ledger = __import__('json').loads(budget_file.read_text(encoding="utf-8")).get(method, {})
        meta_spent = max(meta_spent, ledger.get("meta_spent", 0.))
        adapt_spent = max(adapt_spent, ledger.get("adapt_spent", 0.))
    clock = time.perf_counter()
    def spent():
        used = time.perf_counter()-clock
        return meta_spent+(used if stage == "meta" else 0), adapt_spent+(used if stage == "adaptation" else 0)
    def notify_charged(v):
        ms, ads = spent()
        if spent_callback: spent_callback(method, stage, ms, ads)
        notify(dict(method=method, meta_spent_seconds=ms, adaptation_spent_seconds=ads, **v))
    if spent_callback: spent_callback(method, stage, meta_spent, adapt_spent)
    def save(extra=None):
        ms, ads = spent()
        payload = dict(schema=cfg.version, binding_id=cfg.binding_id, config=cfg.to_dict(), method=method,
            meta=meta.actor_state(), inner=inner.state_dict(), optimizer=optimizer.state_dict(),
            history=history, round_i=round_i, actions=actions, next_u=next_u, next_iid=next_iid,
            state=freeze_collection(state), weights=weights, stage=stage, adapted=adapted,
            meta_spent=ms, adapt_spent=ads, adaptation_restart_debit=adaptation_restart_debit,
            recovery_events=recovery_events, rng_numpy=rng.bit_generator.state,
            rng_torch=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state_all(),
            boundary="after_complete_optimizer_update_and_population_feedback")
        if extra: payload.update(extra)
        save_torch(checkpoint, payload)
        if spent_callback: spent_callback(method, stage, ms, ads)
    if stage == "meta":
        deadline = clock+max(0., cfg.meta_seconds-meta_spent)
        while time.perf_counter() < deadline-5:
            if next_u == 0:
                inner.start_inner(meta); weights = cyclic_tasks(rng).tolist()
                state = dict(next_iid=next_iid, completed_searches=0)
            eps, complete, provenance = collect_closed(pool, split, cfg, inner, meta, weights, state, rng, deadline-3, notify_charged)
            actions += sum(map(len, eps)); next_iid = state["next_iid"]
            if not complete:
                save_torch(root/"unoptimized_tail.pt", dict(episodes=eps, provenance=provenance,
                    scope="resource_truncation_not_optimized"))
                if time.perf_counter() < deadline-3:
                    raise RuntimeError("Incomplete meta collection before external resource deadline")
                break
            metric = update(inner, optimizer, eps, method, cfg, deadline-1)
            iids = sorted({p["iid"] for p in provenance if p.get("optimized_group", True)})
            history.append(dict(round=round_i, inner_update=next_u, iids=iids, weights=weights,
                elapsed_seconds=spent()[0], provenance=provenance, **metric))
            next_u += 1
            if next_u == cfg.inner_updates:
                meta.meta_update(inner, max(0., 1-spent()[0]/cfg.meta_seconds))
                round_i += 1; next_u = 0; state = {}
            save_json(root/"rollouts"/f"meta_batch_{len(history):06d}.json", dict(provenance=provenance,
                episodes=[dict(task=ep[0]["task"], records=[t["diagnostics"] for t in ep]) for ep in eps]))
            save()
            save_json(root/"learning_curve.json", history)
            notify(dict(phase="meta_training", method=method, round=round_i, updates=len(history),
                training_actions=actions, optimized_iids=len({i for h in history if h["steps"] for i in h["iids"]}),
                latest={k:v for k,v in history[-1].items() if k not in ("provenance", "weights", "head_gradients")}))
            if metric.get("resource_truncated_update"): break
        require_resource_cutoff(deadline, 5.)
        # A partial final inner round is disclosed rather than silently lost.
        if next_u > 0:
            meta.meta_update(inner, max(0., 1-spent()[0]/cfg.meta_seconds))
        meta_spent, adapt_spent = spent(); stage = "adaptation"; clock = time.perf_counter()
        state = {}; save()
        save_torch(root/"meta_complete.pt", torch.load(checkpoint, map_location="cpu", weights_only=False))
        saved = None  # Do not reset RNG back to a pre-meta resume point during adaptation.
    order = [15]+list(range(5, 15))+list(range(5))
    parents = {15:None, **{i:15 for i in range(5, 15)}}
    for i in range(5): parents[i] = next(j for j in range(5, 15) if directions()[j, i] > 0)
    models = {}
    for d in adapted:
        data = torch.load(root/"directions"/f"direction_{d['direction']:02d}.pt", map_location=cfg.device, weights_only=True)
        m = Policy(cfg).to(cfg.device); m.load_state_dict(data["model"]); models[d["direction"]] = m
    for sequence_i in range(len(adapted), 16):
        k = order[sequence_i]; parent = meta if parents[k] is None else models[parents[k]]
        actor = Policy(cfg, critic=method == "ppo").to(cfg.device); actor.start_inner(parent)
        optim = torch.optim.Adam(actor.parameters(), lr=cfg.learning_rate)
        # A resume stays within the original cumulative one-hour allocation.
        allocation = adaptation_allocation(cfg.adaptation_seconds, adaptation_restart_debit, sequence_i)
        allocation_end = clock+max(0., allocation-adapt_spent)
        updates = samples = 0; adaptation_history = []
        state = dict(next_iid=(sequence_i*3) % len(split["fit_ids"]), completed_searches=0)
        if saved is not None and saved.get("active_direction") == k:
            actor.load_state_dict(saved["adapt_actor"]); optim.load_state_dict(saved["adapt_optimizer"])
            updates, samples = saved["adapt_updates"], saved["adapt_samples"]
            state = saved["state"]; adaptation_history = saved.get("adaptation_history", [])
        if saved is not None and sequence_i == len(adapted):
            rng.bit_generator.state = saved["rng_numpy"]
            torch.set_rng_state(saved["rng_torch"].cpu())
            torch.cuda.set_rng_state_all([v.cpu() for v in saved["rng_cuda"]])
            saved = None
        while time.perf_counter() < allocation_end-3:
            eps, complete, provenance = collect_closed(pool, split, cfg, actor, parent, [directions()[k]],
                                                       state, rng, allocation_end-2, notify_charged)
            samples += sum(map(len, eps))
            if not complete:
                save_torch(root/f"adapt_tail_{k:02d}.pt", dict(episodes=eps, provenance=provenance,
                    scope="resource_truncation_not_optimized"))
                if time.perf_counter() < allocation_end-2:
                    raise RuntimeError("Incomplete adaptation collection before external resource deadline")
                break
            metric = update(actor, optim, eps, method, cfg, allocation_end-1)
            updates += int(metric["steps"] > 0)
            adaptation_history.append(dict(provenance=provenance, **metric))
            save_json(root/"rollouts"/f"adapt_{k:02d}_batch_{updates:04d}.json", dict(provenance=provenance,
                episodes=[dict(task=ep[0]["task"], records=[t["diagnostics"] for t in ep]) for ep in eps]))
            save(dict(active_direction=k, adapt_actor=actor.state_dict(), adapt_optimizer=optim.state_dict(),
                adapt_updates=updates, adapt_samples=samples, adaptation_history=adaptation_history))
            if metric.get("resource_truncated_update"): break
        model = actor.deploy(); models[k] = model
        path = root/"directions"/f"direction_{k:02d}.pt"
        save_torch(path, dict(schema=cfg.version, config=cfg.to_dict(), method=method,
            model=model.actor_state(), direction=k, weight=directions()[k].tolist(), parent_direction=parents[k],
            updates=updates, interactions=samples, adapted=bool(updates), inheritance_only=not bool(updates)))
        save_json(root/"adaptation"/f"direction_{k:02d}.json", adaptation_history)
        adapted.append(dict(direction=k, parent=parents[k], updates=updates, interactions=samples,
                            checkpoint=str(path), adapted=bool(updates)))
        state = {}; save()
        notify(dict(phase="adaptation", method=method, completed_directions=sequence_i+1,
                    direction=k, updates=updates, interactions=samples, elapsed_seconds=spent()[1]))
    meta_spent, adapt_spent = spent()
    result = dict(method=method, training_seed=cfg.seed, meta_seconds=meta_spent,
        legacy_meta_seconds=cfg.legacy_ppo_seconds if method == "ppo" else 0.,
        adaptation_seconds=adapt_spent, adaptation_restart_debit=adaptation_restart_debit,
        recovery_events=recovery_events, training_actions=actions, rounds=round_i, updates=len(history),
        directions=adapted, allowed_fit_iids=len(split["fit_ids"]),
        optimized_meta_interactions=sum(h["transitions"] for h in history if h["steps"]),
        **training_gates(history, adapted, split),
        scope="single_seed_development; budget_exhaustion_is_not_learning_success")
    save_json(root/"training_report.json", result)
    return [models[i] for i in range(16)], result
