"""Explicit training stages, paired evaluation and resumable checkpoints."""
import json
import random
import time
from pathlib import Path
from dataclasses import asdict

import numpy as np
import torch
from torch.nn import functional as F

from src.core.protocol import RepairAction
from src.data_generation.sharded import _solution
from src.pilot.config import CONFIG, RUN
from src.pilot.data import PilotDataset, save_json, read_json, score, sha
from src.pilot.engine import enumerate_actions, apply_action
from src.pilot.model import RepairNet, tensor_features, repair
from src.pilot.budget import Budget, BudgetExceeded


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)


def cuda_check():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for training; no CPU fallback')
    a = torch.randn(32, 32, device='cuda', requires_grad=True)
    (a @ a).sum().backward()
    print(torch.__version__, torch.cuda.get_device_name(0), flush=True)


def evaluate_tasks(model, ds, ids, path=None, shuffle=False, device='cuda', budget=None):
    rows = []
    weights = [ds.items[t]['preference_weights'] for t in ids]
    shuffled = np.random.default_rng(17073).permutation(len(weights))
    handle = open(path, 'w', encoding='utf-8') if path else None
    try:
        for j, tid in enumerate(ids):
            if budget:
                budget.check()
            inst, state, task = ds.get(tid)
            w = weights[shuffled[j]] if shuffle else weights[j]
            result = repair(model, inst, state, w, device)
            teacher = score(task['label']['objectives'], weights[j], ds.normalization)
            regret = next(c for c in task['teacher_candidates'] if c['teacher_name'] == 'regret3')
            actual = score(result['objectives'], weights[j], ds.normalization) if result['feasible'] else None
            row = dict(task_id=tid, instance_id=inst.instance_id, state_id=state['state_id'],
                       destroy=state['destroy_type'], actual_ratio=len(state['removed_customers'])/inst.n_customers,
                       category=task['preference_category'], preference=weights[j], input_preference=w,
                       teacher_score=teacher, regret_score=score(regret['objectives'], weights[j], ds.normalization),
                       score=actual, gap=None if actual is None else actual-teacher,
                       relative_gap=None if actual is None or teacher <= 1e-6 else (actual-teacher)/teacher,
                       **result)
            if handle:
                handle.write(json.dumps(row, allow_nan=False)+'\n')
                handle.flush()
            rows.append(row)
    finally:
        if handle:
            handle.close()
    return rows


def summary(rows):
    valid = [r for r in rows if r['feasible'] and r['completed']]
    gaps = [r['gap'] for r in valid]
    return dict(tasks=len(rows), failures=len(rows)-len(valid),
                feasible_rate=len(valid)/max(len(rows), 1),
                mean_gap=float(np.mean(gaps)) if gaps else None,
                median_gap=float(np.median(gaps)) if gaps else None,
                p90_gap=float(np.percentile(gaps, 90)) if gaps else None,
                mean_runtime=float(np.mean([r['runtime_seconds'] for r in rows])) if rows else None)


def ranking(s):
    return (s['failures']/max(s['tasks'], 1), s['mean_gap'] if s['mean_gap'] is not None else float('inf'))


def save_checkpoint(path, model, optimizer, ds, progress):
    path = Path(path)
    payload = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), progress=progress,
                   conditioned=model.conditioned, config=CONFIG, manifest_hash=sha(ds.run/'pilot_manifest.json'),
                   torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                   numpy_rng=np.random.get_state(), python_rng=random.getstate())
    torch.save(payload, path.with_suffix('.tmp'))
    path.with_suffix('.tmp').replace(path)


def load_checkpoint(path, ds, device='cuda', optimizer=None):
    # Only load checkpoints generated locally by this pilot.
    ck = torch.load(path, map_location='cpu', weights_only=False)
    if ck['manifest_hash'] != sha(ds.run/'pilot_manifest.json') or ck['config'] != CONFIG:
        raise ValueError('Checkpoint/manifest/config mismatch')
    model = RepairNet(ck['conditioned']).to(device)
    model.load_state_dict(ck['model'])
    return model, ck


def trajectory_backward(model, inst, state, task, device, scale):
    partial, remaining = _solution(state['partial_solution']), list(state['removed_customers'])
    actions_target = task['label']['actions']
    if len(actions_target) != len(remaining):
        raise ValueError('Teacher trajectory length mismatch')
    loss_value, correct = 0., 0
    for target in actions_target:
        actions = enumerate_actions(inst, partial, remaining)
        action = RepairAction(**target)
        idx = actions.actions.index(action)
        if not actions.mask[idx]:
            raise ValueError(f"Masked teacher action: {task['task_id']} {target}")
        features = tensor_features(inst, partial, remaining, actions, device)
        logits = model(features, torch.tensor(task['preference_weights'], device=device, dtype=torch.float32))
        loss = F.cross_entropy(logits[None], torch.tensor([idx], device=device))
        if not torch.isfinite(loss):
            raise ValueError('Non-finite BC loss')
        (loss / len(actions_target) / scale).backward()
        loss_value += loss.item()/len(actions_target)
        correct += int(logits.argmax().item() == idx)/len(actions_target)
        partial, remaining = apply_action(partial, remaining, action)
    return loss_value, correct


def train(run=RUN, seed=17, conditioned=True, overfit=False):
    cuda_check()
    seed_all(seed)
    ds = PilotDataset(run)
    if CONFIG != ds.manifest['config']:
        raise ValueError('Configuration changed after prepare; create a new run')
    if not (ds.run/'stage_a.json').exists() or read_json(ds.run/'stage_a.json')['status'] != 'passed':
        raise RuntimeError('Run stage A before training')
    if not overfit and not (ds.run/'stage_b_decision.json').exists():
        raise RuntimeError('Inspect Stage B and explicitly record decision before Stage C')
    if not overfit and not read_json(ds.run/'stage_b_decision.json')['continue']:
        raise RuntimeError('Stage B decision does not permit expansion')
    name = 'overfit' if overfit else f"{'conditioned' if conditioned else 'unconditioned'}_{seed}"
    out = ds.run/name
    out.mkdir(exist_ok=True)
    model = RepairNet(conditioned).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    progress = dict(update=0, epoch=0, cursor=0, order=[], best=None, stale=0, status='running')
    if (out/'last.pt').exists():
        model, ck = load_checkpoint(out/'last.pt', ds)
        optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
        optimizer.load_state_dict(ck['optimizer'])
        progress = ck['progress']
        if progress['status'] in ('complete', 'early_stopped'):
            print(f'{name} already complete; not overwriting', flush=True)
            return
        torch.set_rng_state(ck['torch_rng']); torch.cuda.set_rng_state_all(ck['cuda_rng'])
        np.random.set_state(ck['numpy_rng']); random.setstate(ck['python_rng'])
    else:
        save_checkpoint(out/'initial.pt', model, optimizer, ds, progress)
    ids = read_json(ds.run/'overfit_tasks.json') if overfit else ds.train
    validation = ids if overfit else ds.dev
    budget = Budget(ds.run)
    used, started, limit = budget.used, budget.start, budget.limit
    def budget_hit():
        return used+time.monotonic()-started >= limit
    try:
        if not (out/'initial_metrics.json').exists():
            rows = evaluate_tasks(model, ds, validation, out/'initial_validation.jsonl', budget=budget)
            save_json(out/'initial_metrics.json', summary(rows))
        training_started = time.monotonic()
        timing_update = progress['update']
        while not budget_hit():
            if overfit and progress['update'] >= CONFIG['overfit_updates']:
                progress['status'] = 'complete'; break
            if not overfit and progress['epoch'] >= CONFIG['max_epochs']:
                progress['status'] = 'complete'; break
            if not progress['order']:
                progress['order'] = random.sample(ids, len(ids))
                progress['cursor'] = 0
            batch = progress['order'][progress['cursor']:progress['cursor']+CONFIG['batch_size']]
            model.train(); optimizer.zero_grad(set_to_none=True)
            losses, accuracies = [], []
            while True:
                try:
                    for tid in batch:
                        budget.check()
                        loss, acc = trajectory_backward(model, *ds.get(tid), 'cuda', len(batch))
                        losses.append(loss); accuracies.append(acc)
                    break
                except torch.cuda.OutOfMemoryError:
                    optimizer.zero_grad(set_to_none=True)
                    losses, accuracies = [], []
                    torch.cuda.empty_cache()
                    if model.chunk_size <= 1:
                        raise
                    model.chunk_size = max(1, model.chunk_size//2)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['gradient_clip'])
            optimizer.step()
            progress['update'] += 1
            progress['cursor'] += len(batch)
            epoch_end = progress['cursor'] >= len(progress['order'])
            if epoch_end:
                progress['order'] = []; progress['cursor'] = 0; progress['epoch'] += 1
            record = dict(update=progress['update'], epoch=progress['epoch'], loss=float(np.mean(losses)), accuracy=float(np.mean(accuracies)),
                          elapsed_seconds=used+time.monotonic()-started)
            with (out/'training.jsonl').open('a') as f:
                f.write(json.dumps(record)+'\n')
            do_eval = progress['update'] % CONFIG['eval_every'] == 0 if overfit else epoch_end
            if do_eval:
                eval_started = time.monotonic()
                rows = evaluate_tasks(model, ds, validation, out/f"validation_{progress['update']:06d}.jsonl", budget=budget)
                metrics = summary(rows)
                save_json(out/'latest_metrics.json', metrics)
                if progress['best'] is None or ranking(metrics) < ranking(progress['best']):
                    progress['best'] = metrics; progress['stale'] = 0
                    save_checkpoint(out/'best.pt', model, optimizer, ds, progress)
                else:
                    progress['stale'] += 1
                print(name, record, metrics, flush=True)
                if progress['update'] - timing_update >= 100 and not (out/'timing_estimate.json').exists():
                    seconds_per_update = (eval_started-training_started)/max(progress['update']-timing_update, 1)
                    validation_seconds = time.monotonic()-eval_started
                    updates_per_epoch = (len(ds.train)+CONFIG['batch_size']-1)//CONFIG['batch_size']
                    save_json(out/'timing_estimate.json', dict(seconds_per_update_upper_estimate=seconds_per_update,
                              validation_seconds=validation_seconds, six_training_runs_upper_hours=
                              6*CONFIG['max_epochs']*(updates_per_epoch*seconds_per_update+validation_seconds)/3600,
                              note='Includes earlier evaluation overhead; excludes final held-out probes.'))
                if not overfit and progress['stale'] >= CONFIG['patience']:
                    progress['status'] = 'early_stopped'; break
            if progress['update'] % 10 == 0 or do_eval:
                save_checkpoint(out/'last.pt', model, optimizer, ds, progress)
                save_json(out/'progress.json', progress)
        else:
            progress['status'] = 'budget_exhausted'
    except KeyboardInterrupt:
        progress['status'] = 'interrupted'
        print('Interrupted; checkpoint saved at last completed update', flush=True)
    except BudgetExceeded:
        progress['status'] = 'budget_exhausted'
    except Exception:
        progress['status'] = 'error'
        raise
    finally:
        save_checkpoint(out/'last.pt', model, optimizer, ds, progress)
        save_json(out/'progress.json', progress)
        budget.close()
    print(name, progress['status'], flush=True)


def evaluate(run=RUN):
    cuda_check()
    ds = PilotDataset(run)
    out = ds.run/'evaluation'
    if (out/'complete.json').exists():
        print('Evaluation already complete; use report stage', flush=True)
        return
    # Freeze all trained models before revealing held-out results.
    checkpoints = []
    for seed in (17, 29, 43):
        for mode in ('conditioned', 'unconditioned'):
            directory = ds.run/f'{mode}_{seed}'
            p = read_json(directory/'progress.json')
            if p['status'] not in ('complete', 'early_stopped'):
                raise RuntimeError(f'Unfinished training: {directory}')
            checkpoints.append(directory/'best.pt')
    out.mkdir(exist_ok=True)
    frozen = {str(p): sha(p) for p in checkpoints}
    frozen.update({str(p.parent/'initial.pt'): sha(p.parent/'initial.pt') for p in checkpoints})
    if (out/'frozen_checkpoints.json').exists() and read_json(out/'frozen_checkpoints.json') != frozen:
        raise ValueError('Checkpoints changed after held-out evaluation was started')
    save_json(out/'frozen_checkpoints.json', frozen)
    budget = Budget(ds.run)
    def run_eval(model, name, shuffle=False):
        path = out/f'{name}.jsonl'
        if path.exists():
            return
        temp = path.with_suffix('.partial')
        evaluate_tasks(model, ds, ds.test, temp, shuffle=shuffle, budget=budget)
        temp.replace(path)
    try:
        for path in checkpoints:
            model, _ = load_checkpoint(path, ds)
            run_eval(model, path.parent.name)
            if model.conditioned:
                run_eval(model, f'{path.parent.name}_shuffled', True)
                baseline, _ = load_checkpoint(path.parent/'initial.pt', ds)
                run_eval(baseline, f'{path.parent.name}_initial')
        evaluate_probes(ds, out, budget)
        save_json(out/'complete.json', dict(status='complete'))
    except BudgetExceeded:
        save_json(out/'status.json', dict(status='budget_exhausted'))
        print('Evaluation budget exhausted; final conclusions remain incomplete', flush=True)
    finally:
        budget.close()


def evaluate_probes(ds, out, budget):
    from collections import defaultdict
    from src.pilot.data import canonical
    grouped = defaultdict(list)
    for tid in ds.test:
        grouped[ds.items[tid]['state_id']].append(tid)
    prefs = ds.manifest['probe_preferences']
    permutation = np.random.default_rng(17074).permutation(len(prefs))
    for seed in (17, 29, 43):
        final_path = out/f'probes_{seed}.jsonl'
        if final_path.exists():
            continue
        conditioned, _ = load_checkpoint(ds.run/f'conditioned_{seed}/best.pt', ds)
        unconditioned, _ = load_checkpoint(ds.run/f'unconditioned_{seed}/best.pt', ds)
        partial_path = final_path.with_suffix('.partial')
        with partial_path.open('w') as f:
            for sid, ids in sorted(grouped.items()):
                if len(ids) < 2:
                    continue
                inst, state, _ = ds.get(ids[0])
                pool = {canonical(c['final_solution']): c for tid in ids for c in ds.items[tid]['teacher_candidates'] if c['is_feasible']}
                candidates = list(pool.values())
                objs = np.asarray([c['objectives'] for c in candidates])
                winners = [min(range(len(candidates)), key=lambda i: score(objs[i], w, ds.normalization)) for w in prefs]
                informative = len(set(winners)) > 1
                for k, w in enumerate(prefs):
                    for mode, model, input_w in [('normal', conditioned, w), ('shuffled', conditioned, prefs[permutation[k]]), ('unconditioned', unconditioned, w)]:
                        budget.check()
                        result = repair(model, inst, state, input_w)
                        actual = score(result['objectives'], w, ds.normalization) if result['feasible'] else None
                        f.write(json.dumps(dict(seed=seed, instance_id=inst.instance_id, state_id=sid, preference_index=k,
                                                preference=w, mode=mode, informative=informative, score=actual,
                                                pool_reference=score(objs[winners[k]], w, ds.normalization), **result))+'\n')
                        f.flush()
        partial_path.replace(final_path)
