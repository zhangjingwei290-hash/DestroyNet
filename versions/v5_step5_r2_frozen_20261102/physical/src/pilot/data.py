"""Read original shards, freeze splits, and create independent pilot labels."""
import gzip
import hashlib
import json
import platform
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from src.data_generation.sharded import _instance_from_row, _solution
from src.pilot.config import CONFIG, ROOT, SOURCE, RUN, SPLIT_SEED, PREFERENCE_SEED


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_rows(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def canonical(routes):
    return tuple(sorted(tuple(r) for r in routes if r))


def score(values, weights, normalization):
    z = (np.asarray(values) - normalization['mins']) / np.asarray(normalization['ranges'])
    x = z * np.asarray(weights)
    return float(x.max() + CONFIG['rho'] * x.sum())


def select_teacher(task, normalization):
    feasible = [c for c in task['teacher_candidates'] if c['is_feasible']]
    if not feasible:
        raise ValueError(f"No feasible teacher: {task['task_id']}")
    return min(feasible, key=lambda c: (score(c['objectives'], task['preference_weights'], normalization), c['teacher_name']))


def load_shard(path):
    rows = list(read_rows(path))
    instances = [r for r in rows if r['kind'] == 'instance']
    if len(instances) != 1:
        raise ValueError(f'Expected one instance in {path}')
    return instances[0], {r['state_id']: r for r in rows if r['kind'] == 'state'}, [r for r in rows if r['kind'] == 'task']


class PilotDataset:
    def __init__(self, run=RUN):
        self.run = Path(run)
        self.manifest = read_json(self.run / 'pilot_manifest.json')
        for relative, expected in self.manifest['code_hashes'].items():
            if sha(ROOT/relative) != expected:
                raise ValueError(f'Code changed after prepare: {relative}; use the frozen snapshot or a new run')
        self.normalization = self.manifest['normalization']
        self.items = {}
        self.instances = {}
        self.states = {}
        for split, files in self.manifest['splits'].items():
            ids = []
            for file in files:
                source_file = Path(self.manifest['source']) / file
                if sha(source_file) != self.manifest['source_hashes'][file]:
                    raise ValueError(f'Source changed after prepare: {source_file}')
                row, states, tasks = load_shard(source_file)
                self.instances[row['instance_id']] = _instance_from_row(row)
                self.states.update(states)
                for task in tasks:
                    task = dict(task, instance_id=row['instance_id'], split=split)
                    task['label'] = select_teacher(task, self.normalization)
                    self.items[task['task_id']] = task
                    ids.append(task['task_id'])
            setattr(self, split, sorted(ids))

    def get(self, task_id):
        task = self.items[task_id]
        return self.instances[task['instance_id']], self.states[task['state_id']], task


def prepare(source=SOURCE, run=RUN):
    source, run = Path(source), Path(run)
    if run.exists():
        raise FileExistsError(f'Use a new RUN directory; refusing to overwrite {run}')
    manifest = read_json(source / 'results/manifest.json')
    groups = defaultdict(list)
    # The first row carries instance metadata; do not load the multi-GB index.
    for rel in sorted(manifest['files']):
        if 'worker_' not in rel:
            continue
        path = source / rel
        first = next(read_rows(path))
        if first['n_customers'] == 50:
            groups[(first['split'], first['instance_type'])].append((first['instance_id'], rel))
    rank = lambda pair: hashlib.sha256(f'{SPLIT_SEED}:{pair[0]}'.encode()).hexdigest()
    counts = {k: len(v) for k, v in groups.items()}
    available = min(counts.get(('train', k), 0) for k in ('R', 'C', 'RC'))
    ntrain = min(60, available * 6 // 7)
    ndev = min(10, available - ntrain)
    ntest = min(20, *(counts.get(('validation', k), 0) for k in ('R', 'C', 'RC')))
    if min(ntrain, ndev, ntest) < 1:
        raise ValueError(f'Insufficient stratified instances: {counts}')
    splits = dict(train=[], dev=[], test=[])
    for kind in ('R', 'C', 'RC'):
        rows = sorted(groups['train', kind], key=rank)
        splits['train'] += [p for _, p in rows[:ntrain]]
        splits['dev'] += [p for _, p in rows[ntrain:ntrain + ndev]]
        splits['test'] += [p for _, p in sorted(groups['validation', kind], key=rank)[:ntest]]
    values, task_ids, state_ids, hashes, instance_ids = [], {}, {}, {}, {}
    for split, files in splits.items():
        task_ids[split], state_ids[split], instance_ids[split] = [], [], []
        for rel in files:
            actual = sha(source / rel)
            if actual.lower() != manifest['files'][rel].lower():
                raise ValueError(f'Source hash mismatch: {rel}')
            hashes[rel] = actual
            row, states, tasks = load_shard(source / rel)
            instance_ids[split].append(row['instance_id'])
            task_ids[split] += [t['task_id'] for t in tasks]
            state_ids[split] += list(states)
            if split == 'train':
                values.append(row['source_objectives'])
                values += [c['objectives'] for t in tasks for c in t['teacher_candidates'] if c['is_feasible']]
    values = np.asarray(values)
    normalization = dict(mins=values.min(0).tolist(), ranges=np.maximum(values.max(0) - values.min(0), 1).tolist(), clipped=False)
    run.mkdir(parents=True)
    rng = np.random.default_rng(PREFERENCE_SEED)
    mixed = []
    while len(mixed) < 10:
        w = rng.dirichlet(np.ones(5))
        if .3 <= w.max() < .5:
            mixed.append(w.tolist())
    preferences = np.eye(5).tolist() + [[.2] * 5] + mixed
    snapshot = run / 'source_snapshot'
    shutil.copytree(ROOT / 'src', snapshot / 'src', ignore=shutil.ignore_patterns('__pycache__'))
    code_hashes = {str(p.relative_to(snapshot)): sha(p) for p in snapshot.rglob('*.py')}
    resolved = dict(source=str(source.resolve()), source_manifest_sha256=sha(source / 'results/manifest.json'),
                    source_protocol_hash=manifest['protocol_hash'], splits=splits, instance_ids=instance_ids,
                    task_ids=task_ids, state_ids=state_ids, source_hashes=hashes, normalization=normalization,
                    config=CONFIG, split_seed=SPLIT_SEED, preference_seed=PREFERENCE_SEED,
                    probe_preferences=preferences, code_hashes=code_hashes,
                    code_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    git_status=subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
                    limitation='Exploratory: original teacher search used train+validation normalization.')
    save_json(run / 'pilot_manifest.json', resolved)
    ds = PilotDataset(run)
    labels, changed, strata, shared = {}, 0, defaultdict(Counter), []
    for tid, task in ds.items.items():
        labels[tid] = task['label']['teacher_name']
        changed += task['selected_teacher'] != labels[tid]
        state = ds.states[task['state_id']]
        strata[task['split']][f"{state['destroy_type']}|{len(state['removed_customers'])/50:.2f}|{task['preference_category']}"] += 1
    by_state = defaultdict(list)
    for task in ds.items.values():
        by_state[task['state_id']].append(task)
    for sid, tasks in by_state.items():
        if len(tasks) < 2:
            continue
        pool = {canonical(c['final_solution']): c for t in tasks for c in t['teacher_candidates'] if c['is_feasible']}
        candidates = list(pool.values())
        objs = np.asarray([c['objectives'] for c in candidates])
        nd = [i for i, a in enumerate(objs) if not any(np.all(b <= a) and np.any(b < a) for b in objs)]
        chosen = [min(range(len(candidates)), key=lambda i: score(objs[i], w, normalization)) for w in preferences]
        shared.append(dict(state_id=sid, split=tasks[0]['split'], instance_id=tasks[0]['instance_id'],
                           unique_solutions=len(pool), nondominated=len(nd), objective_range=np.ptp(objs, axis=0).tolist(),
                           distinct_preference_winners=len(set(chosen))))
    # Greedily cover observed destroy/category strata, then fill deterministically.
    ordered = sorted(ds.train, key=lambda t: hashlib.sha256(f'17:{t}'.encode()).hexdigest())
    chosen, covered = [], set()
    for tid in ordered:
        _, st, task = ds.get(tid)
        keys = {('destroy', st['destroy_type']), ('pref', task['preference_category'])}
        if keys - covered:
            chosen.append(tid)
            covered |= keys
    chosen += [t for t in ordered if t not in chosen][:max(0, 64-len(chosen))]
    save_json(run / 'overfit_tasks.json', chosen[:64])
    save_json(run / 'labels.json', labels)
    save_json(run / 'label_diagnostics.json', dict(changed_labels=changed, tasks=len(labels), strata=dict(strata), shared_states=shared))
    save_json(run / 'environment.json', dict(python=platform.python_version(), platform=platform.platform(), packages=subprocess.check_output(['python', '-m', 'pip', 'freeze'], text=True).splitlines()))
    print(f'Prepared {len(labels)} tasks at {run}', flush=True)
    return resolved
