"""Identity-bound exclusive worker lock and conservative budget recovery."""
import json
import os
from pathlib import Path
import time
import uuid
import psutil
from mo_dpr.artifacts import save_json


def matching_workers(root):
    target = os.path.normcase(str(Path(root).resolve()))
    found = []
    for p in psutil.process_iter(['pid', 'create_time', 'cmdline', 'name']):
        try:
            args = p.info['cmdline'] or []
            if p.pid == os.getpid():
                continue
            if any('emnh_v2_run.py' in x.lower() for x in args) and '--root' in args:
                candidate = os.path.normcase(str(Path(args[args.index('--root')+1]).resolve()))
                if candidate == target:
                    found.append(dict(pid=p.pid, created=p.info['create_time'], command=args,
                        children=[c.pid for c in p.children(recursive=True)]))
        except (psutil.Error, ValueError, IndexError):
            continue
    return found


def acquire(root, cfg, resume):
    root = Path(root)
    if (root/'complete.json').exists():
        raise RuntimeError('Run already complete')
    existing = any((root/name).exists() for name in ('process.json', 'binding.json', 'failure.json'))
    if existing and not resume:
        raise RuntimeError('Existing run is immutable; explicit --resume required')
    if resume and not (root/'binding.json').exists():
        raise RuntimeError('Cannot resume without bound source receipt')
    workers = matching_workers(root)
    if workers:
        raise RuntimeError(f'Another worker or solver is alive: {workers}')
    proc = psutil.Process()
    # Original receipt PID may own solver children even if command lookup differs.
    if (root/'process.json').exists():
        prior = json.loads((root/'process.json').read_text(encoding="utf-8"))
        try:
            old = psutil.Process(prior['pid'])
            if abs(old.create_time()-prior['created']) < .01 and old.pid != proc.pid:
                raise RuntimeError('Previous worker identity remains alive')
        except psutil.NoSuchProcess:
            pass
    lock = root/'worker.lock'
    if lock.exists():
        # Stale locks require an explicit recovery receipt after process audit.
        if not resume:
            raise RuntimeError('Existing worker lock')
        payload = json.loads(lock.read_text(encoding="utf-8"))
        try:
            owner = psutil.Process(payload['pid'])
            if abs(owner.create_time()-payload['created']) < .01:
                raise RuntimeError('Lock owner still alive')
        except psutil.NoSuchProcess:
            pass
        lock.rename(root/f"stale_lock_{int(time.time()*1000)}.json")
    token = dict(pid=proc.pid, created=proc.create_time(), nonce=uuid.uuid4().hex)
    fd = os.open(lock, os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        json.dump(token, f)
    if resume:
        evidence = root/'recovery'; evidence.mkdir(exist_ok=True)
        stamp = str(int(time.time()*1000))
        for name in ('process.json', 'failure.json', 'progress.json', 'health.json', 'budget.json'):
            p = root/name
            if p.exists():
                (evidence/f'{stamp}_{name}').write_bytes(p.read_bytes())
        # Charge lost work since last ledger heartbeat, including abandoned work.
        path = root/'budget.json'
        if path.exists():
            ledger = json.loads(path.read_text(encoding="utf-8"))
            end = json.loads((root/'failure.json').read_text(encoding="utf-8"))['at'] if (root/'failure.json').exists() else time.time()
            lost = max(0., end-ledger['updated'])
            key = 'meta_spent' if ledger['active_stage'] == 'meta' else 'adapt_spent'
            if ledger['active_stage'] in ('meta', 'adaptation'):
                ledger[ledger['active_method']][key] += lost
            ledger['updated'] = time.time()
            save_json(path, ledger)
        (root/'failure.json').unlink(missing_ok=True)
    save_json(root/'process.json', dict(**token, command=proc.cmdline(), root=str(root), device='cuda',
        cpu_threads=cfg.cpu_threads, started=time.time(), explicit_resume=resume, training_seeds=[cfg.seed]))
    return token


def release(root, token):
    p = Path(root)/'worker.lock'
    if p.exists() and json.loads(p.read_text(encoding="utf-8")) == token:
        p.unlink()
