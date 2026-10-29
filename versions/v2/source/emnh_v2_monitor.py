"""Lightweight read-only experiment status; no launches or algorithm changes."""
import argparse
import json
from pathlib import Path
import time
import psutil


def read(path):
    if not path.exists(): return None
    for attempt in range(3):
        try:
            return json.loads(path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            if attempt == 2: raise
            time.sleep(.05*(attempt+1))


def status(root):
    import os
    proc = read(root/'process.json')
    live = []
    target = os.path.normcase(str(root.resolve()))
    for candidate in psutil.process_iter(['pid', 'create_time', 'cmdline']):
        try:
            args = candidate.info['cmdline'] or []
            if '--root' in args and any('emnh_v2_run.py' in x.lower() for x in args):
                run = os.path.normcase(str(Path(args[args.index('--root')+1]).resolve()))
                if run == target:
                    live.append(dict(pid=candidate.pid, created=candidate.info['create_time'],
                                     command=args, children=[c.pid for c in candidate.children(recursive=True)]))
        except (psutil.Error, ValueError, IndexError):
            pass
    identity = False
    cpu = None
    if proc:
        try:
            p = psutil.Process(proc['pid'])
            identity = abs(p.create_time()-proc['created']) < .01
            if identity:
                c = p.cpu_times(); cpu = c.user+c.system
        except psutil.NoSuchProcess:
            pass
    progress = read(root/'progress.json') or {}
    budget = read(root/'budget.json') or {}
    reports = {}
    for method in ('ppo', 'reinforce'):
        r = read(root/method/'training_report.json')
        if r:
            reports[method] = {k:r.get(k) for k in ('meta_seconds','adaptation_seconds','updates',
                'optimized_meta_interactions','optimized_meta_iids','missing_fit_iids','unadapted_directions',
                'training_coverage_passed')}
    result = dict(checked=time.time(), process_receipt=proc, matching_workers=live,
        receipt_identity_alive=identity, cpu_seconds=cpu, phase=progress.get('phase'),
        progress={k:progress.get(k) for k in ('updated','method','round','updates','training_actions',
            'optimized_iids','iid','completed_tasks','target_tasks','actions','completed_directions',
            'direction','interactions','meta_spent_seconds','adaptation_spent_seconds')},
        budget=budget, training_reports=reports, failure=read(root/'failure.json'),
        complete=read(root/'complete.json'))
    log = root/'worker.log'
    if log.exists():
        with log.open('rb') as f:
            f.seek(max(0, log.stat().st_size-7000))
            result['worker_log_tail'] = f.read().decode('utf-8', errors='replace').splitlines()[-8:]
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True)
    args = p.parse_args()
    print(json.dumps(status(Path(args.root)), ensure_ascii=False, allow_nan=False, indent=2))
