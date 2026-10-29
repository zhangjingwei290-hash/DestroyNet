"""Frozen-policy SEARCH-seed replication. This entry never calls training.

Original solver/config/checkpoints stay byte-identical. Each seed is a separate
timed search; archives are never pooled across seeds. See the run's PLAN.md.
"""
import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import uuid

ARMS = ('heuristic', 'ppo')


def validate_plan(plan):
    from mo_dpr.data import fingerprint
    expected = dict(plan)
    identity = expected.pop('id', None)
    if identity != fingerprint(expected):
        raise ValueError('Replication plan changed after preparation')
    ids, seeds = plan['development_ids'], plan['search_seeds']
    if (plan['arms'] != list(ARMS) or len(ids) != 9 or len(set(ids)) != 9
            or len(seeds) != 20 or len(set(seeds)) != 20
            or seeds != list(range(plan['training_seed']+7000, plan['training_seed']+7020))
            or ids != plan['source_binding']['split']['development_ids']
            or plan['automatic_training'] or plan['automatic_next_stage']
            or plan['total_results'] != 360 or plan['new_results'] != 342
            or plan['total_solver_seconds'] != 64800. or plan['new_solver_seconds'] != 61560.):
        raise ValueError('Replication scope/seed/budget differs from authorization')


def read(path):
    for attempt in range(4):
        try:
            with Path(path).open(encoding='utf-8') as stream:
                return json.load(stream)
        except (PermissionError, json.JSONDecodeError):
            if attempt == 3:
                raise
            time.sleep(.05 * (attempt + 1))


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    from mo_dpr.artifacts import save_json
    save_json(path, value)


def result_path(root, seed, iid, arm):
    return Path(root) / 'results' / str(seed) / f'{iid}_{arm}.json'


def validate_result(row, plan, iid, arm, seed):
    import numpy as np
    from moea_emnh_v2.config import search_budget
    expected = (iid, arm, seed, plan['solver_binding_id'], search_budget(int(iid.split('_')[2])))
    actual = tuple(row.get(k) for k in ('iid', 'arm', 'seed', 'binding_id', 'budget_seconds'))
    if actual != expected:
        raise ValueError(f'Result identity/seed/budget/solver mismatch: {actual}')
    box = plan['boxes'][iid]
    for key, values in (('physical_offset', box['offset']), ('physical_scale', box['scale']),
                        ('normalized_reference', box['reference'])):
        np.testing.assert_array_equal(row[key], values)
    if (not np.isfinite(row['hv']) or row['hv'] <= 0 or not row['archive']
            or not np.isfinite(row['search_seconds']) or row['search_seconds'] < expected[-1]):
        raise ValueError('Missing/nonfinite/nonpositive HV or empty feasible archive')


def verify_result(row, env):
    """Offline route/objective/HV verification, outside every solver clock."""
    import moocore
    import numpy as np
    for point in row['archive']:
        np.testing.assert_allclose(env.evaluate(point['routes']), point['objectives'], atol=1e-7, rtol=0)
    points = (np.asarray([p['objectives'] for p in row['archive']]) - env.offset) / env.scales
    ref = np.asarray(env.box['reference'])
    if not np.isfinite(points).all() or np.any(points > ref):
        raise ValueError('Archive exceeds frozen reference or contains nonfinite values')
    hv = float(moocore.hypervolume(points, ref=ref))
    np.testing.assert_allclose(hv, row['hv'], atol=1e-12, rtol=1e-10)
    return dict(archive_solutions=len(points), recomputed_hv=hv,
                max_normalized_objectives=points.max(axis=0).tolist())


def paired_statistics(delta, gap, samples=10000):
    """Rows=IID, columns=shared seed blocks. Preserve method pairing.

    Fixed-nine-IID CI resamples seed columns; exploratory crossed CI resamples
    IID rows and seed columns independently. Neither includes training variance.
    """
    import numpy as np
    delta, gap = np.asarray(delta, dtype=float), np.asarray(gap, dtype=float)
    if delta.shape != gap.shape or delta.ndim != 2 or min(delta.shape) < 1:
        raise ValueError('Expected matching nonempty IID-by-seed matrices')
    if not np.isfinite(delta).all() or not np.isfinite(gap).all():
        raise ValueError('Nonfinite paired metric')
    ni, ns = delta.shape
    rng = np.random.default_rng(20261021)
    sj = rng.integers(ns, size=(samples, ns))
    ij = rng.integers(ni, size=(samples, ni))
    def intervals(matrix):
        if ns < 2:
            return dict(fixed_nine_iid_seed_bootstrap_95_ci=None, crossed_iid_seed_bootstrap_95_ci=None)
        conditional = matrix.mean(axis=0)[sj].mean(axis=1)
        crossed = matrix[ij[:, :, None], sj[:, None, :]].mean(axis=(1, 2))
        return dict(fixed_nine_iid_seed_bootstrap_95_ci=np.quantile(conditional, [.025, .975]).tolist(),
                    crossed_iid_seed_bootstrap_95_ci=np.quantile(crossed, [.025, .975]).tolist())
    return dict(mean_delta_hv=float(delta.mean()), mean_relative_hv_loss_percent=float(gap.mean()),
        delta_hv_intervals=intervals(delta), relative_loss_intervals=intervals(gap),
        search_seed_count=ns, distinct_iids=ni,
        per_seed_iid_mean_relative_loss_percent=gap.mean(axis=0).tolist(),
        fraction_pairs_loss_at_most_1_percent=float((gap <= 1).mean()),
        fraction_pairs_absolute_gap_at_most_1_percent=float((abs(gap) <= 1).mean()),
        wins_ties_losses=[int((delta > 0).sum()), int((delta == 0).sum()), int((delta < 0).sum())],
        equivalence_or_noninferiority_claim=False, training_seeds=1,
        bootstrap_samples=samples, uncertainty_excludes_training_seed_variance=True)


def summarize(root, plan):
    import numpy as np
    ids = plan['development_ids']
    complete_seeds = [s for s in plan['search_seeds'] if all(result_path(root, s, i, a).exists() for i in ids for a in ARMS)]
    if not complete_seeds:
        return None
    rows, by = [], {}
    for iid in ids:
        for seed in complete_seeds:
            pair = {a: read(result_path(root, seed, iid, a)) for a in ARMS}
            for arm, row in pair.items():
                validate_result(row, plan, iid, arm, seed)
                by[iid, seed, arm] = row
                h = pair['heuristic']['hv']
                rows.append(dict(iid=iid, seed=seed, arm=arm, hv=row['hv'],
                    relative_hv_loss_percent=100*(h-row['hv'])/h,
                    search_seconds=row['search_seconds'], inference_seconds=row['inference_seconds'],
                    actions=len(row['records']), archive_solutions=len(row['archive'])))
    delta = [[by[i,s,'ppo']['hv']-by[i,s,'heuristic']['hv'] for s in complete_seeds] for i in ids]
    gap = [[100*(by[i,s,'heuristic']['hv']-by[i,s,'ppo']['hv'])/by[i,s,'heuristic']['hv'] for s in complete_seeds] for i in ids]
    report = dict(updated=time.time(), planned_search_seeds=20, complete_search_seeds=complete_seeds,
        complete=len(complete_seeds)==20, **paired_statistics(delta, gap),
        per_iid={}, mean_hv_by_arm={a:float(np.mean([by[i,s,a]['hv'] for i in ids for s in complete_seeds])) for a in ARMS},
        scope='Frozen one-training-seed policy, nine development IIDs, repeated stochastic search; incomplete training coverage remains')
    report['execution_costs'] = {}
    report['mean_hv_time_curve'] = {}
    report['initial_hv_and_gain'] = {}
    import moocore
    for arm in ARMS:
        group = [by[i,s,arm] for i in ids for s in complete_seeds]
        records = [rec for row in group for rec in row['records']]
        report['execution_costs'][arm] = dict(search_seconds=sum(r['search_seconds'] for r in group),
            inference_seconds=sum(r['inference_seconds'] for r in group), actions=len(records),
            accepted=sum(r['accepted'] for r in records), timeouts=sum(r['timeout'] for r in records),
            failed=sum(not r['feasible'] and not r['timeout'] for r in records),
            repair_seconds=sum(r['repair_seconds'] for r in records))
        report['mean_hv_time_curve'][arm] = [dict(fraction=f,
            hv=float(np.mean([r['hv_time_curve'][j]['hv'] for r in group])))
            for j,f in enumerate((.1,.25,.5,.75,1.))]
        gains = []
        for row in group:
            initial = row['point_events'][:16]
            if len(initial)!=16 or any(p['seconds']>row['initial_seconds'] for p in initial):
                raise ValueError('Cannot identify initialization before local search')
            z=(np.asarray([p['objectives'] for p in initial])-row['physical_offset'])/row['physical_scale']
            hv0=float(moocore.hypervolume(z,ref=row['normalized_reference']))
            gains.append((hv0,row['hv']-hv0,100*(row['hv']-hv0)/hv0))
        report['initial_hv_and_gain'][arm] = dict(mean_initial_hv=float(np.mean([r[0] for r in gains])),
            mean_absolute_hv_gain=float(np.mean([r[1] for r in gains])),
            mean_relative_hv_gain_percent=float(np.mean([r[2] for r in gains])))
    for idx, iid in enumerate(ids):
        report['per_iid'][iid] = dict(mean_relative_loss_percent=float(np.mean(gap[idx])),
            std_relative_loss_percent=float(np.std(gap[idx], ddof=1)) if len(complete_seeds)>1 else None,
            min_relative_loss_percent=float(np.min(gap[idx])), max_relative_loss_percent=float(np.max(gap[idx])),
            mean_hv={a:float(np.mean([by[iid,s,a]['hv'] for s in complete_seeds])) for a in ARMS},
            std_hv={a:float(np.std([by[iid,s,a]['hv'] for s in complete_seeds], ddof=1)) if len(complete_seeds)>1 else None for a in ARMS})
    save(root/'summary.json', report)
    with (root/'per_seed_results.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lines = ['# 冻结 PPO：20 搜索种子稳定性', '',
        f"完整种子 {len(complete_seeds)}/20；9 个固定研发 IID。所有结果按单次预算计算，不合并跨种子档案。", '',
        f"平均相对 HV 损失（正数落后启发式）：{report['mean_relative_hv_loss_percent']:.6f}%。", '',
        f"固定九 IID、按种子配对重采样95%区间：{report['relative_loss_intervals']['fixed_nine_iid_seed_bootstrap_95_ci']}。", '',
        f"探索性 IID×种子交叉重采样95%区间：{report['relative_loss_intervals']['crossed_iid_seed_bootstrap_95_ci']}。", '',
        '仅一个训练种子；训练覆盖不足仍然存在。这里不证明训练稳定、元学习贡献、等效性或最终泛化。未设可接受差距门槛，不自动判通过。', '',
        '| IID | 平均损失% | 标准差% | 最小损失% | 最大损失% |', '|---|---:|---:|---:|---:|']
    for iid, row in report['per_iid'].items():
        sd = '—' if row['std_relative_loss_percent'] is None else f"{row['std_relative_loss_percent']:.4f}"
        lines.append(f"| {iid} | {row['mean_relative_loss_percent']:.4f} | {sd} | {row['min_relative_loss_percent']:.4f} | {row['max_relative_loss_percent']:.4f} |")
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return report


def prepare(root, source):
    from moea_emnh_v2.config import Config
    from moea_emnh_v2.runner import setup, binding
    from mo_dpr.data import fingerprint
    from mo_dpr.artifacts import save_json
    if (root/'plan.json').exists():
        raise RuntimeError('Existing plan is immutable')
    cfg = Config(**read(source/'config.json'))
    pool, split = setup(cfg)
    old = read(source/'binding.json'); current = binding(cfg, split)
    if fingerprint(current) != old['id'] or not (source/'complete.json').exists():
        raise RuntimeError('Completed source campaign binding changed')
    weights = {str(source/'ppo'/'directions'/f'direction_{k:02d}.pt'): digest(source/'ppo'/'directions'/f'direction_{k:02d}.pt') for k in range(16)}
    seeds = list(range(cfg.seed+7000, cfg.seed+7000+20))
    root.mkdir(parents=True, exist_ok=True)
    for iid in split['development_ids']:
        pool.load(iid)
    plan = dict(schema='frozen_ppo_search_seed_stability_v1', approved='2026-10-21 user: 20 search seeds; PPO plus heuristic',
        created=time.time(), source_root=str(source), solver_binding_id=old['id'], training_seed=cfg.seed,
        search_seeds=seeds, development_ids=split['development_ids'], arms=list(ARMS),
        boxes={i:pool.cfg['boxes_by_instance'][i] for i in split['development_ids']},
        checkpoint_files_sha256=weights, source_binding=old, wrapper_sha256=digest(__file__),
        reused_seed=seeds[0], total_results=360, reused_results=18, new_results=342,
        total_solver_seconds=64800., new_solver_seconds=61560., automatic_training=False,
        automatic_next_stage=False, decode='unchanged stochastic', seed_scope='search only; Config.seed stays training seed',
        summary='paired each IID/seed; equal IID weight, equal seed weight; no best-seed selection or cross-seed archive pooling')
    plan['id'] = fingerprint(plan)
    validate_plan(plan)
    save_json(root/'plan.json', plan)
    (root/'PLAN.md').write_text('''# 冻结 PPO 的 20 个搜索 seed 稳定性验证

用户2026-10-21批准20搜索种子，并选择PPO＋启发式。保留现有一个训练种子权重；不训练、不适配、不读原DEV18/holdout/正式45。

沿用9个研发IID、16方向、四修复器、ALNS历史、随机解码、原物理目标/约束/归一化/参考点/严格方向接受及同一Search类。每次50/150/250客户总60/180/300秒，含推理/修复/初始化/外层。GPU推理、CPU1线程，顺序运行不竞争计时预算。原科学契约源码不修改。

总20个搜索seed=20268020..20268039。首个seed复用并重核原18份结果，原始结果字节保留；新增19个seed共342次、61560秒=17小时6分搜索。总计360份、18小时名义搜索；离线审核/IO/预热另外披露。按seed与IID轮换方法先后顺序。同数字种子形成配对区组，不声称策略分歧后每次随机调用完全相同。

每次求解独立初始化真实种群、档案、ALNS、GRU，Torch/CUDA随机按搜索seed重置；Config.seed仍是训练seed，不改变权重契约。仅加载原不可变16方向文件，检查源码/依赖/配置/数据及权重绑定；每次结果落盘可续跑。失败未落盘的单次搜索不得伪造结果，重做需登记额外失败成本。完成缓存只有通过身份/预算/参考点/可行性/HV重核才可复用。

交付每IID每方法20次HV均值/标准差、配对相对损失分布/最坏值、时间曲线、失败/超时/推理/修复成本。先形成IID×seed配对差矩阵，再等权统计。固定9IID按种子区组bootstrap10000次为执行稳定性区间；IID和seed独立交叉重采样区间另报。不是180独立IID，不包含训练种子方差，不合并20份档案，不选最佳seed。原训练覆盖不足继续披露。可接受gap阈值未预设，不能看结果后自动宣称等效/非劣。

每30分钟轻量监控，只重要阶段/故障/恢复/ETA变化通知，不调用子agent。明确失败才保存现场、核对无同worker及所属子进程，最多一次按同入口--resume续评估；不得训练、修改科学规则或停止无关任务。完成报告后暂停回访，不自动追加实验。
''', encoding='utf-8')
    print(json.dumps({k:plan[k] for k in ('search_seeds','total_results','reused_results','new_results','new_solver_seconds')}, ensure_ascii=False))


def matching(root):
    import psutil
    found = []
    target = os.path.normcase(str(root.resolve()))
    for proc in psutil.process_iter(['pid','create_time','cmdline']):
        try:
            args = proc.info['cmdline'] or []
            if proc.pid == os.getpid() or not any(Path(x).name.lower()=='emnh_seed_stability.py' for x in args):
                continue
            if '--root' in args and '--mode' in args and args[args.index('--mode')+1]=='run' and os.path.normcase(str(Path(args[args.index('--root')+1]).resolve())) == target:
                found.append(dict(pid=proc.pid, created=proc.info['create_time'], command=args,
                    children=[dict(pid=c.pid, created=c.create_time(), command=c.cmdline()) for c in proc.children(recursive=True)]))
        except (psutil.Error, ValueError, IndexError):
            continue
    return found


def run(root, resume=False):
    import psutil
    if (root/'complete.json').exists() or matching(root):
        raise RuntimeError('Complete run or existing worker; duplicate launch refused')
    plan = read(root/'plan.json')
    validate_plan(plan)
    if digest(__file__) != plan['wrapper_sha256']:
        raise RuntimeError('Replication wrapper changed')
    if (root/'process.json').exists() and not resume:
        raise RuntimeError('Prior process requires explicit --resume')
    lock = root/'worker.lock'
    if lock.exists():
        owner = read(lock)
        if psutil.pid_exists(owner['pid']) and abs(psutil.Process(owner['pid']).create_time()-owner['created']) < .01:
            raise RuntimeError('Lock owner alive')
        if not resume:
            raise RuntimeError('Stale lock requires audited recovery')
        lock.rename(root/f"stale_lock_{int(time.time()*1000)}.json")
    me = psutil.Process()
    token = dict(pid=me.pid, created=me.create_time(), nonce=uuid.uuid4().hex)
    with lock.open('x', encoding='utf-8') as stream:
        json.dump(token, stream)
    try:
        import torch
        import numpy as np
        from mo_dpr.data import fingerprint
        from moea_emnh_v2.config import Config
        from moea_emnh_v2.runner import setup, binding, load_directions, save_development_result
        from moea_emnh_v2.environment import Environment, Adaptive
        from moea_emnh_v2.outer import Search
        save(root/'process.json', dict(**token, command=me.cmdline(), training_seeds=[plan['training_seed']],
            search_seeds=plan['search_seeds'], cpu_threads=1, device='cuda', resume=resume))
        if resume:
            for name in ('failure.json','progress.json'):
                path=root/name
                if path.exists():
                    (root/'recovery').mkdir(exist_ok=True)
                    path.rename(root/'recovery'/f'{int(time.time()*1000)}_{name}')
        source = Path(plan['source_root']); cfg = Config(**read(source/'config.json'))
        pool, split = setup(cfg)
        if fingerprint(binding(cfg, split)) != plan['solver_binding_id']:
            raise RuntimeError('Solver/source/dependency/data contract changed')
        if {p:digest(p) for p in plan['checkpoint_files_sha256']} != plan['checkpoint_files_sha256']:
            raise RuntimeError('Frozen checkpoint bytes changed')
        # Preload shared input files outside ALL per-arm solver clocks.
        for iid in plan['development_ids']:
            pool.load(iid)
        policies = load_directions(source/'ppo', cfg, plan['solver_binding_id'])
        warm = Environment(pool, plan['development_ids'][0], cfg, np.full(5,.2))
        obs = warm.observation(torch.zeros(1,128), 'greedy', Adaptive(len(warm.q_values)), 60.,60.,10.,10.)
        with torch.no_grad():
            policies[0].act([obs])
        torch.cuda.synchronize()
        completed = {}
        audit = {}
        # Reuse only the original first seed, with explicit provenance and proof.
        for seed in plan['search_seeds']:
            for iid in plan['development_ids']:
                for arm in ARMS:
                    target = result_path(root, seed, iid, arm)
                    original = source/'development'/f'{iid}_{arm}.json'
                    candidate = target if target.exists() else original if seed == plan['reused_seed'] else None
                    if candidate is None:
                        continue
                    row = read(candidate); validate_result(row, plan, iid, arm, seed)
                    env = Environment(pool, iid, cfg, np.full(5,.2))
                    proof = verify_result(row, env)
                    if candidate != target:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(candidate.read_bytes())
                    key = f'{seed}/{iid}/{arm}'
                    audit[key] = dict(origin=str(candidate), sha256=digest(target), **proof)
                    completed[key] = row['budget_seconds']
        save(root/'verification.json', audit)
        def notify(phase, **extra):
            pending = plan['total_solver_seconds']-sum(completed.values())
            payload = dict(phase=phase, updated=time.time(), pid=me.pid, completed_results=len(completed),
                total_results=360, completed_solver_budget_seconds=sum(completed.values()),
                remaining_solver_budget_seconds=pending, nominal_eta_epoch=time.time()+pending,
                training_performed=False, **extra)
            save(root/'progress.json', payload); save(root/'health.json', dict(phase=phase, updated=payload['updated'], pid=me.pid))
            return payload
        notify('evaluation'); summarize(root, plan)
        for si, seed in enumerate(plan['search_seeds']):
            for ii, iid in enumerate(plan['development_ids']):
                order = ARMS if (si+ii)%2==0 else ARMS[::-1]
                for arm in order:
                    key=f'{seed}/{iid}/{arm}'
                    if key in completed:
                        continue
                    path = result_path(root, seed, iid, arm)
                    notify('evaluation', current=dict(seed=seed,iid=iid,arm=arm))
                    save(root/'active_attempt.json',dict(seed=seed,iid=iid,arm=arm,started=time.time(),pid=me.pid))
                    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                    # Same Search + stochastic policy as original evaluation.
                    with torch.no_grad():
                        search = Search(pool, iid, cfg, seed, policies if arm=='ppo' else None)
                        last_notify = [time.perf_counter()]
                        def progress(s):
                            if time.perf_counter()-last_notify[0] >= 30:
                                notify('evaluation', current=dict(seed=seed,iid=iid,arm=arm,
                                    elapsed=time.perf_counter()-s.started, actions=len(s.records)))
                                last_notify[0]=time.perf_counter()
                        search.run(progress=progress)
                        row=dict(arm=arm,binding_id=plan['solver_binding_id'],**search.report())
                    validate_result(row,plan,iid,arm,seed)
                    proof=verify_result(row,search.base)
                    save_development_result(path,row)
                    audit[key]=dict(origin='new_search',sha256=digest(path),**proof)
                    completed[key]=row['budget_seconds']
                    save(root/'verification.json',audit)
                    print(json.dumps(notify('evaluation',last_completed=dict(seed=seed,iid=iid,arm=arm,hv=row['hv'])),ensure_ascii=False),flush=True)
                    (root/'active_attempt.json').unlink(missing_ok=True)
                    del search
            summarize(root,plan)
        report=summarize(root,plan)
        if len(completed)!=360 or not report['complete']:
            raise RuntimeError('Incomplete replication matrix')
        save(root/'complete.json',dict(at=time.time(),results=360,search_seeds=20,training_seeds=1,
            final_report=str(root/'REPORT.md'),automatic_next_stage=False))
        notify('complete')
    except BaseException:
        failed_at=time.time()
        attempt=read(root/'active_attempt.json') if (root/'active_attempt.json').exists() else None
        save(root/'failure.json',dict(at=failed_at,traceback=traceback.format_exc(),active_attempt=attempt,
            failed_attempt_wall_seconds=0. if attempt is None else max(0.,failed_at-attempt['started']),
            instruction='Audit worker/children and failed search cost before explicit resume; never retrain'))
        raise
    finally:
        if lock.exists() and read(lock)==token:
            lock.unlink()


def monitor(root):
    import psutil
    data={name:read(root/name) if (root/name).exists() else None for name in
          ('process.json','progress.json','health.json','failure.json','complete.json')}
    proc=data['process.json']; identity=None
    if proc:
        try:
            p=psutil.Process(proc['pid'])
            identity=dict(pid=p.pid,created=p.create_time(),matches=abs(p.create_time()-proc['created'])<.01,
                command=p.cmdline(),cpu_seconds=sum(p.cpu_times()[:2]),children=[c.pid for c in p.children(recursive=True)])
        except psutil.NoSuchProcess:
            pass
    data['actual_process']=identity
    data['matching_workers']=matching(root)
    log=root/'worker.log'
    if log.exists():
        with log.open('rb') as stream:
            stream.seek(max(0,log.stat().st_size-4000)); data['log_tail']=stream.read().decode('utf-8',errors='replace')
    print(json.dumps(data,ensure_ascii=False,allow_nan=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('prepare','run','monitor'),required=True)
    parser.add_argument('--root',required=True);parser.add_argument('--source')
    parser.add_argument('--resume',action='store_true');parser.add_argument('--log',action='store_true')
    args=parser.parse_args();root=Path(args.root).resolve()
    if args.mode=='prepare':
        if not args.source: parser.error('--source required')
        prepare(root,Path(args.source).resolve())
    elif args.mode=='monitor': monitor(root)
    elif args.log:
        with (root/'worker.log').open('a',encoding='utf-8',buffering=1) as log:
            sys.stdout=sys.stderr=log;run(root,args.resume)
    else:run(root,args.resume)


if __name__=='__main__':
    main()
