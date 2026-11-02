import csv
import json
from collections import defaultdict
import numpy as np
from src.pilot.config import RUN
from src.pilot.data import read_json, save_json
from src.pilot.experiment import summary


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]


def bootstrap(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return dict(instances=len(values), mean=float(values.mean()) if len(values) else None, ci95=None)
    rng = np.random.default_rng(17075)
    means = np.array([rng.choice(values, len(values), replace=True).mean() for _ in range(2000)])
    return dict(instances=len(values), mean=float(values.mean()), ci95=np.quantile(means, [.025, .975]).tolist())


def report(run=RUN):
    out = run/'evaluation'
    complete = (out/'complete.json').exists()
    summaries, comparisons, strata = {}, {}, {}
    for path in sorted(out.glob('*.jsonl')):
        if path.name.startswith('probes'):
            continue
        records = rows(path)
        result = summary(records)
        wins, ties, losses = 0, 0, 0
        grouped = defaultdict(list)
        split_groups = defaultdict(list)
        for r in records:
            if r['score'] is not None:
                delta = r['score']-r['regret_score']
                tol = 1e-6 + 1e-5*abs(r['regret_score'])
                wins += delta < -tol; losses += delta > tol; ties += abs(delta) <= tol
                grouped[r['instance_id']].append(r['gap'])
            for field in ('destroy', 'actual_ratio', 'category'):
                split_groups[f'{field}:{r[field]}'].append(r)
        result.update(regret_wins=int(wins), regret_ties=int(ties), regret_losses=int(losses),
                      instance_gap=bootstrap([np.mean(v) for v in grouped.values()]),
                      feasible_mean_objectives=np.mean([r['objectives'] for r in records if r['feasible']], axis=0).tolist()
                      if any(r['feasible'] for r in records) else None)
        summaries[path.stem] = result
        strata[path.stem] = {k: summary(v) for k, v in split_groups.items()}
    for seed in (17, 29, 43):
        normal_path = out/f'conditioned_{seed}.jsonl'
        if not normal_path.exists():
            continue
        normal = {r['task_id']: r for r in rows(normal_path)}
        for mode in ('shuffled', 'initial', 'unconditioned'):
            other_path = out/(f'unconditioned_{seed}.jsonl' if mode == 'unconditioned' else f'conditioned_{seed}_{mode}.jsonl')
            if not other_path.exists():
                continue
            paired = defaultdict(list)
            other = rows(other_path)
            for b in other:
                a = normal.get(b['task_id'])
                if a and a['score'] is not None and b['score'] is not None:
                    paired[a['instance_id']].append(b['score']-a['score'])
            comparisons[f'{seed}_vs_{mode}'] = bootstrap([np.mean(v) for v in paired.values()])
    probes = {}
    for path in sorted(out.glob('probes_*.jsonl')):
        data = rows(path)
        keyed = {(r['state_id'], r['preference_index'], r['mode']): r for r in data}
        for mode in ('shuffled', 'unconditioned'):
            for pref_group in ('all', 0, 1, 2, 3, 4):
                paired, raw_paired, total, failed = defaultdict(list), defaultdict(list), 0, 0
                for a in data:
                    if a['mode'] != 'normal' or not a['informative'] or (pref_group != 'all' and a['preference_index'] != pref_group):
                        continue
                    b = keyed.get((a['state_id'], a['preference_index'], mode))
                    total += 1
                    if not b or a['score'] is None or b['score'] is None:
                        failed += 1; continue
                    paired[a['instance_id']].append(b['score']-a['score'])
                    raw_paired[a['instance_id']].append(np.asarray(b['objectives'])-np.asarray(a['objectives']))
                objective_ci = [bootstrap([np.mean(v, axis=0)[k] for v in raw_paired.values()]) for k in range(5)]
                probes[f'{path.stem}_{mode}_{pref_group}'] = dict(**bootstrap([np.mean(v) for v in paired.values()]), total_pairs=total, failed_pairs=failed,
                                                                raw_objective_benefits=objective_ci)
    result = dict(evaluation_complete=complete, summaries=summaries, strata=strata, comparisons=comparisons, probes=probes,
                  limitations=['Exploratory 50-customer synthetic data only.', 'Original teacher search used train+validation normalization.',
                               'Quality comparisons exclude failed pairs and must be read with failure rates.',
                               'CPU teacher timing and GPU model timing are not a hardware-matched speedup claim.'])
    feasible = all(summaries.get(f'conditioned_{s}', {}).get('feasible_rate', 0) >= .99 for s in (17, 29, 43))
    learned = all((comparisons.get(f'{s}_vs_initial', {}).get('mean') or 0) > 0 and
                  summaries.get(f'conditioned_{s}', {}).get('failures', 10**9) <=
                  summaries.get(f'conditioned_{s}_initial', {}).get('failures', -1) for s in (17, 29, 43))
    preference = all((probes.get(f'probes_{s}_shuffled_all', {}).get('ci95') or [-1])[0] > 0 and
                     probes.get(f'probes_{s}_shuffled_all', {}).get('failed_pairs', 1) == 0 and
                     (probes.get(f'probes_{s}_unconditioned_all', {}).get('mean') or 0) > 0 and
                     probes.get(f'probes_{s}_unconditioned_all', {}).get('failed_pairs', 1) == 0 for s in (17, 29, 43))
    result['decision'] = ('incomplete' if not complete else 'expand_bc' if feasible and learned and preference else
                          'repair_valid_preference_unresolved' if feasible and learned else 'revise_learning_or_rollout')
    save_json(run/'report.json', result)
    with (run/'summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f); writer.writerow(['method', 'tasks', 'failures', 'feasible_rate', 'mean_gap', 'p90_gap', 'mean_runtime'])
        for name, v in summaries.items():
            writer.writerow([name]+[v[k] for k in ('tasks', 'failures', 'feasible_rate', 'mean_gap', 'p90_gap', 'mean_runtime')])
    labels = dict(incomplete='实验未全部完成，不能判断方案通过。', expand_bc='达到本轮扩大BC的探索性条件。',
                  repair_valid_preference_unresolved='修复学习成立，但偏好能力尚未得到充分证据。',
                  revise_learning_or_rollout='闭环学习或自由修复尚未达到本轮要求。')
    text = '# RepairNet 小实验报告\n\n'+labels[result['decision']]+'\n\n'
    text += '详情见 summary.csv 与 report.json。失败率优先于可行样本上的质量均值。\n\n'
    text += '本轮仅验证50客户的探索性学习能力，不证明跨规模或跨EA通用。原教师搜索使用过含验证信息的归一化，重新评分不消除这一历史限制。\n'
    (run/'REPORT.md').write_text(text, encoding='utf-8')
    print(text, flush=True)
