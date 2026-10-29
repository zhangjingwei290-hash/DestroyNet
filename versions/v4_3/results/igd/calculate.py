"""Offline IGD/IGD+ of completed F archives; no solver imports or training."""
import csv
import json
from pathlib import Path

import moocore
import numpy as np

SOURCE = Path(__file__).resolve().parents[2] / 'experiments/v4_3_continuous_cudagraph_20261025/runs/full_closedloop_20261025_f_continuous_cudagraph_5seed'
OUT = Path(__file__).resolve().parent


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    assert json.loads((SOURCE / 'complete.json').read_text())['completed']
    reports = [json.loads(p.read_text()) for p in sorted(SOURCE.glob('*_seed*_*.json')) if p.name.endswith(('_ppo.json', '_heuristic.json'))]
    assert len(reports) == 90
    lookup = {(d['iid'], d['search_seed'], d['method']): d for d in reports}
    assert len(lookup) == 90
    iids = sorted({d['iid'] for d in reports})
    seeds = sorted({d['search_seed'] for d in reports})
    assert len(iids) == 9 and len(seeds) == 5
    for iid in iids:
        for seed in seeds:
            a, b = [lookup[iid, seed, method] for method in ('heuristic', 'ppo')]
            for key in ('physical_offset', 'physical_scale', 'normalized_reference', 'initial_population_objectives'):
                assert a[key] == b[key], (iid, seed, key)
    rows, references = [], {}
    for iid in iids:
        group = [d for d in reports if d['iid'] == iid]
        sets = [np.asarray([p['objectives'] for p in d['archive']], dtype=float) for d in group]
        assert all(a.ndim == 2 and a.shape[1] == 5 and np.isfinite(a).all() for a in sets)
        union = np.unique(np.vstack(sets), axis=0)
        reference = moocore.filter_dominated(union)
        ideal, nadir = reference.min(0), reference.max(0)
        span = nadir - ideal
        # No arbitrary epsilon or removal of objectives: fail if a protocol decision is needed.
        if np.any(span <= 0):
            raise ValueError(f'{iid}: zero reference-front range; define treatment explicitly')
        references[iid] = dict(unique_union_points=len(union), reference_points=len(reference), ideal=ideal.tolist(), nadir=nadir.tolist(), span=span.tolist())
        write_csv(OUT / f'{iid}_reference.csv', [dict(zip(('vehicles', 'distance', 'makespan', 'waiting', 'tardiness'), p.tolist())) for p in reference])
        for normalization in ('empirical_ideal_nadir', 'fixed_physical'):
            offset = ideal if normalization == 'empirical_ideal_nadir' else np.array(group[0]['physical_offset'])
            scale = span if normalization == 'empirical_ideal_nadir' else np.array(group[0]['physical_scale'])
            ref = (reference - offset) / scale
            for d, points in zip(group, sets):
                normalized = (points - offset) / scale
                rows.append(dict(normalization=normalization, iid=iid, search_seed=d['search_seed'], method=d['method'], archive_points=len(points), reference_points=len(reference), igd=float(moocore.igd(normalized, ref=ref)), igd_plus=float(moocore.igd_plus(normalized, ref=ref))))
    write_csv(OUT / 'per_run.csv', rows)
    summaries, per_iid = {}, []
    for normalization in ('empirical_ideal_nadir', 'fixed_physical'):
        subset = [r for r in rows if r['normalization'] == normalization]
        summaries[normalization] = {}
        for metric in ('igd', 'igd_plus'):
            paired = []
            for iid in iids:
                hv = np.array([next(r[metric] for r in subset if r['iid'] == iid and r['search_seed'] == seed and r['method'] == 'heuristic') for seed in seeds])
                pv = np.array([next(r[metric] for r in subset if r['iid'] == iid and r['search_seed'] == seed and r['method'] == 'ppo') for seed in seeds])
                reduction = 100 * (hv - pv) / hv
                paired.append(reduction)
                per_iid.append(dict(normalization=normalization, metric=metric, iid=iid, heuristic_mean=float(hv.mean()), heuristic_std=float(hv.std(ddof=1)), ppo_mean=float(pv.mean()), ppo_std=float(pv.std(ddof=1)), mean_paired_reduction_percent=float(reduction.mean()), ppo_wins=int((pv < hv).sum()), ties=int((pv == hv).sum()), ppo_losses=int((pv > hv).sum())))
            hmean = float(np.mean([r[metric] for r in subset if r['method'] == 'heuristic']))
            pmean = float(np.mean([r[metric] for r in subset if r['method'] == 'ppo']))
            reduction = np.array(paired)
            rng = np.random.default_rng(20261025)
            bootstrap = reduction[:, rng.integers(0, 5, size=(10000, 5))].mean(axis=(0, 2))
            summaries[normalization][metric] = dict(heuristic_mean=hmean, ppo_mean=pmean, reduction_of_grand_means_percent=100*(hmean-pmean)/hmean, mean_paired_reduction_percent=float(reduction.mean()), seed_block_bootstrap_95_percent=np.quantile(bootstrap, [.025, .975]).tolist(), ppo_wins=int((reduction > 0).sum()), ties=int((reduction == 0).sum()), ppo_losses=int((reduction < 0).sum()))
    write_csv(OUT / 'per_iid.csv', per_iid)
    write_json(OUT / 'reference_protocol.json', dict(source=str(SOURCE), moocore_version=moocore.__version__, objectives='five minimization objectives', reference='Per IID: exact unique nondominated union of both methods across all five search seeds; each unique reference vector has equal weight; no resampling.', reference_dependence='Empirical, evaluated-run-inclusive, density-dependent approximation, not the true Pareto front or an independent reference.', normalizations=['empirical ideal/nadir of shared reference front (no clipping)', 'original fixed physical affine transform'], zero_span='Stop for explicit protocol choice; none encountered.', references=references))
    write_json(OUT / 'summary.json', summaries)
    lines = ['# V4.3 F offline IGD and IGD+', '', '90 existing archives / nine development IIDs / five search seeds / one training seed. No training, new search, source modification or original-result overwrite.', '', 'Reference per IID: nondominated unique union of H/P archives from all five seeds. This combines records only to define a shared empirical reference; every run is scored separately. The reference is not the true Pareto front and depends on included runs and sampling density.', '', 'All metrics are library calls to moocore '+moocore.__version__+'. Both metrics are lower-is-better. Empirical normalization uses reference ideal/nadir; fixed-physical normalization is separately reported; values from the two scales are not interchangeable. No objective has zero reference range.', '', '| Normalization | Metric | H mean | PPO mean | Mean paired reduction % | PPO W/T/L |', '|---|---|---:|---:|---:|---|']
    for normalization, metrics in summaries.items():
        for metric, s in metrics.items():
            lines.append(f"| {normalization} | {metric} | {s['heuristic_mean']:.9f} | {s['ppo_mean']:.9f} | {s['mean_paired_reduction_percent']:.5f} | {s['ppo_wins']}/{s['ties']}/{s['ppo_losses']} |")
    lines += ['', 'The bootstrap resamples the five search-seed blocks on the fixed nine IIDs with the empirical reference held fixed; the interval is exploratory and does not measure training-seed or reference-estimation uncertainty.', '', 'Files: calculate.py, per_run.csv, per_iid.csv, summary.json, reference_protocol.json, and nine *_reference.csv files.']
    (OUT / 'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
