"""External-library HV/IGD+; pair within IID before macro aggregation."""
import csv,json
from pathlib import Path
import numpy as np
import moocore
ROOT=Path(__file__).resolve().parent

def main():
    from evaluate import read,write,assert_pair
    plan=read(ROOT/'plan.json');out=ROOT/'results';pairs=[];instances=[]
    for iid in plan['ids']:
        reports={(seed,m):read(out/f'{iid}_seed{seed}_{m}.json') for seed in plan['search_seeds'] for m in plan['methods']}
        first=next(iter(reports.values()));offset=np.array(first['physical_offset']);scale=np.array(first['physical_scale']);ref=first['normalized_reference']
        normalized={k:(np.array([p['objectives'] for p in r['archive']])-offset)/scale for k,r in reports.items()}
        union=np.unique(np.concatenate(list(normalized.values())),axis=0)
        empirical=union[moocore.is_nondominated(union)]
        write(out/f'{iid}_igd_reference.json',dict(iid=iid,normalized_points=empirical.tolist(),source='nondominated union of all 40 final archives; empirical, not true Pareto front',offset=offset.tolist(),scale=scale.tolist()))
        local=[]
        for seed in plan['search_seeds']:
            h=reports[seed,'heuristic'];v=reports[seed,'v5_meta_frozen'];assert_pair(h,v)
            row=dict(iid=iid,seed=seed)
            for m,r in [('heuristic',h),('v5',v)]:
                if r['physical_offset']!=first['physical_offset'] or r['physical_scale']!=first['physical_scale'] or r['normalized_reference']!=ref:raise ValueError('Inconsistent metric box')
                a=normalized[seed,r['method']];hv=float(moocore.hypervolume(a,ref=ref));np.testing.assert_allclose(hv,r['hv'],rtol=1e-10,atol=1e-12)
                row.update({m+'_hv':hv,m+'_igd_plus':float(moocore.igd_plus(a,ref=empirical)),m+'_actions':len(r['records']),m+'_inference_seconds':r['inference_seconds'],m+'_repair_seconds':sum(c['seconds'] for c in r['repair_costs'].values()),m+'_timeouts':sum(c['timeouts'] for c in r['repair_costs'].values()),m+'_search_seconds':r['search_seconds'],m+'_outside_ref_points':int(np.any(a>=np.array(ref),axis=1).sum())})
            for j in range(5):
                row[f'heuristic_objective_{j+1}_min']=h['objective_minima'][j];row[f'v5_objective_{j+1}_min']=v['objective_minima'][j]
            row['hv_delta']=row['v5_hv']-row['heuristic_hv'];row['hv_relative_gain_percent']=100*row['hv_delta']/row['heuristic_hv'] if row['heuristic_hv'] else None
            row['igd_plus_delta']=row['v5_igd_plus']-row['heuristic_igd_plus'];local.append(row);pairs.append(row)
        stats=dict(iid=iid,n_search_seeds=20,empirical_reference_size=len(empirical))
        for key in (k for k in local[0] if k not in ('iid','seed','hv_relative_gain_percent')):
            a=np.array([r[key] for r in local]);stats[key+'_mean']=float(a.mean());stats[key+'_sd']=float(a.std(ddof=1))
        delta=np.array([r['hv_delta'] for r in local]);rng=np.random.default_rng(20261026)
        ci=np.quantile(delta[rng.integers(0,20,(5000,20))].mean(1),[.025,.975]);stats.update(hv_delta_ci95_low=float(ci[0]),hv_delta_ci95_high=float(ci[1]),hv_wins=int((delta>1e-12).sum()),hv_ties=int((abs(delta)<=1e-12).sum()),hv_losses=int((delta< -1e-12).sum()))
        instances.append(stats)
    for name,rows in [('paired_results.csv',pairs),('main_table.csv',instances)]:
        with (out/name).open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    macro={k:float(np.mean([r[k] for r in instances])) for k in instances[0] if k.endswith('_mean')}
    write(out/'summary.json',dict(scope=plan['scope'],runs=len(pairs)*2,pairs=len(pairs),instances=instances,macro_equal_instance_means=macro,reference_note=plan['igd_plus'],ci_note='Per-IID percentile paired bootstrap over 20 search seeds, 5000 resamples; conditional on one frozen training seed',limitations=plan['limitations']))
    (out/'REPORT.md').write_text('# V5 frozen meta / realworld45\n\n45 instances × 20 paired search seeds, 1800 searches. No adaptation.\n\nSee main_table.csv (45 IID rows), paired_results.csv (900 pairs), summary.json, raw archives and time curves.\n\nHV: moocore, fixed physical normalization and ref=(1.1)^5. Positive HV delta favors V5. IGD+: moocore, per-IID empirical pooled nondominated reference; negative delta favors V5. No clipping or per-method scaling. Main table SD uses ddof=1.\n\n'+ '\n'.join('- '+x for x in plan['limitations']),encoding='utf-8')
if __name__=='__main__':main()
