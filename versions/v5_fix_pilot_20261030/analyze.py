"""Fixed-three-seed paired pilot statistics; baseline files are opened read-only."""
from pilot_common import ROOT,read,write,sha,log
from diagnostics import records_q,BINS,LABELS
import argparse,csv,math,statistics,time
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
import moocore
import psutil

def csv_write(path,rows):
    if not rows:return
    with Path(path).open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def geometry(reports):
    points=np.unique(np.concatenate([np.asarray([p['objectives'] for p in r['archive']]) for r in reports]),axis=0)
    pf=points[moocore.is_nondominated(points)]
    ideal=pf.min(0);span=pf.max(0)-ideal
    zero=span/np.asarray(reports[0]['physical_scale'])<=1e-12
    def transform(a):return np.where(zero,0.,(a-ideal)/np.where(zero,1.,span))
    return transform,transform(pf),dict(ideal=ideal.tolist(),nadir=pf.max(0).tolist(),zero_span_coordinates=np.flatnonzero(zero).tolist(),reference=[1.1]*5)
def signed_rank(values):
    delta=[float(v) for v in values if abs(v)>1e-12]
    if not delta:return dict(W=0.,p=1.)
    order=sorted(range(len(delta)),key=lambda j:abs(delta[j]));doubled=[0]*len(delta);i=0
    while i<len(order):
        j=i+1
        while j<len(order) and abs(delta[order[j]])==abs(delta[order[i]]):j+=1
        for k in order[i:j]:doubled[k]=i+1+j
        i=j
    positive=sum(r for r,d in zip(doubled,delta) if d>0);total=sum(doubled);observed=min(positive,total-positive)
    counts=Counter({0:1})
    for rank in doubled:
        nxt=counts.copy()
        for s,c in counts.items():nxt[s+rank]+=c
        counts=nxt
    return dict(W=observed/2,p=sum(c for s,c in counts.items() if min(s,total-s)<=observed)/2**len(delta))
def wtl(values):return [sum(v>1e-12 for v in values),sum(abs(v)<=1e-12 for v in values),sum(v< -1e-12 for v in values)]
def mean(values):return float(statistics.mean(values)) if values else None

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--through-stage',type=int,choices=(1,2),required=True);args=ap.parse_args()
    psutil.Process().cpu_affinity([14,15]);psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    plan=read(ROOT/'plan.json');base=Path(plan['baseline_root']);methods=['alns_tuned','v5_meta_frozen','step1_adapted']
    if args.through_stage==2:methods+=['step2_meta_only','step2_adapted']
    ids,seeds=plan['test_ids'],plan['search_seeds'];out=ROOT/f'step{args.through_stage}_analysis';out.mkdir(exist_ok=False)
    metrics=[];instance=[];run_q=[];run_direction=[];raw_receipts=[];direction_objects=[]
    baseline_audit=read(base/'AUDIT.json');assert baseline_audit['passed']
    assert read(base/'summary.json')['raw_runs']==1800
    for stage in range(1,args.through_stage+1):
        assert read(ROOT/f'step{stage}_evaluation/complete.json')['runs']==(27 if stage==1 else 54)
        assert read(ROOT/f'step{stage}_training/training_report.json')['all_directions_updated']
    for iid in ids:
        reports={};n=int(iid.split('-')[0])
        for seed in seeds:
            initial=None
            for method in methods:
                path=base/'results'/f'{iid}_seed{seed}_{method}.json' if method in methods[:2] else ROOT/f'step{method[4]}_evaluation'/f'{iid}_seed{seed}_{method}.json'
                before=path.stat()
                cache=ROOT/'baseline_cache'/path.name
                if method in methods[:2] and cache.is_file():
                    r=read(cache)
                    assert (r['raw_bytes'],r['raw_mtime_ns'])==(before.st_size,before.st_mtime_ns)
                else:r=read(path)
                after=path.stat()
                assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
                assert (r['iid'],r['seed'],r['method'])==(iid,seed,method)
                if initial is None:initial=r['initial_population']
                else:assert initial==r['initial_population'],'Unpaired full initial population'
                assert r['budget_seconds']==read(ROOT/'config.json')['solve_seconds_by_instance'][iid]
                assert len(r['per_direction'])==16
                values=np.asarray([p['objectives'] for p in r['archive']]);assert values.ndim==2 and values.shape[1]==5 and np.isfinite(values).all()
                physical=(values-np.asarray(r['physical_offset']))/np.asarray(r['physical_scale'])
                assert math.isclose(float(moocore.hypervolume(physical,ref=r['normalized_reference'])),r['hv'],rel_tol=1e-10,abs_tol=1e-12)
                diagnostic=r.get('q_diagnostics') or records_q(r['records'],n)
                for label,block in diagnostic['blocks'].items():
                    count=block['count'];row=dict(iid=iid,seed=seed,n=n,method=method,condition=label,count=count,mean_q_ratio=block['sum_q_ratio']/count if count else None,small_le15_pct=100*block['small_le15']/count if count else None)
                    row.update({f'bin_{int(100*b)}pct':100*c/count if count else None for b,c in zip(BINS,block['hist'])});run_q.append(row)
                for direction in r['per_direction']:
                    k=direction['direction'];q=diagnostic['by_direction'][str(k)]['overall'];count=q['count']
                    row=dict(iid=iid,seed=seed,method=method,direction=k,scalar=float(direction['scalar']),steps=count,mean_q_ratio=q['sum_q_ratio']/count if count else None)
                    row.update({f'objective_{j}':v for j,v in enumerate(direction['objectives'])})
                    row.update({f'weight_{j}':v for j,v in enumerate(direction['weight'])})
                    run_direction.append(row)
                if '_throughput' not in r:
                    repairs=sum(float(t['repair_seconds']) for t in r['records'])
                    inference=sum(float(t['inference_seconds']) for t in r['records'])
                    assert math.isclose(inference,r['inference_seconds'],rel_tol=1e-10,abs_tol=1e-9)
                    r['_throughput']=dict(steps=len(r['records']),inference_seconds=inference,repair_seconds=repairs,search_seconds=r['search_seconds'])
                reports[seed,method]=r
                raw_receipts.append(dict(iid=iid,seed=seed,method=method,path=str(path),bytes=before.st_size,mtime_ns=before.st_mtime_ns))
        transform,pf,ref=geometry(list(reports.values()))
        write(out/'references'/f'{iid}.json',dict(**ref,empirical_reference=pf.tolist(),methods=methods,seeds=seeds))
        for method in methods:
            local=[]
            for seed in seeds:
                r=reports[seed,method];a=transform(np.asarray([p['objectives'] for p in r['archive']]))
                row=dict(iid=iid,seed=seed,n=n,method=method,hv=float(moocore.hypervolume(a,ref=[1.1]*5)),igd_plus=float(moocore.igd_plus(a,ref=pf)),**r['_throughput'])
                metrics.append(row);local.append(row)
            instance.append(dict(iid=iid,n=n,method=method,hv_mean=mean([r['hv'] for r in local]),hv_sd=statistics.stdev([r['hv'] for r in local]),igd_plus_mean=mean([r['igd_plus'] for r in local]),igd_plus_sd=statistics.stdev([r['igd_plus'] for r in local])))
        log(out/'analysis.log',dict(iid=iid,phase='scored',runs=len(reports)))
        del reports
    lookup={(r['iid'],r['method']):r for r in instance};run_lookup={(r['iid'],r['seed'],r['method']):r for r in metrics}
    comparison=[];tests=[]
    for method in methods:
        row=dict(method=method,hv_mean=mean([lookup[i,method]['hv_mean'] for i in ids]),igd_plus_mean=mean([lookup[i,method]['igd_plus_mean'] for i in ids]))
        for metric,sign in [('hv',1),('igd_plus',-1)]:
            differences=[sign*(lookup[i,method][metric+'_mean']-lookup[i,'alns_tuned'][metric+'_mean']) for i in ids]
            run_differences=[sign*(run_lookup[i,s,method][metric]-run_lookup[i,s,'alns_tuned'][metric]) for i in ids for s in seeds]
            row[metric+'_instance_WTL']=wtl(differences);row[metric+'_run_WTL']=wtl(run_differences)
            if method!='alns_tuned':tests.append(dict(method=method,metric=metric,**signed_rank(differences),instance_WTL=wtl(differences)))
        comparison.append(row)
    previous=0.
    for index,test in enumerate(sorted(tests,key=lambda t:t['p'])):
        previous=max(previous,min(1.,(len(tests)-index)*test['p']));test['holm_p']=previous
    throughput=[]
    for n in (50,150,250):
        for method in methods:
            group=[r for r in metrics if r['n']==n and r['method']==method]
            throughput.append(dict(n=n,method=method,runs=len(group),**{k:mean([r[k] for r in group]) for k in ('steps','inference_seconds','repair_seconds','search_seconds')}))
    qtable=[]
    for method in methods:
        for label in LABELS:
            group=[r for r in run_q if r['method']==method and r['condition']==label and r['count']>0]
            row=dict(domain='evaluation',method=method,condition=label,count=sum(r['count'] for r in group),runs_with_state=len(group))
            keys=['mean_q_ratio','small_le15_pct']+[f'bin_{int(100*b)}pct' for b in BINS]
            for key in keys:
                iid_values=[mean([r[key] for r in group if r['iid']==iid]) for iid in ids]
                row[key]=mean([v for v in iid_values if v is not None])
            qtable.append(row)
    for stage in range(1,args.through_stage+1):
        history=read(ROOT/f'step{stage}_training/training_report.json')['history']
        if stage==1:continue # Its unchanged training logger records the original ten bins; evaluation diagnostics above use the common requested bins.
        for domain in ('meta','adaptation','all_training'):
            hist=history if domain=='all_training' else [h for h in history if h['phase']==domain]
            for label in LABELS:
                blocks=[h['q_breakdown']['pilot_diagnostics']['blocks'][label] for h in hist]
                count=sum(b['count'] for b in blocks)
                row=dict(domain=domain,method=f'step{stage}_{domain}_training',condition=label,count=count,runs_with_state=None,mean_q_ratio=sum(b['sum_q_ratio'] for b in blocks)/count if count else None,small_le15_pct=100*sum(b['small_le15'] for b in blocks)/count if count else None)
                for j,b in enumerate(BINS):row[f'bin_{int(100*b)}pct']=100*sum(v['hist'][j] for v in blocks)/count if count else None
                qtable.append(row)
    direction_table=[]
    for k in range(16):
        for method in methods:
            group=[r for r in run_direction if r['method']==method and r['direction']==k]
            direction_table.append(dict(direction=k,method=method,runs=len(group),scalar_mean=mean([r['scalar'] for r in group]),scalar_sd=statistics.stdev([r['scalar'] for r in group]),q_mean=mean([r['mean_q_ratio'] for r in group if r['mean_q_ratio'] is not None]),mean_steps=mean([r['steps'] for r in group]),**{f'objective_{j}_mean':mean([r[f'objective_{j}'] for r in group]) for j in range(5)}))
    tw0=[]
    for method in methods:
        subset=[lookup[i,method] for i in plan['TW0_ids']]
        tw0.append(dict(method=method,instances=plan['TW0_ids'],hv_mean=mean([r['hv_mean'] for r in subset]),igd_plus_mean=mean([r['igd_plus_mean'] for r in subset])))
    for name,rows in [('per_run_metrics.csv',metrics),('main_table.csv',instance),('q_diagnostics_per_run.csv',run_q),('table1_q_distribution.csv',qtable),('table2_throughput.csv',throughput),('per_direction_per_run.csv',run_direction),('per_direction.csv',direction_table),('wilcoxon_exploratory.csv',tests)]:csv_write(out/name,rows)
    summary=dict(through_stage=args.through_stage,instances=9,seeds=seeds,methods=comparison,tests=tests,raw_runs=len(metrics),TW0=tw0,metric_protocol=plan['reporting'],q_bins=list(BINS),q_bin_note='Nearest descriptive centers, not categorical policy actions. N=50 cannot produce q/N=1%; q<=15% is counted directly.',evaluation_aggregation='Within each IID mean over three seeds; equal-IID macro. Conditional-q means omit genuinely absent states and report counts.',training_aggregation='Counts over optimized training transitions; training and evaluation are separate domains.',no_full_training_launched=True,source_and_load='Reused baseline and new searches on same desktop at different load epochs; new searches three pinned workers.')
    write(out/'summary.json',summary);write(out/'RAW_READONLY_RECEIPT.json',raw_receipts)
    lines=[f'# V5 two-step pilot: completed through stage {args.through_stage}','','9 fixed real instances x3 fixed seeds; per-seed HV/IGD+, shared empirical references for all methods in this report.','','## Overall paired comparison','','|Method|HV mean|IGD+ mean|HV W/T/L (9 IID)|HV W/T/L (27 runs)|','|---|---:|---:|---|---|']
    for r in comparison:lines.append(f"|{r['method']}|{r['hv_mean']:.8f}|{r['igd_plus_mean']:.8f}|{r['hv_instance_WTL']}|{r['hv_run_WTL']}|")
    lines+=['','## Table 1: granularity diagnostics','','Evaluation rows are equal-IID means of conditional per-run proportions; training rows pool optimized transitions. No state is filled with a synthetic zero.','','|Domain|Method|Condition|Actions|Mean q/N %|q/N<=15% %|'+ '|'.join(f'{b*100:g}%' for b in BINS)+'|','|'+'---|'*16]
    fmt=lambda v:'NA' if v is None else f'{v:.3f}'
    for r in qtable:
        cells=[r['domain'],r['method'],r['condition'],str(r['count']),fmt(None if r['mean_q_ratio'] is None else 100*r['mean_q_ratio']),fmt(r['small_le15_pct'])]+[fmt(r[f'bin_{int(100*b)}pct']) for b in BINS]
        lines.append('|'+'|'.join(cells)+'|')
    lines+=['','## Table 2: search throughput','','|N|Method|Runs|Steps|Inference s|Repair s|Search s|','|---|---|---:|---:|---:|---:|---:|']
    for r in throughput:lines.append(f"|{r['n']}|{r['method']}|{r['runs']}|{r['steps']:.2f}|{r['inference_seconds']:.3f}|{r['repair_seconds']:.3f}|{r['search_seconds']:.3f}|")
    lines+=['','## Table 3: nine-instance per-seed metrics (mean +/- sample SD)','','|IID|Method|HV|IGD+|','|---|---|---:|---:|']
    for r in instance:lines.append(f"|{r['iid']}|{r['method']}|{r['hv_mean']:.6f} +/- {r['hv_sd']:.6f}|{r['igd_plus_mean']:.6f} +/- {r['igd_plus_sd']:.6f}|")
    lines+=['','## Direction-specific scalar objectives','','Fixed physical augmented Tchebycheff scores, separately from the empirical metric geometry. Full five objectives and conditional q distributions are in CSV.','','|Direction|Method|Mean scalar|Mean q/N %|Mean steps|','|---:|---|---:|---:|---:|']
    for r in direction_table:lines.append(f"|{r['direction']}|{r['method']}|{r['scalar_mean']:.8f}|{fmt(None if r['q_mean'] is None else 100*r['q_mean'])}|{r['mean_steps']:.2f}|")
    lines+=['','## Conditions','','- Only FIT synthetic inputs are used for optimization. One training seed; this is the authorized bounded pilot.','- The original9-instance list includes TW=0 only at150-2-0 and250-1-0; it has no50-customer TW=0 case.','- All new runs retain the original per-instance outer budget, local<=min(10,T/16,remaining), stochastic sampling, five objectives and hard constraints.','- Original baseline files remain read-only. The original45x20 reference geometry and numbers are not overwritten.','- Stage1 freezes the encoder; stage2 changes granularity parameterization, explicitly reinitializes its final q location branch, and changes the trajectory objective. The stage2 contrast is this combined change.','- Exact signed-rank tests on9 instance means are exploratory, with Holm adjustment within this report. Related instances and repeated inspection are disclosed.','- The1% reporting center is structurally unreachable for N=50 (minimum q/N=2%). No threshold is used to suppress or rerun unfavorable results.','- No formal full training or additional algorithm stage is launched.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write(out/'complete.json',dict(completed=True,stage=args.through_stage,runs=len(metrics),instances=9,seeds=3,finished_unix=time.time()))
    print('PILOT_ANALYSIS_COMPLETE '+str(out),flush=True)

if __name__=='__main__':main()
