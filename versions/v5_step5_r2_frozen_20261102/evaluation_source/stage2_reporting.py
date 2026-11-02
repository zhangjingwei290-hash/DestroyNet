"""Common-reference gate and incremental benchmark reporting, no training/search."""
from stage2_common import ROOT,read,write
from pathlib import Path
from collections import defaultdict
import math,statistics,time

METRICS=('empirical_hv','fixed_box_hv','igd_plus')

def parent_roots():
    p=read(ROOT/'plan.json');return p,Path(p['baseline_root']),Path(p['stage4_root']),Path(read(ROOT/'PARENT_R2.json')['root']),Path(read(ROOT/'PARENT_R2.json')['archive'])
def old_path(iid,seed,method):
    p,baseline,step4,parent,archive=parent_roots()
    if method in ('alns_tuned','v5_meta_frozen'):return baseline/'metric_inputs/results'/f'{iid}_seed{seed}_{method}.json'
    if method=='step4_k4_adapted':return step4/'evaluation/metric_inputs'/f'{iid}_seed{seed}_{method}.json'
    if method=='step5_stage1_zero_shot':return parent/'stage1_evaluation/metric_inputs'/f'{iid}_seed{seed}_{method}.json'
    if method=='step5_stage2_r2':return archive/'stage2_evaluation/metric_inputs'/f'{iid}_seed{seed}_step5_stage2_retrained.json'
    if method=='step5_stage2_r3':return ROOT/'evaluation_core_r3/metric_inputs'/f'{iid}_seed{seed}_{method}.json'
    raise ValueError(method)
def primary_path(iid,seed):
    p=read(ROOT/'plan.json');selection=read(ROOT/'MODEL_SELECTION.json');m=selection['method']
    if iid in p['core_ids'] and seed<=20261026:return old_path(iid,seed,m)
    return ROOT/'stage2_evaluation/metric_inputs'/f'{iid}_seed{seed}_{m}.json'
def canonical(path,method):
    d=read(path);d['method']=method;d['_path']=str(path);return d

def compute(ids,seeds,methods,path_for):
    import numpy as np,moocore
    from stats import geometry
    runs=[];means=[];geometry_rows={};reports=[];missing=[]
    for iid in ids:
        group=[]
        for m in methods:
            for seed in seeds:
                path=path_for(iid,seed,m)
                if not path.exists():missing.append(dict(iid=iid,seed=seed,method=m));continue
                d=canonical(path,m);assert d['iid']==iid and d['seed']==seed;group.append(d);reports.append(d)
        transform,pf,geometry_rows[iid]=geometry(group)
        for d in group:
            F=np.asarray([p['objectives'] for p in d['archive']]);fixed=(F-d['physical_offset'])/d['physical_scale'];ref=(np.asarray(d['physical_reference'])-d['physical_offset'])/d['physical_scale']
            runs.append(dict(iid=iid,n=int(iid.split('-')[0]),seed=d['seed'],method=d['method'],empirical_hv=float(moocore.hypervolume(transform(F),ref=[1.1]*5)),fixed_box_hv=float(moocore.hypervolume(fixed,ref=ref)),igd_plus=float(moocore.igd_plus(transform(F),ref=pf))))
        for m in methods:
            rows=[x for x in runs if x['iid']==iid and x['method']==m]
            if not rows:continue
            row=dict(iid=iid,n=int(iid.split('-')[0]),method=m,seeds=len(rows),complete_seed_coverage=len(rows)==len(seeds))
            for metric in METRICS:
                vals=[x[metric] for x in rows];row[metric+'_mean']=statistics.mean(vals);row[metric+'_sample_sd']=statistics.stdev(vals) if len(vals)>1 else None
            means.append(row)
    macro=[]
    for n in ('overall',50,150,250):
        ii=[i for i in ids if n=='overall' or int(i.split('-')[0])==n]
        for m in methods:
            rows=[x for x in means if x['iid'] in ii and x['method']==m]
            if not rows:continue
            result=dict(n=n,method=m,iids=len(rows),runs=sum(x['seeds'] for x in rows),expected_runs=len(ii)*len(seeds),complete_coverage=all(x['complete_seed_coverage'] for x in rows) and len(rows)==len(ii))
            for metric in METRICS:
                vals=[x[metric+'_mean'] for x in rows];result[metric]=statistics.mean(vals);result[metric+'_between_iid_sample_sd']=statistics.stdev(vals) if len(vals)>1 else None
                per_seed=[]
                for seed in seeds:
                    ss=[x[metric] for x in runs if x['iid'] in ii and x['method']==m and x['seed']==seed]
                    if len(ss)==len(ii):per_seed.append(statistics.mean(ss))
                result[metric+'_macro_seed_sample_sd']=statistics.stdev(per_seed) if len(per_seed)>1 else None
                result[metric+'_complete_macro_seeds']=len(per_seed)
            macro.append(result)
    from stats import signed_rank,wtl
    comparisons=[];target=methods[-1]
    for n in ('overall',50,150,250):
        tests=[]
        for baseline in methods[:-1]:
            for metric in METRICS:
                sign=-1 if metric=='igd_plus' else 1;index={(x['iid'],x['seed'],x['method']):x[metric] for x in runs}
                by_iid=[];by_run=[];paired=0
                for iid in ids:
                    if n!='overall' and int(iid.split('-')[0])!=n:continue
                    diff=[sign*(index[iid,s,target]-index[iid,s,baseline]) for s in seeds if (iid,s,target) in index and (iid,s,baseline) in index]
                    if diff:by_iid.append(statistics.mean(diff));by_run+=diff;paired+=len(diff)
                if not by_iid:continue
                test=signed_rank(by_iid);row=dict(n=n,method=target,baseline=baseline,metric=metric,paired_iids=len(by_iid),paired_runs=paired,iid_wtl=wtl(by_iid),run_wtl=wtl(by_run),**test)
                tests.append(row)
        sorted_tests=sorted(tests,key=lambda x:x['p']);previous=0.
        for j,row in enumerate(sorted_tests):previous=max(previous,min(1.,row['p']*(len(tests)-j)));row['p_holm']=previous;row['holm_family_size']=len(tests)
        comparisons+=tests
    return dict(ids=ids,seeds=seeds,methods=methods,per_run=runs,main_table=means,macro=macro,comparisons=comparisons,geometry=geometry_rows,missing=missing,reports=reports)

def write_tables(out,data):
    from stats import csv_write
    out.mkdir(parents=True,exist_ok=True)
    for key in ('per_run','main_table','macro','comparisons','missing'):csv_write(out/('per_run_metrics.csv' if key=='per_run' else key+'.csv'),data[key])

def diagnostics(out,reports,method):
    from stats import csv_write
    rows=[];mode_rows=[];qrows=[];directions=[];knees=[];states=[];direction_states=[]
    for n in (50,150,250):
        group=[r for r in reports if r['method']==method and int(r['iid'].split('-')[0])==n];steps=sum(r['steps'] for r in group)
        costs=[c for r in group for c in r['repair_costs'].values()];cache=[r['cache_diagnostics'] for r in group]
        rows.append(dict(n=n,method=method,runs=len(group),steps=steps,mean_steps=steps/max(1,len(group)),inference_seconds=sum(r['inference_seconds'] for r in group),repair_seconds=sum(c['seconds'] for c in costs),gnn_refresh_seconds=sum(c['gnn_refresh_seconds'] for c in cache),refreshes=sum(c['refreshes'] for c in cache),cache_hit_rate=sum(c['hits'] for c in cache)/max(1,steps)))
        hist=defaultdict(int)
        for r in group:
            for q,c in r['q_histogram'].items():hist[int(q)]+=int(c)
        bounds=[0,.01,.02,.03,.04,.05,.10,.20,.30,.40,.60,1.]
        for lo,hi in zip(bounds[:-1],bounds[1:]):qrows.append(dict(n=n,lower_open=lo,upper_closed=hi,count=sum(c for q,c in hist.items() if lo<q/n<=hi),total=steps,mean_q_over_n=sum(q/n*c for q,c in hist.items())/max(1,steps)))
        for st in range(10):
            selected=[x for r in group for x in r.get('state_awareness',[]) if x['stagnation']==st];cnt=sum(x['count'] for x in selected)
            states.append(dict(n=n,stagnation=st,label='nine_or_more' if st==9 else str(st),count=cnt,mean_q_over_n=sum(x['q_sum'] for x in selected)/max(1,cnt*n),accepted=sum(x['accepted'] for x in selected),coverage_runs=sum('state_awareness' in r for r in group),total_runs=len(group),**{f'mode{m}_count':sum(x['mode_counts'][m] for x in selected) for m in range(3)}))
        for k in range(16):
            selected=[x for r in group for x in r.get('direction_statistics',[]) if x['direction']==k];cnt=sum(x['count'] for x in selected)
            direction_states.append(dict(n=n,direction=k,count=cnt,mean_q_over_n=sum(x['q_sum'] for x in selected)/max(1,cnt*n),accepted=sum(x['accepted'] for x in selected),**{f'mode{m}_count':sum(x['mode_counts'][m] for x in selected) for m in range(3)}))
        for m in range(3):
            mm=[r['mode_diagnostics'][m] for r in group];count=sum(x['steps'] for x in mm);acc=sum(x['accepted'] for x in mm)
            mode_rows.append(dict(n=n,mode=m,steps=count,proportion=count/max(1,steps),acceptance_rate=acc/max(1,count),vehicle_reductions=sum(x['vehicle_reductions'] for x in mm),accepted_vehicle_reductions=sum(x['accepted_vehicle_reductions'] for x in mm),first_route_cap_exceptions=sum(x['first_route_cap_exceptions'] for x in mm)))
    for r in reports:
        if r['method']!=method:continue
        for row in r['per_direction']:directions.append(dict(iid=r['iid'],seed=r['seed'],**row))
        for vehicles in sorted({int(p['objectives'][0]) for p in r['archive']}):
            points=[p for p in r['archive'] if int(p['objectives'][0])==vehicles];best=min(points,key=lambda p:p['objectives'][1]);timebest=min(points,key=lambda p:p['objectives'][2])
            knees.append(dict(iid=r['iid'],seed=r['seed'],vehicles=vehicles,distance=best['objectives'][1],completion_at_same_distance_point=best['objectives'][2],best_completion=timebest['objectives'][2],distance_at_same_completion_point=timebest['objectives'][1]))
    for name,data in [('throughput',rows),('mode_distribution',mode_rows),('q_distribution',qrows),('per_direction',directions),('vehicle_knee',knees),('state_awareness',states),('direction_behavior',direction_states)]:csv_write(out/(name+'.csv'),data)
    return dict(throughput=rows,modes=mode_rows)

def markdown(data,title):
    lines=['# '+title,'','每IID共同经验ND参考；每seed档案独立HV/IGD+，再跨seed均值/样本SD与IID等权。FixedBoxHV另列。','12核心IID用于模型选择，45全体也已被检查；此为条件于选择流程的描述性基准结果。不同版本参考重算后绝对值不可直接横比。','宏观表±为完整宏观seed均值之间的样本SD，CSV另列IID均值间SD；逐IID表±为seed SD。Step4缺失不补跑、不插补；配对检验只用双方实际共有seed，IID均值同样在共有seed上计算。','', '|N|Method|runs/expected|EmpHV ± macro-seed SD|FixedBoxHV|IGD+ ± macro-seed SD|','|---|---|---:|---:|---:|---:|']
    def val(v):return 'NA' if v is None else f'{v:.8f}'
    for row in data['macro']:lines.append(f"|{row['n']}|{row['method']}|{row['runs']}/{row['expected_runs']}|{val(row['empirical_hv'])} ± {val(row['empirical_hv_macro_seed_sample_sd'])}|{val(row['fixed_box_hv'])}|{val(row['igd_plus'])} ± {val(row['igd_plus_macro_seed_sample_sd'])}|")
    lines+=['','|N|Baseline|Metric|IID W/T/L|run W/T/L|exact p|Holm p|','|---|---|---|---|---|---:|---:|']
    for row in data['comparisons']:lines.append(f"|{row['n']}|{row['baseline']}|{row['metric']}|{row['iid_wtl']}|{row['run_wtl']}|{row['p']:.7g}|{row['p_holm']:.7g}|")
    lines+=['','|IID|Method|seeds|EmpHV mean ± sampleSD|IGD+ mean ± sampleSD|','|---|---|---:|---:|---:|']
    for row in data['main_table']:lines.append(f"|{row['iid']}|{row['method']}|{row['seeds']}|{val(row['empirical_hv_mean'])} ± {val(row['empirical_hv_sample_sd'])}|{val(row['igd_plus_mean'])} ± {val(row['igd_plus_sample_sd'])}|")
    return lines

def gate():
    plan=read(ROOT/'plan.json');methods=['alns_tuned','v5_meta_frozen','step4_k4_adapted','step5_stage1_zero_shot','step5_stage2_r2','step5_stage2_r3']
    data=compute(plan['core_ids'],[20261024,20261025,20261026],methods,old_path);assert not data['missing']
    out=ROOT/'stage2_analysis';write_tables(out,data)
    values={r['method']:r['empirical_hv'] for r in data['macro'] if r['n']=='overall'}
    winner='r3' if values['step5_stage2_r3']>=values['step5_stage2_r2'] else 'r2';method='step5_stage2_'+winner
    selection=dict(selected=winner,method=method,whole_model=True,criterion='overall12IID_own_seed_HV_in_common_six_method_reference',r2_hv=values['step5_stage2_r2'],r3_hv=values['step5_stage2_r3'],old_r2_five_method_hv=.7274359010104822,old_reference_not_direct_threshold=True,core12_is_model_selection_set=True,created_unix=time.time())
    if (ROOT/'MODEL_SELECTION.json').exists():
        prior=read(ROOT/'MODEL_SELECTION.json');assert prior['selected']==winner
    else:write(ROOT/'MODEL_SELECTION.json',selection)
    summary={k:v for k,v in data.items() if k not in ('reports','per_run','main_table')};summary['model_selection']=selection
    if (ROOT/'engineering/replay_guard_direction05_batch003/REPAIR.json').exists():summary['engineering_repair']=read(ROOT/'engineering/replay_guard_direction05_batch003/REPAIR.json')
    write(out/'summary_12iid_3seed.json',summary);text=markdown(data,'Stage2 r3 12-IID model-selection gate')
    text+=['',f"Whole-model selected: {winner}. r2/r3 are recomputed under the SAME six-method reference; the historical .7274 is not a comparable threshold after the reference changes.",'No per-direction checkpoint mixing, no additional training or tuning follows this gate.']
    (ROOT/'STAGE2_R3_12IID_GATE.md').write_text('\n'.join(text)+'\n',encoding='utf-8');return selection

def full(seeds):
    plan=read(ROOT/'plan.json');selection=read(ROOT/'MODEL_SELECTION.json');target=selection['method'];methods=['alns_tuned','v5_meta_frozen','step4_k4_adapted',target]
    def paths(i,s,m):return primary_path(i,s) if m==target else old_path(i,s,m)
    data=compute(plan['test_ids'],seeds,methods,paths)
    assert all(x['method']=='step4_k4_adapted' for x in data['missing'])
    assert sum(r['method']==target for r in data['per_run'])==45*len(seeds)
    out=ROOT/('stage2_analysis_45iid_3seed' if len(seeds)==3 else 'stage2_analysis_45iid_20seed');write_tables(out,data)
    diag=diagnostics(out,data['reports'],target);summary={k:v for k,v in data.items() if k not in ('reports','per_run','main_table')};summary.update(model_selection=selection,diagnostics=diag,completed_seeds=len(seeds),selected_runs=45*len(seeds),target_runs=900,generated_unix=time.time())
    if (ROOT/'engineering/replay_guard_direction05_batch003/REPAIR.json').exists():summary['engineering_repair']=read(ROOT/'engineering/replay_guard_direction05_batch003/REPAIR.json')
    write(out/'summary.json',summary);write(ROOT/'stage2_analysis/summary_full45.json',summary);write(ROOT/'stage2_analysis_45iid_20seed/progress_summary.json',dict(completed_seeds=seeds,selected_runs=45*len(seeds),target_runs=900,step4_missing=len(data['missing']),selected_model=selection['selected'],updated_unix=time.time()))
    text=markdown(data,f'Stage2 full45 benchmark: {len(seeds)} seeds');text+=['',f"Selected whole model: {selection['selected']}. Missing Step4 rows: {len(data['missing'])}; old baseline directories remain unchanged.",'All subsequent seeds use this same model. r3 gate36 results retained even when r2 selected. These inspected, selection-conditioned benchmarks do not isolate the causal effect of any one training change.']
    if 'engineering_repair' in summary:text += ['', 'Training resumed after a guarded replay failure; original offending transition was not retained. A reproducible nearconstant-feature reduction discrepancy was fixed using shared stable nonlearned reductions while the network remainedFP32 and all guards stayed unchanged. Original144meta and direction15 updates were preserved; direction05first2 optimizer steps were reconstructed from preserved traces with identical losses. Historical r2 deployment retains its original float32 feature reductions. See engineering repair receipt for diagnostic limits and the one unoptimized fault group.']
    if len(seeds)==3:(ROOT/'STAGE2_FULL45_3SEED_REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    (ROOT/'STAGE2_FULL45_REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    (ROOT/'STAGE2_FULL45_20SEED_REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    chosen=Path(read(ROOT/'PARENT_R2.json')['archive'])/'stage2_training' if selection['selected']=='r2' else ROOT/'stage2_training'
    import shutil
    shutil.copy2(chosen/'q_behavior.csv',out/'FIT_stagnation0_2_8_behavior.csv')
    if len(seeds)==20:figures(out,data,diag)
    return summary

def figures(out,data,diag):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,ax=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for i,metric in enumerate(('empirical_hv','igd_plus')):
        for j,m in enumerate(data['methods']):
            rows=[next(r for r in data['macro'] if r['n']==n and r['method']==m) for n in (50,150,250)]
            ax[i].bar(np.arange(3)+(j-1.5)*.18,[r[metric] for r in rows],width=.18,label=m)
        ax[i].set_xticks(range(3),['50','150','250']);ax[i].set_title(metric)
    ax[0].legend(fontsize=6);fig.savefig(out/'benchmark.png',dpi=220);fig.savefig(out/'benchmark.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True)
    for m in range(3):ax.bar(np.arange(3)+(m-1)*.23,[r['proportion'] for r in diag['modes'] if r['mode']==m],width=.23,label=f'Mode {m}')
    ax.set_xticks(range(3),['50','150','250']);ax.set_ylabel('Action fraction');ax.legend();fig.savefig(out/'modes.png',dpi=220);fig.savefig(out/'modes.pdf');plt.close(fig)

if __name__=='__main__':
    import argparse,psutil
    psutil.Process().cpu_affinity([14,15]);ap=argparse.ArgumentParser();ap.add_argument('--gate',action='store_true');ap.add_argument('--seeds',nargs='+',type=int);args=ap.parse_args()
    if args.gate:gate()
    else:full(args.seeds)
