"""Descriptive analysis of completed frozen searches. No inference or plotting."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import json, collections, hashlib, itertools, time
from pathlib import Path
import numpy as np

OUT=Path(__file__).resolve().parent
REPO=OUT.parents[1]
ROOT=REPO/'runs_emnh/20261021_ppo_heuristic_20seed_v1'
def read(p):
    with p.open(encoding='utf-8') as f:return json.load(f)
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def phase(f):return 'early' if f<.2 else 'middle' if f<.7 else 'late'
def mean(values):return float(np.mean(values)) if len(values) else None

plan=read(ROOT/'plan.json'); receipts=read(ROOT/'verification.json')
cfg=read(REPO/'runs_emnh/20261020_gpu_v2_seed20261020/config.json')
study=read(Path(cfg['study_root'])/'config.json')
instances={i:read(Path(study['inputs'][i]['path'])) for i in plan['development_ids']}
for i,x in instances.items():assert sha(Path(study['inputs'][i]['path']))==study['inputs'][i]['sha256']
started=time.perf_counter(); actions=[]; sources={}; sessions=0; timing_crossings=0; segment_failures=[]
for seed in plan['search_seeds']:
    for iid in plan['development_ids']:
        inst=instances[iid];n=inst['n_customers'];tw=np.asarray(inst['customer_time_windows'])
        widths=tw[:,1]-tw[:,0];allmean=float(widths.mean());assert allmean>0
        # Fractionally allocate ties at the tightest quartile cutoff so baseline=25%.
        cutoff=np.quantile(widths,.25); below=widths<cutoff;ties=widths==cutoff
        tight=below.astype(float)
        if ties.any():tight[ties]=max(0.,min(1.,(.25*n-below.sum())/ties.sum()))
        for arm in ('heuristic','ppo'):
            path=ROOT/'results'/str(seed)/f'{iid}_{arm}.json';key=f'{seed}/{iid}/{arm}'
            sources[key]=sha(path);assert sources[key]==receipts[key]['sha256']
            row=read(path);records=row['records']; groups=[]; current=[]
            for rec in records:
                if current and (current[-1]['terminal'] or rec['direction']!=current[-1]['direction'] or rec['local_elapsed']<current[-1]['local_elapsed']):
                    groups.append(current);current=[]
                current.append(rec)
            if current:groups.append(current)
            for group in groups:
                sessions+=1
                if len({r['direction'] for r in group})!=1:segment_failures.append(key)
                # outer_elapsed is written AFTER the entire local session.
                # Infer cycle starts from segment completion and cumulative local time.
                base=group[0]['outer_elapsed']-group[-1]['local_elapsed']
                previous=0.;stagnation=0
                for j,r in enumerate(group):
                    assert len(r['removed'])==r['q'] and len(set(r['removed']))==r['q']
                    removed=np.asarray(r['removed'],dtype=int)-1
                    assert np.all((removed>=0)&(removed<n))
                    begin=(base+previous)/row['budget_seconds'];end=(base+r['local_elapsed'])/row['budget_seconds']
                    if phase(begin)!=phase(end):timing_crossings+=1
                    status='new_session' if j==0 else 'after_accept' if stagnation==0 else 'reject_1_4' if stagnation<5 else 'reject_5_plus'
                    actions.append(dict(iid=iid,seed=seed,arm=arm,n=n,direction=r['direction'],
                        qi=r['action'].get('q_index') if arm=='ppo' else None,
                        q_fraction=r['q']/n,phase_start=phase(begin),phase_finish=phase(end),phase_segment_end=phase(r['outer_elapsed']/row['budget_seconds']),
                        stagnation=stagnation,status=status,repair=r['repair'],accepted=bool(r['accepted']),
                        width_ratio=float(widths[removed].mean()/allmean),tight_share=float(tight[removed].mean()),
                        repair_seconds=r['repair_seconds'],timeout=bool(r['timeout'])))
                    previous=r['local_elapsed'];stagnation=0 if r['accepted'] else stagnation+1
assert not segment_failures

def aggregate(items):
    """Equal run weights: mean within IID x seed, then mean over runs."""
    by=collections.defaultdict(list)
    for a in items:by[a['iid'],a['seed']].append(a)
    metrics=('q_fraction','width_ratio','tight_share','accepted','timeout')
    result=dict(actions=len(items),runs=len(by))
    for m in metrics:result[m]=mean([np.mean([a[m] for a in group]) for group in by.values()])
    if items:
        freq=np.asarray([np.bincount([int(np.searchsorted([.025,.075,.15,.3,.5,.7,.9,1.01],a['q_fraction'])) for a in group],minlength=8)/len(group) for group in by.values()]).mean(axis=0)
        result['common_q_fraction_bin_frequencies']=freq.tolist()
    if items and items[0]['arm']=='ppo':
        freq=np.asarray([np.bincount([a['qi'] for a in group],minlength=9)/len(group) for group in by.values()]).mean(axis=0)
        result['q_index_frequencies']=freq.tolist()
        result['entropy_of_mean_q_index_distribution_nats']=float(-np.sum(freq[freq>0]*np.log(freq[freq>0])))
    return result

def contrast(left,right,metric):
    def means(items):
        d=collections.defaultdict(list)
        for a in items:d[a['iid'],a['seed']].append(a[metric])
        return {k:float(np.mean(v)) for k,v in d.items()}
    x,y=means(left),means(right);keys=sorted(set(x)&set(y))
    if not keys:return dict(paired_runs=0)
    # Average within seed over available fixed IID differences, resample seeds.
    seeds=sorted({s for _,s in keys})
    sums=np.asarray([np.sum([y[k]-x[k] for k in keys if k[1]==s]) for s in seeds])
    counts=np.asarray([sum(k[1]==s for k in keys) for s in seeds])
    rng=np.random.default_rng(20261023)
    draws=rng.integers(len(seeds),size=(10000,len(seeds)))
    samples=sums[draws].sum(axis=1)/counts[draws].sum(axis=1)
    return dict(paired_runs=len(keys),mean_right_minus_left=float(np.mean([y[k]-x[k] for k in keys])),
        conditional_search_seed_bootstrap_95_ci=np.quantile(samples,[.025,.975]).tolist(),
        note='Exploratory conditional-on-fixed-IID descriptive contrast; incomplete cells reported, no multiplicity correction or causal claim')

summary=dict(scope='Completed frozen-policy deployment logs; one training seed; no new inference/search/training/plots',
    total_actions=len(actions),ppo_actions=sum(a['arm']=='ppo' for a in actions),sessions=sessions,
    time_semantics='Approximate cycle-start time reconstructed from post-session outer_elapsed and cumulative local_elapsed; completion-time and session-end sensitivity also reported',
    actions_crossing_stage_boundary=timing_crossings,
    grouping='Average actions within each IID x seed cell, then equally average available cells. No action-count weighting across runs.',
    preference={},stages={},stagnation={},static_customer_features={},contrasts={})
P=[a for a in actions if a['arm']=='ppo']
for k,label in ((0,'vehicles'),(1,'distance'),(4,'tardiness'),(15,'center')):
    part=[a for a in P if a['direction']==k]
    summary['preference'][label]=dict(direction=k,adapted=k not in [1,9,11,14],**aggregate(part),by_size={str(n):aggregate([a for a in part if a['n']==n]) for n in (50,150,250)})
    summary['static_customer_features'][label]=dict(all_q=aggregate(part),q_at_most_10_percent=aggregate([a for a in part if a['q_fraction']<=.1]))
for size in ('all',50,150,250):
    summary['stages'][str(size)]={}
    for arm in ('heuristic','ppo'):
        part=[a for a in actions if a['arm']==arm and (size=='all' or a['n']==size)]
        summary['stages'][str(size)][arm]={}
        for clock in ('phase_start','phase_finish','phase_segment_end'):
            summary['stages'][str(size)][arm][clock]={p:aggregate([a for a in part if a[clock]==p]) for p in ('early','middle','late')}
        summary['contrasts'][f'{size}_{arm}_late_minus_early_q_fraction']=contrast([a for a in part if a['phase_start']=='early'],[a for a in part if a['phase_start']=='late'],'q_fraction')
for status in ('new_session','after_accept','reject_1_4','reject_5_plus'):
    summary['stagnation'][status]=aggregate([a for a in P if a['status']==status])
for size in ('all',50,150,250):
    part=[a for a in P if size=='all' or a['n']==size]
    summary['contrasts'][f'{size}_stagnation_5plus_minus_afteraccept_q']=contrast([a for a in part if a['status']=='after_accept'],[a for a in part if a['status']=='reject_5_plus'],'q_fraction')
for label,k in (('vehicles',0),('distance',1),('center',15)):
    part=[a for a in P if a['direction']==k];other=[a for a in P if a['direction']==4]
    summary['contrasts'][f'tardiness_minus_{label}_q']=contrast(part,other,'q_fraction')
    summary['contrasts'][f'tardiness_minus_{label}_width_ratio']=contrast(part,other,'width_ratio')
    summary['contrasts'][f'tardiness_minus_{label}_tight_share']=contrast(part,other,'tight_share')
    summary['contrasts'][f'tardiness_minus_{label}_smallq_width_ratio']=contrast([a for a in part if a['q_fraction']<=.1],[a for a in other if a['q_fraction']<=.1],'width_ratio')
# Total variation between averaged q-index distributions, same nine action bins.
summary['q_distribution_bin_labels']=['q/N<=2.5%','2.5%<q/N<=7.5%','7.5%<q/N<=15%','15%<q/N<=30%','30%<q/N<=50%','50%<q/N<=70%','70%<q/N<=90%','90%<q/N<=100%']
summary['q_distribution_total_variation']={f'{x}_vs_{y}':float(.5*np.abs(np.asarray(summary['preference'][x]['common_q_fraction_bin_frequencies'])-summary['preference'][y]['common_q_fraction_bin_frequencies']).sum()) for x,y in itertools.combinations(summary['preference'],2)}
summary['q_index_note']='Raw q_index support differs with N: 50 customers has 8 legal bins after deduplication; 150/250 have 9. Across-size comparisons use actual q/N or the eight common fraction bins.'
# Control direction and selected repairer when contrasting stagnation groups.
bycell=collections.defaultdict(lambda:collections.defaultdict(list))
for a in P:
    if a['status'] in ('after_accept','reject_5_plus'):
        bycell[a['iid'],a['seed'],a['direction'],a['repair']][a['status']].append(a['q_fraction'])
matched=collections.defaultdict(list)
for key,groups in bycell.items():
    if len(groups)==2:matched[key[:2]].append(float(np.mean(groups['reject_5_plus'])-np.mean(groups['after_accept'])))
summary['stagnation_direction_repair_matched']={
    'matched_cells':sum(map(len,matched.values())), 'runs':len(matched),
    'mean_run_average_q_fraction_difference':mean([np.mean(v) for v in matched.values()]),
    'by_size':{str(n):dict(runs=sum(int(k[0].split('_')[2])==n for k in matched),difference=mean([np.mean(v) for k,v in matched.items() if int(k[0].split('_')[2])==n])) for n in (50,150,250)},
    'note':'Conditioned on observed run/direction/repair cells containing both statuses; stage, route state, and duration censoring remain uncontrolled.'}
summary['ppo_stages_selected_direction_mix']={p:{str(k):sum(a['phase_start']==p and a['direction']==k for a in P) for k in range(16)} for p in ('early','middle','late')}
save('summary.json',summary)
save('verification.json',dict(passed=True,source_reports_sha256=sources,source_reports_matched_original_receipts=360,
    input_instances_verified=9,actions=len(actions),no_new_search=True,no_new_inference=True,no_plots=True,
    missing_dynamic_features=['predecision full routes','per-customer predecision tardiness/slack','predecessor-successor detour'],
    elapsed_seconds=time.perf_counter()-started))
print(json.dumps(dict(preference=summary['preference'],stage_ppo={size:{p:x['q_fraction'] for p,x in arms['ppo']['phase_start'].items()} for size,arms in summary['stages'].items()},
    stages_sensitivity={clock:{p:x['q_fraction'] for p,x in groups.items()} for clock,groups in summary['stages']['all']['ppo'].items()},
    stagnation=summary['stagnation'],contrasts=summary['contrasts'],tv=summary['q_distribution_total_variation'],
    static_small_q={k:v['q_at_most_10_percent'] for k,v in summary['static_customer_features'].items()},
    actions_crossing_boundary=timing_crossings),ensure_ascii=False,indent=2))
