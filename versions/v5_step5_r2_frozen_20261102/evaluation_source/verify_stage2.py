"""Independent physical and metric audit after all36 Stage2 r2 searches."""
from stage2_common import ROOT,read,write,sha,load_cfg,EvalPool,readonly_check
from pathlib import Path
import math,statistics,time


def main():
    import psutil
    psutil.Process().cpu_affinity([14,15])
    import numpy as np
    import moocore
    import csv
    cfg=load_cfg();pool=EvalPool(cfg);plan=read(ROOT/'plan.json');plan['methods']=plan['methods']+['step5_stage2_retrained']
    out=ROOT/'stage2_analysis';summary=read(ROOT/'stage2_summary.json')
    assert (summary['total_runs'],summary['new_runs'],summary['reused_runs'])==(180,36,144)
    with (out/'per_run_metrics.csv').open(encoding='utf-8-sig') as f:metrics=list(csv.DictReader(f))
    with (out/'main_table.csv').open(encoding='utf-8-sig') as f:means=list(csv.DictReader(f))
    assert len(metrics)==len({(r['iid'],r['seed'],r['method']) for r in metrics})==180
    assert len(means)==60
    initial_count=archive_count=record_count=0
    for iid in plan['test_ids']:
        instance,row,box,_=pool.load(iid);reports=[]
        for method in plan['methods']:
            source=(Path(plan['baseline_root'])/'metric_inputs/results' if method in plan['methods'][:2]
                else Path(plan['stage4_root'])/'evaluation/metric_inputs' if method==plan['methods'][2]
                else ROOT/'stage1_evaluation/metric_inputs' if method=='step5_stage1_zero_shot' else ROOT/'stage2_evaluation')
            reports += [read(source/f'{iid}_seed{s}_{method}.json') for s in plan['search_seeds']]
        allpoints=np.unique(np.concatenate([np.asarray([p['objectives'] for p in r['archive']]) for r in reports]),axis=0)
        front=allpoints[moocore.is_nondominated(allpoints)]
        ideal=front.min(0);span=front.max(0)-ideal;zero=span/np.asarray(box['scale'])<=1e-12
        def norm(x):return np.where(zero,0.,(np.asarray(x)-ideal)/np.where(zero,1.,span))
        for r in reports:
            a=np.asarray([p['objectives'] for p in r['archive']]);key=(iid,str(r['seed']),r['method'])
            actual=next(m for m in metrics if (m['iid'],m['seed'],m['method'])==key)
            expected={'empirical_hv':float(moocore.hypervolume(norm(a),ref=[1.1]*5)),
                'igd_plus':float(moocore.igd_plus(norm(a),ref=norm(front))),
                'fixed_box_hv':float(moocore.hypervolume((a-np.asarray(box['offset']))/np.asarray(box['scale']),ref=box['reference']))}
            for name,value in expected.items():assert math.isclose(value,float(actual[name]),rel_tol=1e-10,abs_tol=1e-11)
            if r['method']!=plan['methods'][-1]:continue
            assert r['frozen_tensors_verified'] and not r['technical_smoke']
            assert r['mechanism_revision']=='stage2_r2_structured_modes_unified_scalar' and not r['legacy_u_calibration'] and r['q_head_every_step'] and not r['external_pulse']
            for snapshot in r['dynamic_scalarization_snapshots']:
                assert snapshot['normalization_source']=='archive_only_prefer_at_least_two_in_box'
                assert snapshot['span_floor_fraction']==.05
                np.testing.assert_allclose(snapshot['ideal'],snapshot['selected_min'],atol=0,rtol=0)
                np.testing.assert_allclose(snapshot['span'],np.maximum(np.asarray(snapshot['selected_max'])-snapshot['selected_min'],.05*np.asarray(box['scale'])),atol=1e-10,rtol=0)
            old=next(v for v in reports if v['method']==plan['methods'][0] and v['seed']==r['seed'])
            assert r['initial_population']==old['initial_population'] and r['budget_seconds']==old['budget_seconds']
            for routes,F in zip(r['initial_population']['routes'],r['initial_population']['objectives']):
                sol=pool.api.Solution([pool.api.Route(v) for v in routes]);assert pool.api.feasible(sol,instance)
                np.testing.assert_allclose(pool.api.evaluate(sol,instance).to_array(),F,atol=1e-7,rtol=0.)
                initial_count+=1
            for point in r['archive']:
                sol=pool.api.Solution([pool.api.Route(v) for v in point['routes']]);assert pool.api.feasible(sol,instance)
                np.testing.assert_allclose(pool.api.evaluate(sol,instance).to_array(),point['objectives'],atol=1e-7,rtol=0.)
                archive_count+=1
            for rec in r['records']:
                snapshot=r['dynamic_scalarization_snapshots'][rec['scalar_snapshot']]
                assert snapshot['direction']==rec['direction']
                def scalar(F):
                    rawz=(np.asarray(F)-snapshot['ideal'])/np.asarray(snapshot['span']);z=np.asarray(snapshot['weight'])*rawz
                    ref=np.asarray(box['offset'])+np.asarray(box['scale'])*np.asarray(box['reference'])
                    return float(z.max()+.05*z.sum()+.001*rawz.sum()+snapshot['penalty']*np.maximum((np.asarray(F)-ref)/np.asarray(box['scale']),0.).sum())
                before=scalar(rec['before_objectives']);after=scalar(rec['current_objectives'])
                if rec['accepted']:
                    assert after<before and math.isclose(rec['reward'],before-after,rel_tol=1e-9,abs_tol=1e-10)
                else:assert rec['reward']==0. and rec['before_objectives']==rec['current_objectives']
                assert len(rec['removed'])==len(set(rec['removed']))==rec['q']
                validate_mode(rec['action'],rec['sampling_routes'],instance.n_customers,rec['sampling_node_logits'],instance)
                assert not rec['use_pulse']
                if rec['repair_final_score'] is not None:
                    # Feasible workers enumerate insertion candidates under this identical context.
                    if rec['candidate_objectives'] is not None:assert abs(scalar(rec['candidate_objectives'])-rec['repair_final_score'])<1e-9
                assert rec['inference_seconds']>=0. and rec['repair_seconds']>=0.
                record_count+=1
    for row in means:
        selected=[r for r in metrics if r['iid']==row['iid'] and r['method']==row['method']]
        assert len(selected)==3
        for metric in ('empirical_hv','fixed_box_hv','igd_plus'):
            values=[float(r[metric]) for r in selected]
            assert math.isclose(float(row[metric+'_mean']),statistics.mean(values),rel_tol=1e-10,abs_tol=1e-11)
            assert math.isclose(float(row[metric+'_sample_sd']),statistics.stdev(values),rel_tol=1e-10,abs_tol=1e-11)
    gate=read(ROOT/'stage2_training/Q_GATE.json')
    for k,digest in enumerate(gate['checkpoint_sha256']):assert sha(ROOT/'stage2_training/checkpoints'/f'direction_{k:02d}.pt')==digest
    for k,digest in enumerate(plan['checkpoint_sha256']):assert sha(ROOT/'checkpoints'/f'direction_{k:02d}.pt')==digest
    for iid,digest in plan['input_sha256'].items():assert sha(ROOT/'inputs'/f'{iid}.json')==digest
    train=read(ROOT/'stage2_training/training_report.json')
    assert train['active_training_wall_seconds']<=4500.
    meta=[h for h in train['history'] if h['phase']=='meta']
    assert len(meta)==train['meta_batches']
    assert all(h['initial_macro_ratio_error']<=1e-4 and h['initial_micro_ratio_error']<=1e-4 and h['sample_replay_logit_error']<5e-5 and h['sample_replay_probability_error']<5e-5 for h in train['history'])
    assert train['revision']=='stage2_r2_structured_modes_unified_scalar'
    late=[h for h in meta if h['batch']>=31];adapt=[h for h in train['history'] if h['phase']=='adaptation']
    late_fraction=sum(h['mature_groups'] for h in late)/max(1,sum(h['total_groups'] for h in late));adapt_fraction=sum(h['mature_groups'] for h in adapt)/max(1,len(adapt))
    assert math.isclose(late_fraction,train['late_meta_mature_fraction']) and math.isclose(adapt_fraction,train['adaptation_mature_fraction'])
    for h in adapt:
        if h['direction'] in (3,4,14):assert all(p['tw']>0 for p in h['source_provenance'])
    trace_audit=audit_traces(train)
    frozen_audit=audit_frozen_weights()
    assert train['no_validation_or_test_training'] and train['backbone_and_context_frozen']
    for k in range(16):
        items=[h for h in train['history'] if h['phase']=='adaptation' and h['direction']==k]
        assert len(items)==train['direction_updates'][str(k)]
        assert sum(h['optimizer_steps'] for h in items)==train['direction_optimizer_steps'][str(k)]
    for entry in read(ROOT/'ABORTED_READONLY_RECEIPT.json')['files']:
        st=(ROOT/entry['path']).stat();assert (st.st_size,st.st_mtime_ns)==(entry['size'],entry['mtime_ns'])
    for name in ('STAGE2_R1_READONLY_RECEIPT.json','STAGE2_LOCAL_READONLY_RECEIPT.json'):
        for entry in read(ROOT/name)['files']:
            st=(ROOT/entry['path']).stat();assert (st.st_size,st.st_mtime_ns)==(entry['size'],entry['mtime_ns'])
    for name,digest in read(ROOT/'stage1_coordinator/SOURCE_MANIFEST.json').items():assert sha(ROOT/name)==digest
    readonly=readonly_check()
    write(out/'AUDIT.json',dict(passed=True,unique_metrics=180,means_sd=60,new_archives=36,
        feasible_archive_points=archive_count,initial_individuals=initial_count,scalar_records=record_count,
        frozen_checkpoint_bytes=16,training_coverage_verified=True,training_targets_met=train['meta_switch_gate'] and train['all_direction_targets_met'] and train['mature_quota_passed'],trace_audit=trace_audit,frozen_audit=frozen_audit,aborted_partial_preserved=True,readonly=readonly,checked_unix=time.time()))



def validate_mode(a,routes,n,logits,instance=None):
    import numpy as np
    m=a['mode'];u=float(a['u']);q=a['q'];removed=a['removed'];v=len(routes)
    assert len(removed)==len(set(removed))==q and all(1<=c<=n for c in removed)
    if m in (0,1):
        base=2 if m==0 else 3;maximum=min(max(12,math.floor(.22*n)),45) if m==0 else min(max(10,math.floor(.35*n)),75)
        expected=int(np.clip(np.rint(base*(maximum/base)**u),base,n))
        assert q==expected and a['q_condition']==q and a['effective_selection_steps']==q
        if m==0:
            anchor=a['anchor'];support=a['anchor_candidates'];M=min(n-1,max(2*(q-1),16))
            assert len(support)==len(set(support))==M and anchor not in support and 0 not in support
            assert removed[0]==anchor and removed[1:]==a['micro_selected'] and set(removed[1:])<=set(support)
            if instance is not None:
                # Match neighbor ordering using the same distance matrix, independently of sampling.
                expected_neighbors=np.argsort(instance.distance_matrix[anchor],kind='stable').tolist()
                expected_neighbors=[c for c in expected_neighbors if c not in (0,anchor)][:M]
                assert support==expected_neighbors
        else:assert removed==a['micro_selected']
    else:
        assert m==2
        upper=min(v,max(1,min(4,math.floor(.25*v))));K=int(np.clip(1+math.floor(u*upper),1,upper))
        selected=a['selected_routes'];assert len(selected)==len(set(selected))==K and all(0<=j<v for j in selected)
        assert a['q_condition']==int(np.clip(np.rint(K*n/v),2,n)) and a['effective_selection_steps']==K
        cap=min(int(np.rint(.45*n)),100);expected=list(routes[selected[0]]);over=len(expected)>cap
        if not over:
            for j in selected[1:]:
                left=cap-len(expected)
                if left<=0:break
                route=routes[j]
                if len(route)<=left:expected.extend(route)
                else:
                    order=sorted(range(len(route)),key=lambda k:(-float(logits[route[k]-1]),k))[:left]
                    expected.extend(route[k] for k in order);break
        assert removed==expected and a['q_cap']==cap and bool(a['first_route_exceeds_cap'])==over

def audit_traces(train):
    import gzip,pickle,numpy as np
    files=sorted((ROOT/'stage2_training/trace').glob('*.pkl.gz'));assert len(files)==len(train['history'])
    total=checked=0;modes=[0,0,0];maximum=0.
    def lse(x):
        x=np.asarray(x,dtype=float);M=x.max();return float(M+np.log(np.exp(x-M).sum()))
    def pl(ell,prefix,support):
        remaining=list(support);value=0.
        for j in prefix:value+=float(ell[j])-lse(np.asarray(ell)[remaining]);remaining.remove(j)
        return value
    for path in files:
        with gzip.open(path,'rb') as f:entry=pickle.load(f)
        for ep in entry['episodes']:
            for t in ep:
                validate_mode(t['action'],t['routes'],t['n'],t['logits']);modes[t['mode']]+=1;total+=1
                p=next(p for p in entry['provenance'] if p['group_id']==t['start_group'])
                assert t['source_stage']==p['source_label'] and t['iid']==p['iid']
                if p['required_mature']:assert p['source_label'] in ('Mid','Late')
                if p['source_label']=='Late':assert p['source_initial_stagnation']>=4 or p['source_acceptance_rate']<.15
                if p['source_label']=='Mid':assert len(p['source_routes'])<=.9*p['source_initial_vehicles']
            # Independent literal density on bounded samples; all transitions had online dual-ratio checks.
            for t in ep[::max(1,len(ep)//8)]:
                a=t['action'];m=t['mode'];u=t['u'];alpha=t['alpha'];beta=t['beta'];ml=t['mode_logits']
                macro=float(ml[m])-lse(ml)+(alpha-1)*math.log(u)+(beta-1)*math.log1p(-u)+math.lgamma(alpha+beta)-math.lgamma(alpha)-math.lgamma(beta)
                ell=t['logits']
                if m==0:micro=float(ell[a['anchor']-1])-lse(ell)+pl(ell,[c-1 for c in a['micro_selected']],[c-1 for c in a['anchor_candidates']])
                elif m==1:micro=pl(ell,[c-1 for c in a['micro_selected']],range(t['n']))
                else:micro=pl(t['route_logits'],a['selected_routes'],range(len(t['routes'])))
                maximum=max(maximum,abs(macro-t['logprob_macro']),abs(micro-t['logprob_micro']));checked+=1
    assert total==train['optimized_transitions'] and maximum<1e-7 and all(modes)
    return dict(trajectories_action_transitions=total,mode_counts=modes,independent_density_samples=checked,max_density_error=maximum)

def audit_frozen_weights():
    import torch
    initial=torch.load(ROOT/'stage2_training/initialized_meta.pt',map_location='cpu',weights_only=False)['model']
    names=[k for k in initial if k.startswith(('encoder.','context.','q_history_norm.'))]
    for path in [ROOT/'stage2_training/meta_retrained.pt']+sorted((ROOT/'stage2_training/checkpoints').glob('direction_*.pt')):
        state=torch.load(path,map_location='cpu',weights_only=False)['model']
        assert all(torch.equal(initial[k],state[k]) for k in names)
    return dict(checkpoints=17,frozen_tensor_count=len(names),torch_equal=True)

if __name__=='__main__':main()
