"""Final independent r3 training coverage, frozen-source and metric verification."""
from stage2_common import ROOT,read,write,sha,readonly_check
from pathlib import Path
import statistics,math,time

def json_list(value):
    import ast
    return ast.literal_eval(value)

def main():
    import psutil,torch,csv,numpy as np
    from collections import Counter
    from stats import signed_rank,wtl
    from verify_stage2 import audit_traces
    from stage2_schedule import schedules,cosine
    from stage2_reporting import compute,primary_path,old_path
    psutil.Process().cpu_affinity([14,15]);torch.set_num_threads(1)
    train=read(ROOT/'stage2_training/training_report.json');schedule=schedules();assert train['hard_limit_seconds'] is None and not train['cutoff_reached']
    assert train['meta_batches']==144 and train['meta_merges']==36 and set(train['fit_visit_counts'].values())=={4}
    meta=[h for h in train['history'] if h['phase']=='meta'];adapt=[h for h in train['history'] if h['phase']=='adaptation']
    assert len(meta)==144 and len(adapt)==1440
    actual=[p['iid'] for h in meta for p in h['source_provenance']];assert actual==[x['iid'] for x in schedule['meta']]
    for h in train['history']:
        assert h['initial_macro_ratio_error']<=1e-4 and h['initial_micro_ratio_error']<=1e-4 and h['sample_replay_logit_error']<5e-5 and h['sample_replay_probability_error']<5e-5 and h['optimizer_steps']==2
        index=h['batch']-1 if h['phase']=='meta' else h['update_batches']-1;total=144 if h['phase']=='meta' else 90
        np.testing.assert_allclose(h['learning_rates'],[cosine(index,total,.003,.0005),cosine(index,total,.001,.0002)],atol=1e-15)
    for k in range(16):
        rows=[h for h in adapt if h['direction']==k];assert len(rows)==90 and sum(h['mature_groups'] for h in rows)>=54
        assert [h['source_provenance'][0]['iid'] for h in rows]==[r['iid'] for r in schedule['adaptation'][str(k)]]
        assert len({tuple(p['scheduled']['cell']) for h in rows for p in h['source_provenance']})==(36 if k in (3,4,14) else 45)
        if k in (3,4,14):assert all(p['tw']>0 for h in rows for p in h['source_provenance'])
    assert all(any(h['family_gradient_norms'][family]>0 and h['family_parameter_delta_norms'][family]>0 for h in train['history']) for family in ('context','q_history_norm','film','node','physical_bilinear','route_bilinear'))
    assert all(h['mature_groups']>=3 for h in meta if h['batch']>=19)
    # Every direction has two canonical design-cell passes, with different replica per base cell.
    for k in range(16):
        selected=[h['source_provenance'][0]['scheduled'] for h in adapt if h['direction']==k]
        for cell_pass in (1,2):assert len({tuple(p['design_cell']) for p in selected if p['cell_pass']==cell_pass})==45
        for first,second in zip(selected[:45],selected[45:]):assert first['replica']!=second['replica']
    initial=torch.load(ROOT/'stage2_training/initialized_meta.pt',map_location='cpu',weights_only=False)['model'];encoder=[k for k in initial if k.startswith('encoder.')]
    for path in [ROOT/'stage2_training/meta_retrained.pt']+sorted((ROOT/'stage2_training/checkpoints').glob('direction_*.pt')):
        state=torch.load(path,map_location='cpu',weights_only=False)['model'];assert all(torch.equal(initial[k],state[k]) for k in encoder)
    for path in sorted((ROOT/'stage2_training/direction_initialization').glob('direction_*.pt')):
        item=torch.load(path,map_location='cpu',weights_only=False);meta_state=torch.load(ROOT/'stage2_training/meta_retrained.pt',map_location='cpu',weights_only=False)['model']
        parent=item['parent'];base=torch.load(ROOT/'stage2_training/checkpoints'/f'direction_{parent:02d}.pt',map_location='cpu',weights_only=False)['model'] if parent is not None else None
        for name,value in item['model'].items():
            expected=meta_state[name] if name.startswith('encoder.') or base is None else .7*base[name]+.3*meta_state[name]
            assert torch.equal(value,expected)
    assert len(list((ROOT/'stage2_training/adaptation_boundaries').glob('*.pt')))==16*6
    trace=audit_traces(train)
    plan=read(ROOT/'plan.json');selection=read(ROOT/'MODEL_SELECTION.json');methods=['alns_tuned','v5_meta_frozen','step4_k4_adapted',selection['method']]
    # Independently reconstruct ND geometry and metrics, without the reporting compute helper.
    import moocore
    folder=ROOT/'stage2_analysis_45iid_20seed'
    with (folder/'per_run_metrics.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    assert len(rows)==len({(r['iid'],r['seed'],r['method']) for r in rows})
    expected={};missing=[]
    for iid in plan['test_ids']:
        reports=[]
        for method in methods:
            for seed in plan['search_seeds']:
                path=primary_path(iid,seed) if method==selection['method'] else old_path(iid,seed,method)
                if not path.exists():missing.append(dict(iid=iid,seed=seed,method=method));continue
                reports.append((method,seed,read(path)))
        F=np.unique(np.concatenate([np.asarray([p['objectives'] for p in d['archive']]) for _,_,d in reports]),axis=0)
        nd=F[moocore.is_nondominated(F)];ideal=nd.min(0);span=nd.max(0)-ideal;zero=span/np.asarray(reports[0][2]['physical_scale'])<=1e-12
        def norm(values):return np.where(zero,0.,(np.asarray(values)-ideal)/np.where(zero,1.,span))
        for method,seed,d in reports:
            values=np.asarray([p['objectives'] for p in d['archive']]);physical=(values-d['physical_offset'])/d['physical_scale'];ref=(np.asarray(d['physical_reference'])-d['physical_offset'])/d['physical_scale']
            expected[iid,str(seed),method]=dict(empirical_hv=float(moocore.hypervolume(norm(values),ref=[1.1]*5)),fixed_box_hv=float(moocore.hypervolume(physical,ref=ref)),igd_plus=float(moocore.igd_plus(norm(values),ref=norm(nd))))
    assert len(rows)==len(expected)
    for row in rows:
        e=expected[row['iid'],row['seed'],row['method']]
        for metric in ('empirical_hv','fixed_box_hv','igd_plus'):assert math.isclose(float(row[metric]),e[metric],rel_tol=1e-10,abs_tol=1e-11)
    with (folder/'main_table.csv').open(encoding='utf-8-sig') as f:mean_rows=list(csv.DictReader(f))
    for row in mean_rows:
        subset=[r for r in rows if r['iid']==row['iid'] and r['method']==row['method']];assert int(row['seeds'])==len(subset)
        for metric in ('empirical_hv','fixed_box_hv','igd_plus'):
            values=[float(r[metric]) for r in subset]
            assert math.isclose(float(row[metric+'_mean']),statistics.mean(values),rel_tol=1e-10,abs_tol=1e-11)
            assert math.isclose(float(row[metric+'_sample_sd']),statistics.stdev(values),rel_tol=1e-10,abs_tol=1e-11)
    with (folder/'comparisons.csv').open(encoding='utf-8-sig') as f:comparisons=list(csv.DictReader(f))
    from scipy.stats import rankdata
    for scope in ('overall','50','150','250'):
        family=[c for c in comparisons if c['n']==scope]
        for c in family:
            differences=[];run_differences=[];metric=c['metric'];sign=-1 if metric=='igd_plus' else 1
            for iid in plan['test_ids']:
                if scope!='overall' and int(iid.split('-')[0])!=int(scope):continue
                dd=[sign*(expected[iid,str(seed),selection['method']][metric]-expected[iid,str(seed),c['baseline']][metric]) for seed in plan['search_seeds'] if (iid,str(seed),c['baseline']) in expected]
                if dd:differences.append(statistics.mean(dd));run_differences+=dd
            assert json_list(c['iid_wtl'])==wtl(differences) and json_list(c['run_wtl'])==wtl(run_differences)
            nonzero=np.asarray([d for d in differences if abs(d)>1e-12]);ranks=np.rint(rankdata(abs(nonzero))*2).astype(int) if len(nonzero) else np.asarray([],dtype=int)
            ways=Counter({0:1})
            for rank in ranks:
                next_ways=ways.copy()
                for total,count in ways.items():next_ways[total+int(rank)]+=count
                ways=next_ways
            total=int(ranks.sum());positive=int(ranks[nonzero>0].sum());observed=min(positive,total-positive)
            probability=sum(count for x,count in ways.items() if min(x,total-x)<=observed)/2**len(ranks)
            assert abs(float(c['p'])-probability)<1e-12;c['_p']=probability
        previous=0.
        for j,c in enumerate(sorted(family,key=lambda c:c['_p'])):
            previous=max(previous,min(1.,c['_p']*(len(family)-j)));assert abs(float(c['p_holm'])-previous)<1e-12
    assert sum(r['method']==selection['method'] for r in rows)==900
    assert all(r['method']=='step4_k4_adapted' for r in missing)
    # Every primary search must be covered by a successful raw physical audit, including reused r2 core.
    covered=set()
    for path in (ROOT/'audits').glob('*.json'):
        audit=read(path)
        if audit.get('passed'):
            for row in audit.get('files',[]):
                assert sha(Path(row['raw_path']))==row['raw_sha256'] and sha(Path(row['compact_path']))==row['compact_sha256']
                covered.add((row['iid'],row['seed'],row['compact_path']))
    assert all((i,s,str(primary_path(i,s))) in covered for i in plan['test_ids'] for s in plan['search_seeds'])
    readonly=readonly_check()
    for name,digest in read(ROOT/('stage2_coordinator/EFFECTIVE_SOURCE_MANIFEST.json' if (ROOT/'stage2_coordinator/EFFECTIVE_SOURCE_MANIFEST.json').exists() else 'stage2_coordinator/SOURCE_MANIFEST.json')).items():assert sha(ROOT/name)==digest,name
    write(ROOT/'AUDIT.json',dict(passed=True,revision=train['revision'],selected_model=selection['selected'],selected_runs=900,r3_gate_runs=36,unique_metrics=len(rows),mean_sd_rows=len(mean_rows),step4_missing=len(missing),meta_groups_exact720=True,adaptation_batches1440=True,total_groups2160=True,trace=trace,encoder_equal=True,readonly=readonly,checked_unix=time.time()))
if __name__=='__main__':main()
