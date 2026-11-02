"""Independent physical/scalar/action audit of only newly requested frozen searches."""
from stage2_common import ROOT,read,write,sha,load_cfg,EvalPool
from pathlib import Path
import time,math

def main():
    import psutil,numpy as np,argparse
    from verify_stage2 import validate_mode
    from stage2_reporting import primary_path
    psutil.Process().cpu_affinity([14,15]);ap=argparse.ArgumentParser();ap.add_argument('--ids',nargs='+',required=True);ap.add_argument('--seeds',nargs='+',type=int,required=True);ap.add_argument('--gate-r3',action='store_true');args=ap.parse_args()
    cfg=load_cfg();pool=EvalPool(cfg);plan=read(ROOT/'plan.json');receipt=[];points=initials=records=0
    for iid in args.ids:
        inst,row,box,_=pool.load(iid);n=inst.n_customers
        for seed in args.seeds:
            compact=(ROOT/'evaluation_core_r3/metric_inputs'/f'{iid}_seed{seed}_step5_stage2_r3.json') if args.gate_r3 else primary_path(iid,seed)
            raw=compact.parent.parent/compact.name;digest=sha(raw);d=read(raw)
            assert d['iid']==iid and d['seed']==seed and d['frozen_tensors_verified'] and not d['technical_smoke']
            assert d['budget_seconds']==cfg.solve_seconds_by_instance[iid] and len(d['initial_population']['routes'])==16
            old=read(Path(plan['baseline_root'])/'metric_inputs/results'/f'{iid}_seed{seed}_alns_tuned.json')
            assert d['initial_population']==old['initial_population'] and d['physical_offset']==old['physical_offset'] and d['physical_scale']==old['physical_scale']
            for routes,F in zip(d['initial_population']['routes'],d['initial_population']['objectives']):
                sol=pool.api.Solution([pool.api.Route(v) for v in routes]);assert pool.api.feasible(sol,inst);np.testing.assert_allclose(pool.api.evaluate(sol,inst).to_array(),F,atol=1e-7,rtol=0.);initials+=1
            for point in d['archive']:
                sol=pool.api.Solution([pool.api.Route(v) for v in point['routes']]);assert pool.api.feasible(sol,inst);np.testing.assert_allclose(pool.api.evaluate(sol,inst).to_array(),point['objectives'],atol=1e-7,rtol=0.);points+=1
            mode=[dict(steps=0,accepted=0,vehicle_reductions=0,accepted_vehicle_reductions=0,first_route_cap_exceptions=0) for _ in range(3)]
            for rec in d['records']:
                snap=d['dynamic_scalarization_snapshots'][rec['scalar_snapshot']];assert snap['direction']==rec['direction'] and snap['span_floor_fraction']==.05
                np.testing.assert_allclose(snap['span'],np.maximum(np.asarray(snap['selected_max'])-snap['selected_min'],.05*np.asarray(box['scale'])),atol=1e-10,rtol=0.)
                def scalar(F):
                    z=(np.asarray(F)-snap['ideal'])/snap['span'];wz=z*np.asarray(snap['weight']);ref=np.asarray(box['offset'])+np.asarray(box['scale'])*np.asarray(box['reference'])
                    return float(wz.max()+.05*wz.sum()+.001*z.sum()+np.maximum((np.asarray(F)-ref)/box['scale'],0.).sum())
                before=scalar(rec['before_objectives']);after=scalar(rec['current_objectives'])
                if rec['accepted']:assert after<before and math.isclose(rec['reward'],before-after,rel_tol=1e-9,abs_tol=1e-10)
                else:assert rec['reward']==0 and rec['before_objectives']==rec['current_objectives']
                if rec.get('repair_final_score') is not None and rec['candidate_objectives'] is not None:assert abs(scalar(rec['candidate_objectives'])-rec['repair_final_score'])<1e-9
                validate_mode(rec['action'],rec['sampling_routes'],n,rec.get('sampling_node_logits'),inst)
                refresh=rec['cache_step']==0 or (rec['dirty_since_refresh_before'] and rec['repair_seconds_since_refresh_before']>=5*max(rec['last_gnn_refresh_seconds_before'],1e-4)) or (not rec['dirty_since_refresh_before'] and rec['cache_age_before']>=8)
                assert rec['cache_refresh']==refresh and rec['cache_hit']==(not refresh)
                assert rec['cache_age_after']==(1 if refresh else rec['cache_age_before']+1)
                assert rec['inference_seconds']>=0 and rec['repair_seconds']>=0
                m=mode[rec['mode']];m['steps']+=1;m['accepted']+=int(rec['accepted']);m['first_route_cap_exceptions']+=int(rec['first_route_exceeds_cap'])
                m['vehicle_reductions']+=int(rec['candidate_objectives'] is not None and rec['candidate_objectives'][0]<rec['before_objectives'][0]);m['accepted_vehicle_reductions']+=int(rec['accepted'] and rec['current_objectives'][0]<rec['before_objectives'][0]);records+=1
            assert len(d['records'])==d['steps'] and sum(int(v) for v in d['q_histogram'].values())==d['steps']
            for m in range(3):
                for key in mode[m]:assert mode[m][key]==d['mode_diagnostics'][m][key]
            compact_d=read(compact);assert compact_d['archive']==d['archive'] and compact_d['checkpoint_sha256']==d['checkpoint_sha256']
            receipt.append(dict(iid=iid,seed=seed,raw_path=str(raw),raw_sha256=digest,compact_path=str(compact),compact_sha256=sha(compact),steps=d['steps'],checkpoint_sha256=d['checkpoint_sha256']))
    tag=('gate_r3' if args.gate_r3 else 'selected')+'_'+str(args.seeds[0])+'_'+str(args.seeds[-1])+'_'+str(len(args.ids))
    write(ROOT/'audits'/f'{tag}.json',dict(passed=True,runs=len(receipt),initial_individuals=initials,feasible_archive_points=points,scalar_action_records=records,files=receipt,checked_unix=time.time()))
    print(dict(passed=True,runs=len(receipt),archive_points=points,records=records),flush=True)
if __name__=='__main__':main()
