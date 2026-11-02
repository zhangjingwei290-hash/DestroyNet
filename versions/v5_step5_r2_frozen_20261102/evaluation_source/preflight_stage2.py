"""Finite FIT-only r2 tests; all updated tensors and advanced states are discarded."""
from stage2_common import ROOT,read,write,sha,reset,load_cfg,FitPool,require_stage1
import os,sys,time,subprocess,traceback

def main():
    gate=require_stage1();began=time.perf_counter()
    subprocess.run([sys.executable,'-B','-X','utf8',str(ROOT/'stage2_math_units.py')],cwd=ROOT,check=True)
    import psutil,numpy as np,torch
    from solver.model import Policy
    from solver.config import directions
    from solver.training import optimizer_for,frozen_state,assert_frozen,update,group_advantages
    from stage2_initialize import initialize
    from stage2_runtime import Runtime
    from stage2_cpu import CpuHead,struct_state
    psutil.Process().cpu_affinity([0,1]);torch.set_num_threads(1);torch.set_num_interop_threads(1);reset(20261024)
    cfg=load_cfg();pool=FitPool(cfg);model,transfer=initialize(Policy(cfg,tasks=5).cuda())
    opt=optimizer_for(model);frozen=frozen_state(model);initial={k:v.detach().clone() for k,v in model.state_dict().items()}
    import stage1_mechanisms as mechanism
    assert 'stage2_repo' in str(mechanism.__file__)
    from stage2_state_pool import StatePool
    from pathlib import Path
    from stage2_schedule import schedules,set_lr,cosine
    from stage2_blend import initialize_direction
    schedule=schedules();assert len(schedule['meta'])==720
    dummy=optimizer_for(model);assert set_lr(dummy,0,144)==[.003,.001];np.testing.assert_allclose(set_lr(dummy,143,144),[.0005,.0002],atol=1e-15)
    parent=read(ROOT/'PARENT_R2.json');poolpath=Path(parent['archive'])/'stage2_training/progressive_state_pool.pt';assert sha(poolpath)==parent['r2_pool_sha256']
    statepool=StatePool(pool);restore=statepool.restore(torch.load(poolpath,map_location='cpu',weights_only=False))
    runtime=Runtime();episodes=[];sources=[];store={'progressive_pool':statepool};rng=np.random.default_rng(20261024);refs=directions()
    try:
        ids=pool.manifest['fit_order'][:3]
        for task,iid in enumerate(ids):
            ep,p=runtime.collect(model,pool,store,iid,refs,task,task,int(rng.integers(2**31-1)),rng,time.perf_counter()+90.,f'preflight:{task}')
            assert p['valid_complete_group'];episodes.extend(ep);sources.append(p)
        # A real matured state must be advanced and classified without fabricated stagnation/progress.
        ep,p=runtime.collect(model,pool,store,ids[0],refs,0,0,int(rng.integers(2**31-1)),rng,time.perf_counter()+90.,'preflight:mature',require_mature=True)
        assert p['valid_complete_group'] and p['source_label'] in ('Mid','Late');episodes.extend(ep);sources.append(p)
        from verify_stage2 import validate_mode
        for ep in episodes:
            for t in ep:validate_mode(t['action'],t['routes'],t['n'],t['logits'],pool.load(t['iid'])[0])
        groups,advs,weights=group_advantages(episodes)
        assert len(groups)==4 and all(abs(weights[id(ep)].sum()-.25)<1e-10 for ep in episodes)
        metrics=update(model,opt,episodes,cfg,time.perf_counter()+180.,frozen)
        assert metrics['optimizer_steps']==cfg.ppo_epochs and not metrics.get('cutoff_rolled_back')
        assert any(not torch.equal(initial[k],v) for k,v in model.state_dict().items() if '.q_mu.' in k)
        assert all(v>0 for v in metrics['family_gradient_norms'].values()),metrics['family_gradient_norms']
        assert all(v>0 for v in metrics['family_parameter_delta_norms'].values()),metrics['family_parameter_delta_norms']
        assert_frozen(model,frozen)
        assert all(not p.requires_grad for name,p in model.named_parameters() if name.startswith('encoder.'))
        assert all(p.requires_grad for name,p in model.named_parameters() if name.startswith(('context.','q_history_norm.')))
        assert any(not torch.equal(initial[k],v) for k,v in model.state_dict().items() if k.startswith('context.'))
        assert any(not torch.equal(initial[k],v) for k,v in model.state_dict().items() if k.startswith('q_history_norm.'))
        cpu=CpuHead(model);errors=[];times=[]
        for ep in episodes:
            for t in ep[::max(1,len(ep)//3)]:
                out=model.replay_terms([t]);start=time.perf_counter()
                logits,a,b,_=cpu.outputs(t['hidden'],struct_state(t['physical']),t['raw'],t['n'],t['dynamic_position'],t['task']);times.append((time.perf_counter()-start)*1e6)
                m=t['mode'];errors.extend([abs(float(out['alpha'].detach())-a[m]),abs(float(out['beta'].detach())-b[m])])
        assert max(errors)<5e-5
        # Independent literal credit fixture: no time-efficiency division or large-q penalty.
        fixture=[]
        for j in range(4):
            ep=[]
            for i in range(10):
                reward=1. if (j==0 and i==7) else .2 if (j==1 and i==2) else 0.
                ep.append(dict(start_group='unit',source_routes_at_start=[[1,2]],terminal=i==9,truncated=False,
                    stagnation=5 if i==7 else 0,reward=reward,n=50,
                    diagnostics=dict(local_budget=2.5,ending_reason='local_budget' if i==9 else 'continuing',before_score=3.,current_score=3.-(1. if j==0 else .2 if j==1 else 0.),new_nd_entered=(j==0 and i==7),before_objectives=[2,0,0,0,0],candidate_objectives=[1 if j==2 and i==3 else 2,0,0,0,0])))
            fixture.append(ep)
        _,actual,_=group_advantages(fixture)
        G=np.asarray([[sum(.85**k*ep[i+k]['reward'] for k in range(min(8,len(ep)-i))) for i in range(len(ep))] for ep in fixture]);norm=(G-G.mean())/(G.std()+1e-8)
        positive=[t['reward'] for ep in fixture for t in ep if t['reward']>0];returns=np.asarray([1.+.15*np.mean(positive),.2,0.,0.]);tr=(returns-returns.mean())/(returns.std()+1e-8)
        norm[0,7]+=.75;norm[2,3]+=.75
        for j,ep in enumerate(fixture):np.testing.assert_allclose(actual[id(ep)],.6*tr[j]+.4*norm[j],atol=1e-10,rtol=0.)
        # Literal parent/meta blend, no weighted rounding of the bit-frozen encoder.
        meta=model.deploy(0);parent_model=model.deploy(1);child=model.deploy(0)
        initialize_direction(child,parent_model,meta)
        for key,value in child.state_dict().items():
            expected=meta.state_dict()[key] if key.startswith('encoder.') else .7*parent_model.state_dict()[key]+.3*meta.state_dict()[key]
            assert torch.equal(value,expected)
        # Rebinding to a new preference cannot immediately claim the old preference's Late label.
        statepool=store['progressive_pool'];slot=statepool.get(ids[0],0,refs[1]);assert slot['needs_advance'] and statepool.label(slot)=='Early'
        oldslot=statepool.get(ids[0],0,refs[1]);oldslot['used']=oldslot['budget'];oldgeneration=oldslot.get('generation',0)
        fresh=statepool.get(ids[0],0,refs[1]);assert fresh['used']==0 and fresh['generation']==oldgeneration+1 and fresh['env'].routes==pool.load(ids[0])[1]['source_solution'] and statepool.label(fresh)=='Early'
        eligible=statepool.eligible(50,True);assert eligible and all(int(i.split('_')[4][2:])>0 for i in eligible)
        mode_counts={str(m):sum(t['mode']==m for ep in episodes for t in ep) for m in range(3)};assert all(mode_counts.values())
        model.load_state_dict(initial)
        from solver.outer import Search
        deployed=model.deploy(0);deployed.cpu_head=CpuHead(deployed);deployment=[]
        for iid in ids:
            se=Search(pool,iid,cfg,20261022,policies=[deployed]*16,seconds=2.5);se.run();rep=se.report();assert rep['records'];deployment.append(dict(iid=iid,steps=len(rep['records']),budget=rep['budget_seconds']))
        assert all(torch.equal(initial[k],v) for k,v in model.state_dict().items())
        result=dict(passed=True,revision='stage2_r3_full12pass_co_adapted',span_floor_fraction=.05,gate=gate,
            math_units=read(ROOT/'STAGE2_MATH_UNITS.json'),real_trajectories=len(episodes),groups=4,transitions=sum(map(len,episodes)),provenance=sources,
            warm_transfer=transfer,deployment_fit_smoke=deployment,state_pool_restoration=restore,schedule_checked=True,projection_updates_verified=True,all_micro_families_gradient_and_update_verified=True,horizon_reset_to_true_fresh_source_verified=True,parent_meta_blend_verified=True,no_total_training_cutoff=True,cosine_endpoints_verified=True,ppo_metrics=metrics,cpu_gpu_after_update_max_error=max(errors),cpu_q_head_with_structure_mean_us=float(np.mean(times)),
            all_frozen_parameters_equal=True,preflight_updates_discarded=True,credit_independent_fixture_passed=True,mode_counts=mode_counts,
            mature_state_real_advance_verified=True,preference_rebind_reset_verified=True,dead_direction_tw_positive_filter_verified=True,
            isolated_mechanism_module=str(mechanism.__file__),wall_seconds=time.perf_counter()-began,repair_worker_pids=runtime.worker_pids)
        write(ROOT/'PREFLIGHT_STAGE2.json',result);print(result,flush=True)
    finally:runtime.close()

if __name__=='__main__':
    import multiprocessing as mp
    mp.freeze_support()
    try:main()
    except BaseException:
        write(ROOT/'PREFLIGHT_STAGE2_FAILURE.json',dict(traceback=traceback.format_exc(),unix=time.time()));raise
