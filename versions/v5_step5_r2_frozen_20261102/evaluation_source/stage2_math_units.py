"""Finite numerical tests independent of the long training entry."""
from stage2_common import ROOT,read,write,load_cfg,FitPool,reset,require_stage1
import itertools,time,math,copy

def run():
    import psutil,numpy as np,torch
    from solver.model import Policy,prefix_terms_torch,route_features_torch
    from solver.environment import Environment,Adaptive
    from solver.outer import FixedTchebycheff,RoutedMOEAD
    from stage1_mechanisms import dynamic_coordinates
    from stage2_structured import prefix_logprob,route_context,route_removal
    from stage2_cpu import CpuHead,sample
    from stage2_initialize import initialize
    require_stage1();psutil.Process().cpu_affinity([14,15]);torch.set_num_threads(1);torch.set_num_interop_threads(1);reset(20261024)
    cfg=load_cfg();pool=FitPool(cfg);model,transfer=initialize(Policy(cfg).cuda());model.eval();head=CpuHead(model);rng=np.random.default_rng(20261024)
    from solver.repair import score_candidates
    scalar_max=0.;errors=[];mode_seen=set();route_pool_us=[];states=[]
    for iid in read(ROOT/'FIT_MANIFEST.json')['fit_order'][:3]:
        env=Environment(pool,iid,cfg,np.ones(5)/5)
        ref=env.offset+env.scales*np.asarray(env.box['reference'])
        env.dynamic_ideal,env.dynamic_span=dynamic_coordinates([env.objectives],env.scales,ref)
        context=env.scalar_context();F=np.vstack((env.objectives,ref+env.scales*.1,np.zeros(5)))
        z=(F-env.dynamic_ideal)/env.dynamic_span;w=env.preference
        expected=(z*w).max(-1)+.05*(z*w).sum(-1)+.001*z.sum(-1)+np.maximum((F-ref)/env.scales,0.).sum(-1)
        repair=score_candidates(F,w,context);actual=np.asarray([env.score(v) for v in F])
        decomp=FixedTchebycheff(env.offset,env.scales,ref);decomp.dynamic_ideal=env.dynamic_ideal;decomp.dynamic_span=env.dynamic_span
        decomposition=decomp.do(F,weights=np.tile(w,(len(F),1)))
        scalar_max=max(scalar_max,float(np.max(np.abs(repair-expected))),float(np.max(np.abs(actual-expected))),float(np.max(np.abs(decomposition-expected))))
        assert scalar_max<1e-9
        obs=env.observation(torch.zeros((1,128)),'greedy',Adaptive(len(env.q_values)),100.,200.,5.,10.)
        cache=model.refresh(obs,0,True);physical=obs['x'][1:,[18,17,19,12,13]].numpy();raw=obs['context'].numpy()
        t=time.perf_counter();rc=route_context(env.instance,env.routes,physical);route_pool_us.append((time.perf_counter()-t)*1e6)
        zz=(env.objectives-env.dynamic_ideal)/env.dynamic_span;position=np.asarray([zz.mean(),(w*zz).max()],dtype=np.float32)
        for mode in range(3):
            for u in (.05,.40,.95):
                action,state=sample(cache,physical,raw,env.instance.n_customers,rng,env.neighbors,head,rc,position,0,mode,u)
                state.update(source_routes_at_start=env.routes,stagnation=env.stagnation)
                result=model.replay_terms([state]);macro=float(result['logprob_macro'].detach());micro=float(result['logprob_micro'].detach())
                em=abs(math.exp(macro-state['logprob_macro'])-1);ei=abs(math.exp((micro-state['logprob_micro'])/state['effective_selection_steps'])-1)
                le=float((result['logits'][0].detach().cpu()-torch.tensor(state['logits'])).abs().max())
                pe=float((result['mode_probabilities'][0].detach().cpu()-torch.tensor(state['mode_probabilities'])).abs().max())
                if mode==2:le=max(le,float((result['route_logits'][0,:len(env.routes)].detach().cpu()-torch.tensor(state['route_logits'])).abs().max()))
                assert em<=1e-4 and ei<=1e-4 and le<5e-5 and pe<5e-5,(iid,mode,u,em,ei,le,pe)
                result['logprob_macro'].sum().backward(retain_graph=True);result['logprob_micro'].sum().backward()
                assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
                model.zero_grad(set_to_none=True);errors.append((em,ei,le,pe));states.append(state);mode_seen.add(mode)
        phi,_,_=route_features_torch(torch.tensor(physical)[None].cuda(),torch.tensor(rc['assignment'])[None].cuda(),torch.tensor(rc['demands'])[None].cuda(),torch.tensor([rc['capacity']]).cuda())
        np.testing.assert_allclose(phi[0,:len(env.routes)].cpu(),rc['normalized'],atol=5e-5,rtol=0.)
    logits=np.asarray([.2,-.4,.9],dtype=np.float32)
    for count in (1,2,3):
        orders=list(itertools.permutations(range(3),count));probability=sum(math.exp(prefix_logprob(logits,o)) for o in orders)
        assert abs(probability-1)<1e-12
        gpu,entropy=prefix_terms_torch(torch.tensor(logits)[None].cuda(),[orders[0]],torch.ones((1,3),dtype=torch.bool,device='cuda'))
        assert abs(float(gpu.detach())-prefix_logprob(logits,orders[0]))<1e-6 and torch.isfinite(entropy).all()
    # Approved first-whole priority exception remains deterministic and auditable.
    routes=[list(range(1,31)),list(range(31,51))]
    removed,cap,over=route_removal(routes,[0],np.arange(50),50)
    assert removed==routes[0] and cap==22 and over
    from pymoo.core.population import Population
    alg=object.__new__(RoutedMOEAD);alg.neighbors=np.asarray([[0,1,2,3]]*4);alg.ref_dirs=np.ones((4,5))*.2;alg.ideal=np.zeros(5)
    alg.pop=Population.new('F',np.asarray([[20.]*5,[4.,6.,6.,6.,6.],[4.,10.,10.,10.,10.],[4.,30.,30.,30.,30.]]))
    class D:
        def do(self,F,**kw):return np.asarray(F).sum(-1)
    alg.decomposition=D();off=Population.new('F',np.asarray([[5.]*5]))[0];alg._replace(0,off)
    assert np.array_equal(alg.pop[0].F,off.F) and np.array_equal(alg.pop[3].F,off.F) and np.array_equal(alg.pop[2].F,off.F)
    assert not np.array_equal(alg.pop[1].F,off.F)
    result=dict(passed=True,revision='stage2_r3_full12pass_co_adapted',forced_samples=len(states),modes=sorted(mode_seen),
        scalar_max_error=scalar_max,max_macro_ratio_error=max(v[0] for v in errors),max_micro_ratio_error=max(v[1] for v in errors),
        max_logit_error=max(v[2] for v in errors),max_probability_error=max(v[3] for v in errors),
        route_feature_construction_us=route_pool_us,first_whole_exception_verified=True,prefix_probability_normalizes=True,
        sorted_top2_snapshot_replacement_verified=True,pretrained_sha256=transfer['source_sha256'],gradient_finite=True)
    write(ROOT/'STAGE2_MATH_UNITS.json',result);print(result,flush=True)
    return states

if __name__=='__main__':
    import traceback
    try:run()
    except BaseException:
        write(ROOT/'STAGE2_MATH_UNITS_FAILURE.json',dict(traceback=traceback.format_exc()));raise
