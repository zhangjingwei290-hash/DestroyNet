"""Meaningful step2 contract checks, including real four-replica10s collection."""
from pilot_common import ROOT,read,write,reset
import sys,time,copy,os,math
sys.path.insert(0,str(ROOT/'step2_repo'))
import numpy as np
import torch
import psutil
from run import load_config
from solver.data import TrainPool
from solver.model import Policy,integer_q
from solver.config import directions
from solver.environment import Environment,Adaptive
from solver.training import group_advantages,update
from solver.parallel_runtime import ParallelRuntime

def main():
    out=ROOT/'step2_preflight';out.mkdir(exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);reset(20261024)
    psutil.Process().cpu_affinity(list(range(psutil.cpu_count())))
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    cfg=load_config(ROOT/'config.json');pool=TrainPool(cfg)
    ck=torch.load(ROOT/'checkpoints/meta_complete.pt',map_location='cuda',weights_only=False)
    policy=Policy(cfg).cuda();source={k:v for k,v in ck['model'].items() if k in policy.state_dict()}
    assert set(ck['model'])-set(source)=={'heads.0.q_regime_mu'}
    policy.load_state_dict(source)
    with torch.no_grad():
        policy.heads[0].q_mu.weight.zero_();policy.heads[0].q_mu.bias.fill_(-1.);policy.heads[0].q_state_mu.zero_()
    observations=[];calibration=[];sources=[];kappas=[]
    for n in (50,150,250):
        iid=next(i for i in pool.split['fit_order'] if pool.load(i)[0].n_customers==n)
        env=Environment(pool,iid,cfg,directions()[15]);obs=env.observation(torch.zeros(1,128),'greedy',Adaptive(10),100.,200.,10.,10.)
        observations.append(obs);sources.append((n,iid))
        with torch.no_grad():
            h,nodes,x,raw=policy.encode([obs]);a,b,_=policy.q_beta_params(h,nodes,x,raw,[0])
            kappas.append(float(a+b-2))
    # Calibrate the random Beta mean, not its mode. Calibration uses only these
    # three FIT states and never uses evaluation objectives or pilot outcomes.
    target_u=(.19-.01)/.59
    location=(target_u-np.mean([1/(k+2) for k in kappas]))/np.mean([k/(k+2) for k in kappas])
    probability=(location-.06)/.84
    neutral_bias=math.log(probability/(1-probability))
    with torch.no_grad():policy.heads[0].q_mu.bias.fill_(neutral_bias)
    for (n,iid),obs in zip(sources,observations):
        with torch.no_grad():
            h,nodes,x,raw=policy.encode([obs]);a,b,mu=policy.q_beta_params(h,nodes,x,raw,[0])
            draws=torch.distributions.Beta(a,b).sample((10000,));q=integer_q(draws,n)
            sample_mean=float((q.float()/n).mean());small=float((q.float()/n<=.15).float().mean())
            assert .18<sample_mean<.20,(n,sample_mean)
            original=policy.heads[0].q_mu.bias.clone()
            policy.heads[0].q_mu.bias.fill_(-8.);_,_,low=policy.q_beta_params(h,nodes,x,raw,[0])
            policy.heads[0].q_mu.bias.fill_(8.);_,_,high=policy.q_beta_params(h,nodes,x,raw,[0])
            policy.heads[0].q_mu.bias.copy_(original)
            assert float(high-low)>.8,'Output still constrained to an intermediate box'
        actions,lp,_,_=policy.act([obs]);_,replayed,_,_,_=policy.terms([obs],actions=actions)
        assert abs(lp[0]-float(replayed[0]))<2e-4
        calibration.append(dict(n=n,iid=iid,mean_q_ratio=sample_mean,small_le15_fraction=small,replay_error=abs(lp[0]-float(replayed[0]))))
    obs=observations[0]
    def fake(length,total,regime):
        result=[]
        for index in range(length):
            o=dict(obs);o['search_state']=dict(last_accepted=regime,stagnation=0 if regime else 8)
            result.append(dict(iid='fixture',task=0,start_group='fixture',observation=o,reward=total if index==0 else 0.,terminal=index==length-1,truncated=False,diagnostics=dict(local_budget=10.,repair_seconds=999. if regime else .001)))
        return result
    eps=[fake(length,total,j%2==0) for j,(length,total) in enumerate([(1,1.),(10,2.),(3,3.),(20,4.)])]
    _,adv,weights=group_advantages(eps)
    expected=(np.arange(1,5)-2.5)/(np.std(np.arange(1,5))+1e-8)
    for index,ep in enumerate(eps):
        np.testing.assert_allclose(adv[id(ep)],expected[index]);assert abs(weights[id(ep)].sum()-.25)<1e-12
    for ep in eps:
        for t in ep:t['observation']['search_state']=dict(last_accepted=False,stagnation=99)
    _,changed,_=group_advantages(eps)
    for ep in eps:np.testing.assert_array_equal(adv[id(ep)],changed[id(ep)])
    # Four replicas must receive the exact same full10-second budget even for
    # tight50-customer instances whose deployment T/16 is less than10 seconds.
    iid=next(i for i in pool.split['fit_order'] if pool.load(i)[1]['cell']=={'n':50,'d':0,'tw':0})
    state={iid:dict(next_iid=1)};rngs={iid:np.random.default_rng(3).bit_generator.state};runtime=None
    try:
        runtime=ParallelRuntime(cfg);psutil.Process().cpu_affinity([0,1])
        deadline=time.perf_counter()+60.
        r=runtime.collect(policy,pool,pool.split,[directions()[15]],[iid],state,rngs,deadline)[0]
        assert r['complete'] and len(r['episodes'])==4
        for ep in r['episodes']:
            assert ep[-1]['terminal'] and not ep[-1]['truncated']
            assert all(t['diagnostics']['local_budget']==10. for t in ep)
            assert ep[-1]['diagnostics']['local_elapsed']>=10.
        opt=torch.optim.Adam(policy.parameters(),lr=cfg.learning_rate)
        metric=update(policy,opt,r['episodes'],'ppo',cfg,deadline)
        assert metric['steps']==2 and metric['initial_ratio_error']<2e-4
        receipt=dict(passed=True,calibration=calibration,neutral_q_bias=neutral_bias,neutral_q_target=.19,variable_length_trajectory_equal_weight=True,
            regime_and_manual_duration_invariant_advantage=True,equal_10s_collection=True,
            replica_steps=[len(ep) for ep in r['episodes']],replica_returns=[sum(t['reward'] for t in ep) for ep in r['episodes']],
            initial_ratio_error=metric['initial_ratio_error'],optimizer_steps=metric['steps'],
            no_test_training=True,discarded_preflight_weights=True)
        write(out/'VERIFICATION.json',receipt);print(receipt,flush=True)
    finally:
        if runtime is not None:runtime.close()

if __name__=='__main__':
    import multiprocessing as mp
    mp.freeze_support();main()
