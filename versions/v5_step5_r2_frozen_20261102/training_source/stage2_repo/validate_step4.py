"""Inference-only contract verification: graph parity, schedule, frozen models."""
import os,sys
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from pilot_common import ROOT,read,write,sha,reset
import time,copy,types,traceback
import numpy as np,torch,psutil
from run import load_config
from solver.model import Policy,pack,integer_q
from solver.outer import run_cached_session,pulse_required,cached_sample,cache_refresh_required
from solver.data import TrainPool
from solver.config import directions
from solver.environment import Environment,Adaptive

def main():
    began=time.time();out=ROOT/'step4_analysis';assert not (out/'PREFLIGHT.json').exists()
    psutil.Process().cpu_affinity([14,15]);torch.set_num_threads(1);torch.set_num_interop_threads(1);reset(20261024)
    cfg=load_config(ROOT/'step4_repo/step4_config.json');pool=TrainPool(cfg)
    ck=torch.load(ROOT/'step3_training/meta_pilot.pt',map_location='cuda',weights_only=False)
    policy=Policy(cfg).cuda();policy.load_state_dict(ck['model']);policy.eval()
    before={k:v.detach().cpu().clone() for k,v in policy.state_dict().items()}
    assert [(s,pulse_required(s)) for s in range(21)]==[(s,s in (6,10,14,18)) for s in range(21)]
    # Fake session verifies one refresh+three hits, accepted-route invalidation,
    # and zero model/observation calls in reuse steps.
    class FakePolicy:
        cfg=types.SimpleNamespace(hidden_dim=128)
        def __init__(self):self.calls=0
        def dual_cache(self,obs,ids):
            self.calls+=1
            return torch.tensor([[0.]*128+[3.,8.,.2,1.]+[0.]*5+[8.,3.,.7,2.]+[0.]*5])
    class FakeSession:
        def __init__(self,accept_at=None):
            self.env=types.SimpleNamespace(instance=types.SimpleNamespace(n_customers=5),last_accepted=False,stagnation=0)
            self.cached_distribution=None;self.step_in_session=0;self.cache_age=0;self.cache_rng=np.random.default_rng(4)
            self.cache_hidden=None;self.records=[];self.observations=0;self.pending=None;self.accept_at=accept_at
        def remaining(self):return 1. if self.step_in_session<12 else 0.
        def prepare(self,include_observation):self.observations+=include_observation;self.pending=True;return {} if include_observation else None
        def finish(self,action,hidden,seconds):
            accepted=self.step_in_session==self.accept_at
            r=dict(stagnation_before=self.env.stagnation,last_accepted_before=self.env.last_accepted,q=action['q'],accepted=accepted)
            self.env.stagnation=0 if accepted else self.env.stagnation+1;self.env.last_accepted=accepted;self.pending=None;self.records.append(r);return r
        def finalize(self):pass
    f=FakeSession();p=FakePolicy();run_cached_session(f,p)
    assert [r['cache_step'] for r in f.records if r['cache_refresh']]==[0,4,8] and p.calls==f.observations==3
    assert [r['stagnation_before'] for r in f.records if r['use_pulse']]==[6,10]
    f2=FakeSession(1);p2=FakePolicy();run_cached_session(f2,p2)
    assert f2.records[2]['cache_refresh'] and f2.records[2]['last_accepted_before']
    assert p2.calls==f2.observations
    grid=[];max_error=0.
    for n in (50,150,250):
        iid=next(i for i in pool.split['fit_order'] if pool.load(i)[0].n_customers==n)
        env=Environment(pool,iid,cfg,directions()[15]);obs=env.observation(torch.zeros(1,128),'greedy',Adaptive(10),200.,300.,8.,10.)
        fields,nodes=pack([obs]);fields={k:v.cuda() for k,v in fields.items()}
        with torch.no_grad():
            expected=policy._tensor_dual_cache(fields,nodes,[0]).cpu()
            actual=policy.dual_cache([obs],[0]);torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-5)
            max_error=max(max_error,float((actual-expected).abs().max()))
            h,features,x,raw=policy.encode([obs])
            for mode_index,gate in enumerate((0.,1.)):
                r=raw.clone();r[:,14]=gate;a,b,mu=policy.q_beta_params(h,features,x,r,[0]);q=integer_q(mu,n)
                logits=policy.node_logits_batch(features,x,r,q.float()/n,[0]);offset=128+mode_index*(n+4)
                torch.testing.assert_close(actual[:,offset:offset+3],torch.stack((a,b,mu),1).cpu(),rtol=2e-5,atol=2e-5)
                torch.testing.assert_close(actual[:,offset+4:offset+4+n],logits.cpu(),rtol=2e-5,atol=2e-5)
        cache=actual[0].numpy();rng=np.random.default_rng(4);timings=[]
        for pulse in (False,True):
            qvalues=[];start=time.perf_counter()
            for _ in range(3000):
                action=cached_sample(cache,n,128,rng,pulse);qvalues.append(action['q']/n)
                assert len(action['removed'])==action['q']==len(set(action['removed'])) and all(1<=c<=n for c in action['removed'])
            seconds=time.perf_counter()-start;offset=128+(n+4 if pulse else 0);a,beta=cache[offset:offset+2]
            expected_mean=.01+.59*float(a/(a+beta))
            assert abs(np.mean(qvalues)-expected_mean)<.015
            timings.append(dict(pulse=pulse,mean_q_ratio=float(np.mean(qvalues)),expected_continuous_mean=expected_mean,
                sampling_seconds_per_action=seconds/3000,anchor_q=int(cache[offset+3])))
        # A different direction's weights must refresh the shared capture host.
        other=Policy(cfg).cuda();direction=torch.load(ROOT/'step3_training/checkpoints/direction_00.pt',map_location='cuda',weights_only=False)
        other.load_state_dict(direction['model']);other.eval()
        with torch.no_grad():
            target=other._tensor_dual_cache(fields,nodes,[0]).cpu();got=other.dual_cache([obs],[0])
            torch.testing.assert_close(got,target,rtol=2e-5,atol=2e-5)
            torch.testing.assert_close(policy.dual_cache([obs],[0]),actual,rtol=2e-5,atol=2e-5)
        grid.append(dict(n=n,fit_iid=iid,cache_sampling=timings))
    assert all(torch.equal(before[k],v.detach().cpu()) for k,v in policy.state_dict().items())
    result=dict(passed=True,zero_training=True,weights_unchanged=True,checkpoint_binding=read(ROOT/'step4_repo/STEP4_PLAN.json')['checkpoint_binding'],
        K4_refresh_plus_three_hits_verified=True,acceptance_invalidates=True,pulse_schedule=[6,10,14,18],
        no_observation_or_model_call_on_hits=True,mode_anchored_logits_verified=True,graph_eager_max_abs_error=max_error,
        per_direction_shared_host_binding_verified=True,grid=grid,technical_seconds=time.time()-began)
    write(out/'PREFLIGHT.json',result);print(result,flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        (ROOT/'step4_analysis/preflight_failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
