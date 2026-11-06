import collector as t
import json,hashlib,copy,time
from aligned_cpu import struct_state
def main():
 t.psutil.Process().cpu_affinity([14,15]);t.torch.set_num_threads(1);t.torch.set_num_interop_threads(1);t.torch.manual_seed(20261104)
 cfg=t.load_config(t.F/'evaluation_source/eval_config.json');cfg.project_root=str(t.F/'physical');cfg.cpu_workers=0;cfg.step5_reference_penalty=1.;pool=t.Pool()
 bank=json.loads((t.P/'state_bank/v5p_train_50_d0_tw0_0/state_early.json').read_text());m=t.NewPolicy(cfg).cuda();cp=t.P/'training/direction_07.pt';m.load_state_dict(t.torch.load(cp,map_location='cuda',weights_only=False)['model']);m.eval();env=t.Environment(pool,bank['iid'],cfg,t.directions()[7],routes=bank['population'][7]['routes']);ref=env.offset+env.scales*t.np.asarray(env.box['reference']);env.dynamic_ideal,env.dynamic_span=t.dynamic_coordinates([x['objectives'] for x in bank['archive']],env.scales,ref)
 ad=t.Adaptive(len(env.q_values));obs=env.observation(t.torch.zeros(1,cfg.hidden_dim),'greedy',ad,30.,60.,30.,30.);cache=m.refresh(obs,retain_nodes=True);head=t.NewHead(m);head.state=t.np.asarray(t.Session(env.score(env.objectives),env.score(env.objectives),t.calibrated_temperature(.01,.5),t.calibrated_temperature(.0001,.5)).state(.5),t.np.float32);head.summary=t.summary_for(env,bank['archive']);z=(env.objectives-env.dynamic_ideal)/env.dynamic_span;phys=obs['x'][1:,[18,17,19,12,13]].numpy();args=(cache['hidden'],struct_state(phys),obs['context'].numpy(),50,t.np.asarray([z.mean(),(env.preference*z).max()],t.np.float32))
 base=t.NewHead(m);base.state=head.state;base.summary=head.summary
 a0=base.outputs(*args);a1=head.outputs(*args);assert all(t.np.array_equal(x,y) for x,y in zip(a0,a1));head.q_bias=.1;a2=head.outputs(*args);assert t.np.array_equal(a0[0],a2[0]);assert t.np.allclose(a0[1]+a0[2],a2[1]+a2[2],atol=1e-5,rtol=0);assert t.np.all(a2[1]/(a2[1]+a2[2])>=a0[1]/(a0[1]+a0[2]));assert t.np.all(a2[1]>=1) and t.np.all(a2[2]>=1)
 # Literal original v2 head, not only a second wrapper instance.
 from network import legacy
 original=legacy.NewHead(m);original.state=head.state;original.summary=head.summary;a3=original.outputs(*args);assert all(t.np.array_equal(x,y) for x,y in zip(a0,a3))
 rewards=t.torch.tensor([1.,2.,4.,8.]);advantages=(4*rewards-rewards.sum())/3
 assert t.torch.allclose(advantages,t.torch.tensor([1.-14/3,2.-13/3,4.-11/3,8.-7/3])) and abs(float(advantages.sum()))<1e-5
 (t.O/'preflight').mkdir(exist_ok=True);t.write('preflight/PASSED.json',dict(passed=True,v2_zero_bias_bitwise_equal=True,mode_logits_unchanged=True,beta_concentration_preserved=True,beta_mean_increased=True,rloo_formula_checked=True,training_updates=0,checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest()))
if __name__=='__main__':main()
