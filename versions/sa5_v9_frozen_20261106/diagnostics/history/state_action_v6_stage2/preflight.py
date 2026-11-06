import collector as t
from network import load_base,legacy
from aligned_cpu import struct_state
import json

def main():
 t.psutil.Process().cpu_affinity([14,15]);t.torch.set_num_threads(1);t.torch.set_num_interop_threads(1);t.torch.manual_seed(20261105)
 cfg=t.load_config(t.F/'evaluation_source/eval_config.json');cfg.project_root=str(t.F/'physical');cfg.cpu_workers=0;cfg.step5_reference_penalty=1.;pool=t.Pool();m=t.NewPolicy(cfg).cuda();load_base(m,t.P/'training/direction_07.pt');opt=t.configure(m)
 b=json.loads((t.P/'state_bank/v5p_train_50_d0_tw0_0/state_early.json').read_text());env=t.Environment(pool,b['iid'],cfg,t.directions()[7],routes=b['population'][7]['routes']);ref=env.offset+env.scales*t.np.asarray(env.box['reference']);env.dynamic_ideal,env.dynamic_span=t.dynamic_coordinates([x['objectives'] for x in b['archive']],env.scales,ref)
 ad=t.Adaptive(len(env.q_values));obs=env.observation(t.torch.zeros(1,cfg.hidden_dim),'greedy',ad,30.,60.,30.,30.);cache=m.refresh(obs,retain_nodes=True);head=t.NewHead(m);head.state=t.np.asarray(t.Session(env.score(env.objectives),env.score(env.objectives),t.calibrated_temperature(.01,.5),t.calibrated_temperature(.0001,.5)).state(.5),t.np.float32);head.summary=t.summary_for(env,b['archive']);z=(env.objectives-env.dynamic_ideal)/env.dynamic_span;args=(cache['hidden'],struct_state(obs['x'][1:,[18,17,19,12,13]].numpy()),obs['context'].numpy(),50,t.np.asarray([z.mean(),(env.preference*z).max()],t.np.float32));original=legacy.NewHead(m);original.state=head.state;original.summary=head.summary;baseline=original.outputs(*args)
 for stall in (0,3,30):
  head.gate_enabled=True;head.cross_stall=stall;out=head.outputs(*args);assert all(t.np.array_equal(a,b) for a,b in zip(baseline,out))
 for sign in (-1.,1.):
  with t.torch.no_grad():m.stall_gate.body[-1].bias.fill_(sign)
  head.cross_stall=30;out=head.outputs(*args);assert t.np.array_equal(out[0],baseline[0]);assert t.np.allclose(out[1]+out[2],baseline[1]+baseline[2],atol=1e-5);assert sign*head.gate_shift>0
 with t.torch.no_grad():m.stall_gate.body[-1].bias.zero_()
 assert all(p.requires_grad==n.startswith('stall_gate.') for n,p in m.named_parameters())
 (t.O/'preflight').mkdir(exist_ok=True);t.write('preflight/PASSED.json',dict(passed=True,zero_gate_v2_bitwise=True,signed_shift_and_concentration=True,mode_unchanged=True,trainable_parameters=sum(p.numel() for p in m.parameters() if p.requires_grad)))
if __name__=='__main__':main()
