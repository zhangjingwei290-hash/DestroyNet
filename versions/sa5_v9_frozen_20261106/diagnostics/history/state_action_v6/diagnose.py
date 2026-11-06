import collector as t
import json,time,hashlib,traceback
from pathlib import Path
V=t.O.parent/'sa5_cross_session_v5_20261105'
def main():
 t.psutil.Process().cpu_affinity([0,1,2,3]);t.torch.set_num_threads(1);t.torch.set_num_interop_threads(1);cfg=t.load_config(t.F/'evaluation_source/eval_config.json');cfg.project_root=str(t.F/'physical');cfg.cpu_workers=0;cfg.step5_reference_penalty=1.;pool=t.Pool();end=time.perf_counter()+900;rows=[];models={}
 for n in (50,150,250):
  states=json.loads((V/f'collection/{n}/training_states.json').read_text());ids=t.np.linspace(0,len(states)-1,min(12,len(states))).astype(int)
  for ix in ids:
   b=states[int(ix)];k=b['direction']
   if k not in models:
    m=t.NewPolicy(cfg).cuda();m.load_state_dict(t.torch.load(t.P/f'training/direction_{k:02d}.pt',map_location='cuda',weights_only=False)['model']);m.eval();m.requires_grad_(False);models[k]=m
   for shift in (-.10,0.,.10):
    b['q_intervention']=shift;t.torch.manual_seed(20272000+int(ix));begin=time.perf_counter();eps=t.episodes(models[k],pool,cfg,b,t.directions()[k],20272000+int(ix),end,groups=8);assert len(eps)==8
    for rep,e in enumerate(eps):rows.append(dict(n=n,state=int(ix),direction=k,stall=b['cross_stall'],shift=shift,rep=rep,gain=sum(x['scalar_reward'] for x in e),mean_q_ratio=sum(x['action']['q']/n for x in e)/5,group_seconds=time.perf_counter()-begin))
    t.write('paired_results.json',rows);t.write('progress.json',dict(n=n,state=int(ix),shift=shift,trajectories=len(rows),planned_max=864))
 t.write('PHASE1_COMPLETE.json',dict(passed=True,trajectories=len(rows),repair_steps=len(rows)*5,remaining_seconds=end-time.perf_counter()))
if __name__=='__main__':
 try:main()
 except BaseException:t.write('FAILURE.json',dict(error=traceback.format_exc()));raise
