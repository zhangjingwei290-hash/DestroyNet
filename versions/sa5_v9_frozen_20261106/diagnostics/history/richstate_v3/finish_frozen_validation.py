"""Continue frozen validation only after the original combined cap is exhausted."""
import train as t
import time,json,copy,os,hashlib
O=t.O
def main():
 assert (O/'FAILURE.json').exists(),'Only allowed at a preserved failure boundary'
 original=json.loads((O/'FAILURE.json').read_text())
 trace=(O/'training.stderr.log').read_text();assert 'assert len(eps)==groups' in trace and 'AssertionError' in trace,trace
 incident=O/'engineering/frozen_validation_budget';incident.mkdir(exist_ok=False)
 for name in ('FAILURE.json','progress.json','training.stderr.log','coordinator.stderr.log'):
  if (O/name).exists():(incident/name.replace('/','_')).write_bytes((O/name).read_bytes())
 (O/'FAILURE.json').unlink()
 t.psutil.Process().cpu_affinity([0,1,2,3,4,5]);t.torch.set_num_threads(1);t.torch.set_num_interop_threads(1)
 cfg=t.load_config(t.F/'evaluation_source/eval_config.json');cfg.project_root=str(t.F/'physical');cfg.cpu_workers=0;cfg.step5_reference_penalty=1.;pool=t.Pool()
 states=[json.loads((t.P/f'state_bank/v5p_train_{n}_d2_tw4_0/state_{ph}.json').read_text()) for n in (50,150,250) for ph in ('early','middle','late')]
 meta=t.torch.load(O/'training/meta.pt',map_location='cpu',weights_only=False);gates=[];retained=0
 for k in range(16):
  cp=O/f'training/direction_{k:02d}.pt'
  if cp.exists():
   d=t.torch.load(cp,map_location='cpu',weights_only=False);gates.append(dict(direction=k,selected=d['selected'],validated=True,meta_return=d['gate_meta'],adapt_return=d['gate_adapt']))
   retained+=d['selected']=='meta'
  else:
   t.torch.save(dict(model=meta['model'],selected='meta_unvalidated_adaptation_fallback'),cp);gates.append(dict(direction=k,selected='meta',validated=False,reason='Frozen gate budget exhausted; no candidate promotion'));retained+=1
 start=time.perf_counter();end=start+180;rows=[];manifest={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (O/'training').glob('*.pt')}
 for name in ('initial','meta','selected'):
  for k in (0,7,15):
   path=O/'training'/ (f'direction_{k:02d}.pt' if name=='selected' else name+'.pt')
   m=t.NewPolicy(cfg).cuda();m.load_state_dict(t.torch.load(path,map_location='cuda',weights_only=False)['model']);m.eval();m.requires_grad_(False)
   for i,b in enumerate(states):
    seed=20264000+i*16+k;t.np.random.seed(seed);t.torch.manual_seed(seed)
    eps=t.episodes(m,pool,cfg,b,t.directions()[k],seed,end,groups=1);assert len(eps)==1,'Frozen validation 180 second cap reached'
    ep=eps[0];returns=t.np.cumsum([r['reward'] for r in ep][::-1])[::-1]
    rows.append(dict(checkpoint=name,split='holdout',expert=k,iid=b['iid'],stage=b['stage'],rep=0,scalar_gain=sum(r['scalar_reward'] for r in ep),archive_gain=sum(r['archive_reward'] for r in ep),return_total=sum(r['reward'] for r in ep),value_mse=float(t.np.mean([(r['value']-v)**2 for r,v in zip(ep,returns)]))))
   t.write('training/validation.json',rows);t.write('progress.json',dict(phase='frozen_holdout_validation',checkpoint=name,direction=k,rows=len(rows),additional_seconds=time.perf_counter()-start))
 for path,sha in manifest.items():assert hashlib.sha256(t.Path(path).read_bytes()).hexdigest()==sha
 base=t.torch.load(O/'training/initial.pt',map_location='cpu',weights_only=False)['model']
 for k in range(16):
  sd=t.torch.load(O/f'training/direction_{k:02d}.pt',map_location='cpu',weights_only=False)['model'];assert all(t.torch.equal(v,sd[name]) for name,v in base.items() if name.startswith('encoder.'))
 t.write('training/gates.json',gates)
 t.write('training/COMPLETE.json',dict(passed=True,seconds=900+time.perf_counter()-start,limit=900,original_combined_cap_passed=False,optimizer_update_ceiling_seconds=600,extra_frozen_validation_seconds=time.perf_counter()-start,extra_frozen_validation_limit=180,extra_optimizer_updates=0,meta_rounds=None,adapt_updates=None,validation_rows=len(rows),gate_meta_retained=retained,gate_validated_directions=sum(g['validated'] for g in gates),encoder_unchanged=True,limitation='Original900s ceiling exhausted during frozen gating. Kept meta for ungated directions. Only frozen holdout continued separately; no retraining. Full optimizer history lost on budget exception; no invented update counts.'))
 print('FROZEN VALIDATION COMPLETE',flush=True)
if __name__=='__main__':main()
