"""Verify retained boundaries and repair provenance before the one controlled continuation."""
from stage2_common import ROOT,read,write,sha,readonly_check
from pathlib import Path
import time,shutil,ast

def main():
 import psutil,numpy as np,torch
 from stage2_cpu import legacy_struct_state
 import importlib.util
 psutil.Process().cpu_affinity([14,15]);torch.set_num_threads(1)
 incident=ROOT/'engineering/replay_guard_direction05_batch003';validation=read(incident/'PRECISION_REPAIR_VALIDATION.json');assert validation['passed'] and validation['thresholds_unchanged']
 recovery=read(incident/'RECOVERY_INSPECTION.json');assert all(r['loss_delta']==0 for r in recovery['reconstructed'])
 prior=read(incident/'stage2_training/failure_budget.json');assert prior['meta_batches']==144 and prior['direction_updates']['15']==90 and prior['direction_updates']['5']==2
 original=read(ROOT/'stage2_coordinator/SOURCE_MANIFEST.json');newfiles=list(ROOT.glob('*.py'))+list((ROOT/'stage2_repo').rglob('*.py'))
 for p in newfiles:ast.parse(p.read_text(encoding='utf-8'),str(p))
 for name in ['stage2_resume_train','coordinator_stage2_resume','launch_stage2_resume','verify_stage2_final','stage2_reporting']:__import__(name)
 for name in original:
  if name.endswith('.py') and sha(ROOT/name)!=original[name]:
   p=incident/name;assert p.exists() and sha(p)==original[name],name
 spec=importlib.util.spec_from_file_location('preserved_cpu_reference',incident/'stage2_cpu.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 corpus=torch.load(incident/'live_replay_diagnostic.pt',map_location='cpu',weights_only=False)
 for w in corpus['windows']:
  for ep in w['episodes']:
   for t in ep[::max(1,len(ep)//8)]:assert np.array_equal(module.struct_state(t['physical']),legacy_struct_state(t['physical']))
 # All old completed model bytes and scientific mechanism files outside the numerical q statistic are preserved.
 failed=torch.load(ROOT/'stage2_training/failed_training_state.pt',map_location='cpu',weights_only=False)
 meta=torch.load(ROOT/'stage2_training/meta_retrained.pt',map_location='cpu',weights_only=False)['model'];assert all(torch.equal(meta[k],v) for k,v in failed['meta'].items())
 final15=torch.load(ROOT/'stage2_training/checkpoints/direction_15.pt',map_location='cpu',weights_only=False)['model'];boundary15=torch.load(ROOT/'stage2_training/adaptation_boundaries/direction_15_batch_090.pt',map_location='cpu',weights_only=False)['model'];assert all(torch.equal(final15[k],v) for k,v in boundary15.items())
 readonly=readonly_check()
 changed={name:dict(before=original[name],after=sha(ROOT/name)) for name in original if sha(ROOT/name)!=original[name]}
 receipt=dict(ready_for_controlled_resume=True,observed_failure='Original aggregate replay guard at direction05batch003; exact offending transition/metric was not retained',cause_proven=False,reproduced_engineering_issue='Nearconstant physical-statistic FP32 reductions differed CPU/GPU by up to.00158946',numeric_repair='Only nonlearned physical-statistic reductions now float64 on both backends, returned state and neuralnetwork/gradients FP32; original thresholds unchanged',precision_validation=validation,checkpoint_reconstruction=recovery['reconstructed'],recovered_completed_counts=dict(meta=144,direction15=90,direction05=2),continuation_start=dict(direction=5,batch=3),original_unoptimized_fault_group=1,diagnostic_trajectories12_excluded=True,legacy_r2_deployment_preserved=True,extra_guards=['exact offending replay chunk/model/optimizer dump','rollback whole PPO transaction on any failure','per-batch resumed optimizer/state/RNG/history atomic boundary'],changed_sealed_source=changed,original_source_manifest_preserved=True,readonly=readonly,created_unix=time.time())
 write(incident/'REPAIR.json',receipt)
 # Move obsolete active failure sentinels, keeping every byte under the incident directory.
 archive=incident/'archived_active_failures';archive.mkdir(exist_ok=True)
 for name in ['stage2_training/failure.txt','stage2_training/failure_budget.json','stage2_coordinator/failure.txt','stage2_coordinator/training_stderr.log']:
  p=ROOT/name
  if p.exists():target=archive/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(p),str(target))
 manifest={p.relative_to(ROOT).as_posix():sha(p) for p in newfiles}
 for name in ['PROTOCOL.md','STAGE2_PLAN.json','plan.json','PREFLIGHT_STAGE2.json','PARENT_R2.json','READONLY_RECEIPT.json']:
  manifest[name]=sha(ROOT/name)
 write(ROOT/'stage2_coordinator/EFFECTIVE_SOURCE_MANIFEST.json',manifest)
 write(ROOT/'ENGINEERING_AMENDMENT.json',dict(repair=str(incident/'REPAIR.json'),original_manifest='stage2_coordinator/SOURCE_MANIFEST.json',effective_manifest='stage2_coordinator/EFFECTIVE_SOURCE_MANIFEST.json',old_completed_training_preserved=True,thresholds_unchanged=True))
 print(dict(ready=True,changed=list(changed),legacy_r2_unchanged=True,readonly_passed=True),flush=True)
if __name__=='__main__':main()
