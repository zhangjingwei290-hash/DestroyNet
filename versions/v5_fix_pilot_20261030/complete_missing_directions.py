"""Bounded engineering recovery: train only zero-update direction checkpoints."""
from pilot_common import ROOT,read,write,sha,reset,log
import sys,time,copy,traceback,os,shutil
sys.path.insert(0,str(ROOT/'repo'))
import torch,numpy as np,psutil
from run import load_config
from solver.data import TrainPool
from solver.model import Policy
from solver.config import directions
from solver.training import update,save_checkpoint
from solver.parallel_runtime import ParallelRuntime

def main():
    out=ROOT/'step1_training';report=read(out/'training_report.json');plan=read(ROOT/'plan.json')
    missing=[int(k) for k,v in report['direction_updates'].items() if v==0]
    if missing!=[3,4]:raise RuntimeError('Recovery requires explicitly observed missing directions3/4')
    psutil.Process().cpu_affinity(list(range(psutil.cpu_count())));psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);reset(20261024)
    cfg=load_config(ROOT/'config.json');pool=TrainPool(cfg);refs=directions()
    meta_checkpoint=torch.load(ROOT/'checkpoints/meta_complete.pt',map_location='cpu',weights_only=False)
    encoder={k[len('encoder.'):]:v for k,v in meta_checkpoint['model'].items() if k.startswith('encoder.')}
    used=report['wall_seconds']+plan['step1_retained_technical_run_seconds']+read(ROOT/'step1_smoke/training_report.json')['wall_seconds']
    budget=min(240.,2700.-used-15.)
    if budget<40.:raise RuntimeError('Not enough authorized remaining time for two full groups')
    incident=ROOT/'incidents/direction_admission';incident.mkdir(parents=True,exist_ok=False)
    for name in ('training_report.json','failure.txt','progress.json'):
        shutil.copy2(out/name,incident/name)
    for k in missing:shutil.copy2(out/'checkpoints'/f'direction_{k:02d}.pt',incident/f'direction_{k:02d}_unadapted.pt')
    write(incident/'process.json',dict(pid=os.getpid(),created=psutil.Process().create_time(),missing=missing,budget_seconds=budget,previous_total_seconds=used))
    before={k:sha(out/'checkpoints'/f'direction_{k:02d}.pt') for k in range(16) if k not in missing}
    start=time.perf_counter();deadline=start+budget;runtime=None;new=[]
    # Continue the original FIT order at the next previously unused source.
    index=max(pool.split['fit_order'].index(i) for i in report['fit_ids_used'])+1
    try:
        runtime=ParallelRuntime(cfg);psutil.Process().cpu_affinity([0,1])
        for k in missing:
            parent=next(j for j in range(5,15) if refs[j,k]>0)
            parent_checkpoint=torch.load(out/'checkpoints'/f'direction_{parent:02d}.pt',map_location='cuda',weights_only=False)
            actor=Policy(cfg).cuda();actor.load_state_dict(parent_checkpoint['model']);actor.encoder.requires_grad_(False)
            iid=pool.split['fit_order'][index%180];index+=1
            states={iid:dict(next_iid=index)};rngs={iid:np.random.default_rng(20261024+k).bit_generator.state}
            actor.eval();result=runtime.collect(actor,pool,pool.split,[refs[k]],[iid],states,rngs,deadline)[0]
            if not result['complete']:raise RuntimeError('Recovery did not complete the four-replica group')
            actor.train();optimizer=torch.optim.Adam([p for p in actor.parameters() if p.requires_grad],lr=cfg.learning_rate)
            metric=update(actor,optimizer,result['episodes'],'ppo',cfg,deadline)
            if metric['steps']!=cfg.ppo_epochs or metric['initial_ratio_error']>2e-4:raise RuntimeError('Recovery update/density failed')
            for name,value in actor.encoder.state_dict().items():
                if not torch.equal(value.detach().cpu(),encoder[name]):raise RuntimeError('Frozen encoder differs')
            actor.eval()
            checkpoint=dict(version=cfg.version,direction=k,weight=refs[k].tolist(),model=actor.actor_state(),updates=1,config=cfg.to_dict(),source_checkpoint_sha256=plan['checkpoint_sha256'],parent=parent,recovery='one missing-direction batch within total45minute cap')
            save_checkpoint(out/'checkpoints'/f'direction_{k:02d}.pt',checkpoint)
            record=dict(phase='adaptation',direction=k,updates=1,iid=iid,provenance=result['provenance'],recovery=True,**metric)
            new.append(record);report['direction_updates'][str(k)]=1;report['history'].append(record)
            log(incident/'recovery.log',dict(direction=k,iid=iid,transitions=metric['transitions'],elapsed=time.perf_counter()-start))
        for k,digest in before.items():assert sha(out/'checkpoints'/f'direction_{k:02d}.pt')==digest
        elapsed=time.perf_counter()-start
        report.update(wall_seconds=report['wall_seconds']+elapsed,recovery_seconds=elapsed,
            total_step1_optimization_and_technical_seconds=used+elapsed,
            all_directions_updated=True,encoder_frozen_verified=True,
            optimized_transitions=sum(h['transitions'] for h in report['history']),
            fit_ids_used=sorted({h['iid'] for h in report['history']}))
        assert report['total_step1_optimization_and_technical_seconds']<=2700.
        write(out/'training_report.json',report)
        (out/'failure.txt').unlink() # The original traceback remains in the incident archive.
        write(out/'complete.json',dict(completed=True,stage=1,direction_models=16,seconds=report['wall_seconds'],total_with_technical_seconds=used+elapsed,recovery='only zero-update directions3/4',technical_smoke=False))
        write(out/'progress.json',dict(stage=1,phase='complete',elapsed=report['wall_seconds'],direction_updates=report['direction_updates']))
        write(incident/'complete.json',dict(completed=True,seconds=elapsed,unchanged_previously_adapted_checkpoints=True,total_seconds=used+elapsed))
        print('MISSING_DIRECTIONS_COMPLETED',flush=True)
    except BaseException:
        (incident/'recovery_failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
    finally:
        if runtime is not None:runtime.close()

if __name__=='__main__':
    import multiprocessing as mp
    mp.freeze_support();main()
