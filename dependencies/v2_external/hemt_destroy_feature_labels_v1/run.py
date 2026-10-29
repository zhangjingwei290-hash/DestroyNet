"""One-shot persistent CPU worker; frozen source and TRAIN inputs only."""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
import argparse
import hashlib
import json
import math
import time
import traceback
from pathlib import Path
import psutil
from .collector import save, physical_box, load_instance, capture, branch
from .audit import audit_comparison, audit_branch, digest, checked_objectives
from .repair import METHODS


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def verify(out,cfg):
    manifest=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
    for path,value in manifest['files'].items():
        assert sha(path)==value,('Frozen input/source changed',path)
    for item in cfg['inputs'].values():assert sha(item['path'])==item['sha256']


def run(out):
    out=Path(out).resolve();cfg=json.loads((out/'config.json').read_text(encoding='utf-8'));verify(out,cfg)
    lock=out/'worker.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    p=psutil.Process();save(out/'worker_process.json',dict(pid=p.pid,created=p.create_time(),command=p.cmdline(),
                                                       machine='local Windows host',cpu_threads=1,gpu_used=False))
    started=time.time();done=0;states=[];instances={};branches=[]
    try:
        save(out/'health.json',dict(status='capturing',started_at=started))
        for ordinal,iid in enumerate(cfg['train_instance_ids']):
            inst,row=load_instance(cfg,iid);instances[iid]=inst
            path=out/'states'/f'{iid}.json'
            if path.exists():
                assert sha(path)==json.loads(path.with_suffix('.binding.json').read_text())['sha256'],'Captured state changed'
                captured=json.loads(path.read_text(encoding='utf-8'))
            else:
                captured=capture(inst,row,cfg['boxes_by_instance'][iid],cfg,ordinal);save(path,captured)
                save(path.with_suffix('.binding.json'),dict(sha256=sha(path)))
            states.extend(captured)
            save(out/'progress.json',dict(stage='capture',instances_captured=ordinal+1,total_instances=len(cfg['train_instance_ids']),
                                          states=len(states),branches=done,total_branches=cfg['total_branches']))
        save(out/'capture_complete.json',dict(instances=len(instances),states=len(states),
            unique_source_hashes=len({s['source_hash'] for s in states}),scope='Fresh generic ALNS; identical captures explicitly retained'))
        save(out/'health.json',dict(status='collecting',started_at=started))
        for repetition in range(cfg['repetitions']):
            for state_index,state in enumerate(states):
                iid=state['instance_id'];inst=instances[iid];cap=min(15,math.floor(.15*inst.n_customers))
                for pattern_index,pattern in enumerate(cfg['location_patterns']):
                    destroy_seed=cfg['seed']+state_index*17+pattern_index
                    for q_index,q in enumerate([1,math.ceil(cap/2),cap]):
                        seed=cfg['seed']+100000+state_index*100+repetition*10000+pattern_index*10+q_index
                        offset=(state_index+repetition+q_index+pattern_index)%4
                        order=list(METHODS[offset:]+METHODS[:offset])
                        for method in order:
                            key=f'{iid}_s{state["stage_index"]}_{pattern}_q{q}_r{repetition}_{method}'
                            path=out/'branches'/f'{key}.json'
                            if path.exists():
                                assert sha(path)==json.loads(path.with_suffix('.binding.json').read_text())['sha256'],'Saved branch changed'
                                b=json.loads(path.read_text(encoding='utf-8'))
                                audit_branch(b,state,inst,cfg)
                            else:
                                b=branch(inst,state,q,pattern,method,repetition,seed,destroy_seed,cfg);save(path,b)
                                save(path.with_suffix('.binding.json'),dict(sha256=sha(path)))
                            branches.append(b);done+=1
                            elapsed=time.time()-started
                            save(out/'progress.json',dict(stage='labels',branches=done,total_branches=cfg['total_branches'],
                                repetition=repetition,instance_id=iid,stage_index=state['stage_index'],last_key=key,
                                elapsed_seconds=elapsed,eta_seconds=max(0.,elapsed/done*(cfg['total_branches']-done)),
                                first_success=sum(x['first_repair_success'] for x in branches),
                                positive_gain=sum(x['delta_hv']>1e-12 for x in branches)))
            audit_comparison(branches)
            save(out/f'repetition_{repetition}_complete.json',dict(branches=done,audit=audit_comparison(branches)))
        assert done==cfg['total_branches']
        save(out/'labels_complete.json',dict(branches=done,states=len(states),audit=audit_comparison(branches),
                                             elapsed_seconds=time.time()-started))
        save(out/'health.json',dict(status='classical_feature_probe',started_at=started))
        from .probe import probe
        probe(out,cfg)
        verify(out,cfg)
        save(out/'complete.json',dict(branches=done,states=len(states),elapsed_seconds=time.time()-started,
            scope='TRAIN feature diagnostic; no neural training and no native HEMT quality claim'))
        save(out/'health.json',dict(status='complete',ended_at=time.time()))
    except BaseException:
        save(out/'failure.json',dict(error=traceback.format_exc(),branches=done,at=time.time()))
        save(out/'health.json',dict(status='failed',at=time.time()));raise
    finally:
        lock.unlink(missing_ok=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    run(parser.parse_args().output)
