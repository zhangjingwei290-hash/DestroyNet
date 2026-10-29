"""Prepare a unique immutable CPU pilot binding, without launching it."""
import argparse
import json
import shutil
from pathlib import Path
from collections import defaultdict
from .collector import save,physical_box
from .run import sha
from src.data_generation.sharded import _instance_from_row


ROOT=Path(__file__).resolve().parents[1]


def prepare(out):
    out=Path(out).resolve();out.relative_to(ROOT);assert not out.exists(),'Use a new output; preserve previous attempts'
    inv_path=ROOT/'research_notes/hemt_destroy_features_v1_20261012/inventory.json'
    inv=json.loads(inv_path.read_text(encoding='utf-8'));cells=defaultdict(set)
    for source in inv['sample_sources']:cells[source['cell']].add(source['instance_id'])
    assert len(cells)==18
    ids=[sorted(cells[c])[0] for c in sorted(cells)];assert len(set(ids))==18
    inputs={};boxes={}
    for iid in ids:
        item=inv['train_raw_files'][iid];assert sha(item['path'])==item['sha256']
        row=json.loads(Path(item['path']).read_text(encoding='utf-8'));assert row['split']=='train'
        inputs[iid]=dict(path=str(Path(item['path']).resolve()),sha256=item['sha256'])
        boxes[iid]=physical_box(_instance_from_row(row))
    out.mkdir(parents=True)
    cfg=dict(version='hemt_destroy_feature_labels_v1',seed=26101277,device='cpu',threads=1,
             train_instance_ids=ids,inputs=inputs,boxes_by_instance=boxes,repair_ids=['greedy','regret2','regret3','noisy_greedy'],
             capture_seconds=[0.,2.,5.],horizon_seconds=8.,repetitions=3,location_patterns=['random','related'],
             total_branches=3888,probe=dict(mrmr_k=24,folds=3),
             scope='TRAIN18 CPU generic ALNS feature pilot; no GPU, no neural model, no native HEMT, no notebook actions')
    save(out/'config.json',cfg)
    files={str(out/'config.json'):sha(out/'config.json'),str(inv_path):sha(inv_path)}
    # Execute the snapshot itself, so later project editing cannot silently alter the worker.
    for package in ('src','hemt_destroy_features_v1','hemt_destroy_feature_labels_v1'):
        for path in (ROOT/package).rglob('*.py'):
            target=out/'source_snapshot'/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,target);files[str(target)]=sha(target)
    for name in ('PROTOCOL.md','config_draft.json'):
        path=ROOT/'research_notes/hemt_destroy_feature_labels_v1_20261012'/name
        target=out/'protocol'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target);files[str(target)]=sha(target)
    for item in inputs.values():files[item['path']]=item['sha256']
    save(out/'manifest.json',dict(files=files,source_scope='Executable snapshot plus fixed TRAIN inputs and predeclared config/protocol'))
    save(out/'prepared.json',dict(bound_files=len(files),instances=len(ids),branches=3888,
                                  budget_upper_seconds=3888*8+18*5,neural_training=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);prepare(p.parse_args().output)
