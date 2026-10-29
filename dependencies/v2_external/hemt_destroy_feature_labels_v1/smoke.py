"""Real TRAIN input smoke for collector contracts, not a quality result."""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import json
import math
from pathlib import Path
from .collector import capture,branch,physical_box,save
from .audit import audit_comparison
from .repair import METHODS
from src.data_generation.sharded import _instance_from_row


def smoke():
    root=Path(__file__).resolve().parents[1]
    inv=json.loads((root/'research_notes/hemt_destroy_features_v1_20261012/inventory.json').read_text(encoding='utf-8'))
    reports=[]
    for n in (50,150,250):
        iid=sorted(i for i in inv['selected_train_ids'] if f'_train_{n}_' in i)[0]
        row=json.loads(Path(inv['train_raw_files'][iid]['path']).read_text(encoding='utf-8'));inst=_instance_from_row(row)
        cfg=dict(seed=26101277,capture_seconds=[0.,.1,.2],horizon_seconds=.25,location_patterns=['random','related'],
                 repair_ids=list(METHODS),train_instance_ids=[iid],boxes_by_instance={iid:physical_box(inst)})
        states=capture(inst,row,cfg['boxes_by_instance'][iid],cfg,0);assert len(states)==3
        rows=[]
        for pattern in cfg['location_patterns']:
            for method in METHODS:rows.append(branch(inst,states[0],1,pattern,method,0,26101277,26101278,cfg))
        pair=audit_comparison(rows)
        assert all(b['audit']['passed'] for b in rows)
        # Explicit pre-start deadline case: cannot manufacture a late point.
        tiny=dict(cfg,horizon_seconds=1e-8)
        timeout=branch(inst,states[0],1,'random','greedy',0,3,4,tiny)
        assert not timeout['first_repair_success'] and timeout['delta_hv']==0 and timeout['committed_count']==0
        reports.append(dict(instance_id=iid,scale=n,branches=len(rows),comparison=pair,
                            feasible_first_repairs=sum(b['first_repair_success'] for b in rows),
                            tiny_deadline_rejected=True))
    path=root/'research_notes/hemt_destroy_feature_labels_v1_20261012/SMOKE.json'
    save(path,dict(passed=True,scope='3 TRAIN scale collector smoke; no quality claim',results=reports));print(path)


if __name__=='__main__':smoke()
