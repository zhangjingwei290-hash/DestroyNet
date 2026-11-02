"""Step2-of-Step5 isolated paths; no neural library at import."""
from common import ROOT, read, write, sha, reset, log, EvalPool, readonly_check
import sys, copy
sys.path.insert(0,str(ROOT/'stage2_repo'))

def load_cfg():
    from run import load_config
    cfg=load_config(ROOT/'eval_config.json')
    cfg.step5_reference_penalty=1.
    cfg.learning_rate=.003
    cfg.meta_step_size=.25
    cfg.cpu_workers=4
    return cfg

def require_stage1():
    from pathlib import Path
    parent=read(ROOT/'PARENT_R2.json');r2=Path(parent['archive'])
    assert read(r2/'stage2_analysis/AUDIT.json')['passed']
    assert read(r2/'stage2_coordinator/COMPLETE.json')['complete']
    return dict(r2_archive=str(r2),r2_meta_sha256=sha(r2/'stage2_training/meta_retrained.pt'),r2_audit_passed=True)

class FitPool:
    def __init__(self,cfg):
        from solver.data import project_api
        from solver.config import q_values
        self.cfg=cfg;self.api=project_api(cfg.project_root);self.q_values=q_values
        self.manifest=read(ROOT/'FIT_MANIFEST.json');self.ids=set(self.manifest['fit_ids']);self.cache={}
        assert len(self.ids)==180 and not self.ids.intersection(self.manifest['excluded_validation_ids']+self.manifest['excluded_test_ids'])
    def load(self,iid):
        if iid not in self.ids:raise ValueError('Stage2 training outside180 FIT')
        if iid not in self.cache:
            row=read(ROOT/'fit_inputs'/f'{iid}.json')
            assert row['instance_id']==iid and row['split']=='train'
            inst=self.api.instance(row)
            sol=self.api.Solution([self.api.Route(list(r)) for r in row['source_solution']])
            assert self.api.feasible(sol,inst)
            self.cache[iid]=inst,row,copy.deepcopy(self.manifest['boxes'][iid]),self.q_values(inst.n_customers)
        return self.cache[iid]
