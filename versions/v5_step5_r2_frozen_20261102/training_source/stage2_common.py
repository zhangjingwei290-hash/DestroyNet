"""Step2-of-Step5 isolated paths; no neural library at import."""
from common import ROOT, read, write, sha, reset, log, EvalPool, readonly_check
import sys, copy
sys.path.insert(0,str(ROOT/'stage2_repo'))

def load_cfg():
    from run import load_config
    cfg=load_config(ROOT/'eval_config.json')
    cfg.step5_reference_penalty=1.
    cfg.learning_rate=.003
    cfg.meta_step_size=.5
    cfg.cpu_workers=4
    return cfg

def require_stage1():
    assert read(ROOT/'stage1_coordinator/COMPLETE.json')['complete']
    assert read(ROOT/'stage1_analysis/AUDIT.json')['passed']
    summary=read(ROOT/'stage1_summary.json')
    assert summary['mechanism_revision']=='r2_user_emergency_three_fixes'
    assert (summary['new_runs'],summary['total_runs'])==(36,144)
    report=(ROOT/'STAGE1_REPORT.md').read_text(encoding='utf-8')
    assert report
    return dict(stage1_report_sha256=sha(ROOT/'STAGE1_REPORT.md'),stage1_summary_sha256=sha(ROOT/'stage1_summary.json'),stage1_audit_passed=True)

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
