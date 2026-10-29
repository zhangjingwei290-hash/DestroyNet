"""Budget/identity regression only: no model, training or solver runs."""
import json
import math
from pathlib import Path
import unittest
from solver.config import Config,search_budget
from run import load_config

ROOT=Path(__file__).resolve().parents[1]
TABLE=json.loads((ROOT/'paper_search_budgets.json').read_text())['seconds_by_instance']

def paper_config():
    return Config(meta_seconds=9000,adaptation_seconds=5400,local_seconds=10.,repair_seconds=10.,
                  solve_seconds_by_instance=TABLE.copy())

class SearchBudgetTests(unittest.TestCase):
    def test_all45_real_instance_horizons_match_transcribed_table(self):
        cfg=paper_config();cfg.validate(require_budget=True)
        self.assertEqual(len(TABLE),45)
        for iid,value in TABLE.items():
            self.assertEqual(search_budget(int(iid.split('-')[0]),cfg,iid=iid),value)
        self.assertEqual(sum(TABLE.values()),11413.)

    def test_synthetic_replicas_use_explicit_cell_in_all_splits(self):
        cfg=paper_config()
        for iid,value in TABLE.items():
            n,d,tw=map(int,iid.split('-'))
            for split in ('train','validation'):
                for replica in range(4):
                    identity=f'v5_{split}_{n}_d{d}_tw{tw}_{replica}'
                    row={'cell':{'n':n,'d':d,'tw':tw}}
                    self.assertEqual(search_budget(n,cfg,iid=identity,row=row),value)

    def test_no_old_hemt_or_size_mean_fallback(self):
        cfg=paper_config()
        with self.assertRaisesRegex(ValueError,'No reviewed'):
            search_budget(50,cfg,iid='hemtv2_train_50_C_9')
        with self.assertRaisesRegex(ValueError,'No reviewed'):
            search_budget(250,cfg)

    def test_real_identity_and_cell_must_agree(self):
        with self.assertRaisesRegex(ValueError,'differ'):
            search_budget(50,paper_config(),iid='50-0-0',row={'cell':{'n':50,'d':1,'tw':0}})

    def test_size_mismatch_and_invalid_cell_rejected(self):
        cfg=paper_config()
        for n,iid,row in ((150,'50-0-0',None),(50,'synthetic',{'cell':{'n':150,'d':0,'tw':0}}),
                          (50,'synthetic',{'cell':{'n':50,'d':3,'tw':0}}),
                          (50,'synthetic',{'cell':{'n':50,'d':0.5,'tw':0}})):
            with self.assertRaises(ValueError):search_budget(n,cfg,iid=iid,row=row)

    def test_config_rejects_competing_or_incomplete_budget_tables(self):
        cfg=paper_config();cfg.solve_seconds_by_n={50:10,150:15,250:20}
        with self.assertRaisesRegex(ValueError,'not both'):cfg.validate(require_budget=True)
        cfg=paper_config();cfg.solve_seconds_by_instance.pop('250-2-4')
        with self.assertRaisesRegex(ValueError,'all45'):cfg.validate(require_budget=True)

    def test_nonfinite_or_nonpositive_budget_is_rejected(self):
        for value in (math.inf,math.nan,0.,-1.):
            cfg=paper_config();cfg.solve_seconds_by_instance['50-0-0']=value
            with self.assertRaises(ValueError):cfg.validate(require_budget=True)

    def test_legacy_explicit_size_mode_is_preserved(self):
        cfg=Config(solve_seconds_by_n={50:10.,150:15.,250:20.})
        cfg.validate()
        self.assertEqual([search_budget(n,cfg) for n in (50,150,250)],[10.,15.,20.])
        with self.assertRaises(ValueError):search_budget(50,Config())

    def test_reviewed_full_config_has_V2_inner_and_linked_training_budget(self):
        cfg=load_config(ROOT/'full_config.json');cfg.validate(require_budget=True)
        self.assertEqual((cfg.local_seconds,cfg.repair_seconds),(10.,10.))
        self.assertEqual(cfg.microbatch,32)
        planned=json.loads((ROOT/'training_budget_plan.json').read_text())
        self.assertEqual((cfg.meta_seconds,cfg.adaptation_seconds),(planned['meta_seconds'],planned['adaptation_seconds']))
        self.assertEqual(cfg.meta_seconds,129600)
        self.assertEqual(cfg.meta_target_epochs,3)
        self.assertIsNone(cfg.solve_seconds_by_n)
        self.assertEqual(cfg.solve_seconds_by_instance,TABLE)
        self.assertEqual(search_budget(250,cfg,iid='250-0-3'),407.)

    def test_training_allowance_counts_primary_once_and_three_extra_copies(self):
        from plan_training_budget import training_budget
        # Uniform T=160 gives local cap 10: 180*(160+3*5*3*10).
        b=training_budget({key:160. for key in TABLE})
        self.assertEqual(b['primary_seconds_per_meta_epoch'],28800.)
        self.assertEqual(b['clone_window_envelope_per_meta_epoch'],81000.)
        self.assertEqual(b['meta_seconds'],395280)
        self.assertEqual(b['adaptation_seconds'],216000)

    def test_search_constructor_and_local_session_apply_reviewed_horizons(self):
        # Initialization only; no search step, optimizer or model is run.
        from fixtures import Pool
        from solver.outer import Search
        from solver.environment import LocalSession
        class CellPool(Pool):
            def __init__(self,n,d,tw):
                super().__init__(n);self.cell={'n':n,'d':d,'tw':tw}
            def load(self,iid):
                inst,row,box,qs=super().load(iid)
                return inst,dict(row,cell=self.cell),box,qs
        cfg=paper_config();cfg.device='cpu'
        for n,d,tw,expected in ((50,1,0,114.),(150,0,3,309.),(250,0,3,407.)):
            iid=f'v5_validation_{n}_d{d}_tw{tw}_0'
            search=Search(CellPool(n,d,tw),iid,cfg,91)
            self.assertEqual(search.budget,expected)
            session=LocalSession(search.base,search.adaptive[15],91,expected,expected,cfg.local_seconds)
            self.assertEqual(session.budget,min(10.,expected/16.))
            self.assertEqual(session.env.cfg.repair_seconds,10.)

if __name__=='__main__':unittest.main()
