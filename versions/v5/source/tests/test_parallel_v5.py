import unittest,json,threading,time
from pathlib import Path
import numpy as np
import torch
from run import load_config
from solver.data import TrainPool
from solver.parallel_runtime import ParallelRuntime
from solver.training import group_advantages
ROOT=Path(__file__).resolve().parents[1]

class V5Tests(unittest.TestCase):
    def test_180_fit_45_validation_and_test_is_not_loadable_for_training(self):
        pool=TrainPool(load_config(ROOT/'full_config.json'))
        self.assertEqual((len(pool.split['fit_ids']),len(pool.split['validation_ids'])),(180,45))
        with self.assertRaises(ValueError):pool.load('50-0-0')
        self.assertEqual(pool.load(pool.split['validation_ids'][0])[1]['split'],'validation')
    def test_each_block_covers45_cells_exactly(self):
        cfg=load_config(ROOT/'full_config.json');pool=TrainPool(cfg)
        for start in range(0,180,45):
            keys=[tuple(pool.load(i)[1]['cell'][k] for k in ('n','d','tw')) for i in pool.split['fit_order'][start:start+45]]
            self.assertEqual(len(set(keys)),45)
    def test_instance_groups_and_tasks_are_not_pooled(self):
        eps=[]
        for iid in ('a','b'):
            for task in range(5):
                for r in range(4):
                    eps.append([dict(iid=iid,task=task,reward=float(r+1),terminal=True,truncated=False,
                        observation={'search_state':{'last_accepted':True,'stagnation':0}},
                        diagnostics={'repair_seconds':.01,'inference_seconds':.001})])
        groups,_,weights=group_advantages(eps);self.assertEqual(len(groups),10)
        self.assertAlmostEqual(sum(v.sum() for v in weights.values())/len(groups),1.)
    def test_target_and_process_configuration(self):
        cfg=load_config(ROOT/'full_config.json')
        self.assertEqual((cfg.independent_iid_batch,cfg.cpu_workers,cfg.microbatch,cfg.meta_target_epochs),(8,7,32,3))
    def test_queued_lane_preserves_its_active_deadline_and_event_clock(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        runtime=object.__new__(ParallelRuntime);runtime.active=threading.Semaphore(0);runtime.inactive_wait=0.
        initial=time.perf_counter();session=SimpleNamespace(solver_task_deadline=initial+5,records=[])
        def finish(s,*args):s.records.append({'candidate_finished_perf':time.perf_counter()})
        release=threading.Timer(.04,runtime.active.release);release.start()
        with patch('solver.outer.run_session',finish):wait,end=runtime.run_primary(session,None)
        release.join()
        self.assertGreaterEqual(wait,.025)
        self.assertAlmostEqual(session.solver_task_deadline-(initial+5),wait,places=5)
        active_event=session.records[0]['candidate_finished_perf']-session.records[0]['inactive_pause_before_candidate']
        self.assertLess(active_event-initial,.025)
    def test_concurrent_searches_preserve_each_seed_rng_and_population(self):
        from concurrent.futures import ThreadPoolExecutor
        from fixtures import Pool,CFG
        from solver.outer import Search
        import copy
        cfg=copy.deepcopy(CFG);cfg.solve_seconds_by_n={50:30.,150:30.,250:30.,12:30.}
        def build(seed):return Search(Pool(12),'synthetic',cfg,seed,seconds=30.)
        serial=[build(s) for s in (123,456)]
        concurrent=[build(s) for s in (123,456)]
        def advance(search):
            for _ in range(8):self.assertTrue(search.step(lambda k,p:(p,0.,[])))
        for item in serial:advance(item)
        with ThreadPoolExecutor(2) as executor:list(executor.map(advance,concurrent))
        for left,right in zip(serial,concurrent):
            np.testing.assert_allclose(left.algorithm.pop.get('F'),right.algorithm.pop.get('F'))
            self.assertEqual([p.X[0].routes for p in left.algorithm.pop],[p.X[0].routes for p in right.algorithm.pop])
            np.testing.assert_array_equal(left.numpy_state[1],right.numpy_state[1])

if __name__=='__main__':unittest.main()
