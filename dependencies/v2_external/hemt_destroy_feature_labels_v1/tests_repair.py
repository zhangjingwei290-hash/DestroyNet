"""CPU-only module tests; no dataset regeneration or neural imports."""
import copy
import time
import unittest
from unittest.mock import patch
import numpy as np
from src.core.protocol import VRPTWInstance, Solution, Route, RepairAction
from src.evaluation.vrptw_eval import evaluate_solution, is_solution_feasible
from .repair import METHODS, repair, _regret, _choose


def instance(n=20, capacity=10):
    ids=np.arange(n+1,dtype=float)
    # Directed, non-Euclidean travel independent of coordinates and duration.
    distance=1+np.abs(ids[:,None]-ids[None,:])+.03*ids[:,None]
    duration=.2+.01*np.abs(ids[:,None]-ids[None,:])+.003*ids[None,:]
    np.fill_diagonal(distance,0); np.fill_diagonal(duration,0)
    return VRPTWInstance('cpu_repair',n,capacity,3.,(4.,1000.),np.column_stack([ids[1:],ids[1:]**2]),
        np.ones(n),np.tile([0.,900.],(n,1)),np.full(n,.1),distance,duration)


def source(inst):
    return Solution([Route(list(range(1,inst.n_customers//2+1))),Route(list(range(inst.n_customers//2+1,inst.n_customers+1)))])


def candidate(c,pos,score,new=False):
    return {'action':RepairAction(c,-1 if new else 0,0 if new else pos,new),'score':score}


class RepairTests(unittest.TestCase):
    def test_regret_max_value_and_ties(self):
        rng=np.random.default_rng(1)
        # c2 best10, alt100 has more regret than c1 best1,alt2; choose c2.
        panel=[candidate(1,0,1),candidate(1,1,2),candidate(2,0,10),candidate(2,1,100)]
        self.assertEqual(_choose(panel,'regret2',rng)['action'].customer_id,2)
        panel=[candidate(1,0,1),candidate(1,1,2),candidate(1,2,3),
               candidate(2,0,10),candidate(2,1,11),candidate(2,2,100)]
        self.assertEqual(_choose(panel,'regret3',rng)['action'].customer_id,2)
        self.assertEqual(_regret([3,1,2],3),3)
        # Equal regrets -> better best score; then structural key.
        panel=[candidate(2,0,2),candidate(2,1,4),candidate(1,0,1),candidate(1,1,3)]
        self.assertEqual(_choose(panel,'regret2',rng)['action'].customer_id,1)

    def test_regret_scarcity_is_explicit(self):
        panel=[candidate(1,0,100),candidate(2,0,1),candidate(2,1,500)]
        chosen=_choose(panel,'regret2',np.random.default_rng(1))
        self.assertEqual(chosen['action'].customer_id,1)
        self.assertEqual(_regret([100],3),0)

    def test_q1_and_all_methods_deterministic_no_mutation(self):
        inst=instance(); original=source(inst); snapshot=copy.deepcopy(original)
        for method in METHODS:
            a=repair(inst,original,[3],method,[.2]*5,[20,100,100,100,100],11)
            b=repair(inst,original,[3],method,[.2]*5,[20,100,100,100,100],11)
            self.assertTrue(a['feasible'],a)
            self.assertEqual(a['routes'],b['routes'])
            self.assertEqual(a['trace'],b['trace'])
            self.assertFalse(a['trace'][0]['opens_new_route'])
            self.assertTrue(is_solution_feasible(a['solution'],inst))
            np.testing.assert_allclose(a['objectives'],evaluate_solution(a['solution'],inst).to_array(),rtol=0,atol=1e-12)
            self.assertEqual(a['counters']['final_evaluations'],1)
            self.assertEqual(a['counters']['inserted_customers'],1)
        self.assertEqual(original,snapshot)

    def test_multiple_removals_full_vector(self):
        inst=instance(); original=source(inst)
        for method in METHODS:
            out=repair(inst,original,[1,2,11],method,[0,1,0,0,0],{'ranges':[20,100,100,100,100],'mins':[0]*5},4)
            self.assertTrue(out['feasible'],out)
            self.assertEqual(out['counters']['inserted_customers'],3)
            self.assertEqual(len(out['objectives']),5)
            self.assertGreater(out['counters']['candidate_scores'],3)

    def test_expired_deadline_returns_none_no_fallback(self):
        inst=instance(); original=source(inst)
        out=repair(inst,original,[1],'greedy',[.2]*5,[1]*5,1,time.perf_counter()-1)
        self.assertTrue(out['timeout']); self.assertFalse(out['feasible'])
        self.assertIsNone(out['routes']); self.assertIsNone(out['objectives'])
        self.assertEqual(out['counters']['enumeration_calls'],0)

    def test_deadline_checked_during_enumeration(self):
        import hemt_destroy_feature_labels_v1.repair as module
        # A cooperative engine block crosses cutoff; no candidate is submitted.
        original_enumerate=module.enumerate_actions
        def slow(*args):
            batch=original_enumerate(*args)
            time.sleep(.015)
            return batch
        with patch.object(module,'enumerate_actions',slow):
            out=repair(instance(),source(instance()),[1,2],'regret2',[.2]*5,[1]*5,1,time.perf_counter()+.01)
        self.assertTrue(out['timeout']); self.assertIsNone(out['solution'])
        self.assertLessEqual(out['counters']['enumeration_calls'],1)

    def test_empty_skeleton_reopens_route(self):
        inst=instance(1,capacity=10); src=Solution([Route([1])])
        out=repair(inst,src,[1],'greedy',[.2]*5,[1]*5,7)
        self.assertTrue(out['feasible'],out)
        self.assertEqual(out['routes'],[[1]])
        self.assertEqual(out['counters']['selected_new_routes'],1)

    def test_invalid_cap_and_infeasible_source(self):
        inst=instance(); src=source(inst)
        bad=repair(inst,src,[1,2,3,4],'greedy',[.2]*5,[1]*5,7)
        self.assertEqual(bad['reason'],'removal_cap_exceeded')
        inst.vehicle_capacity=1
        bad=repair(inst,src,[1],'greedy',[.2]*5,[1]*5,7)
        self.assertEqual(bad['reason'],'source_infeasible')
        self.assertIsNone(bad['routes'])
        with self.assertRaises(ValueError): repair(inst,src,[1],'fake_fallback',[.2]*5,[1]*5,7)

    def test_per_customer_new_mask(self):
        # Inject engine-mask legal candidates: c1 fits existing, c2 needs new.
        # Exact route validation and final full validation still use real code.
        import hemt_destroy_feature_labels_v1.repair as module
        inst=instance(); inst.vehicle_capacity=10
        src=source(inst)
        original_enumerate=module.enumerate_actions
        def special(i,p,remaining):
            batch=original_enumerate(i,p,remaining)
            if remaining==[11] and len(p.routes)==2:
                batch.mask=np.asarray([a.opens_new_route for a in batch.actions],dtype=bool)
            return batch
        with patch.object(module,'enumerate_actions',special):
            out=repair(inst,src,[1,11],'regret2',[.2]*5,[1]*5,7)
        self.assertTrue(out['feasible'],out)
        self.assertEqual(out['trace'][0]['customer'],11)
        self.assertTrue(out['trace'][0]['opens_new_route'])
        self.assertFalse(out['trace'][1]['opens_new_route'])


if __name__=='__main__': unittest.main()
