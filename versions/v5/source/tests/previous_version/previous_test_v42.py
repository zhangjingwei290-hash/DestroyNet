"""V4.2 exact-formula checks on synthetic states; no experiment training."""
import copy,math,unittest
import numpy as np
import torch
from test_v4 import CFG,observation
from solver.config import RATIOS
from solver.model import Policy
from solver.q_statistics import REGIMES,regime_name,regime_id
from solver.training import group_advantages,q_summary


class RegimePolicyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(73);self.policy=Policy(CFG);_,self.obs=observation()
        self.hidden,self.nodes,self.x,self.raw=self.policy.encode([self.obs])

    def test_regime_mapping_matches_context_decoder_and_used_table_rows(self):
        cases=[(False,0),(True,0)]+[(False,k) for k in (1,2,3,4,10,25)]
        for accepted,stall in cases:
            expected='refine' if accepted else 'initial' if stall==0 else 'stalled'
            self.assertEqual(regime_name(accepted,stall),expected)
            self.assertEqual(REGIMES[regime_id(accepted,stall)],expected)
            raw=self.raw.clone();raw[:,15]=float(accepted);raw[:,14]=min(1.,stall/10.)
            self.policy.zero_grad(set_to_none=True)
            logits=self.policy.q_logits(self.hidden,self.nodes,self.x,raw,[0]);logits[0,0].backward(retain_graph=True)
            grads=self.policy.heads[0].q_regime_table.grad
            for row in range(3):
                if row==regime_id(accepted,stall):self.assertGreater(float(grads[row].abs().sum()),0)
                else:torch.testing.assert_close(grads[row],torch.zeros(10),atol=0,rtol=0)
        with self.assertRaises(ValueError):regime_name(False,-1)

    def test_expanded_base_keeps_separate_bounds_and_fixture_state_response(self):
        with torch.no_grad():
            self.policy.heads[0].q.weight.mul_(100.)
            self.policy.heads[0].q.bias.mul_(100.)
        raws=self.raw.repeat(2,1);raws[0,15]=1.;raws[0,14]=0.;raws[1,15]=0.;raws[1,14]=.3
        captured=[]
        hook=self.policy.heads[0].q.register_forward_hook(lambda m,a,out:captured.append(out))
        logits=self.policy.q_logits(self.hidden.repeat(2,1),self.nodes.repeat(2,1,1),self.x.repeat(2,1,1),raws,[0,0])
        hook.remove()
        base_raw=torch.stack(captured);base_centered=base_raw-base_raw.mean(-1,keepdim=True)
        base=.75*torch.tanh(base_centered/.75);state=logits-base
        self.assertTrue(bool((base.abs()<=.75).all()))
        self.assertTrue(bool((state.abs()<=1.35+1e-6).all()))
        probabilities=logits.softmax(-1);expected=(probabilities*logits.new_tensor(RATIOS)).sum(-1)
        self.assertLess(float(expected[0].detach()),float(expected[1].detach()))
        self.assertLess(float(probabilities.detach().max()),.70)

    def test_global_probability_bound_is_not_a_universal_point_seven(self):
        # Explicit counterexample to the prose claim; preserve the requested code.
        with torch.no_grad():
            self.policy.heads[0].q.weight.zero_()
            extreme=torch.tensor([1000.]+[-1000.]*9)
            self.policy.heads[0].q.bias.copy_(extreme)
            self.policy.heads[0].q_state_bilinear.zero_()
            self.policy.heads[0].q_regime_table.copy_(extreme.repeat(3,1))
        logits=self.policy.q_logits(self.hidden,self.nodes,self.x,self.raw,[0])
        self.assertTrue(bool((logits.abs()<=2.1+1e-6).all()))
        maximum=float(logits.softmax(-1).detach().max())
        self.assertAlmostEqual(maximum,1/(1+9*math.exp(-4.2)),places=6)
        self.assertGreater(maximum,.70)

    def test_refine_only_batch_table_gradient_is_isolated(self):
        obs=copy.deepcopy(self.obs);obs['context'][15]=1.;obs['context'][14]=0.
        obs['search_state']=dict(last_accepted=True,stagnation=0)
        action=dict(q_index=3,q=obs['q_values'][3],removed=list(range(1,obs['q_values'][3]+1)))
        _,lp,_,entropy,_=self.policy.terms([obs]*4,actions=[action]*4)
        (-lp.mean()-.001*entropy.mean()).backward()
        grad=self.policy.heads[0].q_regime_table.grad
        self.assertGreater(float(grad[1].abs().sum()),0)
        torch.testing.assert_close(grad[0],torch.zeros(10),atol=0,rtol=0)
        torch.testing.assert_close(grad[2],torch.zeros(10),atol=0,rtol=0)


class RegimeAdvantageTests(unittest.TestCase):
    def fixture(self):
        group=[]
        for replica in range(4):
            ep=[]
            entries=[(.05,.01,False,0)]+[(.002,.003 if k%2==0 else .08,True,0) for k in range(20)]+[(0.,.05,False,1)]
            for k,(reward,seconds,accepted,stall) in enumerate(entries):
                ep.append(dict(task=0,reward=reward,terminal=k==len(entries)-1,truncated=False,
                    diagnostics=dict(repair_seconds=seconds),observation=dict(search_state=dict(last_accepted=accepted,stagnation=stall))))
            group.append(ep)
        return group

    def test_regime_normalization_and_equal_group_weight(self):
        group=self.fixture();groups,advantages,weights=group_advantages(group)
        byreg={reg:[] for reg in REGIMES};totals={reg:0. for reg in REGIMES};fast=[];slow=[]
        for ep in group:
            for k,t in enumerate(ep):
                st=t['observation']['search_state'];reg=regime_name(st['last_accepted'],st['stagnation'])
                byreg[reg].append(advantages[id(ep)][k]);totals[reg]+=weights[id(ep)][k]
                if reg=='refine':(fast if t['diagnostics']['repair_seconds']==.003 else slow).append(advantages[id(ep)][k])
        for reg in REGIMES:
            self.assertAlmostEqual(float(np.mean(byreg[reg])),0.,places=10)
            self.assertAlmostEqual(totals[reg],1/3,places=12)
        self.assertGreater(float(np.mean(fast)),0.)
        self.assertGreater(float(np.mean(fast)),float(np.mean(slow)))
        self.assertEqual(len(groups[0]),4)

    def test_exact_net_surplus_formula_and_future_window(self):
        group=self.fixture();_,advantages,weights=group_advantages(group)
        total_r=sum(t['reward'] for ep in group for t in ep)
        total_dt=sum(max(t['diagnostics']['repair_seconds'],.001)+.005 for ep in group for t in ep)
        reference=[];regimes=[]
        for ep in group:
            future=0.;values=[];regs=[]
            for t in reversed(ep):
                r=max(0.,t['reward']);dt=max(t['diagnostics']['repair_seconds'],.001)+.005
                st=t['observation']['search_state'];reg=regime_name(st['last_accepted'],st['stagnation'])
                value=(r/(dt/.02)**.65+.85*future-1.2*(total_r/total_dt)*dt if reg=='refine'
                       else r+.5*future-.25*(total_r/total_dt)*dt)
                values.append(value);regs.append(reg);future=r+.85*future
            reference.append(values[::-1]);regimes.append(regs[::-1])
        for reg in REGIMES:
            vals=np.array([value for vs,rs in zip(reference,regimes) for value,r in zip(vs,rs) if r==reg])
            expected=np.zeros(len(vals)) if vals.std()<=1e-12 else (vals-vals.mean())/(vals.std()+1e-8)
            actual=np.array([advantages[id(ep)][k] for ep,rs in zip(group,regimes) for k,r in enumerate(rs) if r==reg])
            np.testing.assert_allclose(actual,expected)

    def test_zero_variance_missing_regimes_and_task_weights(self):
        episodes=[]
        for task in range(2):
            for _ in range(4):
                episodes.append([dict(task=task,reward=0.,terminal=True,truncated=False,
                    diagnostics=dict(repair_seconds=0.),observation=dict(search_state=dict(last_accepted=False,stagnation=0)))])
        groups,advantages,weights=group_advantages(episodes)
        for ep in episodes:
            np.testing.assert_array_equal(advantages[id(ep)],[0.])
            np.testing.assert_allclose(weights[id(ep)],[.25])
        self.assertAlmostEqual(sum(float(weights[id(ep)].sum())/len(groups) for ep in episodes),1.)

    def test_regime_statistics_derive_predecision_state(self):
        _,obs=observation();episodes=[]
        for accepted,stall in [(False,0),(True,0),(False,1)]:
            o=copy.deepcopy(obs);o['search_state']=dict(last_accepted=accepted,stagnation=stall)
            episodes.append([dict(task=0,observation=o,action=dict(q=1,q_index=0))])
        blocks=q_summary(episodes)['by_regime']
        self.assertEqual([blocks[reg]['count'] for reg in REGIMES],[1,1,1])


if __name__=='__main__':unittest.main(verbosity=2)
