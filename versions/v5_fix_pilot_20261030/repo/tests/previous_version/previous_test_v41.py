"""V4.1 synthetic unit checks; no experiment or optimizer steps."""
import copy,math,unittest
from unittest.mock import patch
import numpy as np
import torch
from test_v4 import CFG,Pool,observation
from solver.model import Policy
from solver.training import group_advantages,q_summary
from solver.environment import LocalSession,Adaptive
from solver.q_statistics import empty_q,finish_q,state_labels,add_q


class StatePolicyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(51);self.policy=Policy(CFG);_,self.obs=observation()
        self.hidden,self.nodes,self.x,self.raw=self.policy.encode([self.obs])

    def logits(self,hidden=None,x=None,raw=None,nodes=None):
        return self.policy.q_logits(self.hidden if hidden is None else hidden,
            self.nodes if nodes is None else nodes,self.x if x is None else x,
            self.raw if raw is None else raw,[0])

    def test_structural_momentum_inputs_and_film_match_formula(self):
        captured={}
        hooks=[self.policy.q_body.register_forward_pre_hook(lambda m,a:captured.update(body=a[0])),
               self.policy.q_film.register_forward_pre_hook(lambda m,a:captured.update(film=a[0]))]
        actual=self.logits()
        for hook in hooks:hook.remove()
        phys=self.x[:,:,[18,17,19,12,13]];mu=phys.mean(1)
        sigma=phys.std(1,unbiased=False)
        cv=(sigma/phys.abs().mean(1).clamp_min(1e-4)).clamp(0.,4.)/4.
        z_peak=((phys.amax(1)-mu)/sigma.clamp_min(1e-4)).clamp(0.,6.)/6.
        top10_share=(phys.topk(max(1,phys.shape[1]//10),dim=1).values.abs().sum(1)
                     /phys.abs().sum(1).clamp_min(1e-4)).clamp(0.,1.)
        stats=torch.cat((cv,z_peak,top10_share),-1)
        momentum=torch.cat((self.raw[:,:5],self.raw[:,11:18]),-1)
        state=torch.cat((stats,momentum),-1)
        self.assertEqual(captured['body'].shape,(1,283))
        torch.testing.assert_close(captured['body'][:,:128],self.policy.q_history_norm(self.hidden))
        torch.testing.assert_close(captured['body'][:,256:],state)
        torch.testing.assert_close(captured['film'],state)
        base=self.policy.q_body(captured['body']);gamma,beta=self.policy.q_film(state).chunk(2,-1)
        indicators=torch.stack((top10_share[:,1:].mean(-1),top10_share[:,0],
            self.raw[:,15],self.raw[:,14],1.-self.raw[:,13],self.raw[:,11]),-1)
        hidden=torch.tanh((1+gamma)*base+beta)
        base_raw=self.policy.heads[0].q(hidden)
        base_centered=base_raw-base_raw.mean(-1,keepdim=True)
        regime=1 if self.raw[0,15]>.5 else 2 if self.raw[0,14]>1e-6 else 0
        state_raw=(self.policy.heads[0].q_regime_table[regime]+indicators@self.policy.heads[0].q_state_bilinear)
        state_centered=state_raw-state_raw.mean(-1,keepdim=True)
        expected=.75*torch.tanh(base_centered/.75)+1.35*torch.tanh(state_centered/1.35)
        torch.testing.assert_close(actual,expected)
        self.assertTrue(bool(((stats>=0)&(stats<=1)).all()))

    def test_no_direct_static_node_pooling_and_hidden_has_gradient(self):
        torch.testing.assert_close(self.logits(),self.logits(nodes=self.nodes+100))
        hidden=self.hidden.detach().clone().requires_grad_()
        logits=self.logits(hidden=hidden);logits[0,0].backward()
        self.assertGreater(float(hidden.grad.abs().sum()),0)
        self.assertFalse(torch.equal(logits.detach(),self.logits(hidden=hidden.detach()+torch.randn_like(hidden))))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in self.policy.q_film.parameters()),0)

    def test_physical_shape_and_each_momentum_channel_reach_q(self):
        base=self.logits()
        changed=self.x.clone();changed[:,0,17]+=1
        self.assertFalse(torch.equal(base,self.logits(x=changed)))
        for index in list(range(5))+list(range(11,18)):
            raw=self.raw.clone();raw[:,index]+=.4
            self.assertFalse(torch.equal(base,self.logits(raw=raw)),index)

    def test_single_customer_stats_finite_and_logits_soft_bounded(self):
        x=self.x[:,:1];self.assertTrue(torch.isfinite(self.logits(x=x)).all())
        with torch.no_grad():
            self.policy.heads[0].q.weight.zero_()
            self.policy.heads[0].q.bias.copy_(torch.tensor([1e6]+[-1e6]*9))
        logits=self.logits();self.assertTrue(bool((logits.abs()<=2.1).all()))
        prob=logits.softmax(-1)
        self.assertLessEqual(float(prob.detach().max()),1/(1+9*math.exp(-4.2))+1e-6)
        self.assertTrue(bool((prob>0).all()))

    def test_structure_is_scale_invariant_above_epsilon_and_zero_is_finite(self):
        # The denominator floors intentionally stop exact scale invariance near zero.
        x=torch.zeros(1,20,20);x[:,:,[18,17,19,12,13]]=torch.arange(1.,21.).reshape(1,20,1)
        captured=[]
        hook=self.policy.q_body.register_forward_pre_hook(lambda m,a:captured.append(a[0][:,256:271].detach()))
        self.logits(x=x);self.logits(x=x*.01)
        torch.testing.assert_close(captured[0],captured[1],atol=2e-6,rtol=2e-6)
        zero=self.logits(x=x*0)
        self.assertTrue(torch.isfinite(zero).all())
        torch.testing.assert_close(captured[-1],torch.zeros(1,15))
        hook.remove()

    def test_centered_ste_gradient_does_not_vanish_at_saturated_logits(self):
        with torch.no_grad():
            self.policy.heads[0].q.weight.zero_()
            self.policy.heads[0].q.bias.copy_(torch.tensor([1000.]+[-1000.]*9))
            self.policy.heads[0].q_state_bilinear.zero_()
        logits=self.logits()
        logits[0,0].backward()
        expected=torch.full((10,),-.1);expected[0]=.9
        torch.testing.assert_close(self.policy.heads[0].q.bias.grad,expected)
        self.assertTrue(bool((logits.abs()<=2.1).all()))
        # Adding a shared raw bias must leave forward probabilities unchanged.
        with torch.no_grad():self.policy.heads[0].q.bias.add_(123.)
        torch.testing.assert_close(logits,self.logits())

    def test_state_bilinear_prior_six_indicators_and_trainable_gradient(self):
        coord=torch.linspace(-1.,1.,10)
        prior=torch.outer(torch.tensor([-.6,-.4,-.5,.9,-.5,-.4]),coord)
        self.assertEqual(self.policy.heads[0].q_state_bilinear.shape,(6,10))
        torch.testing.assert_close(self.policy.heads[0].q_state_bilinear.detach(),prior,atol=.08,rtol=0)
        # Isolate the direct state path; q_film and q_body cannot affect this check.
        with torch.no_grad():
            self.policy.heads[0].q.weight.zero_();self.policy.heads[0].q.bias.zero_()
            self.policy.heads[0].q_state_bilinear.copy_(prior)
        raw=self.raw.clone();raw[:,11]=.4;raw[:,13]=.2;raw[:,14]=.3;raw[:,15]=1.
        logits=self.logits(raw=raw)
        logits[0,0].backward()
        self.assertGreater(float(self.policy.heads[0].q_state_bilinear.grad.abs().sum()),0)
        for index in (11,13,14,15):
            changed=raw.clone();changed[:,index]+=.1
            self.assertFalse(torch.equal(logits,self.logits(raw=changed)))

    def test_task_batch_and_checkpoint_roundtrip(self):
        policy=Policy(CFG,tasks=5);obs=[self.obs]*5
        actions,lp,_,_=policy.act(obs,list(range(5)))
        replica=Policy(CFG,tasks=5);replica.load_state_dict(policy.actor_state())
        _,replay,_,_,_=replica.terms(obs,list(range(5)),actions)
        torch.testing.assert_close(replay,torch.tensor(lp),atol=1e-5,rtol=1e-5)


class StateStatisticsTests(unittest.TestCase):
    def test_state_thresholds_and_empty_empirical_entropy(self):
        for count,label in [(0,'zero'),(1,'one_to_three'),(3,'one_to_three'),(4,'four_or_more'),(99,'four_or_more')]:
            self.assertEqual(state_labels(False,count)['by_stagnation'],label)
        empty=finish_q(empty_q());self.assertIsNone(empty['q_entropy_nats']);self.assertIsNone(empty['single_observed_slot'])
        block=empty_q();add_q(block,1,10,0);add_q(block,3,10,5)
        result=finish_q(block)
        self.assertEqual(result['effective_q_slots'],2);self.assertAlmostEqual(result['q_entropy_nats'],math.log(2))
        self.assertAlmostEqual(result['mean_q_over_n'],.2)

    def test_training_uses_raw_predecision_stagnation_not_reward(self):
        _,obs=observation();episodes=[]
        for accepted,stall,qi in [(True,0,0),(False,3,5),(False,15,9)]:
            observation_=copy.deepcopy(obs);observation_['search_state']=dict(last_accepted=accepted,stagnation=stall)
            episodes.append([dict(task=0,direction=0,observation=observation_,reward=1.,
                                 action=dict(q=obs['q_values'][qi],q_index=qi))])
        result=q_summary(episodes)
        self.assertEqual(result['by_last_accepted']['true']['count'],1)
        self.assertEqual(result['by_last_accepted']['false']['count'],2)
        self.assertEqual([b['count'] for b in result['by_stagnation'].values()],[1,1,1])

    def test_record_keeps_state_from_before_action(self):
        env,obs=observation();env.last_accepted=True;env.stagnation=3
        session=LocalSession(env,Adaptive(10),12,5.,5.,.05)
        prepared=session.prepare();self.assertEqual(prepared['search_state'],dict(last_accepted=True,stagnation=3))
        def step(*a,**kw):
            env.last_accepted=False;env.stagnation=4
            return dict(accepted=False,feasible=False,timeout=True,reward=0.,environment_seconds=0.,repair_seconds=.01)
        with patch.object(env,'step',side_effect=step):
            record=session.finish(dict(q_index=0,q=1,removed=[1]),torch.zeros(1,128))
        self.assertTrue(record['last_accepted_before']);self.assertEqual(record['stagnation_before'],3)
        self.assertEqual(env.stagnation,4)


if __name__=='__main__':unittest.main(verbosity=2)
