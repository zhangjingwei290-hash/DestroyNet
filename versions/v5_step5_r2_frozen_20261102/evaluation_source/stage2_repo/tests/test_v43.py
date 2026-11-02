"""Continuous-Beta/CUDA Graph parity, latency and physical compatibility checks."""
from fixtures import *
import statistics,math
from solver.model import integer_q,selected_logprob
from solver.q_statistics import nearest_q_slot,REGIMES,regime_name
from solver.environment import LocalSession
from solver.config import CONTEXT


class ContinuousTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(82)

    def test_beta_replay_matches_action_logprob_and_has_gradients(self):
        _,obs=observation(12);p=Policy(CFG)
        actions,lp,_,_=p.act([obs]*4)
        _,replayed,_,entropy,_=p.terms([obs]*4,actions=actions)
        torch.testing.assert_close(replayed,torch.tensor(lp),atol=1e-5,rtol=1e-5)
        self.assertTrue(torch.isfinite(entropy).all())
        (-replayed.mean()-.001*entropy.mean()).backward()
        for module in (p.q_body,p.q_film,p.heads[0].q_mu,p.heads[0].q_kappa,p.heads[0].node):
            self.assertGreater(sum(float(v.grad.abs().sum()) for v in module.parameters() if v.grad is not None),0.)
        self.assertGreater(float(p.heads[0].q_state_mu.grad.abs().sum()),0.)

    def test_new_state_center_bounds_and_initial_order(self):
        _,obs=observation(50);p=Policy(CFG);centers=[]
        with torch.no_grad():
            p.heads[0].q_mu.weight.zero_();p.heads[0].q_mu.bias.zero_()
            for accepted,stall,band in ((True,0.,(.08,.27)),(False,0.,(.18,.42)),(False,.3,(.27,.50))):
                o=copy.deepcopy(obs);o['context'][15]=float(accepted);o['context'][14]=stall
                h,nodes,x,raw=p.encode([o]);a,b,mode=p.q_beta_params(h,nodes,x,raw,[0])
                center=float(.01+.59*mode);centers.append(center)
                self.assertGreaterEqual(center,band[0]);self.assertLessEqual(center,band[1])
            self.assertLess(centers[0],centers[1]);self.assertLess(centers[1],centers[2])
            # Arbitrary learned values: independent bounds remain valid.
            for param in (p.heads[0].q_mu.weight,p.heads[0].q_mu.bias,p.heads[0].q_regime_mu,p.heads[0].q_state_mu):param.fill_(1000.)
            for accepted,stall,band in ((True,0.,(.08,.27)),(False,0.,(.18,.42)),(False,.3,(.27,.50))):
                o=copy.deepcopy(obs);o['context'][15]=float(accepted);o['context'][14]=stall
                h,nodes,x,raw=p.encode([o]);_,_,mode=p.q_beta_params(h,nodes,x,raw,[0])
                center=float(.01+.59*mode)
                self.assertGreaterEqual(center,band[0]);self.assertLessEqual(center,band[1])
        print('INITIAL_RHO_CENTERS',centers,flush=True)

    def test_actual_regret_fields_lower_center_with_constant_base(self):
        _,obs=observation(50);p=Policy(CFG)
        with torch.no_grad():
            p.heads[0].q_mu.weight.zero_();p.heads[0].q_mu.bias.zero_()
            centers=[]
            for idx in (21,22,23,24):
                o=copy.deepcopy(obs);o['context'][21:25]=0.;o['context'][idx]=1.
                h,nodes,x,raw=p.encode([o]);_,_,mode=p.q_beta_params(h,nodes,x,raw,[0]);centers.append(float(mode))
        self.assertLess(centers[1],centers[0]);self.assertLess(centers[2],centers[0])
        self.assertAlmostEqual(centers[0],centers[3],places=7)

    def test_overlapping_bounds_do_not_prove_arbitrary_parameter_monotonicity(self):
        _,obs=observation(50);p=Policy(CFG);centers=[]
        with torch.no_grad():
            p.heads[0].q_mu.weight.zero_();p.heads[0].q_mu.bias.zero_()
            p.heads[0].q_regime_mu.copy_(torch.tensor([-100.,100.,.85]))
            p.heads[0].q_state_mu.zero_();p.heads[0].q_state_mu[2]=100.
            for accepted in (True,False):
                o=copy.deepcopy(obs);o['context'][15]=float(accepted);o['context'][14]=0.
                h,nodes,x,raw=p.encode([o]);_,_,mode=p.q_beta_params(h,nodes,x,raw,[0]);centers.append(float(.01+.59*mode))
        self.assertGreater(centers[0],centers[1])
        print('MONOTONICITY_COUNTEREXAMPLE',json.dumps(dict(refine=centers[0],initial=centers[1])),flush=True)

    def test_integer_mapping_and_nearest_bin(self):
        u=torch.tensor([0.,.1,.5,.9,1.])
        for n in (1,12,50,150,250):
            q=integer_q(u,n)
            self.assertTrue(bool(((q>=1)&(q<=n)).all()))
            self.assertEqual(q[-1].item(),max(1,round(.6*n)))
            for value in q.tolist():self.assertTrue(0<=nearest_q_slot(value,n)<10)
        with self.assertRaises(ValueError):nearest_q_slot(0,50)

    def test_many_continuous_integers_and_environment_accepts_non_grid_q(self):
        for n in (150,250):
            env,obs=observation(n);p=Policy(CFG)
            # One encoding, 200 draws: test distribution support without search.
            with torch.no_grad():
                h,nodes,x,raw=p.encode([obs]);a,b,_=p.q_beta_params(h,nodes,x,raw,[0])
                q=integer_q(torch.distributions.Beta(a,b).sample((200,)).flatten(),n)
            unique=set(q.tolist());self.assertGreater(len(unique),15)
            outside=next(k for k in unique if k not in env.q_values)
            result=env.step(dict(q=outside,removed=list(range(1,outside+1))),'greedy',2,time.perf_counter()+5)
            self.assertIn('feasible',result)
            with self.assertRaises(ValueError):env.step(dict(q=True,removed=[1]),'greedy',2,time.perf_counter()+1)

    def test_selected_logprob_equals_explicit_pl_prefix(self):
        torch.manual_seed(2);logits=torch.randn(3,15);orders=torch.stack([torch.randperm(15) for _ in range(3)]);qs=torch.tensor([1,7,14])
        actual=selected_logprob(logits,orders,qs)
        for i in range(3):
            used=torch.zeros(15,dtype=torch.bool);expected=0.
            for c in orders[i,:qs[i]]:
                dist=torch.distributions.Categorical(logits=logits[i].masked_fill(used,-torch.inf))
                expected=expected+dist.log_prob(c);used=used.clone();used[c]=True
            torch.testing.assert_close(actual[i],expected/qs[i],atol=2e-6,rtol=2e-6)

    def test_route_edge_shape_invariant_and_cache_reset(self):
        from solver.route_ops import build_edge_index_routes
        env,obs=observation(12)
        for routes in ([[c] for c in range(1,13)],[[1,2,3,4],[5,6,7,8],[9,10,11,12]]):
            self.assertEqual(build_edge_index_routes([routes])[0].shape,(2,12))
        env.steps=1;env.last_accepted=False
        with patch.object(env,'node_features',side_effect=AssertionError('Rejected unchanged state recomputed')):
            cached=env.observation(torch.zeros(1,128),'greedy',Adaptive(10),5.,5.,.05,.05)
        env._observation_cache=None
        recomputed=env.observation(torch.zeros(1,128),'greedy',Adaptive(10),5.,5.,.05,.05)
        for key in ('x','edge_index_r0','edge_index_r1'):torch.testing.assert_close(cached[key],recomputed[key],atol=0,rtol=0)
        env.routes=[r[::-1] for r in env.routes]
        with patch.object(env,'node_features',wraps=env.node_features) as call:
            env.observation(torch.zeros(1,128),'greedy',Adaptive(10),5.,5.,.05,.05)
        self.assertEqual(call.call_count,1)

    def test_regime_formula_values_are_recorded_not_assumed(self):
        _,obs=observation(50);p=Policy(CFG);records=[]
        with torch.no_grad():
            p.heads[0].q_mu.weight.zero_();p.heads[0].q_mu.bias.zero_()
            for accepted,stall in ((True,0.),(False,0.),(False,.3)):
                changed=copy.deepcopy(obs);changed['context'][15]=float(accepted);changed['context'][14]=stall
                h,nodes,x,raw=p.encode([changed]);a,b,mode=p.q_beta_params(h,nodes,x,raw,[0])
                records.append(dict(accepted=accepted,stall=stall,rho_mode=float(.01+.59*mode),rho_mean=float(.01+.59*a/(a+b))))
        print('BETA_CENTERS',json.dumps(records),flush=True)
        self.assertLess(records[0]['rho_mode'],records[1]['rho_mode']);self.assertLess(records[1]['rho_mode'],records[2]['rho_mode'])
        indexes={name:i for i,name in enumerate(CONTEXT)}
        self.assertEqual(indexes['repair_is_regret2'],22);self.assertEqual(indexes['repair_is_regret3'],23)


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class GraphTests(unittest.TestCase):
    def test_cuda_ppo_replay_gradient_and_initial_ratio(self):
        from solver.training import update
        _,obs=observation(50);p=Policy(Config()).cuda()
        class NoStep:
            def zero_grad(self,**kwargs):p.zero_grad(**kwargs)
            def step(self):pass
        actions,lp,_,_=p.act([obs]*4)
        episodes=[[dict(task=0,observation=obs,action=action,logprob=old_lp,reward=.001*i,
            terminal=True,truncated=False,diagnostics=dict(timeout=False,repair_seconds=.01,inference_seconds=.001))]
            for i,(action,old_lp) in enumerate(zip(actions,lp))]
        saved=p.actor_state()
        for method in ('ppo','reinforce'):
            metrics=update(p,NoStep(),episodes,method,p.cfg)
            self.assertLess(metrics['initial_ratio_error'],1e-5)
            self.assertTrue(all(torch.isfinite(v.grad).all() for v in p.parameters() if v.grad is not None))
            self.assertGreater(float(p.heads[0].q_kappa.weight.grad.abs().sum()),0.)
            for name,value in saved.items():torch.testing.assert_close(value,p.state_dict()[name],atol=0,rtol=0)

    def test_training_version_change_invalidates_host_weights(self):
        from solver.cuda_actor import runner_for
        _,obs=observation(50);p=Policy(Config()).cuda()
        p.act([obs]);runner=runner_for(p,50,1,False);before=runner.outputs[5].clone()
        captures=runner.capture_count
        with torch.no_grad():p.heads[0].q_mu.bias.add_(.3)
        p.act([obs]);self.assertFalse(torch.equal(before,runner.outputs[5]))
        self.assertEqual(runner.capture_count,captures)
        actions,lp,_,_=p.greedy([obs])
        _,replay,_,_,_=p.terms([obs],actions=actions)
        torch.testing.assert_close(torch.tensor(lp,device='cuda'),replay,atol=1e-5,rtol=1e-5)

    def test_actual_captured_actions_cover_continuous_integer_support(self):
        for n in (150,250):
            _,obs=observation(n);p=Policy(Config()).cuda();seen=set()
            for _ in range(200):
                actions,_,_,_=p.act([obs]);seen.add(actions[0]['q'])
            self.assertGreater(len(seen),15)
            self.assertTrue(seen-set(obs['q_values']))

    def test_capture_beta_parity_and_latency_single_four(self):
        from solver.cuda_actor import _CUDA_GRAPH_RUNNERS,runner_for
        for n in (50,150,250):
            torch.manual_seed(28);_,obs=observation(n);p=Policy(Config()).cuda()
            for batch in (1,4):
                observations=[obs]*batch
                p.act(observations);p.act(observations);torch.cuda.synchronize()
                runner=runner_for(p,n,batch,False)
                captures=runner.capture_count
                # New policy, changed parameters, routes, features/context/history.
                alternate=Policy(Config()).cuda()
                changed=copy.deepcopy(obs);changed['state_'].fill_(.02);changed['x'][1,17]+=.01;changed['context'][15]=1.
                from solver.route_ops import build_edge_index_routes
                routes=[[c] for c in range(1,n+1)]
                changed['edge_index_r0']=torch.as_tensor(build_edge_index_routes([routes])[0],dtype=torch.long)
                changed['edge_index_r1']=torch.as_tensor(build_edge_index_routes([routes],inverse=True)[0],dtype=torch.long)
                actions,lp,_,hidden=alternate.act([changed]*batch)
                with torch.no_grad():
                    h,nodes,x,raw=alternate.encode([changed]*batch)
                    a,b,mode=alternate.q_beta_params(h,nodes,x,raw,[0]*batch)
                    qs=torch.tensor([v['q']/n for v in actions],device='cuda')
                    node=alternate.node_logits_batch(nodes,x,raw,qs,[0]*batch)
                    torch.testing.assert_close(hidden,h.cpu(),atol=1e-5,rtol=1e-5)
                    for actual,expected in zip(runner.outputs[5:8],(a,b,mode)):torch.testing.assert_close(actual,expected,atol=1e-5,rtol=1e-5)
                    torch.testing.assert_close(runner.outputs[8],node,atol=1e-5,rtol=1e-5)
                    _,replay,_,_,_=alternate.terms([changed]*batch,actions=actions)
                    torch.testing.assert_close(torch.tensor(lp,device='cuda'),replay,atol=1e-5,rtol=1e-5)
                self.assertEqual(runner.capture_count,captures)
                # Return to p and include actual caller CPU outputs in timing.
                for _ in range(3):p.act(observations)
                timings=[]
                for _ in range(20):
                    start=time.perf_counter();p.act(observations);torch.cuda.synchronize();timings.append(1000*(time.perf_counter()-start))
                median=statistics.median(timings);print('GRAPH_LATENCY',json.dumps(dict(n=n,batch=batch,median_ms=median,per_action_ms=median/batch,captures=captures)),flush=True)
                if batch==1:self.assertLess(median,2.)


class SurplusTests(unittest.TestCase):
    def test_subtractive_formula_includes_inference_time(self):
        from solver.training import group_advantages
        episodes=[]
        for i in range(4):
            ep=[]
            for j,(accepted,stall) in enumerate(((False,0),(True,0),(False,1))):
                ep.append(dict(task=0,reward=.01*(i+1),terminal=j==2,truncated=False,
                    observation=dict(search_state=dict(last_accepted=accepted,stagnation=stall)),
                    diagnostics=dict(repair_seconds=.01,inference_seconds=.001 if i%2==0 else .10)))
            episodes.append(ep)
        _,adv,weights=group_advantages(episodes)
        total_r=sum(t['reward'] for ep in episodes for t in ep);total_dt=sum(.01+t['diagnostics']['inference_seconds']+.001 for ep in episodes for t in ep)
        ref={reg:[] for reg in REGIMES}
        for ep in episodes:
            future=0.
            for t in reversed(ep):
                reg=regime_name(**t['observation']['search_state']);dt=.01+t['diagnostics']['inference_seconds']+.001
                value=t['reward']+(.85 if reg=='refine' else .5)*future-(1 if reg=='refine' else .25)*(total_r/total_dt)*dt
                ref[reg].append(value);future=t['reward']+.85*future
        for reg in REGIMES:
            values=np.array(ref[reg]);expected=(values-values.mean())/(values.std()+1e-8)
            actual=[]
            for ep in episodes:
                for k in reversed(range(len(ep))):
                    if regime_name(**ep[k]['observation']['search_state'])==reg:actual.append(adv[id(ep)][k])
            np.testing.assert_allclose(actual,expected)
        self.assertAlmostEqual(sum(float(weights[id(ep)].sum()) for ep in episodes),1.)


if __name__=='__main__':unittest.main(verbosity=2)
