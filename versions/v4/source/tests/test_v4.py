"""Synthetic regression checks only; no FIT/development search or training run."""
import copy,importlib.util,json,sys,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from solver.config import Config,RATIOS,q_values,directions
from solver.data import project_api
CFG=Config(device='cpu',local_seconds=.05,repair_seconds=.05,solve_seconds_by_n={50:5.,150:5.,250:5.})
API=project_api(CFG.project_root)
from src.core.protocol import VRPTWInstance,Route,Solution
from solver.repair import repair,_score
from solver.environment import Environment,Adaptive
from solver.model import Policy
from solver.training import balanced_fit_ids,group_advantages,collect_closed
from solver.outer import Search


def fixture(n=12,seed=7,nonmetric=False):
    rng=np.random.default_rng(seed);coords=rng.uniform(0,10,(n,2));xy=np.vstack(([0,0],coords))
    distance=np.linalg.norm(xy[:,None]-xy[None,:],axis=-1)
    if nonmetric:
        distance=rng.uniform(.1,10,(n+1,n+1));np.fill_diagonal(distance,0)
    inst=VRPTWInstance('synthetic',n,5.,100.,(0.,1000.),coords,np.ones(n),
        np.column_stack((rng.uniform(0,10,n),rng.uniform(30,80,n))),np.ones(n),distance,distance.copy())
    routes=[list(range(i+1,min(i+4,n)+1)) for i in range(0,n,4)]
    return inst,routes

class Pool:
    def __init__(self,n=12):
        self.inst,self.routes=fixture(n);self.api=API
    def load(self,iid):return self.inst,dict(source_solution=self.routes),dict(scale=[10.,100.,1000.,1000.,1000.],offset=[0.]*5,reference=[2.]*5),q_values(self.inst.n_customers)

def observation(n=12):
    env=Environment(Pool(n),'synthetic',CFG,np.full(5,.2))
    return env,env.observation(torch.zeros(1,128),'greedy',Adaptive(10),5.,5.,.05,.05)

class RepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source=ROOT.parents[1]/'moea_emnh_v2/repair.py'
        spec=importlib.util.spec_from_file_location('reference_repair',source)
        cls.reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.reference)
    def test_four_methods_match_original(self):
        for nonmetric in (False,True):
            for seed in range(3):
                inst,routes=fixture(seed=seed,nonmetric=nonmetric)
                for q in (1,4,8,12):
                    removal=np.random.default_rng(seed).choice(np.arange(1,13),q,replace=False).tolist()
                    for method in ('greedy','regret2','regret3','noisy_greedy'):
                        pref=np.random.default_rng(seed+11).dirichlet(np.ones(5))
                        args=(inst,routes,removal,method,pref,[10.,100.,1000.,1000.,1000.],42)
                        old=self.reference.repair(*args);new=repair(*args)
                        self.assertEqual(old['reason'],new['reason'])
                        self.assertEqual(old['routes'],new['routes'])
                        self.assertEqual(old['trace'],new['trace'])
                        self.assertEqual(old['objectives'],new['objectives'])
    def test_cache_reuses_unchanged_routes(self):
        inst,routes=fixture(20)
        args=(inst,routes,list(range(1,17)),'regret3',np.full(5,.2),[10.,100.,1000.,1000.,1000.],17)
        old=self.reference.repair(*args);new=repair(*args)
        self.assertEqual(new['reason'],'ok')
        self.assertGreater(new['counters']['route_blocks_reused'],0)
        self.assertLess(new['counters']['changed_route_evaluations'],old['counters']['changed_route_evaluations'])
        self.assertEqual(new['counters']['enumeration_calls'],0)
        self.assertEqual(new['counters']['vectorized_score_batches'],16)
    def test_only_changed_route_rebuilt(self):
        import solver.repair as module
        from src.pilot.engine import apply_action
        with patch.object(module,'apply_action',wraps=apply_action) as apply:
            inst,routes=fixture(20)
            out=repair(inst,routes,list(range(1,15)),'greedy',np.full(5,.2),[10.,100.,1000.,1000.,1000.],1)
        self.assertEqual(apply.call_count,14)
        # Initial table + at most one changed-route block per remaining customer;
        # singleton blocks persist even when an actual new route is appended.
        initial_routes=sum(bool([c for c in r if c not in range(1,15)]) for r in routes)
        limit=14*(initial_routes+1)+sum(range(14))
        self.assertLessEqual(out['counters']['route_blocks_built'],limit)
    def test_timeout_and_legality(self):
        inst,routes=fixture()
        args=(inst,routes,[1],'greedy',np.full(5,.2),[10.,100.,1000.,1000.,1000.],1)
        result=repair(*args,deadline=time.perf_counter()-1)
        self.assertTrue(result['timeout']);self.assertIsNone(result['routes'])
        self.assertEqual(repair(inst,routes,[1,1],*args[3:])['reason'],'invalid_removal_set')
    def test_vector_scalar_score_identity(self):
        rng=np.random.default_rng(11);objectives=rng.uniform(0,100,(37,5));w=rng.dirichlet(np.ones(5));mins=np.arange(5);scale=np.arange(1,6)
        weighted=w*((objectives-mins)/scale)
        np.testing.assert_array_equal(weighted.max(1)+.05*weighted.sum(1),[_score(v,w,mins,scale) for v in objectives])

class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19);self.env,self.obs=observation();self.policy=Policy(CFG)
    def test_features_twenty_and_physical_proxies(self):
        x=self.obs['x'].numpy();self.assertEqual(x.shape,(13,20))
        for route in self.env.routes:
            for c in route:self.assertAlmostEqual(x[c,18],len(route)/12,places=6)
        self.assertTrue(np.isfinite(x).all());self.assertGreater(x[1:,17].std(),0)
    def test_ten_q_slots_cover_full_range(self):
        self.assertEqual(len(RATIOS),10);self.assertEqual(RATIOS[0],.01);self.assertEqual(RATIOS[-1],1)
        for n in (12,50,150,250):
            self.assertEqual(len(q_values(n)),10);self.assertEqual(q_values(n)[-1],n)
            self.assertTrue(all(1<=q<=n for q in q_values(n)))
    def test_every_q_replays_joint_likelihood(self):
        for qi,q in enumerate(self.obs['q_values']):
            action=dict(q_index=qi,q=q,removed=list(range(1,q+1)))
            out,lp,_,entropy,_=self.policy.terms([self.obs],actions=[action])
            self.assertEqual(out,[action]);self.assertTrue(torch.isfinite(lp).all());self.assertTrue(torch.isfinite(entropy).all())
        actions,lp,_,_=self.policy.act([self.obs]*3)
        _,replayed,_,_,_=self.policy.terms([self.obs]*3,actions=actions)
        np.testing.assert_allclose(replayed.detach().numpy(),lp,atol=1e-5)
    def test_joint_q_and_node_gradients(self):
        action=dict(q_index=3,q=self.obs['q_values'][3],removed=[1,2])
        _,lp,_,_,_=self.policy.terms([self.obs],actions=[action]);(-lp.sum()).backward()
        for module in (self.policy.q_body,self.policy.heads[0].q,self.policy.heads[0].node,self.policy.film):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None),0)
        self.assertGreater(float(self.policy.heads[0].physical_bilinear.grad.abs().sum()),0)
    def test_full_q_disables_all_position_gradients(self):
        action=dict(q_index=9,q=12,removed=list(range(1,13)))
        with patch.object(self.policy,'node_logits',side_effect=AssertionError('position head called for full removal')):
            _,lp,_,ent,_=self.policy.terms([self.obs],actions=[action])
            hidden,nodes,x,raw=self.policy.encode([self.obs])
            d=torch.distributions.Categorical(logits=self.policy.q_logits(nodes,raw,[0]))
            torch.testing.assert_close(lp,d.log_prob(torch.tensor([9])));torch.testing.assert_close(ent,5*d.entropy())
            (-lp.sum()).backward()
        for p in list(self.policy.film.parameters())+list(self.policy.heads[0].node.parameters())+[self.policy.heads[0].physical_bilinear]:self.assertIsNone(p.grad)
        self.assertGreater(float(self.policy.heads[0].q.weight.grad.abs().sum()),0)
    def test_q_explicit_preference_stage_stall_path(self):
        _,nodes,_,raw=self.policy.encode([self.obs]);base=self.policy.q_logits(nodes,raw,[0])
        for index in (0,11,13,14):
            changed=raw.clone();changed[:,index]+=.5
            self.assertFalse(torch.equal(base,self.policy.q_logits(nodes,changed,[0])))
    def test_residual_cold_start_node_variation(self):
        _,nodes,_,_=self.policy.encode([self.obs]);self.assertGreater(float(nodes.detach().std(1).mean()),.001)
    def test_length_normalized_node_score(self):
        action=dict(q_index=5,q=4,removed=[1,2,3,4])
        _,lp,_,_,_=self.policy.terms([self.obs],actions=[action])
        _,nodes,x,raw=self.policy.encode([self.obs])
        qdist=torch.distributions.Categorical(logits=self.policy.q_logits(nodes,raw,[0]))
        logits=self.policy.node_logits(nodes[0],x[0],raw[0,:5],4,12,0,raw[0,14])
        used=torch.zeros(12,dtype=torch.bool);total=logits.new_tensor(0.)
        for c in action['removed']:
            dist=torch.distributions.Categorical(logits=logits.masked_fill(used,-torch.inf))
            total=total+dist.log_prob(torch.tensor(c-1));used[c-1]=True
        torch.testing.assert_close(lp[0],qdist.log_prob(torch.tensor([5]))[0]+total/4)
    def test_vectorized_likelihood_entropy_and_gradients_match_loop(self):
        for qi in (0,3,5,8):
            q=self.obs['q_values'][qi]
            action=dict(q_index=qi,q=q,removed=list(range(1,q+1)))
            _,lp,_,entropy,_=self.policy.terms([self.obs],actions=[action])
            actual_grads=torch.autograd.grad(lp.sum()+.001*entropy.sum(),tuple(self.policy.parameters()),allow_unused=True)
            _,nodes,x,raw=self.policy.encode([self.obs])
            qdist=torch.distributions.Categorical(logits=self.policy.q_logits(nodes,raw,[0]))
            logits=self.policy.node_logits(nodes[0],x[0],raw[0,:5],q,12,0,raw[0,14])
            used=torch.zeros(12,dtype=torch.bool)
            reference=qdist.log_prob(torch.tensor([qi]))[0];ent=5*qdist.entropy()[0]
            for c in action['removed']:
                dist=torch.distributions.Categorical(logits=logits.masked_fill(used,-torch.inf))
                reference=reference+dist.log_prob(torch.tensor(c-1))/q;ent=ent+dist.entropy()/q
                used=used.clone();used[c-1]=True
            torch.testing.assert_close(lp[0],reference,atol=2e-6,rtol=2e-6)
            torch.testing.assert_close(entropy[0],ent,atol=2e-6,rtol=2e-6)
            expected_grads=torch.autograd.grad(reference+.001*ent,tuple(self.policy.parameters()),allow_unused=True)
            for a,b in zip(actual_grads,expected_grads):
                if a is None or b is None:self.assertIs(a,b)
                else:torch.testing.assert_close(a,b,atol=2e-5,rtol=2e-4)
    def test_gumbel_topk_ordered_sampling_distribution(self):
        torch.manual_seed(1701);weights=torch.tensor([1.,2.,4.]);logits=weights.log()
        g=-torch.empty(30000,3).exponential_().log()
        selected=(logits+g).topk(2,dim=-1).indices
        self.assertTrue(bool((selected[:,0]!=selected[:,1]).all()))
        for a in range(3):
            for b in range(3):
                if a==b:continue
                actual=float(((selected[:,0]==a)&(selected[:,1]==b)).float().mean())
                expected=float(weights[a]/weights.sum()*weights[b]/(weights.sum()-weights[a]))
                self.assertAlmostEqual(actual,expected,delta=.012)
    def test_physical_prior_signs_and_stall_film(self):
        diag=self.policy.heads[0].physical_bilinear.diag().detach()
        self.assertEqual(diag.sign().tolist(),[-1.,1.,-1.,1.,1.])
        _,nodes,x,raw=self.policy.encode([self.obs])
        a=self.policy.node_logits(nodes[0],x[0],raw[0,:5],3,12,0,raw.new_tensor(0.))
        b=self.policy.node_logits(nodes[0],x[0],raw[0,:5],3,12,0,raw.new_tensor(1.))
        self.assertFalse(torch.equal(a,b))
    def test_stall_ten_rejection_ramp(self):
        for count in (0,1,5,10,20):
            self.env.stagnation=count;self.env.steps=max(1,count)
            obs=self.env.observation(torch.zeros(1,128),'greedy',Adaptive(10),5.,5.,.05,.05)
            self.assertAlmostEqual(float(obs['context'][14]),min(1.,count/10.),places=6)
    def test_meta_heads_copy_and_mean(self):
        inner=Policy(CFG,tasks=5);inner.start_inner(self.policy)
        for h in inner.heads:torch.testing.assert_close(h.q.weight,self.policy.heads[0].q.weight)
        with torch.no_grad():
            for i,h in enumerate(inner.heads):h.q.bias.fill_(i)
        self.policy.meta_update(inner,1.)
        torch.testing.assert_close(self.policy.heads[0].q.bias,torch.full((10,),2.))
        deployed=inner.deploy(4);torch.testing.assert_close(deployed.heads[0].q.bias,torch.full((10,),4.))

class TrainingTests(unittest.TestCase):
    def test_fit_round_robin(self):
        split=json.loads((ROOT/'split.json').read_text());order=balanced_fit_ids(split['fit_ids'])
        self.assertEqual(set(order),set(split['fit_ids']));self.assertEqual(len(order),45)
        for j in range(0,45,9):self.assertEqual(len({tuple(i.split('_')[2:4]) for i in order[j:j+9]}),9)
    def test_normalized_group_advantages(self):
        episodes=[[dict(task=0,reward=r,terminal=True,truncated=False,diagnostics=dict(repair_seconds=0.))] for r in (0.,.0001,.0002,.0003)]
        _,advantages=group_advantages(episodes);values=np.asarray(list(advantages.values()))
        self.assertAlmostEqual(values.mean(),0,places=10);self.assertGreater(values.std(),.999)
        equal=[[dict(task=0,reward=0.,terminal=True,truncated=False,diagnostics=dict(repair_seconds=0.))] for _ in range(4)]
        self.assertTrue(all(v==0 for v in group_advantages(equal)[1].values()))
    def test_discounted_step_credit(self):
        ep=[dict(task=0,reward=r,terminal=(i==3),truncated=False,diagnostics=dict(repair_seconds=0.)) for i,r in enumerate((0.,1.,0.,0.))]
        episodes=[copy.deepcopy(ep) for _ in range(4)]
        _,adv=group_advantages(episodes)
        values=adv[id(episodes[0])]
        self.assertGreater(values[1],values[0]);self.assertGreater(values[0],values[2])
        raw=np.asarray([.3,1.,0.,0.]);expected=(raw-raw.mean())/(raw.std()+1e-8)
        np.testing.assert_allclose(values,expected)
    def test_repair_time_efficiency_changes_step_credit(self):
        episodes=[[dict(task=0,reward=1.,terminal=True,truncated=False,diagnostics=dict(repair_seconds=seconds))] for seconds in (0.,.1,.5,1.)]
        _,advantages=group_advantages(episodes)
        actual=np.concatenate([advantages[id(ep)] for ep in episodes])
        signals=1/(1+2*np.asarray([0.,.1,.5,1.]))
        np.testing.assert_allclose(actual,(signals-signals.mean())/(signals.std()+1e-8))
        self.assertTrue(bool((np.diff(actual)<0).all()))
    def test_background_does_not_run_local_search(self):
        from solver.training import skip_background
        self.assertTrue(skip_background(5,{0:0},{}))
        self.assertTrue(skip_background(0,{0:0},{0:[]}))
        self.assertFalse(skip_background(0,{0:0},{}))
    def test_progress_log_ignores_broken_stdout(self):
        import tempfile
        from full_validate import log_event
        class Broken:
            def write(self,*a):raise OSError(22,'Invalid argument')
            def flush(self):raise OSError(22,'Invalid argument')
        with tempfile.TemporaryDirectory() as d,patch('sys.stdout',Broken()):
            log_event(d,dict(phase='meta',updates=1))
            result=json.loads((Path(d)/'worker.log').read_text(encoding='utf-8'))
        self.assertEqual(result['updates'],1)
    def test_atomic_checkpoint_retains_previous_on_failed_write(self):
        import tempfile
        from solver.training import save_checkpoint
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'checkpoint.pt'
            save_checkpoint(path,dict(model=torch.tensor([1.,2.]),optimizer={'step':3},next_iid=4))
            self.assertFalse(path.with_suffix('.pt.tmp').exists())
            with patch('solver.training.torch.save',side_effect=OSError('disk write failed')):
                with self.assertRaises(OSError):save_checkpoint(path,dict(model=torch.tensor([9.])))
            restored=torch.load(path,weights_only=False)
            torch.testing.assert_close(restored['model'],torch.tensor([1.,2.]))
            self.assertEqual(restored['next_iid'],4)
    def test_no_default_experiment_budget(self):
        cfg=Config();cfg.validate()
        with self.assertRaisesRegex(ValueError,'Budget not set'):cfg.validate(require_budget=True)
        self.assertFalse(hasattr(cfg,'fixed_q'))
    def test_ppo_and_reinforce_joint_loss_without_weight_update(self):
        from solver.training import update
        _,obs=observation();policy=Policy(CFG)
        class NoStepOptimizer:
            def zero_grad(self,**kw):policy.zero_grad(**kw)
            def step(self):pass  # Validate objective and gradients, no training.
        episodes=[]
        for i,qi in enumerate((1,3,5,9)):
            q=obs['q_values'][qi];action=dict(q_index=qi,q=q,removed=list(range(1,q+1)))
            with torch.no_grad():_,lp,_,_,_=policy.terms([obs],actions=[action])
            episodes.append([dict(task=0,observation=obs,action=action,logprob=float(lp[0]),reward=i*.0001,
                terminal=True,truncated=False,diagnostics=dict(timeout=False,repair_seconds=0.))])
        before=policy.actor_state()
        for method in ('ppo','reinforce'):
            metric=update(policy,NoStepOptimizer(),episodes,method,CFG)
            self.assertEqual(metric['transitions'],4)
            self.assertGreater(float(policy.heads[0].q.weight.grad.abs().sum()),0)
            self.assertGreater(float(policy.heads[0].node[0].weight.grad.abs().sum()),0)
            for k,v in policy.state_dict().items():torch.testing.assert_close(v,before[k])
    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_forward_backward_without_optimizer(self):
        _,obs=observation();policy=Policy(Config()).cuda()
        action=dict(q_index=3,q=obs['q_values'][3],removed=[1,2])
        _,lp,_,_,_=policy.terms([obs],actions=[action]);(-lp.sum()).backward()
        self.assertEqual(lp.device.type,'cuda')
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None))
    def test_adaptation_parent_quota_shared_cursor_and_global_deadline(self):
        import solver.training as module
        import tempfile
        class Clock:
            now=0.
            def read(self):return self.now
        clock=Clock();calls=[]
        class FakePolicy(torch.nn.Module):
            def __init__(self,*args,**kw):super().__init__();self.w=torch.nn.Parameter(torch.tensor(0.))
            def start_inner(self,parent):self.w.data.copy_(parent.w)
            def meta_update(self,inner,eps):self.w.data.copy_(inner.w)
            def actor_state(self):return dict(w=self.w.detach().clone())
        def collect(pool,split,cfg,policy,meta,weights,state,rng,deadline,notify):
            calls.append((len(weights),id(state),state['next_iid'],deadline))
            state['last_iid']=split['fit_ids'][state['next_iid']%45];state['next_iid']+=1
            clock.now+=.6
            return [[]],True,[]
        def fake_update(policy,*args):
            policy.w.data.add_(1);clock.now+=.1
            return dict(steps=1)
        cfg=Config(device='cpu',meta_seconds=.1,adaptation_seconds=16.,repair_seconds=.1,local_seconds=.1,solve_seconds_by_n={50:1.,150:1.,250:1.})
        split=json.loads((ROOT/'split.json').read_text())
        with tempfile.TemporaryDirectory() as tmp,patch.object(module,'Policy',FakePolicy),patch.object(module,'collect_closed',collect),patch.object(module,'update',fake_update),patch.object(module.time,'perf_counter',clock.read),patch.object(module.torch,'save'),patch.object(module,'save_checkpoint'):
            report=module.train_method(None,split,cfg,'ppo',Path(tmp)/'run')
        adaptation=[v for v in calls if v[0]==1]
        self.assertEqual(len({v[1] for v in adaptation}),1)
        self.assertEqual([v[2] for v in adaptation],list(range(1,1+len(adaptation))))
        self.assertEqual(len({v[3] for v in adaptation}),1)
        self.assertEqual([v['direction'] for v in report['adaptation_initializations']],[15]+list(range(5,15))+list(range(5)))
        self.assertGreater(report['direction_spent'][15],0.)
        self.assertGreaterEqual(report['adaptation_initializations'][1]['parent_updates'],1)
        self.assertTrue(all(v>0 for v in report['direction_updates'].values()))
    def test_search_restore_preserves_population_clock_pending_and_rng(self):
        from solver.outer import search_state,restore_search
        pool=Pool(8);policy=Policy(CFG);_,obs=observation(8);policy.act([obs])
        search=Search(pool,'synthetic',CFG,91,[policy]*16,seconds=3.)
        self.assertTrue(search.step());snapshot=search_state(search)
        time.sleep(.03)
        restored=restore_search(snapshot,pool,CFG,[policy]*16,time.perf_counter()+5)
        frozen=search_state(restored)
        self.assertAlmostEqual(frozen['elapsed'],snapshot['elapsed'],delta=.03)
        self.assertEqual(frozen['population'],snapshot['population'])
        self.assertEqual(frozen['archive'],snapshot['archive'])
        self.assertEqual(frozen['pending'],snapshot['pending'])
        self.assertEqual(frozen['cursor'],snapshot['cursor'])
        np.testing.assert_array_equal(frozen['numpy_state'][1],snapshot['numpy_state'][1])
        self.assertEqual(len(frozen['snapshots']),len(snapshot['snapshots']))
        before=[p.X[0].routes for p in restored.algorithm.pop]
        newrefs=directions();newrefs[:5]=np.roll(np.eye(5),1,axis=0)
        counts=[a.count.copy() for a in restored.adaptive]
        restored.set_preferences(newrefs,[policy]*16)
        self.assertEqual(before,[p.X[0].routes for p in restored.algorithm.pop])
        for c,a in zip(counts,restored.adaptive):np.testing.assert_array_equal(c,a.count)
        self.assertTrue(restored.step())
    def test_restore_without_pending_offspring(self):
        from solver.outer import search_state,restore_search
        pool=Pool(8);policy=Policy(CFG);_,obs=observation(8);policy.act([obs])
        search=Search(pool,'synthetic',CFG,99,[policy]*16,seconds=2.)
        snapshot=search_state(search)
        self.assertIsNone(snapshot['pending'])
        restored=restore_search(snapshot,pool,CFG,[policy]*16,time.perf_counter()+3)
        self.assertTrue(restored.step())
        self.assertEqual(len(restored.algorithm.pop),16)
    def test_continued_collection_three_actual_stages(self):
        from solver.training import collect_closed,q_summary
        cfg=Config(device='cpu',local_seconds=.04,repair_seconds=.04,solve_seconds_by_n={8:.8})
        pool=Pool(8);policy=Policy(cfg,tasks=5);meta=Policy(cfg)
        _,obs=observation(8);policy.act([obs],[0])
        split=json.loads((ROOT/'split.json').read_text());state={};all_eps=[];elapsed=[]
        for phase in ('early','middle','late'):
            episodes,complete,provenance=collect_closed(pool,split,cfg,policy,meta,np.eye(5),state,np.random.default_rng(4),time.perf_counter()+10)
            self.assertTrue(complete);self.assertEqual(len(episodes),20)
            self.assertEqual({p['source_stage'] for p in provenance if p['optimized_group']},{phase})
            self.assertEqual(state['next_iid'],1)
            elapsed.append(state['last_continuation']['elapsed_after']);all_eps+=episodes
            if phase!='late':self.assertIn(state['last_iid'],state['search_pool'])
        self.assertGreater(elapsed[1],elapsed[0]);self.assertGreaterEqual(elapsed[2],.8)
        self.assertEqual(state['completed_cycles'],1);self.assertNotIn('active_iid',state)
        summary=q_summary(all_eps)
        self.assertTrue(all(v['count']>0 for v in summary['by_step_stage'].values()))
        source_counts=[v['count'] for v in summary['by_source_stage'].values()]
        self.assertEqual(sum(source_counts),sum(map(len,all_eps)))
    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_continued_collection_reaches_all_three_stages(self):
        from solver.training import collect_closed
        cfg=Config(device='cuda',local_seconds=.08,repair_seconds=.08,solve_seconds_by_n={8:1.6})
        pool=Pool(8);policy=Policy(cfg,tasks=5).cuda();meta=Policy(cfg).cuda()
        _,obs=observation(8)
        policy.act([obs]*5,list(range(5)));torch.cuda.synchronize()
        split=json.loads((ROOT/'split.json').read_text());state={};source_stages=[]
        for phase in ('early','middle','late'):
            episodes,complete,provenance=collect_closed(pool,split,cfg,policy,meta,np.eye(5),state,np.random.default_rng(7),time.perf_counter()+10)
            self.assertTrue(complete)
            source_stages.extend(p['source_stage'] for p in provenance if p['optimized_group'])
            self.assertEqual({p['source_stage'] for p in provenance if p['optimized_group']},{phase})
        self.assertEqual([source_stages.count(p) for p in ('early','middle','late')],[5,5,5])
        self.assertEqual(state['completed_cycles'],1)
    def test_q_stage_and_direction_statistics_use_actual_observation(self):
        from solver.training import q_summary
        _,obs=observation()
        episodes=[]
        for stage,fraction,direction,q in [('early',.1,0,1),('middle',.5,1,4),('late',.8,15,12)]:
            copied=copy.deepcopy(obs);copied['context'][11]=fraction
            episodes.append([dict(task=0,direction=direction,source_stage=stage,observation=copied,action=dict(q=q,q_index=obs['q_values'].index(q)))])
        summary=q_summary(episodes)
        self.assertEqual([v['count'] for v in summary['by_step_stage'].values()],[1,1,1])
        self.assertEqual(summary['by_direction']['15']['q_ratio_sum'],1.)
        self.assertEqual(summary['by_direction']['1']['q_slot_counts'][5],1)
    def test_five_seed_report_preserves_pairs_and_stage_distributions(self):
        import tempfile
        from make_stage_report import main as make_report
        from solver.training import q_summary
        from run import write_json
        _,obs=observation();episodes=[]
        for phase,fraction in zip(('early','middle','late'),(.1,.5,.8)):
            copied=copy.deepcopy(obs);copied['context'][11]=fraction
            episodes.append([dict(task=0,direction=0,source_stage=phase,observation=copied,action=dict(q=1,q_index=0))])
        iid='hemtv2_train_50_C_9';fit='hemtv2_train_50_C_4'
        h=dict(iid=fit,steps=1,q_breakdown=q_summary(episodes),provenance=[dict(optimized_group=True,source_stage=p) for p in ('early','middle','late')])
        t=dict(history=[h],adaptation_history=[h],meta_seconds=1.,adaptation_seconds=1.,optimized_fit_ids=[fit],
               direction_updates={str(k):1 for k in range(16)},meta_completed_search_cycles=1,adapt_completed_search_cycles=1,
               meta_traversal_only_steps=0,adapt_traversal_only_steps=0)
        summary=dict(wall_seconds=3.,training_seed=20261024,search_seeds=list(range(5)),comparison=[],
                     meta_sampling=dict(batches=1,transitions=3,optimizer_steps=1),
                     adaptation_sampling=dict(batches=1,transitions=3,optimizer_steps=1,optimized_iids=[fit]))
        objectives=dict(improvement_percent=[1.]*5,zero_initial_objectives=[])
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'training').mkdir()
            for seed in range(5):
                for method in ('heuristic','ppo'):
                    row=dict(iid=iid,search_seed=seed,method=method,initial_population_objectives=[[1.]*5],
                        records=[dict(direction=0,global_used_fraction_before=fraction,q=1,action=dict(q_index=0),
                                      timeout=False,ending_reason='continuing',repair_seconds=.01) for fraction in (.1,.5,.8)],
                        inference_seconds=.01,objective_component_minima=objectives)
                    write_json(root/f'{iid}_seed{seed}_{method}.json',row)
                summary['comparison'].append(dict(iid=iid,search_seed=seed,initial_hv=.4,heuristic_hv=.5,ppo_hv=.51,hv_gap_percent=-2.))
            write_json(root/'summary.json',summary);write_json(root/'training/training_report.json',t)
            make_report(root)
            analysis=json.loads((root/'analysis.json').read_text())
            self.assertEqual(analysis['paired_hv']['ppo_wins'],5)
            self.assertEqual(analysis['per_iid'][0]['seeds'],5)
            self.assertEqual(analysis['validation_q']['by_step_stage']['late']['count'],5)
            self.assertTrue(analysis['stage_balance']['meta']['actual_step_25_to_40_passed'])
            self.assertEqual(len((root/'per_seed_results.csv').read_text().splitlines()),6)
            self.assertTrue((root/'REPORT.md').exists())
    def test_synthetic_moead_uses_neural_action_and_population_feedback(self):
        env,obs=observation(8);policy=Policy(CFG)
        policy.act([obs]) # prewarm before synthetic search clock
        search=Search(Pool(8),'synthetic',CFG,51,[policy]*16,seconds=2.)
        self.assertTrue(search.step())
        self.assertEqual(len(search.algorithm.pop),16)
        self.assertEqual(len(search.snapshots),1)
        for p in search.algorithm.pop:self.assertTrue(API.feasible(env.solution(p.X[0].routes),env.instance))
        for r in search.records:self.assertIn('q_index',r['action'])

if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
