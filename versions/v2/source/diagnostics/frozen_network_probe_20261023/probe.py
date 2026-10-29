"""Bounded frozen diagnostic; never calls an optimizer or starts a campaign."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import copy
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.distributions import Categorical, Beta
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
ROOT = Path(__file__).resolve().parent
SOURCE = REPO/'runs_emnh/20261020_gpu_v2_seed20261020'
from moea_emnh_v2.config import Config, CONTEXT, directions, search_budget
from moea_emnh_v2.runner import setup, binding
from mo_dpr.data import fingerprint
from moea_emnh_v2.model import Policy
from moea_emnh_v2.environment import Environment, Adaptive

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def save(name, obj):
    (ROOT/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def tv(a, b):
    return float(np.abs(np.asarray(a)-np.asarray(b)).sum()/2)

@torch.no_grad()
def distributions(model, obs, q):
    _, nodes, context, glob = model.encode_batch([obs])
    head = model.heads[0]
    qp = torch.softmax(head.q(model.q_body(glob))[0, :len(obs['q_values'])], -1)
    qf = nodes.new_tensor([[q/nodes.shape[1], q/max(obs['q_values'])]])
    loc = torch.tanh(nodes + torch.tanh(model.location(torch.cat([context, qf], -1)))[:, None])
    logits = head.anchor(loc[0]).squeeze(-1)
    alpha = F.elu(head.alpha(loc[0]).squeeze(-1))+2
    beta = F.elu(head.beta(loc[0]).squeeze(-1).abs())+2
    return qp, logits, alpha, beta

@torch.no_grad()
def conditional_action(model, obs, q, seed):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    qp, logits, alpha, beta = distributions(model, obs, q)
    used = torch.zeros_like(logits, dtype=torch.bool)
    act = dict(q=q, q_index=list(obs['q_values']).index(q), anchors=[], coefficients=[])
    lp = torch.log(qp[act['q_index']])
    for _ in range(min(model.cfg.anchors, q)):
        dist = Categorical(logits=logits.masked_fill(used, -torch.inf))
        ix = dist.sample(); bd = Beta(alpha[ix], beta[ix]); coef = bd.sample()
        lp += dist.log_prob(ix)+bd.log_prob(coef)
        act['anchors'].append(int(ix)+1); act['coefficients'].append(float(coef))
        used[ix] = True
    # Original implementation evaluates the exact same conditional action.
    _, orig_lp, _, _, _ = model.terms([obs], actions=[act])
    assert torch.allclose(lp, orig_lp[0], atol=2e-5, rtol=1e-5), (lp,orig_lp)
    return act

def feature_means(env, removed):
    x = env.node_features()
    detours = np.zeros(env.instance.n_customers+1)
    dm = env.instance.distance_matrix
    for route in env.routes:
        rr = [0]+route+[0]
        for i in range(1,len(rr)-1):
            a,c,b = rr[i-1:i+2]; detours[c] = dm[a,c]+dm[c,b]-dm[a,b]
    ids = np.asarray(removed)
    return dict(tardiness=float(x[ids,13].mean()), hard_slack=float(x[ids,14].mean()),
                detour=float(detours[ids].mean()))

def main():
    started = time.perf_counter()
    cfg = Config(**read(SOURCE/'config.json')); pool, split = setup(cfg)
    assert fingerprint(binding(cfg, split)) == cfg.binding_id, 'Source contract changed'
    files = [SOURCE/'ppo/meta_complete.pt']+[SOURCE/f'ppo/directions/direction_{k:02d}.pt' for k in (0,1,4,15)]
    before = {str(p):digest(p) for p in files}
    torch.manual_seed(cfg.seed); torch.cuda.manual_seed_all(cfg.seed)
    models = {'initial_seed_reconstruction': Policy(cfg).cuda().eval()}
    ck = torch.load(files[0],map_location='cpu',weights_only=False)
    m=Policy(cfg).cuda(); m.load_state_dict(ck['meta']); models['meta']=m.eval()
    for k in (0,1,4,15):
        p=SOURCE/f'ppo/directions/direction_{k:02d}.pt'
        payload=torch.load(p,map_location='cpu',weights_only=False)
        assert payload['config']['binding_id']==cfg.binding_id
        m=Policy(cfg).cuda();m.load_state_dict(payload['model']);models[f'deployed_{k}']=m.eval()
    states={}
    wanted=[f'hemtv2_train_{n}_{c}_4' for n in (50,150,250) for c in ('C','R')]
    for p in sorted((SOURCE/'ppo/rollouts').glob('meta*.json')):
        for prov in read(p)['provenance']:
            iid=prov['iid']
            if iid in wanted and iid not in states:
                states[iid]=dict(iid=iid,routes=prov['source_routes'],source_file=str(p),
                                 source_direction=prov['direction'])
    assert set(states)==set(wanted)
    save('plan.json',dict(scope='Frozen FIT-state diagnostic, not deployment result or new training',
        states=list(states.values()),models=list(models),binding_id=cfg.binding_id,
        histories='Controlled zero-hidden/new-session baseline; context-only stagnation perturbations are synthetic probes',
        preferences=[0,1,4,15],repair_panel='C_4 at each scale, q closest to 5% and 40%; all four original repairers; 2 paired seeds; trained-center/initial/random/worst-distance',
        max_repair_calls=192,repair_wall_cap_seconds=600,per_call_cap='min(10, original search budget /16)',
        sampling='Conditional q diagnostic only; original stochastic anchor/Beta distributions and exact removal function',
        initial='Reconstruct original meta initialization by original seed and constructor; not recovered initial checkpoint',
        source_hashes=before))
    observations={}; envs={}; sensitivity=[]; deletion_features=[]
    for iid,st in states.items():
        env=Environment(pool,iid,cfg,directions()[15],st['routes']);envs[iid]=env
        n=env.instance.n_customers; budget=search_budget(n);local=min(10.,budget/16)
        obs=env.observation(torch.zeros(1,cfg.hidden_dim),'greedy',Adaptive(len(env.q_values)),budget*.5,budget,local,local)
        observations[iid]=obs
        q=min(env.q_values,key=lambda q:abs(q/n-.05))
        variants=[]
        for k in (0,1,4,15):
            o=copy.deepcopy(obs);o['context'][:5]=torch.tensor(directions()[k],dtype=torch.float32)
            variants.append((f'preference_{k}',o))
        for used in (.1,.5,.9):
            o=copy.deepcopy(obs);o['context'][10]=(1-used)*budget/300;o['context'][11]=used
            variants.append((f'global_used_{used}',o))
        for fraction in (.1,.5,1.):
            o=copy.deepcopy(obs);o['context'][12]=local*fraction/300;o['context'][13]=fraction
            variants.append((f'local_remaining_{fraction}',o))
        for ratio in (0.,.5,1.):
            o=copy.deepcopy(obs);o['context'][14]=ratio
            variants.append((f'stagnation_ratio_{ratio}',o))
        for name,m in models.items():
            qp,al,aa,bb=distributions(m,obs,q);baseq=qp.cpu().numpy();basea=al.softmax(-1).cpu().numpy()
            for label,o in variants:
                qp,al,aa,bb=distributions(m,o,q);pq=qp.cpu().numpy();pa=al.softmax(-1).cpu().numpy()
                sensitivity.append(dict(iid=iid,model=name,variant=label,q=q,q_prob=pq.tolist(),
                    q_entropy=float(Categorical(probs=qp).entropy()),q_entropy_max=float(np.log(len(pq))),
                    expected_q_fraction=float(np.dot(pq,np.asarray(obs['q_values'])/n)),
                    q_tv_to_center=tv(pq,baseq),anchor_tv_to_center=tv(pa,basea),
                    anchor_normalized_entropy=float(Categorical(logits=al).entropy()/np.log(n)),
                    coefficient_mean=float((aa/(aa+bb)).mean())))
        # Same trained model, exact same route/history/q, paired sampling randomness.
        for k in (0,1,4,15):
            o=copy.deepcopy(obs);o['context'][:5]=torch.tensor(directions()[k],dtype=torch.float32)
            for rep in range(24):
                act=conditional_action(models['deployed_15'],o,q,73000+rep)
                deleted=env.deletion(act)
                deletion_features.append(dict(iid=iid,preference=k,rep=rep,q=q,removed=deleted,**feature_means(env,deleted)))
        save('sensitivity.json',sensitivity);save('deletion_features.json',deletion_features)
        print('sensitivity_done',iid,round(time.perf_counter()-started,2),flush=True)
    repair_started=time.perf_counter();hard_end=repair_started+600
    arms=['trained_center','initial','random','worst_distance'];repairs=['greedy','regret2','regret3','noisy_greedy']
    cells=[(n,ratio,r,rep) for rep in range(2) for ratio in (.05,.4) for r in repairs for n in (50,150,250)]
    records=[];cells_done=0
    for ci,(n,ratio,r,rep) in enumerate(cells):
        cap=min(10.,search_budget(n)/16)
        if time.perf_counter()+4*(cap+.5)>hard_end:break
        iid=f'hemtv2_train_{n}_C_4';base=envs[iid];obs=copy.deepcopy(observations[iid])
        for j,rr in enumerate(repairs):obs['context'][CONTEXT.index(f'repair_is_{rr}')]=float(rr==r)
        q=min(base.q_values,key=lambda q:abs(q/n-ratio));seed=83000+rep
        selected={};actions={}
        for arm,name in [('trained_center','deployed_15'),('initial','initial_seed_reconstruction')]:
            action=conditional_action(models[name],obs,q,91000+ci)
            actions[arm]=action;selected[arm]=base.deletion(action)
        rng=np.random.default_rng(91000+ci)
        selected['random']=[int(c) for c in rng.choice(np.arange(1,n+1),q,replace=False)]
        costs={};dm=base.instance.distance_matrix
        for route in base.routes:
            rr=[0]+route+[0]
            for j in range(1,len(rr)-1):
                a,c,b=rr[j-1:j+2];costs[c]=dm[a,c]+dm[c,b]-dm[a,b]
        selected['worst_distance']=sorted(costs,key=lambda c:(-costs[c],c))[:q]
        order=list(np.random.default_rng(ci+77).permutation(arms))
        for arm in order:
            env=Environment(pool,iid,cfg,directions()[15],states[iid]['routes'])
            t=time.perf_counter()
            result=env.step(dict(q=q),r,seed,min(hard_end,t+cap),removal_override=selected[arm])
            assert np.isfinite(result['reward']) and len(set(selected[arm]))==q
            records.append(dict(cell=ci,iid=iid,q=q,q_fraction=q/n,repair=r,rep=rep,arm=arm,
                seed=seed,cap=cap,removed=selected[arm],action=actions.get(arm),
                reward=result['reward'],accepted=result['accepted'],timeout=result['timeout'],
                failed=result['failed'],reason=result['reason'],repair_seconds=result['repair_seconds'],
                call_seconds=time.perf_counter()-t,before=result['before_objectives'],after=result['candidate_objectives']))
        cells_done+=1;save('repair_records.json',records)
        if cells_done==3:save('pilot.json',dict(calls=len(records),seconds=time.perf_counter()-repair_started,
            note='First 12 calls counted inside 600-second total; proceed with unchanged 192-call cap.'))
        print('repair_cell',cells_done,'of',len(cells),n,ratio,r,rep,'elapsed',round(time.perf_counter()-repair_started,2),flush=True)
    after={str(p):digest(p) for p in files};assert before==after
    for name,m in models.items():assert all(p.grad is None for p in m.parameters())
    save('verification.json',dict(passed=True,source_checkpoint_hashes_unchanged=True,
        source_binding_checked_at_start=True,no_optimizer_created=True,no_parameter_gradients=True,
        conditional_likelihood_checked=True,sensitivity_states=len(states),sensitivity_rows=len(sensitivity),
        deletion_samples=len(deletion_features),repair_calls=len(records),complete_repair_cells=cells_done,
        planned_repair_cells=len(cells),repair_wall_seconds=time.perf_counter()-repair_started,
        total_seconds=time.perf_counter()-started,source_hashes=after))
    print('COMPLETE',len(records),round(time.perf_counter()-started,2),flush=True)

if __name__=='__main__':main()
