"""Bounded generic ALNS captures and explicitly timed counterfactual branches."""
import copy
import json
import math
import time
from pathlib import Path
import numpy as np
from src.core.protocol import Route, Solution
from src.data_generation.sharded import _instance_from_row
from hemt_destroy_features_v1.features_v2 import extract_features, extract_action_features
from .repair import repair, METHODS
from .audit import checked_objectives, digest, exact_hv, audit_branch


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def solution(routes):
    return Solution([Route(list(r)) for r in routes])


def point(instance, routes):
    return dict(routes=copy.deepcopy(routes), objectives=checked_objectives(routes, instance))


def archive_add(archive, candidate):
    v = candidate['objectives']
    if any(all(a <= b for a, b in zip(p['objectives'], v)) for p in archive):
        return list(archive)
    return [p for p in archive if not all(a <= b for a, b in zip(v, p['objectives']))] + [candidate]


def physical_box(instance):
    n = instance.n_customers
    # Inputs only; no realized solver outcomes. Positive floors handle zero dimensions.
    bound = [n, 2*n*float(np.max(instance.distance_matrix)), float(instance.depot_time_window[1]),
             n*max(0., float(np.max(instance.customer_time_windows[:, 0]))), n*instance.max_delay]
    h=max(float(instance.depot_time_window[1]),1.)
    floors=[1.,max(float(np.max(instance.distance_matrix)),1e-9),h,h,h]
    return dict(scale=[max(float(x),float(f)) for x,f in zip(bound,floors)], offset=[0.]*5, reference=[1.1]*5)


def scalar(p, w, box):
    a = np.asarray(p['objectives']) / np.asarray(box['scale']) * w
    return float(a.max() + .05*a.sum())


def deletion(source, q, pattern, instance, seed):
    rng = np.random.default_rng(seed)
    n = instance.n_customers
    if pattern == 'random':
        ranking = rng.permutation(np.arange(1, n+1))
    elif pattern == 'related':
        anchor = int(rng.integers(1, n+1))
        d = instance.distance_matrix
        ranking = sorted(range(1, n+1), key=lambda c: (0 if c == anchor else 1, d[anchor,c]+d[c,anchor], c))
    else:
        raise ValueError(pattern)
    removed = sorted(int(c) for c in ranking[:q]); removed_set = set(removed)
    partial = [[c for c in r if c not in removed_set] for r in source['routes']]
    return dict(partial_routes=[r for r in partial if r], removed_customers=removed)


def vector_features(instance, state, destroy, method):
    f = extract_features(instance, solution(state['source']['routes']), state['feature_context'])
    g = extract_action_features(f, destroy['removed_customers'], METHODS.index(method), instance=instance)['group']
    names = ['global:'+x for x in f['global']['names']] + ['group:'+x for x in g['names']]
    values = list(f['global']['normalized'].ravel()) + list(g['normalized'].ravel())
    for level in ('node', 'route'):
        block = f[level]['normalized']
        valid = np.isfinite(block); count = valid.sum(axis=0)
        avg = np.divide(np.where(valid,block,0.).sum(axis=0),count,out=np.full(block.shape[1],np.nan),where=count>0)
        names += [level+':mean:'+x for x in f[level]['names']]; values += list(avg)
    # No IID, seed, stage name, or future outcome in predictive columns.
    names += ['control:q_fraction','control:is_related'] + ['control:repair:'+x for x in METHODS]
    values += [len(destroy['removed_customers'])/instance.n_customers, float(state['_pattern']=='related')]
    values += [float(method==x) for x in METHODS]
    return names, [float(x) if np.isfinite(x) else None for x in values]


def attempt(instance, current, archive, q, pattern, method, w, box, seed, started, deadline):
    d = deletion(current, q, pattern, instance, seed)
    result = repair(instance, current['routes'], d['removed_customers'], method, w, box['scale'], seed, deadline=deadline)
    outcome = dict(status=result['reason'], feasible=result['feasible'], committed=False,
                   commit_elapsed_seconds=time.perf_counter()-started, repair_id=method, q=q,
                   reason=result['reason'], timeout=result['timeout'], repair_seconds=result['seconds'],
                   counters=result['counters'], timing=result['timing'])
    proposal = archive
    candidate = None
    if result['feasible']:
        candidate = point(instance, result['routes'])
        np.testing.assert_allclose(candidate['objectives'],result['objectives'],rtol=0.,atol=1e-7)
        outcome.update(candidate)
        proposal = archive_add(archive,candidate)
    # Validation and prospective archive update precede the actual admission cutoff.
    next_current=candidate if candidate is not None and scalar(candidate,w,box)<=scalar(current,w,box) else current
    finished = time.perf_counter()
    outcome['commit_elapsed_seconds'] = finished-started
    if candidate is not None and finished <= deadline:
        outcome.update(status='success',committed=True)
        current=next_current
        archive = proposal
    elif candidate is not None:
        outcome['status'] = 'completed_after_deadline'
    return current, archive, outcome


def capture(instance, row, box, cfg, ordinal):
    initial = point(instance,row['source_solution'])
    archive = [initial]; current = initial; rng = np.random.default_rng(cfg['seed']+ordinal)
    n = instance.n_customers; cap=min(15,math.floor(.15*n)); qs=[1,math.ceil(cap/2),cap]
    started=time.perf_counter(); history=[]; states=[]; iteration=0; stagnation=0; last_acceptance=0
    for stage,target in enumerate(cfg['capture_seconds']):
        deadline=started+target
        while time.perf_counter()<deadline:
            iteration+=1; before=digest(current); size=len(archive)
            q=int(rng.choice(qs)); pattern=str(rng.choice(cfg['location_patterns'])); method=str(rng.choice(METHODS))
            current,archive,outcome=attempt(instance,current,archive,q,pattern,method,np.full(5,.2),box,
                                           int(rng.integers(1,2**31)),started,deadline)
            last_acceptance=int(digest(current)!=before)
            stagnation=0 if len(archive)!=size or last_acceptance else stagnation+1
            history.append(outcome)
        elapsed=time.perf_counter()-started
        preference=np.eye(5)[(ordinal*3+stage)%6].tolist() if (ordinal*3+stage)%6<5 else [.2]*5
        ctx=dict(iteration=iteration,progress=min(1.,elapsed/max(cfg['capture_seconds'][-1],1e-9)),
                 elapsed_seconds=elapsed,budget_seconds=cfg['capture_seconds'][-1],stagnation=stagnation,
                 last_acceptance=last_acceptance,archive_size=len(archive),archive_hv=exact_hv(archive,box))
        ctx.update({f'preference_{i+1}':v for i,v in enumerate(preference)})
        state=dict(instance_id=instance.instance_id,split='train',stage_scope='generic_ALNS',
                   stage_index=stage,capture_target_seconds=target,capture_actual_seconds=elapsed,
                   query_horizon_seconds=cfg['horizon_seconds'],preference=preference,
                   source=copy.deepcopy(current),pre_archive=copy.deepcopy(archive),box=box,feature_context=ctx,
                   observed_history=copy.deepcopy(history),source_hash=digest(current))
        state['pre_archive_hash']=digest(state['pre_archive']); state['reference_hash']=digest(box)
        payload={k:state[k] for k in ('instance_id','source','pre_archive_hash','reference_hash','feature_context')}
        state['state_hash']=digest(payload); states.append(state)
    return states


def branch(instance, state, q, pattern, method, repetition, seed, destroy_seed, cfg):
    source=copy.deepcopy(state['source']); archive=copy.deepcopy(state['pre_archive'])
    box=state['box']; w=np.asarray(state['preference']); h=cfg['horizon_seconds']
    started=time.perf_counter(); deadline=started+h
    d=deletion(source,q,pattern,instance,destroy_seed)
    names,features=vector_features(instance,dict(state,_pattern=pattern),d,method)
    # First D stays fixed across repair arms; continuation randomness is paired.
    result=repair(instance,source['routes'],d['removed_customers'],method,w,box['scale'],seed,deadline=deadline)
    first_action_seconds=time.perf_counter()-started
    outcome=dict(status=result['reason'],feasible=result['feasible'],committed=False,
                 commit_elapsed_seconds=first_action_seconds,reason=result['reason'],timeout=result['timeout'],
                 repair_seconds=result['seconds'],counters=result['counters'],timing=result['timing'],repair_id=method,q=q)
    proposal=archive
    if result['feasible']:
        candidate=point(instance,result['routes']); outcome.update(candidate)
        np.testing.assert_allclose(candidate['objectives'],result['objectives'],rtol=0.,atol=1e-7)
        proposal=archive_add(archive,candidate)
    next_source=candidate if result['feasible'] and scalar(candidate,w,box)<=scalar(source,w,box) else source
    first_commit=time.perf_counter()-started; outcome['commit_elapsed_seconds']=first_commit
    success=result['feasible'] and first_commit<=h
    if success:
        outcome.update(status='success',committed=True); archive=proposal
        source=next_source
    elif result['feasible']:outcome['status']='completed_after_deadline'
    immediate_hv=exact_hv(archive,box); outcomes=[outcome]
    rng=np.random.default_rng(seed); cap=min(15,math.floor(.15*instance.n_customers))
    while success and time.perf_counter()<deadline:
        cq=int(rng.choice([1,math.ceil(cap/2),cap])); cp=str(rng.choice(cfg['location_patterns']))
        cm=str(rng.choice(METHODS)); cs=int(rng.integers(1,2**31))
        source,archive,o=attempt(instance,source,archive,cq,cp,cm,w,box,cs,started,deadline);outcomes.append(o)
    elapsed=time.perf_counter()-started
    pre=exact_hv(state['pre_archive'],box); post=exact_hv(archive,box)
    b=dict(instance_id=instance.instance_id,state_hash=state['state_hash'],pre_archive_hash=state['pre_archive_hash'],
           reference_hash=state['reference_hash'],repair_id=method,repetition=repetition,seed=seed,q=q,
           location_pattern=pattern,destroy=d,destroy_hash=digest(d),horizon_seconds=h,
           timing=dict(includes_first_action=True,elapsed_seconds=elapsed,first_action_seconds=first_action_seconds,
                       first_commit_seconds=first_commit),outcomes=outcomes,post_archive=archive,
           post_archive_hash=digest(archive),attempted_count=len(outcomes),
           committed_count=sum(o['committed'] for o in outcomes),pre_hv=pre,post_hv=post,delta_hv=post-pre,
           immediate_delta_hv=immediate_hv-pre,first_repair_success=bool(success),label_success=bool(success),
           feature_names=names,features=features,censor_reason=None if success else outcome['status'])
    b['audit']=audit_branch(b,state,instance,cfg)
    return b


def load_instance(cfg,iid):
    row=json.loads(Path(cfg['inputs'][iid]['path']).read_text(encoding='utf-8'))
    assert row['split']=='train' and row['instance_id']==iid
    return _instance_from_row(row),row
