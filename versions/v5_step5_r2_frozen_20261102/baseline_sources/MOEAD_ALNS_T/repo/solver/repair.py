from collections.abc import Mapping
import math
import time
import numpy as np
from src.core.protocol import Route, Solution, RepairAction
from src.core.task_protocol import DEFAULT_PROTOCOL
from src.pilot.engine import insertion_context, apply_action
from src.evaluation.vrptw_eval import evaluate_route, evaluate_solution, is_solution_feasible

METHODS = ("greedy", "regret2", "regret3", "noisy_greedy")
NOISE_RELATIVE_AMPLITUDE = .1
RHO = DEFAULT_PROTOCOL.tchebycheff_rho


def _scales(scale):
    if isinstance(scale, Mapping):
        if "ranges" not in scale:
            raise ValueError("scale mapping requires ranges (and optional mins)")
        ranges = np.asarray(scale["ranges"], dtype=float)
        mins = np.asarray(scale.get("mins", np.zeros(5)), dtype=float)
    elif hasattr(scale, "ranges") and hasattr(scale, "mins"):
        ranges, mins = np.asarray(scale.ranges,dtype=float), np.asarray(scale.mins,dtype=float)
    else:
        ranges, mins = np.asarray(scale,dtype=float), np.zeros(5)
    if ranges.shape != (5,) or mins.shape != (5,) or not np.isfinite(ranges).all() or not np.isfinite(mins).all() or (ranges <= 0).any():
        raise ValueError("fixed objective ranges must be five positive finite values")
    return mins.copy(), ranges.copy()


def _score(objectives, weights, mins, ranges):
    weighted = weights * ((np.asarray(objectives)-mins)/ranges)
    return float(weighted.max() + RHO * weighted.sum())


def _action_key(action):
    return (action.customer_id, int(action.opens_new_route), action.route_index, action.insert_position)


def _regret(scores, k):
    """Finite regret-k: pad scarce alternatives with worst available score."""
    scores=sorted(float(x) for x in scores)
    if not scores or k < 2:
        raise ValueError("regret requires at least one legal score and k>=2")
    padded=scores+[scores[-1]]*max(0,k-len(scores))
    return float(sum(x-scores[0] for x in padded[1:k]))


def _repair_raw(instance, source, D, name, preference, scale, seed, deadline=None):
    """Return explicit successful/failed/timeout result; never silently fallback.

    source is a complete feasible Solution (list routes also accepted). D respects
    1..N in this separately authorized broad-granularity experiment. A customer with
    a legal existing insertion has its own new-vehicle alternatives masked.
    Regret chooses maximum score regret, then minimum best score and structural
    deterministic tie keys. Noisy greedy adds stated relative uniform score noise.
    """
    started=time.perf_counter()
    if name not in METHODS:
        raise ValueError(f"unknown repair {name!r}; supported {METHODS}")
    weights=np.asarray(preference,dtype=float)
    if weights.shape != (5,) or not np.isfinite(weights).all() or (weights < 0).any() or not np.isclose(weights.sum(),1.,atol=1e-6,rtol=0):
        raise ValueError("preference must be five finite nonnegative weights summing to one")
    mins,ranges=_scales(scale)
    if deadline is not None and not math.isfinite(float(deadline)):
        raise ValueError("deadline must be None or an absolute finite perf_counter value")
    counters={k:0 for k in ('source_feasibility_checks','enumeration_calls','enumerated_actions','engine_legal_actions',
        'candidate_route_legality_checks','candidate_scores','changed_route_evaluations','initial_route_evaluations',
        'final_evaluations','final_feasibility_checks','timeout_checks','inserted_customers','new_route_actions_masked','selected_new_routes',
        'route_blocks_built','route_blocks_reused','route_contexts_built','vectorized_score_batches')}
    timing={k:0. for k in ('validation_seconds','enumeration_seconds','scoring_seconds','final_evaluation_seconds')}
    trace=[]
    rng=np.random.default_rng(seed)
    def expired():
        counters['timeout_checks']+=1
        return deadline is not None and time.perf_counter() >= float(deadline)
    def result(reason,sol=None,objectives=None,timeout=False):
        ok=reason=='ok'
        return {'method':name,'routes':None if sol is None else [list(r.customers) for r in sol.routes],
                'solution':sol,'objectives':None if objectives is None else np.asarray(objectives,dtype=float).tolist(),
                'feasible':ok,'failed':not ok and not timeout,'timeout':timeout,'reason':reason,
                'seconds':time.perf_counter()-started,'counters':dict(counters),'timing':dict(timing),'trace':list(trace),
                'config':{'rho':RHO,'score':'fixed_affine_unclipped_augmented_tchebycheff',
                          'new_vehicle_mask':'per_customer_any_legal_existing',
                          'regret_shortfall':'scarcity_first_then_finite_regret_repeat_worst',
                          'regret_tie_key':'min(n_legal,k),-regret,best_score,customer_id',
                          'noise_relative_amplitude':NOISE_RELATIVE_AMPLITUDE if name=='noisy_greedy' else 0.,
                          'seed':int(seed),'deadline_clock':'time.perf_counter_absolute'}}
    if expired(): return result('deadline_before_validation',timeout=True)
    validation_start=time.perf_counter()
    if not isinstance(source,Solution):
        source=Solution([Route(list(r)) for r in source])
    else:
        source=Solution([Route(list(r.customers)) for r in source.routes])
    n=int(instance.n_customers)
    raw=list(D)
    if not raw or any(isinstance(c,bool) or not isinstance(c,(int,np.integer)) for c in raw) or len(set(raw)) != len(raw) or any(c<1 or c>n for c in raw):
        return result('invalid_removal_set')
    if len(raw)>n:
        return result('removal_cap_exceeded')
    counters['source_feasibility_checks']+=1
    if not is_solution_feasible(source,instance): return result('source_infeasible')
    remove=set(raw)
    skeleton=[Route([c for c in r.customers if c not in remove]) for r in source.routes]
    partial=Solution([r for r in skeleton if r.customers])
    remaining=sorted(remove)
    metrics=[]
    for route in partial.routes:
        if expired(): return result('deadline_initial_metrics',timeout=True)
        metrics.append(evaluate_route(route,instance)); counters['initial_route_evaluations']+=1
    timing['validation_seconds']+=time.perf_counter()-validation_start
    # Cache LOCAL metrics, not stale global scalar scores. Global makespan and
    # totals change after every insertion and are refreshed in one NumPy pass.
    cache = {}                   # (unassigned customer, rid) -> legal positions
    route_contexts = {}          # rid -> prefix/suffix/load; -1 is singleton

    def build_route_blocks(rid,customers):
        if expired():raise TimeoutError
        route=[] if rid==-1 else partial.routes[rid].customers
        length=len(route)
        if rid not in route_contexts:
            context=insertion_context(instance,route)
            departures,latest=context
            prefix_ok=np.ones(length+1,dtype=bool)
            for j,c in enumerate(route):
                prev=route[j-1] if j else 0
                arrival=departures[j]+instance.duration_matrix[prev,c]
                prefix_ok[j+1]=prefix_ok[j] and arrival<=instance.customer_time_windows[c-1,1]+instance.max_delay+1e-6
            suffix_ok=np.ones(length+1,dtype=bool)
            for j in range(length-1,-1,-1):
                suffix_ok[j]=suffix_ok[j+1] and instance.customer_time_windows[route[j]-1,0]<=latest[j]+1e-6
            route_contexts[rid]=(context,float(sum(instance.customer_demands[c-1] for c in route)),prefix_ok,suffix_ok)
            counters['route_contexts_built']+=1
        (departures,latest),load,prefix_ok,suffix_ok=route_contexts[rid]
        counters['route_blocks_built']+=len(customers)
        cs=np.repeat(np.asarray(customers,dtype=int),length+1)
        positions=np.tile(np.arange(length+1),len(customers))
        counters['enumerated_actions']+=len(cs)
        seq=np.asarray([0]+route+[0],dtype=int)
        arrival=departures[positions]+instance.duration_matrix[seq[positions],cs]
        departure=np.maximum(arrival,instance.customer_time_windows[cs-1,0])+instance.customer_service_times[cs-1]
        limits=np.r_[latest,instance.depot_time_window[1]]
        legal=((load+instance.customer_demands[cs-1]<=instance.vehicle_capacity+1e-6)
               &prefix_ok[positions]&suffix_ok[positions]
               &(arrival<=instance.customer_time_windows[cs-1,1]+instance.max_delay+1e-6)
               &(departure+instance.duration_matrix[cs,seq[positions+1]]<=limits[positions]+1e-6))
        cs=cs[legal];positions=positions[legal]
        counters['engine_legal_actions']+=len(cs)
        # Batch exact evaluate_route arithmetic over every legal insertion on
        # THIS route. Only the position loop remains; no candidate Route objects.
        dist=np.zeros(len(cs));clock=np.zeros(len(cs));waiting=np.zeros(len(cs));tardy=np.zeros(len(cs))
        prev=np.zeros(len(cs),dtype=int)
        for j in range(length+1):
            if j%16==0 and expired():raise TimeoutError
            # Existing path shifts right after the inserted customer.
            indexes=j-(j>positions)
            current=np.where(j==positions,cs,seq[np.minimum(indexes+1,length)])
            dist+=instance.distance_matrix[prev,current]
            clock+=instance.duration_matrix[prev,current]
            early=instance.customer_time_windows[current-1,0];late=instance.customer_time_windows[current-1,1]
            is_wait=clock<early
            waiting+=np.where(is_wait,early-clock,0.)
            tardy+=np.where(~is_wait & (clock>late),clock-late,0.)
            clock=np.where(is_wait,early,clock)+instance.customer_service_times[current-1]
            prev=current
        dist+=instance.distance_matrix[prev,0];clock+=instance.duration_matrix[prev,0]
        values=np.column_stack((dist,clock,waiting,tardy))
        metadata=np.column_stack((cs,np.full(len(cs),rid,dtype=int),positions))
        counters['changed_route_evaluations']+=len(cs)
        for c in customers:
            left,right=np.searchsorted(cs,[c,c+1])
            cache[c,rid]=(metadata[left:right],values[left:right])

    while remaining:
        if expired():return result('deadline_before_customer_block',timeout=True)
        enumeration_started=time.perf_counter()
        rids_order=list(range(len(partial.routes)))+[-1]
        try:
            for rid in rids_order:
                missing=[c for c in remaining if (c,rid) not in cache]
                counters['route_blocks_reused']+=len(remaining)-len(missing)
                if missing:build_route_blocks(rid,missing)
        except TimeoutError:
            timing['enumeration_seconds']+=time.perf_counter()-enumeration_started
            return result('deadline_during_incremental_enumeration',timeout=True)
        blocks=[cache[c,rid] for c in remaining for rid in rids_order]
        metadata=np.concatenate([v[0] for v in blocks]);changed=np.concatenate([v[1] for v in blocks])
        timing['enumeration_seconds']+=time.perf_counter()-enumeration_started
        if not len(metadata):return result('no_legal_insertion')
        if expired():return result('deadline_after_enumeration',timeout=True)
        scoring_started=time.perf_counter()
        metric_array=np.asarray(metrics,dtype=float).reshape(-1,4)
        totals=metric_array.sum(axis=0) if len(metrics) else np.zeros(4)
        other_max=np.asarray([max((m[1] for j,m in enumerate(metrics) if j!=i),default=0.) for i in range(len(metrics))])
        current_max=max((m[1] for m in metrics),default=0.)
        rids=metadata[:,1]
        new=rids==-1
        old=np.zeros_like(changed);old[~new]=metric_array[rids[~new]]
        updated=totals-old+changed
        competing=np.full(len(metadata),current_max);competing[~new]=other_max[rids[~new]]
        objectives=np.column_stack((len(metrics)+new,updated[:,0],np.maximum(competing,changed[:,1]),updated[:,2],updated[:,3]))
        weighted=weights*((objectives-mins)/ranges)
        scores=weighted.max(axis=1)+RHO*weighted.sum(axis=1)
        counters['candidate_scores']+=len(metadata);counters['vectorized_score_batches']+=1
        if not np.isfinite(scores).all():return result('nonfinite_candidate_score')
        # Mask singleton alternatives BEFORE creating Python selection records.
        customers=metadata[:,0]
        has_existing=np.zeros(n+1,dtype=bool);has_existing[customers[~new]]=True
        keep=~(new & has_existing[customers])
        counters['new_route_actions_masked']+=int((~keep).sum())
        metadata=metadata[keep];scores=scores[keep];changed=changed[keep]
        if not len(scores):return result('no_legal_insertion')
        chosen_index=_choose_array(metadata,scores,name,rng)
        customer,rid,pos=map(int,metadata[chosen_index])
        action=RepairAction(customer,rid,pos,rid==-1)
        chosen=dict(action=action,score=float(scores[chosen_index]),changed_metrics=tuple(changed[chosen_index]))
        timing['scoring_seconds']+=time.perf_counter()-scoring_started
        if expired():return result('deadline_after_scoring',timeout=True)
        changed_rid=len(partial.routes) if action.opens_new_route else action.route_index
        partial,remaining=apply_action(partial,remaining,action)
        if action.opens_new_route:
            metrics.append(chosen['changed_metrics']);counters['selected_new_routes']+=1
        else:metrics[changed_rid]=chosen['changed_metrics']
        # This is the only route invalidated. Every unchanged route is reused.
        route_contexts.pop(changed_rid,None)
        for key in list(cache):
            if key[0]==action.customer_id or key[1]==changed_rid:del cache[key]
        counters['inserted_customers']+=1
        trace.append(dict(customer=action.customer_id,route_index=action.route_index,position=action.insert_position,
                          opens_new_route=action.opens_new_route,score=chosen['score']))
    if expired(): return result('deadline_before_final_validation',timeout=True)
    final_started=time.perf_counter()
    counters['final_feasibility_checks']+=1
    feasible=is_solution_feasible(partial,instance)
    counters['final_evaluations']+=1
    objectives=evaluate_solution(partial,instance).to_array()
    timing['final_evaluation_seconds']+=time.perf_counter()-final_started
    if expired(): return result('deadline_during_final_validation',timeout=True)
    if not feasible: return result('final_solution_infeasible')
    return result('ok',partial,objectives)



def _choose_array(meta,scores,method,rng):
    customers,rids,positions=meta.T;new=rids==-1
    if method=='noisy_greedy':
        noisy=scores+rng.uniform(-NOISE_RELATIVE_AMPLITUDE,NOISE_RELATIVE_AMPLITUDE,len(scores))*np.maximum(np.abs(scores),1e-12)
        return int(np.lexsort((positions,rids,new,customers,scores,noisy))[0])
    # Metadata is already in exact customer/new/rid/position tie order.
    if method=='greedy':return int(np.argmin(scores))
    k=2 if method=='regret2' else 3
    order=np.lexsort((positions,rids,new,scores,customers))
    sorted_customers=customers[order]
    starts=np.r_[0,np.flatnonzero(np.diff(sorted_customers))+1];ends=np.r_[starts[1:],len(order)]
    choices=[]
    for begin,end in zip(starts,ends):
        options=order[begin:end];best=int(options[0])
        regret=_regret(scores[options[:k]],k)
        choices.append((min(end-begin,k),-regret,float(scores[best]),int(customers[best]),best))
    return min(choices)[-1]

def repair(instance,source,D,name,preference,scale,seed,deadline=None):
    return _repair_raw(instance,source,D,name,preference,scale,seed,deadline)
