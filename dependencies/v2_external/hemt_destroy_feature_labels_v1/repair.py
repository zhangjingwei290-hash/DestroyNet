"""Independent CPU classical repair panel, with no policy/teacher fallback.

Scoring uses all five candidate objectives with fixed supplied affine scales and
augmented Tchebycheff (rho=.05). No clipping, fitted scaling, or outcome input.
deadline is an absolute time.perf_counter() cutoff, checked during enumeration
per customer and scoring every 16 candidates. One engine enumeration or route
evaluation is indivisible; this is a cooperative cutoff, not process preemption.
"""
from collections.abc import Mapping
import math
import time
import numpy as np
from src.core.protocol import Route, Solution
from src.core.task_protocol import DEFAULT_PROTOCOL
from src.pilot.engine import enumerate_actions, apply_action, route_trace
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


def _choose(candidates, method, rng):
    """Candidate list entries: action, score, changed_metrics, objectives."""
    if method == "noisy_greedy":
        keys=[(c['score']+rng.uniform(-NOISE_RELATIVE_AMPLITUDE,NOISE_RELATIVE_AMPLITUDE)*max(abs(c['score']),1e-12),
               c['score'],_action_key(c['action']),i) for i,c in enumerate(candidates)]
        return candidates[min(keys)[-1]]
    if method == "greedy":
        return min(candidates,key=lambda c:(c['score'],_action_key(c['action'])))
    k=2 if method == "regret2" else 3
    by_customer={}
    for candidate in candidates:
        by_customer.setdefault(candidate['action'].customer_id,[]).append(candidate)
    choices=[]
    for customer, options in sorted(by_customer.items()):
        options.sort(key=lambda c:(c['score'],_action_key(c['action'])))
        regret=_regret([c['score'] for c in options],k)
        # Scarce (<k) insertion alternatives take priority; then max regret.
        # Counts >=k are tied so extra legal alternatives do not reverse regret.
        choices.append((min(len(options),k), -regret, options[0]['score'], customer, options[0]))
    return min(choices,key=lambda x:x[:4])[4]


def repair(instance, source, D, name, preference, scale, seed, deadline=None):
    """Return explicit successful/failed/timeout result; never silently fallback.

    source is a complete feasible Solution (list routes also accepted). D respects
    min(15,floor(.15*N)), with q=1 allowed for tiny CPU fixtures. A customer with
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
        'final_evaluations','final_feasibility_checks','timeout_checks','inserted_customers','new_route_actions_masked','selected_new_routes')}
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
    if len(raw)>max(1,min(15,math.floor(.15*n))):
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
    while remaining:
        if expired(): return result('deadline_before_customer_block',timeout=True)
        enumeration_started=time.perf_counter()
        legal=[]
        # Enumerating per customer bounds one indivisible engine call to O(N+R)
        # insertion alternatives; no full |D|*N giant enumeration hides the cutoff.
        for customer in remaining:
            if expired():
                timing['enumeration_seconds']+=time.perf_counter()-enumeration_started
                return result('deadline_during_enumeration',timeout=True)
            batch=enumerate_actions(instance,partial,[customer]); counters['enumeration_calls']+=1
            counters['enumerated_actions']+=len(batch.actions)
            for action,valid in zip(batch.actions,batch.mask):
                if valid:
                    legal.append(action); counters['engine_legal_actions']+=1
        timing['enumeration_seconds']+=time.perf_counter()-enumeration_started
        if expired(): return result('deadline_after_enumeration',timeout=True)
        # Validate changed route, including prefix feasibility; the engine's suffix
        # mask alone does not prove a partial nonmetric shortcut prefix feasible.
        scoring_started=time.perf_counter()
        metric_array=np.asarray(metrics,dtype=float).reshape(-1,4)
        totals=metric_array.sum(axis=0) if len(metrics) else np.zeros(4)
        other_max=[max((m[1] for j,m in enumerate(metrics) if j!=i),default=0.) for i in range(len(metrics))]
        current_max=max((m[1] for m in metrics),default=0.)
        candidates=[]
        for index,action in enumerate(legal):
            if index%16==0 and expired():
                timing['scoring_seconds']+=time.perf_counter()-scoring_started
                return result('deadline_during_scoring',timeout=True)
            changed=[] if action.opens_new_route else list(partial.routes[action.route_index].customers)
            changed.insert(action.insert_position,action.customer_id)
            counters['candidate_route_legality_checks']+=1
            if not route_trace(instance,changed)[1]: continue
            changed_metrics=np.asarray(evaluate_route(Route(changed),instance),dtype=float)
            counters['changed_route_evaluations']+=1
            old=np.zeros(4) if action.opens_new_route else metric_array[action.route_index]
            updated=totals-old+changed_metrics
            makespan=max(current_max if action.opens_new_route else other_max[action.route_index],changed_metrics[1])
            objectives=np.array([len(partial.routes)+int(action.opens_new_route),updated[0],makespan,updated[2],updated[3]])
            score=_score(objectives,weights,mins,ranges)
            if not math.isfinite(score): return result('nonfinite_candidate_score')
            candidates.append({'action':action,'score':score,'changed_metrics':tuple(changed_metrics),'objectives':objectives})
            counters['candidate_scores']+=1
        timing['scoring_seconds']+=time.perf_counter()-scoring_started
        if expired(): return result('deadline_after_scoring',timeout=True)
        existing_customers={c['action'].customer_id for c in candidates if not c['action'].opens_new_route}
        masked=[c for c in candidates if c['action'].opens_new_route and c['action'].customer_id in existing_customers]
        counters['new_route_actions_masked']+=len(masked)
        candidates=[c for c in candidates if not (c['action'].opens_new_route and c['action'].customer_id in existing_customers)]
        if not candidates: return result('no_legal_insertion')
        chosen=_choose(candidates,name,rng)
        action=chosen['action']
        partial,remaining=apply_action(partial,remaining,action)
        if action.opens_new_route:
            metrics.append(chosen['changed_metrics']); counters['selected_new_routes']+=1
        else: metrics[action.route_index]=chosen['changed_metrics']
        counters['inserted_customers']+=1
        trace.append({'customer':action.customer_id,'route_index':action.route_index,'position':action.insert_position,
                      'opens_new_route':action.opens_new_route,'score':chosen['score']})
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
