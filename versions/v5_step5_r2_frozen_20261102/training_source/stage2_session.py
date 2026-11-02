"""One shared execution/sampling path for Stage2 collection and deployment."""
import time
import numpy as np
from stage1_mechanisms import refresh_required,physical_features
from stage2_cpu import CpuHead,sample
from stage2_structured import route_context

def raw_context(session):
    from solver.config import REPAIRERS,CONTEXT
    env=session.env;repair_id=REPAIRERS[session.pending[1]]
    global_remaining=max(0.,session.global_start-session.elapsed)
    values=list(env.preference)+list((env.objectives-env.offset)/env.scales)
    values += [global_remaining/300.,1-global_remaining/session.global_budget,
        session.remaining()/300.,session.remaining()/session.budget,min(1.,env.stagnation/10.),
        float(env.last_accepted),env.last_reward,len(env.archive)/(env.steps+1),
        env.instance.max_delay/max(env.instance.depot_time_window[1],1.),env.instance.n_customers/250.,max(env.q_values)/env.instance.n_customers]
    values += [float(repair_id==r) for r in REPAIRERS]
    values += session.adaptive.features()
    result=np.asarray(values,dtype=np.float32)
    if result.shape!=(len(CONTEXT),) or not np.isfinite(result).all():raise ValueError('Stage2 current context invalid')
    return result

def run_session(session,policy,task=0,episode=None):
    import torch
    n=session.env.instance.n_customers
    if not hasattr(policy,'cpu_head'):policy.cpu_head=CpuHead(policy)
    session.repair_seconds_since_refresh=0.;session.last_gnn_refresh_seconds=0.;session.dirty_since_refresh=False
    physical=None;routes_context=None;retain_nodes=episode is not None
    while session.remaining()>0:
        service_start=time.perf_counter();session.dirty_since_refresh|=bool(session.env.last_accepted)
        refresh=refresh_required(session);age_before=session.cache_age
        repair_before=session.repair_seconds_since_refresh;gnn_before=session.last_gnn_refresh_seconds;dirty_before=session.dirty_since_refresh
        obs=session.prepare(include_observation=refresh)
        if session.remaining()<=0:session.pending=None;break
        started=time.perf_counter();gnn_seconds=physical_seconds=0.
        if refresh:
            cache=policy.refresh(obs,task=task,retain_nodes=retain_nodes)
            session.cached_distribution=cache;session.cache_hidden=torch.from_numpy(cache['hidden'].copy()).reshape(1,-1)
            physical=obs['x'][1:,[18,17,19,12,13]].numpy().copy()
            routes_context=route_context(session.env.instance,session.env.routes,physical)
            session.cache_age=0;gnn_seconds=time.perf_counter()-started
            session.last_gnn_refresh_seconds=gnn_seconds;session.repair_seconds_since_refresh=0.;session.dirty_since_refresh=False
        elif session.env.last_accepted:
            pstart=time.perf_counter();physical=physical_features(session.env.instance,session.env.routes)
            routes_context=route_context(session.env.instance,session.env.routes,physical)
            physical_seconds=time.perf_counter()-pstart
        raw=raw_context(session)
        z=(session.env.objectives-session.env.dynamic_ideal)/session.env.dynamic_span
        dynamic_position=np.asarray([z.mean(),(session.env.preference*z).max()],dtype=np.float32)
        action,replay=sample(session.cached_distribution,physical,raw,n,session.cache_rng,session.env.neighbors,policy.cpu_head,routes_context,dynamic_position,task)
        session.cache_age+=1
        seconds=time.perf_counter()-started;full_seconds=time.perf_counter()-service_start
        result=session.finish(action,session.cache_hidden,seconds)
        session.repair_seconds_since_refresh+=result['repair_seconds']
        result.update(cache_hit=not refresh,cache_refresh=refresh,cache_age_before=age_before,cache_age_after=session.cache_age,
            cache_step=session.step_in_session,gnn_refresh_seconds=gnn_seconds,physical_refresh_seconds=physical_seconds,
            physical_refresh=bool(not refresh and session.decision_last_accepted),
            cached_sampling_seconds=max(0.,seconds-gnn_seconds-physical_seconds),
            q_head_seconds=replay['q_head_seconds'],full_policy_service_seconds=full_seconds,
            repair_seconds_since_refresh_before=repair_before,last_gnn_refresh_seconds_before=gnn_before,
            dirty_since_refresh_before=dirty_before,use_pulse=False,mode=action['mode'],mode_name=action['mode_name'],
            sampling_routes=[list(r) for r in routes_context['routes']],sampling_node_logits=replay['logits'].tolist() if action['mode']==2 else None,q_condition=action['q_condition'],first_route_exceeds_cap=action['first_route_exceeds_cap'],alpha=replay['alpha'],beta=replay['beta'],logprob_macro=replay['logprob_macro'],logprob_micro=replay['logprob_micro'],effective_selection_steps=replay['effective_selection_steps'])
        if episode is not None:
            replay.update(reward=result['reward'],terminal=result['terminated'],truncated=result['truncated'],diagnostics=result,
                stagnation=session.decision_stagnation,source_routes_at_start=getattr(session,'group_start_routes',None))
            episode.append(replay)
        session.step_in_session+=1
    ending=session.finalize()
    if episode:episode[-1].update(terminal=ending['terminated'],truncated=ending['truncated'])
    return episode
