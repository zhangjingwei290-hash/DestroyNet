"""Closed-loop EMNH, balanced FIT/direction scheduling, normalized group rewards."""
import copy,time,os
from pathlib import Path
import numpy as np
import torch
from .config import directions
from .model import Policy
from .environment import Environment,LocalSession
from .outer import Search,TaskPolicy,run_session,search_state,restore_search
from .q_statistics import empty_state_q,add_q,state_labels,nearest_q_slot


def balanced_fit_ids(ids):
    cells={(n,d):sorted(i for i in ids if i.split('_')[2:4]==[str(n),d])
           for d in ('C','R','RC') for n in (50,150,250)}
    if len(ids)!=45 or any(len(v)!=5 for v in cells.values()):raise ValueError('Require five FIT IIDs per size/distribution cell')
    return [cells[cell][round_i] for round_i in range(5) for cell in cells]


def cyclic_tasks(rng):
    base=rng.dirichlet(np.ones(5))
    return np.stack([np.roll(base,i) for i in range(5)])


def skip_background(k,mapping,episodes):
    return k not in mapping or mapping[k] in episodes


STAGES=('early','middle','late')


def search_stage(used):
    return STAGES[min(2,max(0,int(float(used)*3)))]


def collect_closed(pool,split,cfg,policy,meta,weights,state,rng,deadline,notify=lambda _:None,runtime=None):
    weights=np.asarray(weights)
    if len(weights)==5:
        refs=directions();refs[:5]=weights;targets=list(range(5))
    elif len(weights)==1:
        refs=directions();targets=[int(np.argmin(np.linalg.norm(refs-weights[0],axis=1)))]
        if not np.allclose(refs[targets[0]],weights[0]):raise ValueError('Adapt a deployment direction')
    else:raise ValueError('Collect five meta tasks or one adaptation task')
    mapping={k:t for t,k in enumerate(targets)}
    policies=[TaskPolicy(policy,mapping[k]) if k in mapping else None for k in range(16)]
    order=split['fit_order'] if 'fit_order' in split else balanced_fit_ids(split['fit_ids'])
    search_pool=state.setdefault('search_pool',{})
    iid=state.get('active_iid')
    if iid is None:
        iid=order[state.get('next_iid',0)%len(order)]
        state['next_iid']=state.get('next_iid',0)+1;state['active_iid']=iid
    if iid not in split['fit_ids']:raise ValueError('Collection outside FIT')
    entry=search_pool.get(iid)
    if entry is None:
        search=Search(pool,iid,cfg,int(rng.integers(0,2**31-1)),policies,resource_deadline=deadline,ref_dirs=refs)
        requested_stage=0;cycle=state.get('completed_cycles',0)
    else:
        search=restore_search(entry['search'],pool,cfg,policies,deadline)
        search.set_preferences(refs,policies)
        requested_stage=entry['next_stage'];cycle=entry['cycle']
    elapsed_before=max(0.,search.budget-search.remaining())
    low=requested_stage/3;high=(requested_stage+1)/3
    episodes={};provenance=[];skipped=0;traversal_steps=0

    def fork(k,primary):
        nonlocal skipped,traversal_steps
        if k not in mapping:
            skipped+=1;return primary,0.,[]
        used=1-primary.global_start/primary.global_budget
        if used<low or len(episodes)==len(weights):
            # Real neural search work advances the actual population toward the
            # next sampling window (or finishes the final cycle). No sleeping,
            # progress relabeling, or fictional solver-clock debit.
            if used<low:
                primary.budget=min(primary.budget,max(0.,(low-used)*search.budget))
            paused=0.
            if runtime is None:run_session(primary,policies[k])
            else:paused=runtime.run_primary(primary,policies[k])[0]
            traversal_steps+=len(primary.records)
            return primary,paused,[]
        if mapping[k] in episodes:
            skipped+=1;return primary,0.,[]
        if used>=high and requested_stage<2:
            raise RuntimeError('Search crossed requested phase before all task groups were sampled')
        task=mapping[k];original=copy.deepcopy(primary.adaptive);routes=copy.deepcopy(primary.env.routes)
        # Do not let early task groups consume another task's remaining phase.
        # This only shortens a local segment when the real phase boundary is near.
        missing=len(weights)-len(episodes)
        phase_remaining=max(0.,(high-used)*search.budget)
        primary.budget=min(primary.budget,phase_remaining/(missing+1))
        if primary.budget<=0:raise RuntimeError('No real solver time for phase group')
        if runtime is None:
            group=[[]];run_session(primary,policies[k],task,group[0])
            started=time.perf_counter()
            for _ in range(cfg.replicas-1):
                env=Environment(pool,iid,cfg,refs[k],routes)
                clone=LocalSession(env,copy.deepcopy(original),int(rng.integers(0,2**31-1)),
                    primary.global_budget,primary.global_start,primary.budget,deadline)
                ep=[];run_session(clone,policies[k],task,ep);group.append(ep)
            paused=time.perf_counter()-started
        else:
            setup=time.perf_counter();sessions=[primary]
            for _ in range(cfg.replicas-1):
                env=Environment(pool,iid,cfg,refs[k],routes)
                sessions.append(LocalSession(env,copy.deepcopy(original),int(rng.integers(0,2**31-1)),
                    primary.global_budget,primary.global_start,primary.budget,deadline))
            group,paused=runtime.run_group(sessions,policies[k],task,time.perf_counter()-setup)
        valid=all(ep and ep[-1]['terminal'] and not ep[-1]['truncated'] for ep in group)
        if valid:
            for ep in group:
                for t in ep:
                    t['direction']=k;t['iid']=iid;t['source_stage']=STAGES[requested_stage]
            episodes[task]=group
        else:state.setdefault('unoptimized_groups',[]).append(group)
        provenance.append(dict(iid=iid,direction=k,weight=refs[k].tolist(),optimized_group=valid,
            source_routes=routes,source='continued_actual_MOEAD_offspring',primary_committed=True,
            cycle=cycle,source_stage=search_stage(used),requested_stage=STAGES[requested_stage],
            global_remaining=primary.global_start,global_budget=primary.global_budget,
            rewards=[sum(t['reward'] for t in ep) for ep in group],steps=[len(ep) for ep in group]))
        return primary,paused,group

    while len(episodes)<len(weights) and search.remaining()>0 and time.perf_counter()<deadline:
        if not search.step(fork):break
    complete=len(episodes)==len(weights)
    if complete and requested_stage==2:
        # Finish the remaining REAL search budget before creating another cycle.
        # Foreground trajectories here are traversal-only and reported separately.
        while search.remaining()>0 and time.perf_counter()<deadline:
            if not search.step(fork):break
    snapshot=search_state(search)
    state['last_iid']=iid;state['background_skipped']=state.get('background_skipped',0)+skipped
    state['traversal_only_steps']=state.get('traversal_only_steps',0)+traversal_steps
    state['last_population']=copy.deepcopy(snapshot['population']);state['last_archive']=copy.deepcopy(search.archive)
    if complete and requested_stage==2 and search.remaining()<=0:
        search_pool.pop(iid,None);state.pop('active_iid',None)
        state['completed_cycles']=state.get('completed_cycles',0)+1
    else:
        search_pool[iid]=dict(search=snapshot,next_stage=requested_stage+int(complete),cycle=cycle)
    state['last_continuation']=dict(iid=iid,cycle=cycle,requested_stage=STAGES[requested_stage],
        resumed=entry is not None,elapsed_before=elapsed_before,elapsed_after=snapshot['elapsed'],
        search_budget=search.budget,complete=complete,traversal_only_steps=traversal_steps)
    flat=[ep for t in sorted(episodes) for ep in episodes[t]]
    notify(dict(phase='collection',iid=iid,complete=complete,background_skipped=skipped,**{
        k:v for k,v in state['last_continuation'].items() if k not in ('iid','complete')}))
    return flat,complete,provenance


def group_advantages(episodes):
    from .q_statistics import regime_name, REGIMES
    groups = {}
    for ep in episodes:
        if not ep or not ep[-1]['terminal'] or ep[-1].get('truncated'):
            raise ValueError('Require complete same-start trajectories')
        groups.setdefault((ep[0].get('iid'),ep[0]['task'],ep[0].get('start_group')), []).append(ep)
    advantages = {}
    regime_weights = {}
    for task, group in groups.items():
        if len(group) != 4:
            raise ValueError('Require four SAME-START replicas per task')
        total_r = sum(max(0., float(t['reward'])) for ep in group for t in ep)
        total_dt = sum((max(float(t['diagnostics'].get('repair_seconds',0.)),0.) + max(float(t['diagnostics'].get('inference_seconds',0.)),0.) + .001) for ep in group for t in ep)
        v_ref = total_r / max(total_dt, 1e-3)  # 该起点组实际每秒目标改进速率

        # 第一步：计算每条轨迹的剩余窗口累计实际改进 G_next（gamma=0.85，捕获快步策略为后续留出更多步数的累计收益）
        raw_scores = []
        regimes = []
        for ep in group:
            n_steps = len(ep)
            future_r = 0.0
            ep_scores = [0.0] * n_steps
            ep_regs = [''] * n_steps
            for k in range(n_steps - 1, -1, -1):
                t = ep[k]
                r = max(0., float(t['reward']))
                dt = (max(float(t['diagnostics'].get('repair_seconds',0.)),0.) + max(float(t['diagnostics'].get('inference_seconds',0.)),0.) + .001)
                st = t['observation']['search_state']
                reg = regime_name(st['last_accepted'], st['stagnation'])
                if reg == 'refine':
                    # 顺风精修态：即时改进 + 0.85未来累计 - 1.00倍实测时间机会成本
                    u = r + 0.85 * future_r - 1.00 * v_ref * dt
                else:
                    # 首步重构态（initial）与受阻突围态（stalled）：比拼单步破局幅度与后续解锁空间，采用温和线性时间减法成本
                    u = r + 0.50 * future_r - 0.25 * v_ref * dt
                ep_scores[k] = float(u)
                ep_regs[k] = reg
                future_r = r + 0.85 * future_r
            raw_scores.append(np.asarray(ep_scores, dtype=np.float64))
            regimes.append(ep_regs)

        # 第二步：按 (initial, refine, stalled) 三个考场各自独立做组内零均值标准化（绝不跨状态混池！）
        bucket_vals = {reg: [] for reg in REGIMES}
        for ep_s, ep_r in zip(raw_scores, regimes):
            for u, reg in zip(ep_s, ep_r):
                bucket_vals[reg].append(u)
        stats = {}
        active_regimes = [reg for reg in REGIMES if len(bucket_vals[reg]) > 0]
        for reg in REGIMES:
            vals = np.asarray(bucket_vals[reg], dtype=np.float64)
            if len(vals) >= 2 and vals.std() > 1e-12:
                stats[reg] = (float(vals.mean()), float(vals.std()), len(vals))
            else:
                stats[reg] = (float(vals.mean()) if len(vals) else 0.0, None, len(vals))

        # 第三步：赋标准化优势，并为每个样本计算“三考场等权防刷票权重”
        # 步数均匀权重与活跃考场均衡权重各占50%，每组总权重为1
        n_active = max(1, len(active_regimes))
        total_group_steps=sum(len(ep) for ep in group)
        for ep, ep_s, ep_r in zip(group, raw_scores, regimes):
            adv_arr = np.zeros(len(ep), dtype=np.float64)
            w_arr = np.zeros(len(ep), dtype=np.float64)
            for k, (u, reg) in enumerate(zip(ep_s, ep_r)):
                mean_r, std_r, count_r = stats[reg]
                adv_arr[k] = 0.0 if std_r is None else (u - mean_r) / (std_r + 1e-8)
                w_arr[k] = 0.5/max(1,total_group_steps) + 0.5/(n_active * max(1,count_r))
            advantages[id(ep)] = adv_arr
            regime_weights[id(ep)] = w_arr
    return groups, advantages, regime_weights


def q_summary(episodes):
    def empty():return dict(count=0,q_ratio_sum=0.,q_slot_counts=[0]*10)
    result=dict(by_step_stage={name:empty() for name in STAGES},
                by_source_stage={name:empty() for name in STAGES},by_direction={str(k):empty() for k in range(16)},
                **empty_state_q())
    for ep in episodes:
        for t in ep:
            obs=t['observation'];stage=search_stage(float(obs['context'][11]));n=len(obs['x'])-1
            d=str(t.get('direction',t['task']))
            labels=state_labels(obs['search_state']['last_accepted'],obs['search_state']['stagnation'])
            for name,label in labels.items():add_q(result[name][label],t['action']['q'],n,nearest_q_slot(t['action']['q'],n))
            for block in (result['by_step_stage'][stage],result['by_source_stage'][t.get('source_stage',stage)],result['by_direction'][d]):
                block['count']+=1;block['q_ratio_sum']+=t['action']['q']/n;block['q_slot_counts'][nearest_q_slot(t['action']['q'],n)]+=1
    return result

def update(policy,optimizer,episodes,method,cfg,deadline=None):
    if method not in ('ppo','reinforce'):raise ValueError('Unknown inner optimizer')
    groups,advantages,regime_weights=group_advantages(episodes)
    samples=[]
    for ep in episodes:
        for t,adv,rw in zip(ep,advantages[id(ep)],regime_weights[id(ep)]):
            samples.append((t,float(adv),float(rw)/len(groups)))
    buckets={}
    for s in samples:buckets.setdefault(len(s[0]['observation']['x']),[]).append(s)
    epochs=cfg.ppo_epochs if method=='ppo' else 1
    metrics=dict(steps=0,transitions=len(samples),trajectories=len(episodes),loss=0.,
        reward_sum=sum(t['reward'] for t,_,_ in samples),baseline='same_start_three_regime_continuous_beta_cudagraph_subtractive_surplus',
        initial_ratio_error=0.,q_histogram={},q_slot_histogram={},positive_reward_fraction=sum(t['reward']>0 for t,_,_ in samples)/len(samples),timeout_fraction=sum(t['diagnostics']['timeout'] for t,_,_ in samples)/len(samples))
    metrics['q_breakdown']=q_summary(episodes)
    for t,_,_ in samples:
        q=t['action']['q'];metrics['q_histogram'][q]=metrics['q_histogram'].get(q,0)+1
        slot=t['action']['q_index'];metrics['q_slot_histogram'][slot]=metrics['q_slot_histogram'].get(slot,0)+1
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        for bucket in buckets.values():
            for offset in range(0,len(bucket),cfg.microbatch):
                if deadline is not None and time.perf_counter()>=deadline:
                    optimizer.zero_grad(set_to_none=True);metrics['resource_truncated_update']=True;return metrics
                chunk=bucket[offset:offset+cfg.microbatch];ts=[s[0] for s in chunk]
                _,lp,_,entropy,_=policy.terms([t['observation'] for t in ts],[t['task'] for t in ts],[t['action'] for t in ts])
                adv=lp.new_tensor([s[1] for s in chunk]);weight=lp.new_tensor([s[2] for s in chunk])
                if method=='ppo':
                    ratio=(lp-lp.new_tensor([t['logprob'] for t in ts])).exp()
                    actor=-torch.minimum(ratio*adv,ratio.clamp(1-cfg.clip_epsilon,1+cfg.clip_epsilon)*adv)
                    if epoch==0:metrics['initial_ratio_error']=max(metrics['initial_ratio_error'],float((ratio.detach()-1).abs().max()))
                else:actor=-lp*adv
                loss=((actor-cfg.entropy_weight*entropy)*weight).sum()
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite actor loss')
                loss.backward();metrics['loss']+=float(loss.detach())/epochs
        gradients={n:p.grad for n,p in policy.named_parameters() if p.grad is not None}
        if not gradients or any(not torch.isfinite(g).all() for g in gradients.values()):raise FloatingPointError('Missing/nonfinite gradient')
        metrics['gradient_norms']={n:float(g.norm()) for n,g in gradients.items() if n.startswith(('q_body.','heads.'))}
        if deadline is not None and time.perf_counter()>=deadline:
            optimizer.zero_grad(set_to_none=True);metrics['resource_truncated_update']=True;return metrics
        # No new gradient clipping, KL cap, frozen backbone or low-rate branch.
        optimizer.step();metrics['steps']+=1
    return metrics



def save_checkpoint(path,payload):
    """Publish a complete checkpoint atomically; retain previous on write error."""
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    torch.save(payload,temporary)
    os.replace(temporary,path)


def rng_state(rng):
    return dict(numpy_generator=rng.bit_generator.state,numpy_global=np.random.get_state(),
                torch_cpu=torch.get_rng_state(),torch_cuda=torch.cuda.get_rng_state_all())

def train_method(pool,split,cfg,method,root,notify=lambda _:None):
    cfg.validate(require_budget=True)
    if cfg.independent_iid_batch!=1:raise ValueError('Use train_parallel_meta.py for independent-IID batch8; no silent serial fallback')
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(cfg.seed);torch.cuda.manual_seed_all(cfg.seed);rng=np.random.default_rng(cfg.seed)
    meta=Policy(cfg).to(cfg.device);inner=Policy(cfg,tasks=5).to(cfg.device)
    optimizer=torch.optim.Adam(inner.parameters(),lr=cfg.learning_rate)
    state={'next_iid':0};history=[];round_i=0;partial_updates=0
    started=time.perf_counter();deadline=started+cfg.meta_seconds
    while time.perf_counter()<deadline:
        inner.start_inner(meta)
        weights=cyclic_tasks(rng);partial_updates=0
        for u in range(cfg.inner_updates):
            if time.perf_counter()>=deadline:break
            episodes,complete,provenance=collect_closed(pool,split,cfg,inner,meta,weights,state,rng,deadline,notify)
            if not complete:
                torch.save(dict(episodes=episodes,unoptimized_groups=state.pop('unoptimized_groups',[]),provenance=provenance),root/'meta_unoptimized_tail.pt')
                if time.perf_counter()<deadline:raise RuntimeError('Incomplete task group before resource cutoff; do not silently switch phase')
                break
            metric=update(inner,optimizer,episodes,method,cfg,deadline)
            history.append(dict(round=round_i,inner_update=u,iid=state['last_iid'],provenance=provenance,continuation=state.get('last_continuation'),**metric))
            if metric['steps']:partial_updates+=1
            save_checkpoint(root/'latest_checkpoint.pt',dict(version=cfg.version,config=cfg.to_dict(),
                stage='meta',boundary='after_optimizer_before_meta_merge',meta=meta.actor_state(),
                inner=inner.actor_state(),optimizer=optimizer.state_dict(),history=history,
                round_i=round_i,next_inner_update=u+1,partial_updates=partial_updates,
                weights=weights,state=state,meta_spent=time.perf_counter()-started,rng=rng_state(rng)))
            notify(dict(phase='meta',round=round_i,batches=len(history),iid=state['last_iid'],**metric))
            if metric.get('resource_truncated_update'):break
        if partial_updates:meta.meta_update(inner,cfg.meta_step_size)
        round_i+=1
    meta_seconds=time.perf_counter()-started
    save_checkpoint(root/'meta_complete.pt',dict(version=cfg.version,config=cfg.to_dict(),model=meta.actor_state(),history=history,meta_spent=meta_seconds,state=state,rng=rng_state(rng)))
    # Finish each parent's soft quota before initializing its children.
    order=[15]+list(range(5,15))+list(range(5));refs=directions()
    parent={15:None,**{k:15 for k in range(5,15)}}
    for k in range(5):parent[k]=next(j for j in range(5,15) if refs[j,k]>0)
    actors={};adapt_optimizers={};counts={k:0 for k in order};spent={k:0. for k in order};adapt_history=[]
    # Separate meta/adaptation solver cycles; keep the balanced FIT cursor.
    adapt_state={'next_iid':state['next_iid']}
    started=time.perf_counter();deadline=started+cfg.adaptation_seconds
    batch_seconds=[];initializations=[];quotas={}
    for idx,k in enumerate(order):
        remaining_dirs=len(order)-idx
        quota_k=max(0.,deadline-time.perf_counter())/remaining_dirs
        quotas[k]=quota_k
        actor=Policy(cfg).to(cfg.device)
        actor.start_inner(meta if parent[k] is None else actors[parent[k]])
        actors[k]=actor;optim=torch.optim.Adam(actor.parameters(),lr=cfg.learning_rate);adapt_optimizers[k]=optim
        initializations.append(dict(direction=k,parent=parent[k],parent_updates=0 if parent[k] is None else counts[parent[k]],
                                    next_iid=adapt_state['next_iid']))
        while spent[k]<quota_k and time.perf_counter()<deadline:
            # Admission estimate only; direction quota NEVER truncates a batch.
            # All admitted collection and update work sees the global deadline.
            estimate=max(batch_seconds[-3:],default=4*cfg.local_seconds)+.1
            if counts[k]>0 and spent[k]+.5*estimate>=quota_k:break
            if deadline-time.perf_counter()<estimate:break
            began=time.perf_counter()
            episodes,complete,provenance=collect_closed(pool,split,cfg,actor,meta,[refs[k]],adapt_state,rng,deadline,notify)
            if complete:
                metric=update(actor,optim,episodes,method,cfg,deadline)
                counts[k]+=int(metric['steps']>0)
                adapt_history.append(dict(direction=k,iid=adapt_state['last_iid'],provenance=provenance,continuation=adapt_state.get('last_continuation'),**metric))

            else:
                torch.save(dict(episodes=episodes,unoptimized_groups=adapt_state.pop('unoptimized_groups',[]),provenance=provenance),root/f'adapt_unoptimized_tail_{k:02d}.pt')
                if time.perf_counter()<deadline:raise RuntimeError('Incomplete adaptation group before resource cutoff')
            duration=time.perf_counter()-began;spent[k]+=duration;batch_seconds.append(duration)
            save_checkpoint(root/'latest_checkpoint.pt',dict(version=cfg.version,config=cfg.to_dict(),
                stage='adaptation',boundary='after_optimizer_and_direction_accounting',meta=meta.actor_state(),
                actors={d:m.actor_state() for d,m in actors.items()},
                optimizers={d:o.state_dict() for d,o in adapt_optimizers.items()},
                active_direction=k,order_index=idx,counts=counts,spent=spent,quotas=quotas,
                batch_seconds=batch_seconds,initializations=initializations,adapt_state=adapt_state,
                history=history,adaptation_history=adapt_history,meta_spent=meta_seconds,
                adaptation_spent=time.perf_counter()-started,rng=rng_state(rng)))
            if complete:notify(dict(phase='adaptation',direction=k,updates=counts[k],iid=adapt_state['last_iid'],**metric))
    
    adaptation_seconds=time.perf_counter()-started
    for k in order:
        actor=actors.get(k,meta)
        torch.save(dict(version=cfg.version,config=cfg.to_dict(),direction=k,weight=refs[k].tolist(),model=actor.actor_state(),updates=counts[k]),root/f'direction_{k:02d}.pt')
    covered=sorted({h['iid'] for h in history if h['steps']>0})
    return dict(version=cfg.version,method=method,meta_seconds=meta_seconds,adaptation_seconds=adaptation_seconds,
        meta_batches=len(history),optimized_fit_ids=covered,missing_fit_ids=sorted(set(split['fit_ids'])-set(covered)),
        direction_updates=counts,direction_spent=spent,adaptation_initializations=initializations,
        adaptation_next_iid=adapt_state['next_iid'],direction_quota_seconds=quotas,fit_coverage_passed=len(covered)==len(split['fit_ids']),
        adaptation_coverage_passed=all(counts.values()),history=history,adaptation_history=adapt_history,
        final_partial_meta_round_updates=partial_updates,
        meta_completed_search_cycles=state.get('completed_cycles',0),adapt_completed_search_cycles=adapt_state.get('completed_cycles',0),
        meta_traversal_only_steps=state.get('traversal_only_steps',0),adapt_traversal_only_steps=adapt_state.get('traversal_only_steps',0))
