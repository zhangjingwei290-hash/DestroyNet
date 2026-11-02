"""Net-return four-replica credit and cached-state PPO; GPU gradients only."""
import time,copy
import numpy as np
import torch

def group_advantages(episodes):
    groups={}
    for ep in episodes:
        if not ep or not ep[-1]['terminal'] or ep[-1].get('truncated') or ep[-1]['diagnostics']['ending_reason']!='local_budget':raise ValueError('Require complete same-start2.5s groups')
        if any(abs(t['diagnostics']['local_budget']-2.5)>1e-8 for t in ep):raise ValueError('Unequal window')
        groups.setdefault(ep[0]['start_group'],[]).append(ep)
    advantages={};weights={}
    for key,group in groups.items():
        if len(group)!=4 or any(ep[0]['source_routes_at_start']!=group[0][0]['source_routes_at_start'] for ep in group):raise ValueError('Different source routes or replica count')
        positive=[float(t['reward']) for ep in group for t in ep if t['reward']>0]
        rbar=float(np.mean(positive)) if positive else 0.
        returns=np.asarray([ep[0]['diagnostics']['before_score']-ep[-1]['diagnostics']['current_score']+.15*sum(bool(t['diagnostics']['new_nd_entered']) for t in ep)*rbar for ep in group])
        traj=(returns-returns.mean())/(returns.std()+1e-8)
        Gs=[]
        for ep in group:
            rewards=np.asarray([float(t['reward']) for t in ep]);G=np.zeros(len(ep))
            for k in range(8):
                if k<len(ep):G[:len(ep)-k]+=.85**k*rewards[k:]
            Gs.append(G)
        joined=np.concatenate(Gs);average=joined.mean();sd=joined.std()
        for j,(ep,G) in enumerate(zip(group,Gs)):
            step=(G-average)/(sd+1e-8)
            for i,t in enumerate(ep):
                d=t['diagnostics'];candidate=d.get('candidate_objectives')
                fewer=candidate is not None and candidate[0]<d['before_objectives'][0]
                if (t['stagnation']>=4 and t['reward']>0) or fewer:step[i]+=.75
            advantages[id(ep)]=.60*traj[j]+.40*step
            weights[id(ep)]=np.full(len(ep),.25/len(ep))
    return groups,advantages,weights

def optimizer_for(policy):
    fast=[];slow=[]
    for name,p in policy.named_parameters():
        q=name.startswith(('q_body.','q_film.')) or (name.startswith('heads.') and ('.q_mu.' in name or '.q_kappa.' in name or '.mode_logits.' in name))
        node=name.startswith(('context.','q_history_norm.','film.')) or (name.startswith('heads.') and ('.node.' in name or name.endswith(('.physical_bilinear','.route_bilinear','.route_score_scale'))))
        p.requires_grad_(q or node)
        if q:fast.append(p)
        if node:slow.append(p)
    assert fast and slow
    return torch.optim.Adam([{'params':fast,'lr':.003},{'params':slow,'lr':.001}])

def frozen_state(policy):return {k:v.detach().clone() for k,v in policy.state_dict().items() if k.startswith('encoder.')}

def assert_frozen(policy,reference):
    assert all(torch.equal(v,policy.state_dict()[k]) for k,v in reference.items()),'Frozen backbone/context changed'

def validate_action(t):
    from stage2_structured import q_for_mode,route_removal
    a=t['action'];mode=a['mode'];n=t['n'];routes=t['routes'];size,q_condition=q_for_mode(mode,a['u'],n,len(routes))
    assert a['q_condition']==q_condition and a['q']==len(a['removed']) and len(set(a['removed']))==a['q']
    assert all(1<=c<=n for c in a['removed'])
    if mode==0:
        assert a['q']==size and a['removed'][0]==a['anchor'] and a['micro_selected']==a['removed'][1:]
        assert a['anchor'] not in a['anchor_candidates'] and 0 not in a['anchor_candidates']
        assert set(a['removed'][1:])<=set(a['anchor_candidates']) and a['effective_selection_steps']==size
    elif mode==1:assert a['q']==size and a['micro_selected']==a['removed'] and a['effective_selection_steps']==size
    else:
        assert len(a['selected_routes'])==len(set(a['selected_routes']))==size
        expected,cap,over=route_removal(routes,a['selected_routes'],t['logits'],n)
        assert expected==a['removed'] and cap==a['q_cap'] and over==a['first_route_exceeds_cap'] and a['effective_selection_steps']==size
        assert set(routes[a['selected_routes'][0]])<=set(a['removed'])

def update(policy,opt,episodes,cfg,deadline,frozen):
    groups,advs,ws=group_advantages(episodes);samples=[]
    for ep in episodes:
        for t,a,w in zip(ep,advs[id(ep)],ws[id(ep)]):validate_action(t);samples.append((t,float(a),float(w)/len(groups)))
    buckets={}
    for item in samples:buckets.setdefault(item[0]['n'],[]).append(item)
    tracked=('context.','q_history_norm.','film.','node','physical_bilinear','route_bilinear')
    def family(name):
        if name.startswith('context.'):return 'context'
        if name.startswith('q_history_norm.'):return 'q_history_norm'
        if name.startswith('film.'):return 'film'
        if '.node.' in name:return 'node'
        if name.endswith('.physical_bilinear'):return 'physical_bilinear'
        if name.endswith('.route_bilinear'):return 'route_bilinear'
        return None
    before={k:v.detach().clone() for k,v in policy.state_dict().items()};old_opt=copy.deepcopy(opt.state_dict())
    metrics=dict(family_gradient_norms={k:0. for k in ('context','q_history_norm','film','node','physical_bilinear','route_bilinear')},optimizer_steps=0,transitions=len(samples),groups=len(groups),trajectories=len(episodes),loss=0.,initial_ratio_error=0.,initial_macro_ratio_error=0.,initial_micro_ratio_error=0.,sample_replay_alpha_error=0.,sample_replay_logit_error=0.,sample_replay_logprob_error=0.,gradient_norms={},source='2.5s_net_return_plus_ND_eight_step_credit',density='structured_mode_beta_and_selected_prefix_exact')
    try:
        for epoch in range(cfg.ppo_epochs):
            opt.zero_grad(set_to_none=True)
            for bucket in buckets.values():
                for offset in range(0,len(bucket),cfg.microbatch):
                    if time.perf_counter()>=deadline-1.:raise TimeoutError('Training cutoff inside PPO batch')
                    chunk=bucket[offset:offset+cfg.microbatch];states=[t for t,_,_ in chunk]
                    result=policy.replay_terms(states);macro=result['logprob_macro'];micro=result['logprob_micro']
                    old_macro=macro.new_tensor([t['logprob_macro'] for t in states]);old_micro=micro.new_tensor([t['logprob_micro'] for t in states])
                    effective=micro.new_tensor([t['effective_selection_steps'] for t in states])
                    r_macro=(macro-old_macro).exp();r_micro=((micro-old_micro)/effective).exp()
                    if epoch==0:
                        me=float((r_macro.detach()-1).abs().max());mi=float((r_micro.detach()-1).abs().max())
                        metrics['initial_macro_ratio_error']=max(metrics['initial_macro_ratio_error'],me);metrics['initial_micro_ratio_error']=max(metrics['initial_micro_ratio_error'],mi)
                        metrics['initial_ratio_error']=max(metrics['initial_ratio_error'],me,mi)
                        ae=max(float((result[name].detach()-macro.new_tensor([t[name] for t in states])).abs().max()) for name in ('alpha','beta'))
                        le=float((result['logits'].detach()-torch.as_tensor(np.stack([t['logits'] for t in states]),device=policy.device)).abs().max())
                        pe=float((result['mode_probabilities'].detach()-torch.as_tensor(np.stack([t['mode_probabilities'] for t in states]),device=policy.device)).abs().max())
                        for j,t in enumerate(states):
                            if t['mode']==2:le=max(le,float((result['route_logits'][j,:len(t['routes'])].detach()-macro.new_tensor(t['route_logits'])).abs().max()))
                        metrics['sample_replay_alpha_error']=max(metrics['sample_replay_alpha_error'],ae);metrics['sample_replay_logit_error']=max(metrics['sample_replay_logit_error'],le)
                        metrics['sample_replay_probability_error']=max(metrics.get('sample_replay_probability_error',0.),pe)
                        if me>1e-4 or mi>1e-4 or ae>=5e-5 or le>=5e-5 or pe>=5e-5:
                            from pathlib import Path
                            diagnostic=dict(macro_ratio_error=me,micro_ratio_error=mi,alpha_beta_error=ae,logit_error=le,probability_error=pe,epoch=epoch,offset=offset)
                            target=Path(cfg.study_root)/'engineering/replay_mismatch_dumps';target.mkdir(parents=True,exist_ok=True)
                            stamp=time.time_ns();path=target/f'mismatch_{stamp}.pt'
                            torch.save(dict(states=states,episodes=episodes,model=before,optimizer=old_opt,metrics=metrics,diagnostic=diagnostic),path)
                            raise RuntimeError('Structured macro/micro replay mismatch '+str(diagnostic)+' dump='+str(path))
                    adv=macro.new_tensor([a for _,a,_ in chunk]);weight=macro.new_tensor([w for _,_,w in chunk])
                    def clipped(ratio):return torch.minimum(ratio*adv,ratio.clamp(1-cfg.clip_epsilon,1+cfg.clip_epsilon)*adv)
                    loss=((-clipped(r_macro)-clipped(r_micro)-cfg.entropy_weight*result['entropy'])*weight).sum()
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite PPO loss')
                    loss.backward();metrics['loss']+=float(loss.detach())/cfg.ppo_epochs
            grads={name:p.grad for name,p in policy.named_parameters() if p.grad is not None}
            if not grads or any(not torch.isfinite(g).all() for g in grads.values()):raise FloatingPointError('Missing/nonfinite gradients')
            metrics['gradient_norms']={name:float(g.norm()) for name,g in grads.items() if '.q_mu.' in name or '.mode_logits.' in name or '.route_' in name or name.startswith(('context.0','q_history_norm.','q_body.0','q_film.0','film.0'))}
            for name,g in grads.items():
                key=family(name)
                if key is not None:metrics['family_gradient_norms'][key]+=float(g.detach().square().sum())
            if time.perf_counter()>=deadline-1.:raise TimeoutError('Training cutoff before optimizer')
            opt.step();torch.cuda.synchronize();metrics['optimizer_steps']+=1
            assert_frozen(policy,frozen)
            if time.perf_counter()>=deadline:raise TimeoutError('Training cutoff after optimizer; rollback')
    except TimeoutError:
        policy.load_state_dict(before);opt.load_state_dict(old_opt);opt.zero_grad(set_to_none=True)
        metrics['discarded_optimizer_steps']=metrics['optimizer_steps'];metrics['optimizer_steps']=0;metrics['cutoff_rolled_back']=True
    except BaseException:
        policy.load_state_dict(before);opt.load_state_dict(old_opt);opt.zero_grad(set_to_none=True)
        raise
    metrics['family_parameter_delta_norms']={key:float(sum((policy.state_dict()[name]-old).detach().square().sum() for name,old in before.items() if family(name)==key).sqrt()) for key in metrics['family_gradient_norms']}
    metrics['family_gradient_norms']={key:value**.5 for key,value in metrics['family_gradient_norms'].items()}
    return metrics
