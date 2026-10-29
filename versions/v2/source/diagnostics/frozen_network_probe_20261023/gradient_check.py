"""Gradient-only diagnostic at saved inner-policy boundary; no optimizer/updates."""
from probe import *
from moea_emnh_v2.training import returns

def main():
    cfg=Config(**read(SOURCE/'config.json'));torch.set_num_threads(1)
    ck=torch.load(SOURCE/'ppo/meta_complete.pt',map_location='cpu',weights_only=False)
    data=torch.load(SOURCE/'ppo/unoptimized_tail.pt',map_location='cpu',weights_only=False)
    m=Policy(cfg,tasks=5,critic=True).cuda();m.load_state_dict(ck['inner']);m.eval()
    before={k:v.detach().clone() for k,v in m.state_dict().items()}
    samples=[]
    for ep in data['episodes']:
        if not ep:continue
        for t,target in zip(ep,returns(ep)):
            samples.append((t,target))
    # Last incomplete collection is diagnostic only, never claimed optimized training data.
    rows=[]
    for n in sorted({len(t['observation']['x'])-1 for t,_ in samples}):
        sub=[(t,target) for t,target in samples if len(t['observation']['x'])-1==n][:16]
        ts=[t for t,_ in sub];targets=torch.tensor([r for _,r in sub],device='cuda')
        _,lp,value,entropy,_=m.terms([t['observation'] for t in ts],[t['task'] for t in ts],[t['action'] for t in ts])
        old=torch.tensor([t['logprob'] for t in ts],device='cuda')
        oldv=torch.tensor([t['value'] for t in ts],device='cuda');adv=targets-oldv
        ratio=(lp-old).exp();ratio_error=float((ratio-1).abs().max())
        if ratio_error>1e-3:
            rows.append(dict(n=n,samples=len(sub),matched_policy=False,ratio_max_error=ratio_error));continue
        actor=-torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv).mean()
        critic=.5*(value-targets).square().mean();exploration=-cfg.entropy_weight*entropy.mean()
        params=list(m.named_parameters());vectors={};norms={}
        for label,loss in [('actor',actor),('critic',critic),('entropy',exploration)]:
            grads=torch.autograd.grad(loss,[p for _,p in params],retain_graph=True,allow_unused=True)
            vectors[label]=grads;norms[label]={}
            for prefix in ['all','encoder.','context.','q_body.','location.','heads.','critics.']:
                sq=sum(float(g.square().sum()) for (name,p),g in zip(params,grads) if g is not None and (prefix=='all' or name.startswith(prefix)))
                norms[label][prefix]=sq**.5
        cos={}
        for other in ['critic','entropy']:
            dot=sum(float(a.mul(b).sum()) for a,b in zip(vectors['actor'],vectors[other]) if a is not None and b is not None)
            cos[other]=dot/max(norms['actor']['all']*norms[other]['all'],1e-20)
        variance=float(targets.var(unbiased=False))
        ev=None if variance<=1e-20 else 1-float((targets-value).var(unbiased=False))/variance
        rows.append(dict(n=n,samples=len(sub),matched_policy=True,ratio_max_error=ratio_error,
            mean_abs_advantage=float(adv.abs().mean()),mean_return=float(targets.mean()),return_std=variance**.5,
            value_mse=float((value-targets).square().mean()),explained_variance=ev,
            loss_values=dict(actor=float(actor),critic=float(critic),entropy=float(exploration)),
            gradient_norms=norms,actor_gradient_cosine=cos))
    assert all(torch.equal(v,before[k]) for k,v in m.state_dict().items())
    save('gradient_check.json',dict(scope='Last unoptimized meta collection; at most 16 transitions per available size; gradients at matched saved inner model only. Equal transition weights within this diagnostic; not the full original task-weighted update. No optimizer, backward update or parameter changes.',
        rows=rows,weights_unchanged=True))
    print('GRADIENT_CHECK_COMPLETE',flush=True)

if __name__=='__main__':main()
