"""Frozen probability and representation checks on stored real observations."""
from probe import *

@torch.no_grad()
def representation(m, obs):
    enc=m.encoder;n=len(obs['x']);x=obs['x'].to('cuda')
    en=obs['edge_index_n'].to('cuda');e0=obs['edge_index_r0'].to('cuda');e1=obs['edge_index_r1'].to('cuda')
    def spread(v):
        v=v.reshape(n,-1)[1:]
        return dict(rms_node_variation=float(((v-v.mean(0))**2).mean().sqrt()),
                    rms_activation=float(v.square().mean().sqrt()))
    vals={'input':spread(x)}
    a=enc.gat_0(x,en);vals['gat0']=spread(a)
    b=enc.gat_1(a,e1);c=enc.gat_1(b,e1);d=enc.gat_n(c,en)+a
    vals['first_route_block']=spread(d)
    e=enc.gat_2(d,e0);f=enc.gat_2(e,e0);g=enc.gat_n(f,en)+d
    vals['pre_GRU_gate']=spread(g)
    _,nodes,ctx,glob=m.encode_batch([obs]);v=nodes[0]
    vals['post_GRU_gate']=dict(rms_node_variation=float(((v-v.mean(0))**2).mean().sqrt()),rms_activation=float(v.square().mean().sqrt()))
    return vals

def main():
    cfg=Config(**read(SOURCE/'config.json'));pool,split=setup(cfg)
    torch.manual_seed(cfg.seed);torch.cuda.manual_seed_all(cfg.seed)
    initial=Policy(cfg).cuda().eval()
    models={'initial_seed_reconstruction':initial}
    ck=torch.load(SOURCE/'ppo/meta_complete.pt',map_location='cpu',weights_only=False)
    meta=Policy(cfg).cuda();meta.load_state_dict(ck['meta']);models['meta']=meta.eval()
    for k in (0,1,4,15):
        m=Policy(cfg).cuda();payload=torch.load(SOURCE/f'ppo/directions/direction_{k:02d}.pt',map_location='cpu',weights_only=False)
        m.load_state_dict(payload['model']);models[f'deployed_{k}']=m.eval()
    selected={};total=0
    for p in [SOURCE/'ppo/unoptimized_tail.pt']+sorted((SOURCE/'ppo').glob('adapt_tail*.pt')):
        data=torch.load(p,map_location='cpu',weights_only=False)
        for ei,ep in enumerate(data['episodes']):
            for ti,t in enumerate(ep):
                o=t['observation'];n=len(o['x'])-1;total+=1
                # At most 12 actual observations per size, retaining both nonzero and zero history.
                key=(n, bool(o['state_'].abs().sum()>0))
                bucket=selected.setdefault(key,[])
                if len(bucket)<6:bucket.append((o,dict(file=str(p),episode=ei,transition=ti,n=n,hidden_norm=float(o['state_'].norm()))))
    rows=[];reps=[]
    with torch.no_grad():
        for group in selected.values():
            for o,identity in group:
                q=min(o['q_values'],key=lambda q:abs(q/identity['n']-.05))
                for name,m in models.items():
                    qp,al,aa,bb=distributions(m,o,q)
                    oq=qp.cpu().numpy();oa=al.softmax(-1).cpu().numpy()
                    changed=copy.deepcopy(o);changed['context'][:5]=torch.tensor([0.,0.,0.,0.,1.])
                    q2,a2,_,_=distributions(m,changed,q)
                    rows.append(dict(**identity,model=name,q_entropy_ratio=float(Categorical(probs=qp).entropy()/np.log(len(qp))),
                        anchor_entropy_ratio=float(Categorical(logits=al).entropy()/np.log(len(al))),
                        anchor_logit_range=float(al.max()-al.min()),anchor_tv_uniform=tv(oa,np.ones(len(oa))/len(oa)),
                        q_tv_uniform=tv(oq,np.ones(len(oq))/len(oq)),
                        preference_to_tardy_q_tv=tv(oq,q2.cpu().numpy()),preference_to_tardy_anchor_tv=tv(oa,a2.softmax(-1).cpu().numpy())))
                reps.append(dict(**identity,representation=representation(models['deployed_15'],o)))
    parameter_changes={}
    init=initial.state_dict()
    for name,m in models.items():
        parameter_changes[name]={}
        for prefix in ('encoder.','context.','q_body.','location.','heads.0.q.','heads.0.anchor.','heads.0.alpha.','heads.0.beta.'):
            pairs=[(v,init[k]) for k,v in m.state_dict().items() if k.startswith(prefix)]
            diff=sum(float((a-b).square().sum()) for a,b in pairs)**.5
            norm=sum(float(b.square().sum()) for _,b in pairs)**.5
            parameter_changes[name][prefix]=dict(l2_change=diff,relative_l2=diff/max(norm,1e-12))
    save('real_history_check.json',dict(scope='Actual saved training observations, including nonzero recurrent states; frozen models tested on the SAME stored history. Not exact deployment replay; initial model receives trained-policy history as a diagnostic control.',
        available_transitions=total,selected_observations=len(reps),rows=rows,representations=reps,parameter_changes=parameter_changes))
    print('REAL_HISTORY_CHECK_COMPLETE',len(reps),flush=True)

if __name__=='__main__':main()
