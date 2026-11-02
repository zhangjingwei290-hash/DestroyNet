"""FIT-only finite behavioral diagnostics; no checkpoint selection or extra updates."""
from stage2_common import ROOT,read,write,load_cfg,FitPool,sha,require_stage1
import time,math

def main():
    import psutil,numpy as np,torch
    from scipy.special import betainc
    from solver.model import Policy
    from solver.config import directions
    from solver.environment import Environment,Adaptive
    from stage2_cpu import CpuHead,struct_state
    from stage2_structured import softmax,route_context,interpolate,route_logits,route_removal,top_prefix,q_for_mode,MODE_NAMES
    from stage1_mechanisms import dynamic_coordinates,normalize_physical
    from stats import csv_write
    require_stage1();psutil.Process().cpu_affinity([14,15]);torch.set_num_threads(1);torch.set_num_interop_threads(1)
    report=read(ROOT/'stage2_training/training_report.json');cfg=load_cfg();pool=FitPool(cfg);refs=directions()
    ids=pool.manifest['fit_order'][:3];rows=[];hashes=[];rng=np.random.default_rng(20261024);policy=Policy(cfg).cuda().eval()
    for k in range(16):
        path=ROOT/'stage2_training/checkpoints'/f'direction_{k:02d}.pt';digest=sha(path);hashes.append(digest)
        ck=torch.load(path,map_location='cuda',weights_only=False);policy.load_state_dict(ck['model'],strict=True);cpu=CpuHead(policy)
        for iid in ids:
            env=Environment(pool,iid,cfg,refs[k]);n=env.instance.n_customers
            env.dynamic_ideal,env.dynamic_span=dynamic_coordinates([env.objectives],env.scales,env.offset+env.scales*np.asarray(env.box['reference']))
            obs=env.observation(torch.zeros((1,128)),'greedy',Adaptive(len(env.q_values)),100.,200.,5.,10.)
            cache=policy.refresh(obs,0,False);phys=obs['x'][1:,[18,17,19,12,13]].numpy();rc=route_context(env.instance,env.routes,phys)
            z=(env.objectives-env.dynamic_ideal)/env.dynamic_span;position=np.asarray([z.mean(),(refs[k]*z).max()],dtype=np.float32)
            for stag in (0,2,8):
                raw=obs['context'].numpy().copy();raw[14]=stag/10.
                logits,aa,bb,_=cpu.outputs(cache['hidden'],struct_state(phys),raw,n,position);probs=softmax(logits)
                for m in range(3):
                    a,b=float(aa[m]),float(bb[m])
                    if m<2:
                        base=2. if m==0 else 3.;upper=min(max(12,math.floor(.22*n)),45) if m==0 else min(max(10,math.floor(.35*n)),75)
                        qs=np.arange(int(base),upper+1,dtype=float);den=math.log(upper/base)
                        low=np.clip(np.log(np.maximum(base,qs-.5)/base)/den,0.,1.);high=np.clip(np.log((qs+.5)/base)/den,0.,1.);low[0]=0.;high[-1]=1.
                        mass=betainc(a,b,high)-betainc(a,b,low);assert abs(mass.sum()-1)<1e-9
                        eq=float(qs@mass);se=0.;definition='exact_discrete_Beta_integral'
                    else:
                        v=len(env.routes);km=min(v,max(1,min(4,math.floor(.25*v))));estimates=[];variances=[]
                        for kk in range(1,km+1):
                            mass=float(betainc(a,b,kk/km)-betainc(a,b,(kk-1)/km));qc=int(np.clip(np.rint(kk*n/v),2,n))
                            ell=interpolate(cache['values_lo'],cache['values_hi'],qc,n)+normalize_physical(phys)@cache['physical_weights']
                            rl=route_logits(ell,rc,refs[k],cpu.params['heads.0.route_bilinear'],cpu.params['heads.0.route_score_scale'])
                            sampled=np.asarray([len(route_removal(env.routes,top_prefix(rl,kk,rng),ell,n)[0]) for _ in range(128)])
                            estimates.append(mass*sampled.mean());variances.append(mass*mass*sampled.var(ddof=1)/128)
                        eq=float(sum(estimates));se=float(math.sqrt(sum(variances)));definition='exact_K_mass_with_128_route_prefix_MC_per_K'
                    rows.append(dict(direction=k,fit_iid=iid,n=n,stagnation=stag,mode=m,mode_name=MODE_NAMES[m],mode_probability=float(probs[m]),alpha=a,beta=b,
                        expected_q=eq,expected_q_ratio=eq/n,expected_q_standard_error=se,expectation_definition=definition,updates=ck['updates'],optimizer_steps=ck['optimizer_steps'],delta_q_mu_weight=report['q_mu_weight_delta_norm'][str(k)]))
        assert sha(path)==digest
    csv_write(ROOT/'stage2_training/q_behavior.csv',rows)
    result=dict(revision='stage2_r2_structured_modes_unified_scalar',all_behavior_targets_met=None,direction_diagnostics=[dict(direction=k,q_mu_changed=report['q_mu_weight_delta_norm'][str(k)]>0) for k in range(16)],
        table=rows,checkpoint_sha256=hashes,interpretation='Finite FIT-source diagnostic; conditional action densities exact; route-coherent expected q is a separately labelled Monte Carlo estimate, never used to select or retrain',
        actor_structural_gate=report['all_directions_updated'],training_coverage_target=report['all_direction_targets_met'] and report['meta_switch_gate'] and report['mature_quota_passed'],checked_unix=time.time())
    write(ROOT/'stage2_training/Q_GATE.json',result);print(dict(valid_updated_directions=report['all_directions_updated'],coverage_targets_met=result['training_coverage_target']),flush=True)
if __name__=='__main__':main()
