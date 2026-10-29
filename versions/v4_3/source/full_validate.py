"""Authorized full-budget PPO closed-loop run and paired development evaluation."""
import argparse,os,time,traceback,json,sys
from pathlib import Path
import numpy as np
import torch
from run import load_config,write_json
from solver.data import TrainPool



def log_event(root,message):
    # Own short-lived file handle; never print progress to a launcher/PTY pipe.
    with (Path(root)/'worker.log').open('a',encoding='utf-8') as handle:
        handle.write(json.dumps(message,ensure_ascii=False,allow_nan=False)+'\n')

def main(root,search_seeds=1):
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    sys.stdout=(root/'worker_stdout.log').open('a',encoding='utf-8',buffering=1)
    sys.stderr=sys.stdout
    cfg=load_config(Path(__file__).with_name('full_config.json'));cfg.validate(require_budget=True)
    import shutil
    write_json(root/'full_config.json',cfg.to_dict())
    snapshot=root/'source';snapshot.mkdir()
    for name in ('solver','tests'):
        shutil.copytree(Path(__file__).parent/name,snapshot/name,ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('run.py','full_validate.py','make_full_report.py','make_stage_report.py','full_config.json','split.json'):
        shutil.copy2(Path(__file__).with_name(name),snapshot/name)
    torch.set_num_threads(1)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    pool=TrainPool(cfg)
    from solver.training import train_method
    from solver.model import Policy
    from solver.outer import Search
    import moocore
    started=time.perf_counter();hard_deadline=float("inf")
    write_json(root/'process.json',dict(pid=os.getpid(),config=cfg.to_dict(),start_unix=time.time(),training_budget_seconds=cfg.meta_seconds+cfg.adaptation_seconds,validation_search_budget_seconds=270.*search_seeds,validation_search_seeds=[cfg.seed+s for s in range(search_seeds)],collection_protocol="continued_three_stage_actual_search"))
    def notify(message):
        compact={k:v for k,v in message.items() if k not in ('gradient_norms','provenance')}
        compact['elapsed']=time.perf_counter()-started
        write_json(root/'progress.json',compact)
        log_event(root,compact)
    try:
        training=train_method(pool,pool.split,cfg,'ppo',root/'training',notify)
        write_json(root/'training'/'training_report.json',training)
        policies=[]
        for k in range(16):
            data=torch.load(root/'training'/f'direction_{k:02d}.pt',map_location='cuda',weights_only=False)
            model=Policy(cfg).cuda();model.load_state_dict(data['model']);model.eval();policies.append(model)
        frozen_states=[model.actor_state() for model in policies]
        reports=[];comparison=[]
        for seed_offset in range(search_seeds):
            search_seed=cfg.seed+seed_offset
            for iid in pool.split['development_ids']:
                pair={}
                for method in ('heuristic','ppo'):
                    budget=cfg.solve_seconds_by_n[pool.load(iid)[0].n_customers]
                    if hard_deadline-time.perf_counter()<budget+1:raise TimeoutError('Insufficient room for full validation search')
                    torch.manual_seed(search_seed);torch.cuda.manual_seed_all(search_seed);np.random.seed(search_seed)
                    search=Search(pool,iid,cfg,search_seed,None if method=='heuristic' else policies,resource_deadline=hard_deadline)
                    initial=np.asarray([p.F.copy() for p in search.algorithm.pop])
                    initial_hv=float(moocore.hypervolume((initial-search.base.offset)/search.base.scales,ref=np.asarray(search.base.box['reference'])))
                    search.run();report=search.report();report['method']=method;report['initial_hv']=initial_hv;report['search_seed']=search_seed
                    final=np.asarray([p['objectives'] for p in report['archive']])
                    initial_best=initial.min(0);final_best=final.min(0)
                    report['objective_component_minima']=dict(initial=initial_best.tolist(),final=final_best.tolist(),
                        improvement=(initial_best-final_best).tolist(),improvement_percent=np.divide(100*(initial_best-final_best),initial_best,out=np.zeros(5),where=initial_best!=0).tolist(),
                        zero_initial_objectives=np.flatnonzero(initial_best==0).tolist(),note='Per-objective archive minima; may be different solutions')
                    report['initial_population_objectives']=initial.tolist()
                    write_json(root/f'{iid}_seed{search_seed}_{method}.json',report);reports.append(report);pair[method]=report
                    notify(dict(phase='validation',iid=iid,search_seed=search_seed,method=method,hv=report['hv'],initial_hv=initial_hv,
                        actions=len(report['records']),timeouts=sum(r['timeout'] for r in report['records'])))
                h,p=pair['heuristic'],pair['ppo']
                if h['initial_population_objectives']!=p['initial_population_objectives']:raise ValueError('Paired initial populations differ')
                comparison.append(dict(iid=iid,search_seed=search_seed,initial_hv=p['initial_hv'],heuristic_hv=h['hv'],ppo_hv=p['hv'],
                    hv_gap_percent=100*(h['hv']-p['hv'])/h['hv'],ppo_hv_gain_percent=100*(p['hv']-p['initial_hv'])/p['initial_hv'],
                    heuristic_hv_gain_percent=100*(h['hv']-h['initial_hv'])/h['initial_hv'],
                    ppo_objective_minima=p['objective_component_minima'],heuristic_objective_minima=h['objective_component_minima']))
        if any(not torch.equal(value,policies[k].state_dict()[name]) for k,state in enumerate(frozen_states) for name,value in state.items()):
            raise RuntimeError('Frozen model parameters changed during validation')
        summary=dict(frozen_validation_weights_verified=True,config=cfg.to_dict(),wall_seconds=time.perf_counter()-started,comparison=comparison,
            mean_paired_hv_gap_percent=float(np.mean([r['hv_gap_percent'] for r in comparison])),
            training={k:v for k,v in training.items() if k not in ('history','adaptation_history')},
            search_seeds=[cfg.seed+s for s in range(search_seeds)],training_seed=cfg.seed,
            collection_protocol='continued_three_stage_actual_search',
            scope='35min meta + 25min adaptation; one training seed, five independent paired search seeds; nine development IIDs')
        for phase,key in (('meta','history'),('adaptation','adaptation_history')):
            hs=training[key];hist=np.zeros(10,dtype=int)
            for h in hs:
                for slot,count in h['q_slot_histogram'].items():hist[int(slot)]+=count
            count=sum(h['transitions'] for h in hs)
            summary[phase+'_sampling']=dict(batches=len(hs),transitions=count,optimizer_steps=sum(h['steps'] for h in hs),
                q_slot_counts=hist.tolist(),q_slot_percent=(100*hist/max(1,count)).tolist(),
                timeout_fraction=sum(h['transitions']*h['timeout_fraction'] for h in hs)/max(1,count),
                positive_reward_fraction=sum(h['transitions']*h['positive_reward_fraction'] for h in hs)/max(1,count),
                optimized_iids=sorted({h['iid'] for h in hs if h['steps']>0}))
        write_json(root/'summary.json',summary)
        from make_stage_report import main as report_main
        report_main(root)
        write_json(root/'complete.json',dict(completed=True,wall_seconds=summary['wall_seconds']))
        log_event(root,dict(phase='complete',mean_paired_hv_gap_percent=summary['mean_paired_hv_gap_percent']))
    except BaseException:
        (root/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    parser.add_argument('--search-seeds',type=int,default=1)
    args=parser.parse_args()
    if args.search_seeds<1:parser.error('search-seeds must be positive')
    main(args.root,args.search_seeds)
