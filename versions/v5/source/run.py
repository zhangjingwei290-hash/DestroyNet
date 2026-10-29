"""Reviewable V4 entry. Inspect does no training; budgets have no defaults."""
import argparse,json
from pathlib import Path
from dataclasses import fields
from solver.config import Config,RATIOS,q_values


def write_json(path,value):
    import numpy as np
    def convert(obj):
        if isinstance(obj,np.ndarray):return obj.tolist()
        if isinstance(obj,np.generic):return obj.item()
        raise TypeError(type(obj).__name__)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=convert),encoding='utf-8')


def load_config(path=None):
    cfg=Config()
    if path:
        values=json.loads(Path(path).read_text(encoding='utf-8'))
        known={f.name for f in fields(cfg)}
        if set(values)-known:raise ValueError(f'Unknown settings: {sorted(set(values)-known)}')
        for key,value in values.items():setattr(cfg,key,value)
    cfg.validate()
    return cfg


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=('inspect','train','validate'),default='inspect')
    parser.add_argument('--config',help='Explicit settings/budgets JSON; required for train/validate')
    parser.add_argument('--method',choices=('ppo','reinforce'),default='ppo')
    parser.add_argument('--root',help='New output directory')
    parser.add_argument('--checkpoint-root',help='Reviewed V4 direction checkpoints for validation')
    args=parser.parse_args();cfg=load_config(args.config)
    if args.mode=='inspect':
        print(json.dumps(dict(config=cfg.to_dict(),ratios=RATIOS,q_support={n:q_values(n) for n in (50,150,250)},
            decoder='stochastic_q_then_direct_without_replacement; full_removal_q_only',training_started=False),indent=2))
        return
    if not args.config or not args.root:parser.error('--config and --root required; no experiment budgets are assumed')
    cfg.validate(require_budget=True)
    import torch
    if cfg.device!='cuda' or not torch.cuda.is_available():raise RuntimeError('Neural train/validate requires CUDA')
    torch.set_num_threads(cfg.cpu_threads)
    from solver.data import TrainPool
    pool=TrainPool(cfg)  # initializes physical src imports before outer/repair
    from solver.training import train_method
    from solver.model import Policy
    from solver.outer import Search
    if args.mode=='train':
        report=train_method(pool,pool.split,cfg,args.method,args.root,
            lambda p:print(json.dumps(p,ensure_ascii=False,default=str),flush=True))
        write_json(Path(args.root)/'training_report.json',report)
    else:
        if not args.checkpoint_root:parser.error('--checkpoint-root required for validation')
        root=Path(args.root);root.mkdir(parents=True,exist_ok=False)
        policies=[]
        for k in range(16):
            state=torch.load(Path(args.checkpoint_root)/f'direction_{k:02d}.pt',map_location=cfg.device,weights_only=False)
            if state['version']!=cfg.version or state['direction']!=k:raise ValueError('Wrong V4 checkpoint/direction')
            policy=Policy(cfg).to(cfg.device);policy.load_state_dict(state['model']);policy.eval();policies.append(policy)
        reports=[]
        for iid in pool.split['development_ids']:
            # Same act() as collection: q is always selected by q_body.
            torch.manual_seed(cfg.seed);torch.cuda.manual_seed_all(cfg.seed)
            search=Search(pool,iid,cfg,cfg.seed,policies).run()
            report=search.report();write_json(root/f'{iid}.json',report);reports.append(report)
        write_json(root/'validation.json',dict(config=cfg.to_dict(),reports=reports))

if __name__=='__main__':main()
