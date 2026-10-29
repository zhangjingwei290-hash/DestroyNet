"""V4 architecture; all runtime experiment budgets are supplied after review."""
from dataclasses import dataclass,asdict
from itertools import combinations
from pathlib import Path
import math
import numpy as np
RATIOS=(.01,.02,.05,.10,.20,.30,.40,.60,.80,1.)
REPAIRERS=("greedy","regret2","regret3","noisy_greedy")
CONTEXT = tuple([f"weight_{i}" for i in range(5)] + [f"objective_{i}" for i in range(5)]
    + ["global_remaining_seconds/300", "global_used_fraction", "local_remaining_seconds/300",
       "local_remaining_fraction", "stagnation_fraction", "last_accepted", "last_reward",
       "archive_fraction", "max_delay/horizon", "N/250", "max_q/N"]
    + [f"repair_is_{r}" for r in REPAIRERS]
    + [f"repair_{metric}_{r}" for metric in ("probability", "recent_seconds/300", "recent_success_rate",
                                            "recent_improvement", "count/1000") for r in REPAIRERS])


def directions():
    result = list(np.eye(5))
    for i, j in combinations(range(5), 2):
        v = np.zeros(5); v[[i, j]] = .5; result.append(v)
    return np.asarray(result + [np.full(5, .2)])


def q_values(n):
    if n<1:raise ValueError("N must be positive")
    # Keep ten categorical slots, including aliases for small synthetic graphs.
    return tuple(max(1,min(n,math.ceil(n*r))) for r in RATIOS)

@dataclass
class Config:
    version:str="v4_3_continuous_cudagraph_20261025"
    project_root:str=r"D:\WorkSpace\versions\version0.7"
    study_root:str=r"D:\WorkSpace\versions\version0.7\artifacts\hemt_feature_stage_v3_20261013"
    split_manifest:str=str(Path(__file__).resolve().parents[1]/"split.json")
    device:str="cuda"
    seed:int=20261024
    hidden_dim:int=128
    near_neighbors:int=10
    episode_steps:int=150
    learning_rate:float=3e-3
    entropy_weight:float=.001
    clip_epsilon:float=.2
    ppo_epochs:int=2
    task_count:int=5
    replicas:int=4
    inner_updates:int=4
    microbatch:int=8
    meta_step_size:float=1.
    cpu_threads:int=1
    independent_iid_batch:int=1
    cpu_workers:int=1
    meta_target_epochs:int|None=None
    meta_seconds:float|None=None
    adaptation_seconds:float|None=None
    repair_seconds:float|None=None
    local_seconds:float|None=None
    solve_seconds_by_n:dict|None=None
    solve_seconds_by_instance:dict|None=None

    def validate(self,require_budget=False):
        if self.task_count!=5 or self.replicas!=4:raise ValueError("Five cyclic tasks / four same-start replicas required")
        if self.hidden_dim!=128 or self.cpu_threads!=1:raise ValueError("128D encoder and single-thread repair")
        if not 0<self.meta_step_size<=1 or self.learning_rate<=0:raise ValueError("Invalid optimizer configuration")
        if any(type(v) is not int or v<1 for v in (self.independent_iid_batch,self.cpu_workers)):
            raise ValueError('Positive integer IID batch/CPU workers required')
        if self.meta_target_epochs is not None and (type(self.meta_target_epochs) is not int or self.meta_target_epochs<1):
            raise ValueError('Positive integer epoch target required')
        if min(self.inner_updates,self.microbatch,self.ppo_epochs)<1:raise ValueError("Invalid batch configuration")
        for name in ('meta_seconds','adaptation_seconds','repair_seconds','local_seconds'):
            value=getattr(self,name)
            if value is None:
                if require_budget:raise ValueError(f"Budget not set: {name}")
            elif not math.isfinite(value) or value<=0:raise ValueError(f"Invalid budget: {name}")
        if self.solve_seconds_by_n is not None and self.solve_seconds_by_instance is not None:
            raise ValueError("Choose instance or size search budgets, not both")
        if require_budget and self.solve_seconds_by_n is None and self.solve_seconds_by_instance is None:
            raise ValueError("Search budgets not set")
        if self.solve_seconds_by_n is not None:
            self.solve_seconds_by_n={int(k):float(v) for k,v in self.solve_seconds_by_n.items()}
            if set(self.solve_seconds_by_n)!={50,150,250} or any(not math.isfinite(v) or v<=0 for v in self.solve_seconds_by_n.values()):raise ValueError("Specify positive search seconds for N=50/150/250")
        if self.solve_seconds_by_instance is not None:
            self.solve_seconds_by_instance={str(k):float(v) for k,v in self.solve_seconds_by_instance.items()}
            required={f'{n}-{d}-{tw}' for n in (50,150,250) for d in range(3) for tw in range(5)}
            if set(self.solve_seconds_by_instance)!=required or any(not math.isfinite(v) or v<=0 for v in self.solve_seconds_by_instance.values()):
                raise ValueError("Specify positive search seconds for all45 N-d-tw cells")

    def to_dict(self):return asdict(self)

def search_budget(n,cfg,iid=None,row=None):
    """Bind a real IID or explicit synthetic cell to its reviewed wall horizon.

    Synthetic replicas inherit their (N, capacity class, time-window class) cell.
    Do not infer that cell from old C/R/RC names or silently use a size mean.
    """
    per_instance=cfg.solve_seconds_by_instance
    if per_instance is not None:
        if cfg.solve_seconds_by_n is not None:
            raise ValueError("Ambiguous search budgets")
        key=iid if iid in per_instance else None
        cell=None if row is None else row.get('cell')
        if cell is not None:
            if any(type(cell.get(k)) is not int for k in ('n','d','tw')):
                raise ValueError("Synthetic budget cell requires integer n/d/tw")
            cell_key=f"{cell['n']}-{cell['d']}-{cell['tw']}"
            if cell['n']!=int(n) or (key is not None and key!=cell_key):
                raise ValueError("Instance identity and search budget cell differ")
            key=cell_key
        if key is None or key not in per_instance or int(key.split('-')[0])!=int(n):
            raise ValueError(f"No reviewed per-instance budget for {iid}; require real IID or explicit synthetic cell")
        value=float(per_instance[key])
    else:
        if cfg.solve_seconds_by_n is None:raise ValueError("Search budget pending user decision")
        value=float(cfg.solve_seconds_by_n[int(n)])
    if not math.isfinite(value) or value<=0:raise ValueError("Invalid search horizon")
    return value
