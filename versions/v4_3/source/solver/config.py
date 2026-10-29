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
    meta_seconds:float|None=None
    adaptation_seconds:float|None=None
    repair_seconds:float|None=None
    local_seconds:float|None=None
    solve_seconds_by_n:dict|None=None

    def validate(self,require_budget=False):
        if self.task_count!=5 or self.replicas!=4:raise ValueError("Five cyclic tasks / four same-start replicas required")
        if self.hidden_dim!=128 or self.cpu_threads!=1:raise ValueError("128D encoder and single-thread repair")
        if not 0<self.meta_step_size<=1 or self.learning_rate<=0:raise ValueError("Invalid optimizer configuration")
        if min(self.inner_updates,self.microbatch,self.ppo_epochs)<1:raise ValueError("Invalid batch configuration")
        for name in ('meta_seconds','adaptation_seconds','repair_seconds','local_seconds'):
            value=getattr(self,name)
            if value is None:
                if require_budget:raise ValueError(f"Budget not set: {name}")
            elif not math.isfinite(value) or value<=0:raise ValueError(f"Invalid budget: {name}")
        if require_budget and self.solve_seconds_by_n is None:raise ValueError("Search budgets not set")
        if self.solve_seconds_by_n is not None:
            self.solve_seconds_by_n={int(k):float(v) for k,v in self.solve_seconds_by_n.items()}
            if set(self.solve_seconds_by_n)!={50,150,250} or any(not math.isfinite(v) or v<=0 for v in self.solve_seconds_by_n.values()):raise ValueError("Specify positive search seconds for N=50/150/250")

    def to_dict(self):return asdict(self)

def search_budget(n,cfg):
    if cfg.solve_seconds_by_n is None:raise ValueError("Search budget pending user decision")
    return float(cfg.solve_seconds_by_n[int(n)])
