"""Synthetic regression checks only; no FIT/development search or training run."""
import copy,importlib.util,json,sys,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from solver.config import Config,RATIOS,q_values,directions
from solver.data import project_api
CFG=Config(device='cpu',local_seconds=.05,repair_seconds=.05,solve_seconds_by_n={50:5.,150:5.,250:5.})
API=project_api(CFG.project_root)
from src.core.protocol import VRPTWInstance,Route,Solution
from solver.repair import repair,_score
from solver.environment import Environment,Adaptive
from solver.model import Policy
from solver.training import balanced_fit_ids,group_advantages,collect_closed
from solver.outer import Search


def fixture(n=12,seed=7,nonmetric=False):
    rng=np.random.default_rng(seed);coords=rng.uniform(0,10,(n,2));xy=np.vstack(([0,0],coords))
    distance=np.linalg.norm(xy[:,None]-xy[None,:],axis=-1)
    if nonmetric:
        distance=rng.uniform(.1,10,(n+1,n+1));np.fill_diagonal(distance,0)
    inst=VRPTWInstance('synthetic',n,5.,100.,(0.,1000.),coords,np.ones(n),
        np.column_stack((rng.uniform(0,10,n),rng.uniform(30,80,n))),np.ones(n),distance,distance.copy())
    routes=[list(range(i+1,min(i+4,n)+1)) for i in range(0,n,4)]
    return inst,routes

class Pool:
    def __init__(self,n=12):
        self.inst,self.routes=fixture(n);self.api=API
    def load(self,iid):return self.inst,dict(source_solution=self.routes),dict(scale=[10.,100.,1000.,1000.,1000.],offset=[0.]*5,reference=[2.]*5),q_values(self.inst.n_customers)

def observation(n=12):
    env=Environment(Pool(n),'synthetic',CFG,np.full(5,.2))
    return env,env.observation(torch.zeros(1,128),'greedy',Adaptive(10),5.,5.,.05,.05)

