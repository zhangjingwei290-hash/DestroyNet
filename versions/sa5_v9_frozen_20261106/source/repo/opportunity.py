"""Causal EMA in the current scalar coordinates and scale-balanced PPO helpers."""
from collections import deque
import numpy as np
import torch

class OpportunityEMA:
    def __init__(self, window=32):
        if window<1: raise ValueError('Positive EMA window required')
        self.window=window;self.alpha=2./(window+1);self.history=deque(maxlen=window)
    def value(self,score):
        # Re-evaluate saved objective pairs in exactly this batch's scalar context.
        value=0.
        for sessions,seconds in self.history:
            gains=[max(0.,float(score(s['before'],s['weight'])-score(s['after'],s['weight']))) for s in sessions]
            value=(1-self.alpha)*value+self.alpha*float(np.mean(gains))/seconds
        return value
    def observe(self,sessions,seconds):
        if not np.isfinite(seconds) or seconds<=0:raise ValueError('Nonpositive wall time')
        if sessions:self.history.append((sessions,seconds))

def balanced_advantages(target,value,scales):
    if set(scales)!={50,150,250}:raise ValueError('Collect all three scales before any update')
    adv=(target-value).detach();weights=torch.empty_like(adv)
    for n in (50,150,250):
        ids=torch.tensor([i for i,k in enumerate(scales) if k==n],device=adv.device)
        a=adv[ids];adv[ids]=(a-a.mean())/(a.std(unbiased=False)+1e-8)
        weights[ids]=1./(3*len(ids))
    return adv,weights
