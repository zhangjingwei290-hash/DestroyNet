"""Predecision state group counts; empirical entropy in natural-log units."""
import math
from .config import RATIOS

REGIMES=('initial','refine','stalled')

def regime_name(last_accepted,stagnation):
    if stagnation<0:
        raise ValueError('Negative consecutive stagnation count')
    if bool(last_accepted):
        return 'refine'
    return 'initial' if int(stagnation)==0 else 'stalled'

def regime_id(last_accepted,stagnation):
    name=regime_name(last_accepted,stagnation)
    return 0 if name=='initial' else (1 if name=='refine' else 2)


def empty_q():return dict(count=0,q_ratio_sum=0.,q_slot_counts=[0]*len(RATIOS))


def empty_state_q():
    return dict(by_last_accepted={key:empty_q() for key in ('true','false')},
                by_stagnation={key:empty_q() for key in ('zero','one_to_three','four_or_more')},
                by_regime={key:empty_q() for key in REGIMES})


def state_labels(last_accepted,stagnation):
    if stagnation<0:raise ValueError('Negative consecutive stagnation count')
    return dict(by_last_accepted='true' if last_accepted else 'false',
                by_stagnation='zero' if stagnation==0 else 'one_to_three' if stagnation<=3 else 'four_or_more',
                by_regime=regime_name(last_accepted,stagnation))


def add_q(block,q,n,slot):
    block['count']+=1;block['q_ratio_sum']+=q/n;block['q_slot_counts'][slot]+=1


def finish_q(block):
    n=block['count'];counts=block['q_slot_counts']
    probabilities=[v/n for v in counts if v] if n else []
    return dict(**block,mean_q_over_n=block['q_ratio_sum']/n if n else None,
                q_slot_percent=[100*v/n if n else 0. for v in counts],
                effective_q_slots=sum(v>0 for v in counts),
                q_entropy_nats=-sum(p*math.log(p) for p in probabilities) if n else None,
                dominant_q_slot_fraction=max(counts)/n if n else None,
                single_observed_slot=(sum(v>0 for v in counts)==1) if n else None)


def nearest_q_slot(q,n):
    if n<1 or not 1<=int(q)<=int(n):
        raise ValueError('q must be in 1..N')
    ratio=float(q)/float(n)
    return min(range(len(RATIOS)),key=lambda i:abs(RATIOS[i]-ratio))
