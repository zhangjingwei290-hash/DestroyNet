"""Descriptive q bins only; no change to the continuous action support."""
import math
BINS=(.01,.02,.05,.10,.15,.20,.30,.40,.50,.60)
LABELS=('overall','initial','refine','stalled','early','middle','late')

def empty():return {k:dict(count=0,sum_q_ratio=0.,small_le15=0,hist=[0]*10) for k in LABELS}
def add(blocks,q,n,accepted,stagnation,used):
    ratio=float(q)/n
    regime='refine' if accepted else ('initial' if stagnation==0 else 'stalled')
    phase=('early','middle','late')[min(2,max(0,int(float(used)*3)))]
    slot=min(range(10),key=lambda k:abs(BINS[k]-ratio))
    for key in ('overall',regime,phase):
        b=blocks[key];b['count']+=1;b['sum_q_ratio']+=ratio;b['small_le15']+=int(ratio<=.15+1e-12);b['hist'][slot]+=1
def episodes_q(episodes):
    blocks=empty();directions={str(k):empty() for k in range(16)}
    for ep in episodes:
        for t in ep:
            obs=t['observation'];st=obs['search_state'];q=t['action']['q'];n=len(obs['x'])-1;used=float(obs['context'][11])
            add(blocks,q,n,st['last_accepted'],st['stagnation'],used)
            add(directions[str(t.get('direction',t['task']))],q,n,st['last_accepted'],st['stagnation'],used)
    return dict(blocks=blocks,by_direction=directions,bins=list(BINS),binning='nearest reporting center; q<=15% is an exact separate threshold')
def records_q(records,n):
    blocks=empty();directions={str(k):empty() for k in range(16)}
    for t in records:
        values=(t['q'],n,t['last_accepted_before'],t['stagnation_before'],t['global_used_fraction_before'])
        add(blocks,*values);add(directions[str(t['direction'])],*values)
    return dict(blocks=blocks,by_direction=directions,bins=list(BINS))
