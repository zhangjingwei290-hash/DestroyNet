"""Step3 descriptive centers plus explicitly bounded ratio histograms."""
from diagnostics import BINS,LABELS
EXACT_LIMITS=(.01,.02,.05,.10,.15,.20,.30,.40,.50,.60)
EXACT_LABELS=('q<=1%','1%<q<=2%','2%<q<=5%','5%<q<=10%','10%<q<=15%',
    '15%<q<=20%','20%<q<=30%','30%<q<=40%','40%<q<=50%','50%<q<=60%','q>60%')
STAGNATION=('zero','one_to_three','four_or_more','deep_seven_or_more')

def block():return dict(count=0,sum_q_ratio=0.,small_le15=0,hist=[0]*10,exact_hist=[0]*11)
def empty():return dict(blocks={k:block() for k in LABELS},by_stagnation={k:block() for k in STAGNATION})
def add(target,q,n,accepted,stagnation,used):
    ratio=q/n;regime='refine' if accepted else ('initial' if stagnation==0 else 'stalled')
    phase=('early','middle','late')[min(2,max(0,int(used*3)))];slot=min(range(10),key=lambda j:abs(BINS[j]-ratio))
    exact=next((j for j,b in enumerate(EXACT_LIMITS) if ratio<=b+1e-12),10)
    labels=['zero' if stagnation==0 else ('one_to_three' if stagnation<=3 else 'four_or_more')]
    if stagnation>=7:labels.append('deep_seven_or_more')
    for b in [target['blocks'][k] for k in ('overall',regime,phase)]+[target['by_stagnation'][k] for k in labels]:
        b['count']+=1;b['sum_q_ratio']+=ratio;b['small_le15']+=int(ratio<=.15+1e-12);b['hist'][slot]+=1;b['exact_hist'][exact]+=1
def build(entries):
    out=empty();out['by_direction']={str(k):empty() for k in range(16)}
    for q,n,accepted,stagnation,used,direction in entries:
        add(out,q,n,accepted,stagnation,used);add(out['by_direction'][str(direction)],q,n,accepted,stagnation,used)
    out.update(bins=list(BINS),exact_labels=list(EXACT_LABELS),binning='nearest centers and exact upper-inclusive ratio intervals, separately reported')
    return out
def records_q(records,n):
    return build((t['q'],n,t['last_accepted_before'],t['stagnation_before'],t['global_used_fraction_before'],t['direction']) for t in records)
def episodes_q(episodes):
    return build((t['action']['q'],len(t['observation']['x'])-1,t['observation']['search_state']['last_accepted'],
        t['observation']['search_state']['stagnation'],float(t['observation']['context'][11]),int(t.get('direction',t['task']))) for ep in episodes for t in ep)
