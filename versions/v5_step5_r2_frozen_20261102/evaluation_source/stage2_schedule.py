"""Deterministic180FIT x4meta andtwo45-cell per-direction adaptation schedules."""
from stage2_common import ROOT,read
import math

def cell(iid):
    p=iid.split('_');return (int(p[2]),int(p[3][1:]),int(p[4][2:]))
def replica(iid):return int(iid.split('_')[-1])
def schedules():
    ids=read(ROOT/'FIT_MANIFEST.json')['fit_ids'];lookup={(*cell(i),replica(i)):i for i in ids}
    assert len(lookup)==180
    cells=[(n,d,tw) for tw in range(5) for d in range(3) for n in (50,150,250)]
    meta=[]
    for epoch in range(4):
        for rep in range(4):
            for j in range(45):
                c=cells[(7*j+11*epoch+5*rep)%45];index=len(meta);batch=index//5+1;task=index%5
                meta.append(dict(group=index+1,batch=batch,task=task,epoch=epoch+1,iid=lookup[(*c,(rep+epoch)%4)],cell=list(c),replica=(rep+epoch)%4,require_mature=batch>=19 and task in (0,2,4)))
    adaptation={}
    for k in range(16):
        rows=[]
        for cell_pass in range(2):
            for j in range(45):
                n,d,tw=cells[(7*j+5*k)%45];design_cell=[n,d,tw]
                replaced=tw==0 and k in (3,4,14)
                if replaced:tw=1+((j+k+d+(50,150,250).index(n)+cell_pass)%4)
                rep=(j+k+cell_pass)%4;c=(n,d,tw);index=45*cell_pass+j
                rows.append(dict(direction=k,batch=index+1,cell_pass=cell_pass+1,design_cell=design_cell,iid=lookup[(*c,rep)],cell=list(c),replica=rep,tw0_cell_replaced=replaced,require_mature=index%5 in (0,2,4)))
        adaptation[str(k)]=rows
    assert len(meta)==720
    from collections import Counter
    for epoch in range(1,5):assert Counter(x['iid'] for x in meta if x['epoch']==epoch)==Counter({i:1 for i in ids})
    for k,rows in adaptation.items():
        assert len(rows)==90 and sum(x['require_mature'] for x in rows)==54
        assert len({tuple(x['cell']) for x in rows})==(36 if int(k) in (3,4,14) else 45)
        for cell_pass in (1,2):
            selected=[x for x in rows if x['cell_pass']==cell_pass]
            assert len({tuple(x['design_cell']) for x in selected})==45
        for first,second in zip(rows[:45],rows[45:]):assert first['replica']!=second['replica']
        if int(k) in (3,4,14):assert all(x['cell'][2]>0 for x in rows)
    return dict(meta=meta,adaptation=adaptation)
def cosine(index,total,initial,minimum):return minimum+(initial-minimum)*.5*(1+math.cos(math.pi*index/max(1,total-1)))
def set_lr(opt,index,total):
    opt.param_groups[0]['lr']=cosine(index,total,.003,.0005)
    opt.param_groups[1]['lr']=cosine(index,total,.001,.0002)
    return [g['lr'] for g in opt.param_groups]
