"""Read-only, bounded-memory behavioral analysis of completed V5 reports."""
import os
os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
import csv, json, math, random, time
from collections import defaultdict
from pathlib import Path
import psutil

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent

def save(name,value):
    p=OUT/name;t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');t.replace(p)

def add(d,k,v):
    a=d.setdefault(k,[0.,0]);a[0]+=v;a[1]+=1

def mean(v):return sum(v)/len(v) if v else None

def signed_rank(values):
    vals=[v for v in values if abs(v)>1e-12]
    if not vals:return dict(n=0,W=0,p=1.,positive=0,negative=0,ties=len(values))
    order=sorted(range(len(vals)),key=lambda i:abs(vals[i]));ranks=[0]*len(vals);j=0
    while j<len(order):
        k=j+1
        while k<len(order) and abs(vals[order[k]])==abs(vals[order[j]]):k+=1
        for i in order[j:k]:ranks[i]=j+1+k
        j=k
    positive=sum(r for r,v in zip(ranks,vals) if v>0);total=sum(ranks);w=min(positive,total-positive)
    dp=[1]+[0]*total;used=0
    for r in ranks:
        for j in range(used,-1,-1):dp[j+r]+=dp[j]
        used+=r
    p=min(1.,2*sum(dp[:w+1])/(2**len(vals)))
    return dict(n=len(vals),W=w/2,p=p,positive=sum(v>0 for v in vals),negative=sum(v<0 for v in vals),ties=len(values)-len(vals))

def describe(values):
    values=[x for x in values if x is not None]
    if not values:return None
    rng=random.Random(20261029);n=len(values);boots=sorted(sum(values[rng.randrange(n)] for _ in range(n))/n for _ in range(10000))
    return dict(instances=n,mean=mean(values),iid_bootstrap_ci95=[boots[249],boots[9749]],signed_rank=signed_rank(values))

def compare(groups,key_a,key_b):
    bases={k[:-1] for k in groups};diff=[]
    for base in bases:
        a=groups.get(base+(key_a,));b=groups.get(base+(key_b,))
        if a and b:diff.append(a[0]/a[1]-b[0]/b[1])
    return mean(diff),len(diff)

def main():
    proc=psutil.Process();proc.cpu_affinity([14,15]);proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    files=sorted((ROOT/'results').glob('*_v5_meta_frozen.json'))
    assert len(files)==900
    save('PROTOCOL.json',dict(frozen_before_scanning=time.time(),reports=900,phase_bounds=[0,1/3,2/3,1],metric='actual q/N',
        fresh='last_accepted_before=false and stagnation_before=0',success='last_accepted_before=true',deep_stall='last_accepted_before=false and stagnation_before>=5',
        primary_contrasts=['late_minus_early_controlled','deep_stall_minus_fresh_controlled','success_minus_fresh_controlled'],
        matching='within IID/seed, direction/repair; state contrasts also phase; phase contrasts also success/stall/fresh regime. Equal matched cells per run, equal available seeds per IID, equal IIDs.',
        tests='two-sided exact conditional signed-rank, doubled average ranks, absdiff<=1e-12 dropped; Holm over three primary contrasts; IID bootstrap exploratory; related instance variants not independent families',
        exclusions='No training, inference, solver execution, source edits, weight loading or result rewriting.'))
    run_rows=[];provenance=[];seen=set();total_actions=0;phase_counts=[0]*3;state_counts=defaultdict(int)
    for index,path in enumerate(files):
        before=path.stat()
        with path.open(encoding='utf-8-sig') as f:report=json.load(f)
        iid=report['iid'];seed=report['seed'];n=int(iid.split('-')[0]);key=(iid,seed)
        assert key not in seen;seen.add(key)
        assert seed in range(20261024,20261044)
        phases={};states={};phase_matched={};state_matched={};depth_matched={};run_count=0
        for rec in report['records']:
            q=rec['q'];assert 1<=q<=n
            ratio=q/n;u=rec['global_used_fraction_before'];assert -1e-6<=u<=1+1e-5
            phase=min(2,max(0,int(u*3)));direction=rec['direction'];repair=rec['repair']
            accepted=rec['last_accepted_before'];stall=rec['stagnation_before'];assert stall>=0
            state='success' if accepted else ('stall' if stall>0 else 'fresh')
            state_detail='success' if accepted else ('deep_stall' if stall>=5 else ('shallow_stall' if stall>0 else 'fresh'))
            add(phases,phase,ratio);add(states,state_detail,ratio)
            add(phase_matched,(direction,repair,state,phase),ratio)
            add(state_matched,(direction,repair,phase,state_detail),ratio)
            add(depth_matched,(direction,repair,phase,state_detail),ratio)
            phase_counts[phase]+=1;state_counts[state_detail]+=1;run_count+=1
        after=path.stat();assert before.st_size==after.st_size and before.st_mtime_ns==after.st_mtime_ns
        row=dict(iid=iid,seed=seed,n=n,actions=run_count)
        for p,label in enumerate(['early','middle','late']):row[label]=phases[p][0]/phases[p][1] if p in phases else None
        for label in ['fresh','success','shallow_stall','deep_stall']:row[label]=states[label][0]/states[label][1] if label in states else None
        row['late_minus_early']=row['late']-row['early'] if row['late'] is not None and row['early'] is not None else None
        for name,d,a,b in [('late_minus_early_controlled',phase_matched,2,0),('deep_stall_minus_fresh_controlled',state_matched,'deep_stall','fresh'),('success_minus_fresh_controlled',state_matched,'success','fresh'),('deep_minus_shallow_stall_controlled',depth_matched,'deep_stall','shallow_stall')]:
            row[name],row[name+'_cells']=compare(d,a,b)
        run_rows.append(row);total_actions+=run_count
        provenance.append(dict(file=path.name,bytes=before.st_size,mtime_ns=before.st_mtime_ns,actions=run_count))
        if (index+1)%30==0:
            save('progress.json',dict(completed=index+1,total=900,actions=total_actions,time=time.time(),available_ram_gb=psutil.virtual_memory().available/2**30))
            print(json.dumps(dict(completed=index+1,actions=total_actions)),flush=True)
        del report
    assert len({k[0] for k in seen})==45
    assert all(sum(i==iid for i,s in seen)==20 for iid in {k[0] for k in seen})
    metrics=['early','middle','late','fresh','success','shallow_stall','deep_stall','late_minus_early','late_minus_early_controlled','deep_stall_minus_fresh_controlled','success_minus_fresh_controlled','deep_minus_shallow_stall_controlled']
    iid_rows=[]
    for iid in sorted({r['iid'] for r in run_rows}):
        rows=[r for r in run_rows if r['iid']==iid];out=dict(iid=iid,n=rows[0]['n'])
        for m in metrics:
            vals=[r[m] for r in rows if r[m] is not None];out[m]=mean(vals);out[m+'_runs']=len(vals)
        iid_rows.append(out)
    summary=dict(reports=len(files),instances=45,seeds=20,total_actions=total_actions,phase_action_counts=phase_counts,state_action_counts=dict(state_counts),
        metrics={m:describe([r[m] for r in iid_rows]) for m in metrics},by_scale={str(n):{m:mean([r[m] for r in iid_rows if r['n']==n and r[m] is not None]) for m in metrics} for n in [50,150,250]},
        matched_runs={m:sum(r[m] is not None for r in run_rows) for m in metrics},limitations=['Behavioral associations, not causal training benefit.','V5 has explicit success/stagnation regime anchors; these differences cannot by themselves establish learned behavior.','Related IID variants limit independence assumptions. No counterfactual inference performed.'])
    for m in ['early','middle','late','fresh','success','shallow_stall','deep_stall']:
        summary['metrics'][m].pop('signed_rank',None)
    prim=['late_minus_early_controlled','deep_stall_minus_fresh_controlled','success_minus_fresh_controlled'];ordered=sorted(prim,key=lambda m:summary['metrics'][m]['signed_rank']['p']);last=0
    for i,m in enumerate(ordered):
        last=max(last,min(1.,(3-i)*summary['metrics'][m]['signed_rank']['p']));summary['metrics'][m]['holm_p_three_primary']=last
    for name,rows in [('per_run.csv',run_rows),('per_instance.csv',iid_rows)]:
        with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    save('source_receipt.json',provenance);save('summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
