"""Finite synthetic checks for pooled-reference per-seed metrics and missing pairs."""
from stage2_common import ROOT,write

def main():
    import psutil,math
    from stage2_reporting import compute,write_tables
    psutil.Process().cpu_affinity([14,15]);out=ROOT/'engineering/report_units';out.mkdir(parents=True,exist_ok=True)
    ids=['50-0-0','50-0-1'];seeds=[1,2,3];methods=['baseline','missing_baseline','candidate']
    for ii,iid in enumerate(ids):
        for m,method in enumerate(methods):
            for seed in seeds:
                if method=='missing_baseline' and seed==3:continue
                F=[2+ii,.5+.1*seed+.02*m,1+.1*m,.3+.1*seed,.8-.01*m]
                d=dict(iid=iid,seed=seed,method=method,archive=[dict(objectives=F)],physical_offset=[0]*5,physical_scale=[10]*5,physical_reference=[11]*5)
                write(out/f'{iid}_{seed}_{method}.json',d)
    data=compute(ids,seeds,methods,lambda i,s,m:out/f'{i}_{s}_{m}.json');write_tables(out/'tables',data)
    assert len(data['per_run'])==16 and len(data['main_table'])==6 and len(data['missing'])==2
    assert all(c['paired_runs']==4 for c in data['comparisons'] if c['n']=='overall' and c['baseline']=='missing_baseline')
    assert all(c['paired_runs']==6 for c in data['comparisons'] if c['n']=='overall' and c['baseline']=='baseline')
    assert all(math.isfinite(r['empirical_hv']) for r in data['per_run'])
    assert all(r['holm_family_size']==6 for r in data['comparisons'] if r['n']=='overall')
    write(ROOT/'REPORTING_UNITS.json',dict(passed=True,own_seed_metric_rows=16,means=6,missing=2,paired_missing_runs=4,complete_pair_runs=6,common_reference_no_seed_archive_merge=True))
if __name__=='__main__':main()
