"""Compact descriptive summaries of the54 reused raw reports, read-only."""
from pilot_common import ROOT,read,write,sha
from diagnostics import records_q
from pathlib import Path
import math,psutil
def main():
    psutil.Process().cpu_affinity([14,15]);psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    plan=read(ROOT/'plan.json');base=Path(plan['baseline_root']);receipts=[]
    for iid in plan['test_ids']:
        for seed in plan['search_seeds']:
            for method in ('alns_tuned','v5_meta_frozen'):
                path=base/'results'/f'{iid}_seed{seed}_{method}.json';before=path.stat();r=read(path)
                q=records_q(r['records'],int(iid.split('-')[0]))
                throughput=dict(steps=len(r['records']),inference_seconds=sum(t['inference_seconds'] for t in r['records']),repair_seconds=sum(t['repair_seconds'] for t in r['records']),search_seconds=r['search_seconds'])
                assert math.isclose(throughput['inference_seconds'],r['inference_seconds'],rel_tol=1e-10,abs_tol=1e-9)
                compact={k:v for k,v in r.items() if k not in ('records','variation_events','point_events')}
                compact.update(q_diagnostics=q,_throughput=throughput,raw_bytes=before.st_size,raw_mtime_ns=before.st_mtime_ns,raw_sha256=sha(path))
                after=path.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
                write(ROOT/'baseline_cache'/path.name,compact)
                receipts.append(dict(path=str(path),bytes=before.st_size,mtime_ns=before.st_mtime_ns,sha256=compact['raw_sha256']))
        print('BASELINE_CACHE '+iid,flush=True)
    write(ROOT/'baseline_cache/COPY_RECEIPT.json',dict(passed=True,reports=len(receipts),files=receipts))
if __name__=='__main__':main()
