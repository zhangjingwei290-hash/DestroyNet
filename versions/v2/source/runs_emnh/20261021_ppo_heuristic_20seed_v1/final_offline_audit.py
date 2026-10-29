"""Read saved results after worker exit; never instantiate a policy or search."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
import json, time, csv, hashlib
import numpy as np
import moocore
import psutil
from emnh_seed_stability import read, validate_plan, validate_result, verify_result, matching
from moea_emnh_v2.config import Config
from moea_emnh_v2.runner import setup, binding
from moea_emnh_v2.environment import Environment
from mo_dpr.data import fingerprint

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

started = time.perf_counter()
plan, summary, receipts = (read(ROOT / n) for n in ('plan.json', 'summary.json', 'verification.json'))
process = read(ROOT / 'process.json')
assert not matching(ROOT)
assert not (psutil.pid_exists(process['pid']) and abs(psutil.Process(process['pid']).create_time()-process['created']) < .01)
assert read(ROOT / 'complete.json')['results'] == 360
assert not (ROOT/'worker.lock').exists() and not (ROOT/'failure.json').exists()
validate_plan(plan)
source = Path(plan['source_root'])
cfg = Config(**read(source/'config.json'))
pool, split = setup(cfg)
assert fingerprint(binding(cfg, split)) == plan['solver_binding_id']
assert sha(ROOT.parents[1]/'emnh_seed_stability.py') == plan['wrapper_sha256']
assert all(sha(p) == expected for p, expected in plan['checkpoint_files_sha256'].items())
ids, seeds, arms = plan['development_ids'], plan['search_seeds'], plan['arms']
assert len(list((ROOT/'results').glob('*/*.json'))) == 360 and len(receipts) == 360
hv, initial = np.zeros((9,20,2)), np.zeros((9,20,2))
curves = np.zeros((9,20,2,5))
costs = {a:dict(search_seconds=0., inference_seconds=0., repair_seconds=0., actions=0, accepted=0, timeouts=0, failed=0) for a in arms}
archive_count = record_count = event_count = reused = 0
for i,iid in enumerate(ids):
    env = Environment(pool, iid, cfg, np.full(5,.2))
    for j,seed in enumerate(seeds):
        initial_points = []
        for k,arm in enumerate(arms):
            path = ROOT/'results'/str(seed)/f'{iid}_{arm}.json'
            row = read(path)
            validate_result(row, plan, iid, arm, seed)
            proof = verify_result(row, env)
            receipt = receipts[f'{seed}/{iid}/{arm}']
            assert sha(path) == receipt['sha256']
            np.testing.assert_allclose(proof['recomputed_hv'],receipt['recomputed_hv'],rtol=0,atol=1e-12)
            if seed == plan['reused_seed']:
                assert path.read_bytes() == (source/'development'/f'{iid}_{arm}.json').read_bytes()
                reused += 1
            np.testing.assert_allclose(row['physical_reference'],env.offset+env.scales*np.asarray(env.box['reference']),rtol=0,atol=1e-9)
            init = row['point_events'][:16]
            assert len(init)==16 and all(p['seconds'] <= row['initial_seconds'] for p in init)
            values = np.asarray([p['objectives'] for p in init])
            initial_points.append(values)
            initial[i,j,k] = moocore.hypervolume((values-env.offset)/env.scales,ref=env.box['reference'])
            hv[i,j,k] = row['hv']
            for q,f in enumerate((.1,.25,.5,.75,1.)):
                cutoff = row['budget_seconds']*f
                values = np.asarray([p['objectives'] for p in row['point_events'] if p['seconds'] <= cutoff])
                value = moocore.hypervolume((values-env.offset)/env.scales,ref=env.box['reference']) if len(values) else 0.
                np.testing.assert_allclose(row['hv_time_curve'][q]['seconds'],cutoff,rtol=0,atol=1e-12)
                np.testing.assert_allclose(row['hv_time_curve'][q]['hv'],value,rtol=1e-10,atol=1e-12)
                curves[i,j,k,q] = value
            np.testing.assert_allclose(curves[i,j,k,-1],row['hv'],rtol=1e-10,atol=1e-12)
            assert np.all(np.diff(curves[i,j,k]) >= -1e-12)
            c = costs[arm]
            c['search_seconds'] += row['search_seconds']
            c['inference_seconds'] += row['inference_seconds']
            c['repair_seconds'] += sum(p['repair_seconds'] for p in row['records'])
            c['actions'] += len(row['records'])
            c['accepted'] += sum(p['accepted'] for p in row['records'])
            c['timeouts'] += sum(p['timeout'] for p in row['records'])
            c['failed'] += sum(not p['feasible'] and not p['timeout'] for p in row['records'])
            archive_count += len(row['archive']); record_count += len(row['records']); event_count += len(row['point_events'])
        np.testing.assert_array_equal(*initial_points)
delta, gap = hv[:,:,1]-hv[:,:,0], 100*(hv[:,:,0]-hv[:,:,1])/hv[:,:,0]
np.testing.assert_allclose(gap.mean(),summary['mean_relative_hv_loss_percent'],rtol=0,atol=1e-12)
np.testing.assert_allclose(delta.mean(),summary['mean_delta_hv'],rtol=0,atol=1e-12)
rng = np.random.default_rng(20261021)
sj=rng.integers(20,size=(10000,20)); ij=rng.integers(9,size=(10000,9))
for matrix,key in ((gap,'relative_loss_intervals'),(delta,'delta_hv_intervals')):
    fixed=np.quantile(matrix.mean(axis=0)[sj].mean(axis=1),[.025,.975])
    crossed=np.quantile(matrix[ij[:,:,None],sj[:,None,:]].mean(axis=(1,2)),[.025,.975])
    np.testing.assert_allclose(fixed,summary[key]['fixed_nine_iid_seed_bootstrap_95_ci'],rtol=0,atol=1e-12)
    np.testing.assert_allclose(crossed,summary[key]['crossed_iid_seed_bootstrap_95_ci'],rtol=0,atol=1e-12)
for k,arm in enumerate(arms):
    for name,value in costs[arm].items():
        np.testing.assert_allclose(value,summary['execution_costs'][arm][name],rtol=1e-12,atol=1e-7)
    np.testing.assert_allclose(curves[:,:,k,:].mean(axis=(0,1)),[p['hv'] for p in summary['mean_hv_time_curve'][arm]],rtol=0,atol=1e-12)
    np.testing.assert_allclose(hv[:,:,k].mean(),summary['mean_hv_by_arm'][arm],rtol=0,atol=1e-12)
    gain=summary['initial_hv_and_gain'][arm]
    np.testing.assert_allclose([initial[:,:,k].mean(),(hv[:,:,k]-initial[:,:,k]).mean(),(100*(hv[:,:,k]-initial[:,:,k])/initial[:,:,k]).mean()],list(gain.values()),rtol=0,atol=1e-10)
for i,iid in enumerate(ids):
    row=summary['per_iid'][iid]
    np.testing.assert_allclose([gap[i].mean(),gap[i].std(ddof=1),gap[i].min(),gap[i].max()],[row[n] for n in ('mean_relative_loss_percent','std_relative_loss_percent','min_relative_loss_percent','max_relative_loss_percent')],rtol=0,atol=1e-12)
    for k,arm in enumerate(arms):
        np.testing.assert_allclose([hv[i,:,k].mean(),hv[i,:,k].std(ddof=1)],[row['mean_hv'][arm],row['std_hv'][arm]],rtol=0,atol=1e-12)
with (ROOT/'per_seed_results.csv').open(encoding='utf-8-sig',newline='') as f:
    csv_rows=list(csv.DictReader(f))
assert len(csv_rows)==360
for row in csv_rows:
    i,j,k=ids.index(row['iid']),seeds.index(int(row['seed'])),arms.index(row['arm'])
    np.testing.assert_allclose(float(row['hv']),hv[i,j,k],rtol=0,atol=1e-12)
assert len({(r['iid'],r['seed'],r['arm']) for r in csv_rows})==360
audit=dict(passed=True,at=time.time(),results=360,complete_search_seeds=20,distinct_iids=9,paired_comparisons=180,
    archived_solutions_rechecked=archive_count,decision_records=record_count,point_events_rechecked=event_count,
    hv_curve_points_recomputed=1800,cached_results_byte_identical=reused,checkpoint_files_unchanged=16,
    solver_binding_matches=True,wrapper_matches=True,worker_exited=True,failed_searches=0,recoveries=0,
    elapsed_seconds=time.perf_counter()-started,
    verification_scope='Independent reread of all published artifacts using frozen physical evaluator and moocore; not a second physical implementation',
    gap_percent_quantiles=dict(zip(['min','p05','p25','median','p75','p95','max'],np.quantile(gap,[0,.05,.25,.5,.75,.95,1]).tolist())),
    per_seed_mean_gap_sample_sd=float(gap.mean(axis=0).std(ddof=1)),
    mean_curve_relative_loss_percent=(100*(curves[:,:,0,:]-curves[:,:,1,:])/curves[:,:,0,:]).mean(axis=(0,1)).tolist(),
    new_run_wall_seconds=read(ROOT/'complete.json')['at']-process['created'],
    total_actual_search_seconds=sum(c['search_seconds'] for c in costs.values()),
    new_search_seconds=sum(c['search_seconds'] for c in costs.values())-sum(read(source/'development'/f'{i}_{a}.json')['search_seconds'] for i in ids for a in arms))
(ROOT/'offline_verification.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(audit,ensure_ascii=False))
