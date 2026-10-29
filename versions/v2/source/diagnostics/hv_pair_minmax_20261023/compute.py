"""User-requested offline HV: shared bounds within each IID x search-seed pair.

Only saved feasible archives are read. No solver, policy, or training is called.
All per-method archive points enter the bounds pool (no additional global filter).
"""
import os
for n in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[n]='1'
import json, csv, hashlib, time
from pathlib import Path
import numpy as np
import moocore

REPO=Path(__file__).resolve().parents[2]
SOURCE=REPO/'runs_emnh/20261021_ppo_heuristic_20seed_v1'
OUT=Path(__file__).resolve().parent
def read(p):
    with p.open(encoding='utf-8') as f:return json.load(f)
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

start=time.perf_counter()
plan=read(SOURCE/'plan.json'); receipts=read(SOURCE/'verification.json')
assert read(SOURCE/'offline_verification.json')['passed']
assert read(SOURCE/'complete.json')['results']==360
np.testing.assert_allclose(moocore.hypervolume([[.2,.8],[.8,.2]],ref=[1.1]*2),.45,atol=1e-12,rtol=0)
rows=[]; inputs={}; zero_dimensions=[]; small_dimensions=[]
for iid in plan['development_ids']:
    for seed in plan['search_seeds']:
        data={}; points={}
        for arm in plan['arms']:
            p=SOURCE/'results'/str(seed)/f'{iid}_{arm}.json'
            digest=sha(p); key=f'{seed}/{iid}/{arm}'
            assert digest==receipts[key]['sha256']; inputs[key]=digest
            row=read(p)
            assert (row['iid'],row['seed'],row['arm'])==(iid,seed,arm)
            data[arm]=row
            f=np.asarray([x['objectives'] for x in row['archive']],dtype=np.float64)
            assert f.ndim==2 and f.shape[1]==5 and len(f)>0 and np.isfinite(f).all()
            original=moocore.hypervolume((f-row['physical_offset'])/row['physical_scale'],ref=row['normalized_reference'])
            np.testing.assert_allclose(original,row['hv'],rtol=1e-10,atol=1e-12)
            points[arm]=f
        pool=np.concatenate(list(points.values()),axis=0)
        lo,hi=pool.min(axis=0),pool.max(axis=0); span=hi-lo
        active=span>0
        for j in np.flatnonzero(~active):zero_dimensions.append(dict(iid=iid,seed=seed,objective=int(j),value=float(lo[j])))
        for j in np.flatnonzero(active & (span<1e-8*np.maximum(1.,np.maximum(abs(lo),abs(hi))))):
            small_dimensions.append(dict(iid=iid,seed=seed,objective=int(j),span=float(span[j])))
        denom=np.where(active,span,1.)
        values={}
        for arm,f in points.items():
            z=(f-lo)/denom
            assert np.all(z>=0) and np.all(z<=1)
            value=float(moocore.hypervolume(z,ref=[1.1]*5))
            assert np.isfinite(value) and value>0
            # Verify against physical-space HV and the affine volume Jacobian.
            if active.any():
                physical=float(moocore.hypervolume(f[:,active],ref=(lo+1.1*span)[active]))
                check=physical/float(np.prod(span[active]))*1.1**int((~active).sum())
            else:check=1.1**5
            np.testing.assert_allclose(value,check,rtol=1e-9,atol=1e-12)
            values[arm]=value
        gap=100*(values['heuristic']-values['ppo'])/values['heuristic']
        oldgap=100*(data['heuristic']['hv']-data['ppo']['hv'])/data['heuristic']['hv']
        rows.append(dict(iid=iid,seed=seed,heuristic_hv=values['heuristic'],ppo_hv=values['ppo'],
            gap_percent=gap,delta_hv=values['ppo']-values['heuristic'],original_gap_percent=oldgap,
            lower=lo.tolist(),upper=hi.tolist(),physical_reference=(lo+1.1*span).tolist(),
            zero_range_dimensions=np.flatnonzero(~active).tolist()))
gap=np.asarray([[r['gap_percent'] for r in rows if r['iid']==i] for i in plan['development_ids']])
assert gap.shape==(9,20)
summary=dict(schema='user_pair_pooled_archive_minmax_hv_v1',source=str(SOURCE),
    scope='Shared min/max per IID x seed from both method archives; bounds differ across seeds',
    pool='Union of both per-run archives; not globally nondominated-filtered again',
    normalized_reference=[1.1]*5,normalization='(f-pair_min)/(pair_max-pair_min)',
    zero_range_rule='Exactly constant dimensions map to zero for both methods; keep five dimensions (common factor 1.1 per constant dimension)',
    results=360,pairs=180,distinct_iids=9,search_seeds=20,training_seeds=1,
    mean_relative_hv_loss_percent=float(gap.mean()),median_relative_hv_loss_percent=float(np.median(gap)),
    gap_sample_sd_percent=float(gap.std(ddof=1)),min_gap_percent=float(gap.min()),max_gap_percent=float(gap.max()),
    wins_ties_losses=[int((gap<0).sum()),int((gap==0).sum()),int((gap>0).sum())],
    mean_hv={a:float(np.mean([r[a+'_hv'] for r in rows])) for a in plan['arms']},
    original_mean_relative_hv_loss_percent=float(np.mean([r['original_gap_percent'] for r in rows])),
    per_iid={},zero_range_dimensions=zero_dimensions,near_zero_nonconstant_dimensions=small_dimensions,
    caveat='Pair-specific scales/references support within-pair comparison; across-seed absolute HV SD is not a common-reference stability measure. Post-hoc diagnostic, old results retained; no equivalence claim.')
rng=np.random.default_rng(20261021)
sj=rng.integers(20,size=(10000,20));ij=rng.integers(9,size=(10000,9))
summary['conditional_seed_bootstrap_95_ci_percent']=np.quantile(gap.mean(axis=0)[sj].mean(axis=1),[.025,.975]).tolist()
summary['exploratory_crossed_bootstrap_95_ci_percent']=np.quantile(gap[ij[:,:,None],sj[:,None,:]].mean(axis=(1,2)),[.025,.975]).tolist()
for i,iid in enumerate(plan['development_ids']):
    rr=[r for r in rows if r['iid']==iid]
    summary['per_iid'][iid]=dict(mean_gap_percent=float(gap[i].mean()),median_gap_percent=float(np.median(gap[i])),
        std_gap_percent=float(gap[i].std(ddof=1)),min_gap_percent=float(gap[i].min()),max_gap_percent=float(gap[i].max()),
        mean_hv={a:float(np.mean([r[a+'_hv'] for r in rr])) for a in plan['arms']})
assert all(sha(SOURCE/'results'/key.split('/')[0]/f'{key.split("/")[1]}_{key.split("/")[2]}.json')==v for key,v in inputs.items())
save('summary.json',summary);save('paired_results.json',rows)
with (OUT/'paired_results.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
save('verification.json',dict(passed=True,source_reports_sha256=inputs,original_hv_recomputed=360,
    minmax_hv_affine_crosschecks=360,analytical_2d_test=True,source_unchanged_after_read=True,
    physical_feasibility='Inherited original completed offline audit and exact archive-byte identity',
    elapsed_seconds=time.perf_counter()-start))
lines=['# 用户指定：同实例、同 seed 的两方法共同 min–max HV','',
    '每一对单独汇集两方法最终档案，逐目标取共同 min/max，归一化后参考点为五维 1.1；两方法各自算 HV。未再做跨方法全局非支配过滤。所有原始结果字节不变。','',
    f"平均配对 gap：{summary['mean_relative_hv_loss_percent']:.6f}%（正数 PPO 落后）；原物理参考点口径为 {summary['original_mean_relative_hv_loss_percent']:.6f}%。",'',
    f"中位数 {summary['median_relative_hv_loss_percent']:.6f}%；范围 [{summary['min_gap_percent']:.6f}%, {summary['max_gap_percent']:.6f}%]；PPO 胜/平/负 {summary['wins_ties_losses']}。",'',
    f"条件搜索 seed 区组重采样 95% 区间：{summary['conditional_seed_bootstrap_95_ci_percent']}%；探索性交叉重采样区间：{summary['exploratory_crossed_bootstrap_95_ci_percent']}%。",'',
    '| IID | 启发式平均 HV | PPO 平均 HV | 平均配对 gap (%) |','|---|---:|---:|---:|']
for iid,p in summary['per_iid'].items():lines.append(f"| {iid} | {p['mean_hv']['heuristic']:.6f} | {p['mean_hv']['ppo']:.6f} | {p['mean_gap_percent']:.4f} |")
lines+=['',f'零范围维度共 {len(zero_dimensions)} 个（两方法均置 0，公共体积因子不影响相对 gap）；近零但非零范围维度 {len(small_dimensions)} 个，明细见 summary.json。','',
    '这是一项按用户指定口径的事后离线诊断：每个 seed 的尺子不同，表中跨 seed 平均 HV 对应不同边界；主要比较每对的 gap。不能把这里的跨 seed 绝对 HV 方差当作固定参考点下的搜索稳定性。单训练 seed、38/45 元覆盖及 12/16 有效适配限制继续适用。']
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
