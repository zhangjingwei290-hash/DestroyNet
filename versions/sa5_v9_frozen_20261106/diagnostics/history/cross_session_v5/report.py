from pathlib import Path
import json,hashlib,statistics,csv,collections
import numpy as np,moocore
O=Path(__file__).resolve().parent

def write(name,x):(O/name).write_text(json.dumps(x,indent=2),encoding='utf-8')
def main():
 methods=['roulette_sa5','v2_baseline','v2_stallq','gate_untrained','gate_trained'];runs=[];means=[];bins=[]
 for n in (50,150,250):
  iid=f'{n}-2-4';rows=[]
  for seed in (20261104,20261105,20261106):
   root=O/f'evaluation/{iid}/{seed}';assert json.loads((root/'analysis/AUDIT.json').read_text())['passed']
   for method in methods:
    r=json.loads((root/f'results/{method}.json').read_text());assert r['physical_audit_passed'];rows.append(r)
  pts=np.unique(np.concatenate([np.asarray([p['objectives'] for p in r['archive']]) for r in rows]),axis=0);pf=pts[moocore.is_nondominated(pts)];lo=pf.min(0);span=pf.max(0)-lo
  # Same physical zero-span convention as original evaluator.
  F=Path('D:/workspace/destroy-repair-V0.01-publication/versions/v5_step5_r2_frozen_20261102');scale=np.asarray(json.loads((F/'evaluation_source/boxes.json').read_text())[iid]['scale']);zero=span/scale<=1e-12
  def norm(x):return np.where(zero,0,(np.asarray(x)-lo)/np.where(zero,1,span))
  ref=norm(pf);write(f'reference_{n}.json',dict(ideal=lo.tolist(),span=span.tolist(),zero=np.flatnonzero(zero).tolist(),reference_pf=pf.tolist()))
  for r in rows:
   x=norm([p['objectives'] for p in r['archive']]);h=float(moocore.hypervolume(x,ref=[1.1]*5));g=float(moocore.igd_plus(x,ref=ref));runs.append(dict(iid=iid,n=n,seed=r['seed'],method=r['method'],hv=h,igd_plus=g,attempts=len(r['records']),batches=len(r['batches'])))
   for label,keep in [('0',lambda s:s==0),('1-2',lambda s:1<=s<=2),('3-5',lambda s:3<=s<=5),('6+',lambda s:s>=6)]:
    rr=[t for t in r['records'] if keep(t['direction_stall'])]
    if rr:bins.append(dict(iid=iid,seed=r['seed'],method=r['method'],stall_bin=label,count=len(rr),mean_q_ratio=statistics.mean(t['q']/n for t in rr),mean_gate_shift=statistics.mean(t.get('gate_shift',0) for t in rr),mode_proportions=[sum(t.get('mode')==k for t in rr)/len(rr) for k in (0,1,2)]))
  for m in methods:
   rr=[r for r in runs if r['n']==n and r['method']==m];assert len(rr)==3
   means.append(dict(n=n,iid=iid,method=m,**{key:statistics.mean(t[key] for t in rr) for key in ('hv','igd_plus','attempts','batches')},hv_sd=statistics.stdev(t['hv'] for t in rr),igd_plus_sd=statistics.stdev(t['igd_plus'] for t in rr)))
 lines=['# V5 跨会话停滞：粒度分支开发对照','', '3实例×3seed×5组；每次60秒、16方向并行、5步SA。共同经验参考按每IID所有5组×3seed重建，各seed独立评分，再报告均值与样本SD。均为已使用开发实例，3IID不足以支持总体显著性或未见泛化结论。','', '|N|组|HV均值±SD|IGD+均值±SD|相对v2 HV|尝试数均值|','|---|---|---:|---:|---:|---:|']
 for r in means:
  b=next(x for x in means if x['n']==r['n'] and x['method']=='v2_baseline');r['hv_change_vs_v2_pct']=100*(r['hv']/b['hv']-1)
  lines.append(f"|{r['n']}|{r['method']}|{r['hv']:.6f} ± {r['hv_sd']:.6f}|{r['igd_plus']:.6f} ± {r['igd_plus_sd']:.6f}|{r['hv_change_vs_v2_pct']:+.2f}%|{r['attempts']:.1f}|")
 lines+=['','跨会话停滞在新IID/run归零，在方向完整5步调用结束后更新；会话内停滞仍单独归零。真实fit外层轨迹记录用于训练，没有给旧快照补造历史标签。新分支只控制Beta均值，可正可负，并保留浓度；原专家全部冻结。新分支末层初始化0，所以未训练分支是零作用一致性对照；它与原v2的实测差异包含时间预算、调度和采样波动。','', '训练仅为共享粒度分支继续适配：原标量最好值改善、原固定Critic估计GAE、PPO均值目标；未重跑元训练，没有最大值/赢家奖励。训练与部署同真实lambda、N和跨会话计数。','', 'training/response.json是固定偏好/规模下仅改变跨会话停滞的Beta均值偏移；stall_bins.json是运行中实际q/N条件统计，受所到状态分布影响，不能单独当因果证明。分析包含负向结果，不自动提升权重为正式版本。']
 for stage in ('collection','evaluation'):
  lock=json.loads((O/f'{stage}_SOURCE_MANIFEST.json').read_text());assert all(hashlib.sha256(Path(f['windows_path']).read_bytes()).hexdigest()==f['sha256'] for f in lock['files'])
 write('per_run.json',runs);write('stall_bins.json',bins);write('summary.json',dict(means=means,training=json.loads((O/'training/COMPLETE.json').read_text()),searches=45));write('AUDIT.json',dict(passed=True,searches=45,physical_case_audits=9,common_reference_runs_per_iid=15,source_locks_unchanged=True));(O/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
 with (O/'main_table.csv').open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(means[0]));w.writeheader();w.writerows(means)
 print(json.dumps(means,indent=2))
if __name__=='__main__':main()
