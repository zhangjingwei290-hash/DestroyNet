import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import json,hashlib,numpy as np,moocore,psutil
psutil.Process().cpu_affinity([14,15]);O=Path('D:/workspace/fuxian/DPR/development/sa5_global_return_v7_20261105');D=Path(__file__).resolve().parent;F=Path('D:/workspace/destroy-repair-V0.01-publication/versions/v5_step5_r2_frozen_20261102');boxes=json.loads((F/'data/FIT_MANIFEST.json').read_text())['boxes'];rows=[]
assert json.loads((O/'COMPLETE.json').read_text())['passed']
for n in (50,150,250):
 runs=[]
 for kind in ('initial','trained'):
  root=O/f'cases/heldout_{n}_{kind}';assert json.loads((root/'analysis/AUDIT.json').read_text())['passed'];r=json.loads((root/'results/gate_trained.json').read_text());assert r['physical_audit_passed'];runs.append((kind,r))
 assert runs[0][1]['initial_population']==runs[1][1]['initial_population']
 allp=np.unique(np.concatenate([np.asarray([p['objectives'] for p in r['archive']]) for _,r in runs]),axis=0);pf=allp[moocore.is_nondominated(allp)];lo=pf.min(0);span=pf.max(0)-lo;iid=f'v5p_train_{n}_d1_tw2_0';zero=span/np.asarray(boxes[iid]['scale'])<=1e-12
 def norm(x):return np.where(zero,0.,(np.asarray(x)-lo)/np.where(zero,1.,span))
 for kind,r in runs:
  x=norm([p['objectives'] for p in r['archive']]);rows.append(dict(n=n,kind=kind,hv=float(moocore.hypervolume(x,ref=[1.1]*5)),igd_plus=float(moocore.igd_plus(x,ref=norm(pf))),batches=len(r['batches']),actions=len(r['records']),actual_seconds=r['online_seconds']))
for n in (50,150,250):
 b=next(r for r in rows if r['n']==n and r['kind']=='initial');a=next(r for r in rows if r['n']==n and r['kind']=='trained');a['hv_change_pct']=100*(a['hv']/b['hv']-1);a['igd_reduction_pct']=100*(1-a['igd_plus']/b['igd_plus']) if b['igd_plus'] else None
lock=json.loads((O/'SOURCE_MANIFEST.json').read_text());assert all(hashlib.sha256(Path(x['windows_path']).read_bytes()).hexdigest()==x['sha256'] for x in lock['files']);history=json.loads((O/'training_history.json').read_text());assert all(x['outer_batches']>1 and x['max_density_error']<.002 for x in history)
lines=['# 全局回报训练小型验证','','每规模1个留出实例×1seed，两组各30秒，16方向并行。比较V6初始粒度分支与经过全局回报继续训练后的同一分支。每实例只用这两组最终档案建立共同经验前沿，各组独立评分。训练6条完整宿主轨迹、6次PPO更新、每次2epoch。所有旧专家冻结，新粒度分支与32维输入Critic学习。','','|N|初始HV|训练后HV|HV变化|初始IGD+|训练后IGD+|外层批次 初始→训练后|','|---|---:|---:|---:|---:|---:|---|']
for n in (50,150,250):
 b=next(r for r in rows if r['n']==n and r['kind']=='initial');a=next(r for r in rows if r['n']==n and r['kind']=='trained');lines.append(f"|{n}|{b['hv']:.6f}|{a['hv']:.6f}|{a['hv_change_pct']:+.2f}%|{b['igd_plus']:.6f}|{a['igd_plus']:.6f}|{b['batches']}→{a['batches']}|")
lines+=['','训练目标是预算结束时档案在16个固定物理归一化标量方向上的平均最好值，不是HV。动作目标包含之后外层生成的收益，5步局搜返回不截断回报，只有全局预算结束才封闭。日志/特征/推理/修复/外层计入在线墙钟；模型加载与优化另计。','', '结论范围：完成时间闭环技术验证，但6次更新、每规模1seed不足以证明性能提升或已学会最优粒度/时间分配。全局共同回报对每个动作的归因噪声较大；没有通过该快检证明延长训练必然有效。跨版本或跨实例不同参考HV不直接比较。']
(D/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8');(D/'summary.json').write_text(json.dumps(rows,indent=2));(D/'AUDIT.json').write_text(json.dumps(dict(passed=True,heldout_runs=6,paired_initial_populations=True,physical_audits=True,source_lock_unchanged=True,full_host_rollouts=True,max_density_error=max(x['max_density_error'] for x in history),training_actions=sum(x['actions'] for x in history)),indent=2));print(json.dumps(rows,indent=2))
