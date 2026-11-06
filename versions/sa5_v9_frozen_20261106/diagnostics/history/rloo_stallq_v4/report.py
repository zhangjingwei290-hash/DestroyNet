from pathlib import Path
import json,csv,statistics,collections,hashlib
O=Path(__file__).resolve().parent
def write(name,x):(O/name).write_text(json.dumps(x,indent=2),encoding='utf-8')
def main():
 allmetrics=[];changes=[];budget=json.loads((O/'training/COMPLETE.json').read_text());assert budget['passed'];lines=['# V4 停滞粒度与RLOO两项开发快检','', '三开发IID，各1seed，每组60秒，16方向并行修复。两阶段各自重新计算共同经验参考，各seed档案独立评分。不同阶段绝对HV不横比。没有未见泛化或显著性结论。','']
 for stage in ('quick1','quick2'):
  lines+=['## '+stage,'','|IID|方法|HV↑|IGD+↓|相对本阶段v2的HV变化|尝试数|批次数|','|---|---|---:|---:|---:|---:|---:|']
  for iid in ('50-2-4','150-2-4','250-2-4'):
   case=O/f'{stage}/{iid}/20261104';assert json.loads((case/'analysis/AUDIT.json').read_text())['passed'];ms=json.loads((case/'analysis/summary.json').read_text())['metrics'];base=next(m for m in ms if m['method']=='v2_baseline')
   for m in ms:
    delta=100*(m['hv']/base['hv']-1);allmetrics.append(dict(stage=stage,iid=iid,hv_change_pct=delta,**m))
    if m['method']!='v2_baseline':changes.append(dict(stage=stage,iid=iid,method=m['method'],hv_change_pct=delta,igd_reduction_pct=100*(1-m['igd_plus']/base['igd_plus']) if base['igd_plus'] else None,attempt_ratio=m['attempts']/base['attempts'],batch_ratio=m['batches']/base['batches']))
    lines.append(f'|{iid}|{m["method"]}|{m["hv"]:.6f}|{m["igd_plus"]:.6f}|{delta:+.2f}%|{m["attempts"]}|{m["batches"]}|')
  if stage=='quick1':
   lines+=['','M=3完整局搜无改进后，Beta均值+0.10，浓度与模式概率不变，刷新记录解后恢复。不保证零时间开销。记录解在当前动态坐标下重评分；只保存一个历史记录向量，不冒称对全部历史重新求最优。该规则是人为停滞控制，不是网络学会自适应。','']
  else:lines+=['','两种继续训练均从相同v2专家权重出发、同起点四轨迹组、相同数据/更新数/优化器。PPO-GAE使用原Critic，PPO-RLOO使用轨迹总标量回报的留一基线、无价值损失、Critic权重冻结。保留PPO截断与样本标准化，不宣称整个更新是无偏REINFORCE。未重跑元训练，比较范围是继续适配。','']
 valid=json.loads((O/'training/validation.json').read_text());macro=[]
 for kind in ('initial','ppo','rloo'):
  rows=[r for r in valid if r['kind']==kind];macro.append(dict(kind=kind,mean_scalar_gain=statistics.mean(r['scalar_gain'] for r in rows),positive_trajectories=sum(r['scalar_gain']>0 for r in rows),total=len(rows)))
 lines+=['## 固定留出开发状态','',f'训练和验证{budget["seconds"]:.1f}秒，上限900秒；每方向每学习器9次更新，共{budget["updates"]}次，{budget["transitions"]}步。原始Critic和编码器在RLOO组逐值不变。','', '|模型|平均标量改善|找到改善的轨迹|','|---|---:|---:|']
 for m in macro:lines.append(f'|{m["kind"]}|{m["mean_scalar_gain"]:.6f}|{m["positive_trajectories"]}/{m["total"]}|')
 lines+=['','训练d0/tw0与验证d2/tw4按IID分开，但都是已反复使用的开发数据。每组合一条验证轨迹，不能把均值差当显著改进。','', '所有负向结果与原始轨迹保留，旧队列不恢复；没有新增预算、测试阈值扫描或追加seed。']
 for stage in ('quick1','quick2'):
  manifest=json.loads((O/f'{stage}_SOURCE_MANIFEST.json').read_text());assert all(hashlib.sha256(Path(f['windows_path']).read_bytes()).hexdigest()==f['sha256'] for f in manifest['files'])
 (O/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8');write('summary.json',dict(metrics=allmetrics,changes=changes,validation=macro,training=budget));write('AUDIT.json',dict(passed=True,physical_case_audits=6,searches=15,source_locks_unchanged=True,training_counts_matched=True,no_significance_claim=True))
 print(json.dumps(dict(changes=changes,validation=macro),indent=2))
if __name__=='__main__':main()
