from pathlib import Path
import json,csv,subprocess,sys,os,hashlib,statistics,collections
import numpy as np,moocore
O=Path(__file__).resolve().parent
def dump(name,x):(O/name).write_text(json.dumps(x,indent=2),encoding='utf-8')
def mean(x):return statistics.mean(x)
def grouped(rows,keys):
 out=collections.defaultdict(list)
 for r in rows:out[tuple(r[k] for k in keys)].append(r)
 return out
def main():
 completion=json.loads((O/'training/COMPLETE.json').read_text());assert completion['passed']
 valid=json.loads((O/'training/validation.json').read_text());holds=[r for r in valid if r['split']=='holdout'];holdmacro=[]
 for (checkpoint,),rr in grouped(holds,['checkpoint']).items():
  holdmacro.append(dict(checkpoint=checkpoint,scalar_gain=mean([r['scalar_gain'] for r in rr]),archive_gain=mean([r['archive_gain'] for r in rr]),return_total=mean([r['return_total'] for r in rr]),value_mse=mean([r['value_mse'] for r in rr])))
 metrics=[];performance=[]
 for iid in ['50-2-4','150-2-4','250-2-4']:
  case=O/f'cases/{iid}/20261104';assert json.loads((case/'analysis/AUDIT.json').read_text())['passed'];summary=json.loads((case/'analysis/summary.json').read_text());by={m['method']:m for m in summary['metrics']}
  ref=json.loads((case/'analysis/reference.json').read_text());lo=np.asarray(ref['ideal']);span=np.asarray(ref['nadir'])-lo;zero=np.zeros(5,dtype=bool);zero[ref['zero_coordinates']]=True
  initial=json.loads((case/'results/roulette_sa5.json').read_text())['initial_archive'];points=np.asarray([r['objectives'] for r in initial]);z=np.where(zero,0.,(points-lo)/np.where(zero,1.,span));initial_hv=float(moocore.hypervolume(z,ref=[1.1]*5))
  for m in summary['metrics']:
   m['initial_hv']=initial_hv;m['hv_gain_per_second']=(m['hv']-initial_hv)/60.;metrics.append(dict(iid=iid,**m))
  for method in ['neural_v3_sa5','neural_v3_sa10']:
   b=by['neural_v2_sa5'];m=by[method];performance.append(dict(iid=iid,method=method,hv_change_pct=100*(m['hv']/b['hv']-1),igd_reduction_pct=100*(1-m['igd_plus']/b['igd_plus']) if b['igd_plus'] else None,attempts=m['attempts'],batches=m['batches']))
 (O/'response').mkdir(exist_ok=True)
 with (O/'response.stdout.log').open('w') as out,(O/'response.stderr.log').open('w') as err:
  result=subprocess.run([sys.executable,'-B','-X','utf8',str(O/'response_probe.py')],env=dict(os.environ,DIAG_OUTPUT=str(O/'response'),DIAG_DRAWS='512'),stdout=out,stderr=err)
 assert result.returncode==0,'Frozen response probe failed; see logs'
 response=json.loads((O/'response/summary.json').read_text());assert json.loads((O/'response/AUDIT.json').read_text())['passed']
 dump('summary.json',dict(training=completion,holdout=holdmacro,performance=performance,metrics=metrics,response=response,scope='3 reused development IIDs x1seed x4methods x60seconds; no significance/generalization claim'))
 lines=['# Rich-state v3 开发试验','', '用户批准的组合修改：显式偏好/物理状态Critic、方向相关解集特征、残差FiLM、标量+有界档案新颖性奖励、同起点组残差基线、方向适配验证门控。图编码器保持冻结。组合试验不能识别单个组件因果贡献。','',f'训练与验证累计记账{completion["seconds"]:.1f}秒。原900秒组合上限是否通过：{completion.get("original_combined_cap_passed",True)}；优化更新不超过600秒，超出的冻结验证单列，无新增更新。16方向中{completion["gate_meta_retained"]}个保留元模型；未完成门控方向不提升适配模型。精确更新历史若未落盘则标null，不虚构轮数。','', '## 按IID划分的留出开发状态','', '训练d0/tw0、门控d1/tw2、最终快检d2/tw4；每规模各一个IID，三真实阶段。仍为反复使用的开发数据，不当未见测试。','', '|模型|标量收益|档案奖励|复合回报|价值预测MSE|','|---|---:|---:|---:|---:|']
 for r in holdmacro:lines.append(f'|{r["checkpoint"]}|{r["scalar_gain"]:.6f}|{r["archive_gain"]:.6f}|{r["return_total"]:.6f}|{r["value_mse"]:.6f}|')
 lines+=['','## 同60秒预算的完整宿主开发比较','', '每个IID四方法共用重新计算的经验前沿，各自独立评分；16方向并行修复。新旧5步用于组合修改比较，新5/10步用于长度比较。10步同时在10步内完成同样温度起止值，属于长度与降温步速的联合变化。10步使用同一份5步训练权重，历史长度与决策阶段分布会改变，尚非重新训练的10步策略。每方法单seed，无显著性检验。','', '|IID|方法|HV↑|IGD+↓|HV净增/秒|尝试数|外层批次数|','|---|---|---:|---:|---:|---:|---:|']
 for m in metrics:lines.append(f'|{m["iid"]}|{m["method"]}|{m["hv"]:.6f}|{m["igd_plus"]:.6f}|{m["hv_gain_per_second"]:.6f}|{m["attempts"]}|{m["batches"]}|')
 lines+=['','## 相对原v2五步','', '|IID|新方法|HV变化|IGD+降低|','|---|---|---:|---:|']
 for r in performance:
  igd='NA' if r['igd_reduction_pct'] is None else f'{r["igd_reduction_pct"]:+.2f}%'
  lines.append(f'|{r["iid"]}|{r["method"]}|{r["hv_change_pct"]:+.2f}%|{igd}|')
 lines+=['','## 条件响应与结论边界','', 'response/保留固定状态的条件响应。没有要求变化达到1%—3%，也没有以.055样本最大收益作为训练验收。响应增大本身不等于搜索效果更好。','', '该试验回答这份组合改动在三个60秒开发实例上是否改善及5/10步如何取舍，不能证明新奖励、FiLM或Critic单独有效。原v2当前实例完成后停在完整边界，原结果保留。本版没有自动扩大54次搜索或恢复历史900队列。']
 (O/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
 manifest=json.loads((O/'SOURCE_MANIFEST.json').read_text());assert all(hashlib.sha256(Path(x['windows_path']).read_bytes()).hexdigest()==x['sha256'] for x in manifest['files'])
 dump('AUDIT.json',dict(passed=True,physical_and_per_case_audits=3,methods_per_case=4,searches=12,source_locks_unchanged=True,response_audit=True,holdout_rows=len(holds),no_significance_claim=True,full_graph_encoder_frozen=True))
 print(json.dumps(dict(training=completion,holdout=holdmacro,performance=performance),indent=2),flush=True)
if __name__=='__main__':main()
