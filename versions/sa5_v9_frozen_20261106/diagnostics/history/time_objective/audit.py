from pathlib import Path
import json,statistics
D=Path(__file__).resolve().parent;R=Path('/mnt/d/workspace/fuxian/DPR/development');a=json.loads((R/'sa5_state_action_v6_20261105/paired_results.json').read_text());rows=[]
for n in (50,150,250):
 for shift in (-.1,0,.1):
  r=[x for x in a if x['n']==n and x['shift']==shift];groups={}
  for x in r:groups.setdefault(x['state'],[]).append(x)
  gain=sum(x['gain'] for x in r);seconds=sum(max(x['group_seconds'] for x in g) for g in groups.values());rows.append(dict(n=n,shift=shift,mean_gain=gain/len(r),group_wall_seconds=seconds,gain_per_second=gain/seconds,mean_trajectory_seconds=seconds/len(r)))
findings=dict(global_remaining_already_in_actor=True,old_critic_has_global_remaining=True,five_step_terminal_bootstrap_zero=True,no_following_outer_generations_in_training=True,repair_deadline_uses_training_run_deadline=True,remaining_budget_feature_does_not_enforce_captured_solver_deadline=True,conclusion='Time input exists, but training return ends at5 steps and does not price lost outer evolution opportunities. Real deployment obeys total clock; training and deployment budget semantics differ.')
(D/'summary.json').write_text(json.dumps(dict(findings=findings,paired_costs=rows),indent=2))
lines=['# 时间目标核验','','已核验：Actor状态含global_remaining_fraction，原Critic也通过7维状态接收它。训练在第5步真终止，bootstrap=0，回报没有接到后续外层。修复截止使用整个训练任务deadline，快照剩余solver预算只更新输入，不作为该采样轨迹独立硬截止。因此不能把当前实现称为完整预算感知训练。','','下表为已有单进程配对诊断的墙钟效率，包含该诊断的特征/网络/修复等开销；不是16方向并行吞吐，不替代同时间HV。','', '|N|Beta均值偏移|平均五步收益|每轨迹秒|累计收益/累计秒|','|---|---:|---:|---:|---:|']
for x in rows:lines.append(f"|{x['n']}|{x['shift']:+.2f}|{x['mean_gain']:.6f}|{x['mean_trajectory_seconds']:.4f}|{x['gain_per_second']:.6f}|")
lines+=['','需要验证的新训练契约：保留单次最多5步，训练片段跨多个真实外层更新；全局耗时连续扣减，局搜返回外层不作为训练回报终点；真正到总预算才终止，工程截断独立标记。按16方向真实批次记录墙钟，禁止把worker累计秒当墙钟。优化器、验证和模型预热成本单列。价值估计需随新训练目标学习。新实现尚未启动训练，本文件只报告已完成的核验。']
(D/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8');print(json.dumps(rows,indent=2))
