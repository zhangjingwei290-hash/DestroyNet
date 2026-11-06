import analyze as a
import json,statistics,collections
O=a.O
assert json.loads((O/'FIXED_PREFERENCE_AUDIT.json').read_text())['passed']
rr=a.load('fixed_preference.csv')
for row in a.rows:
 if row['expert']==7:
  rr.append(dict(iid=row['iid'],n=row['n'],stage=row['stage'],preference=7,policy_expert=7,rep=row['rep'],arm=row['arm'],positive_gain=row['positive_gain'],q_ratio=row['q_ratio'],mode_probabilities=row['mode_probabilities']))
assert len(rr)==2160
for r in rr:
 for key in ('n','preference','rep'):r[key]=int(r[key])
 for key in ('positive_gain','q_ratio'):r[key]=float(r[key])
blocks=[];cross=[]
for (iid,stage,pref),rows in a.groups(rr,['iid','stage','preference']).items():
 arms=a.groups(rows,['arm']);means={arm[0]:a.mean([r['positive_gain'] for r in v]) for arm,v in arms.items()}
 winner=max(means,key=means.get);net=arms[('network',)]
 probs=[json.loads(r['mode_probabilities']) for r in net]
 blocks.append(dict(iid=iid,n=int(net[0]['n']),stage=stage,preference=pref,best_sample_arm=winner,best_mean_gain=means[winner],network_gain=means['network'],network_q_ratio=a.mean([r['q_ratio'] for r in net]),network_mode_probabilities=[a.mean([p[k] for p in probs]) for k in range(3)]))
 for fold in (0,1):
  choose=lambda r:(r['rep']<4)==(fold==0)
  train={arm[0]:a.mean([r['positive_gain'] for r in v if choose(r)]) for arm,v in arms.items() if arm[0]!='network'}
  win=max(train,key=lambda k:(train[k],k))
  test=lambda arm:a.mean([r['positive_gain'] for r in arms[(arm,)] if not choose(r)])
  cross.append(dict(n=int(net[0]['n']),iid=iid,stage=stage,preference=pref,fold=fold,selected=win,selected_gain=test(win),network_gain=test('network'),difference=test(win)-test('network')))
agg=[]
for (n,),rows in a.groups(cross,['n']).items():agg.append(dict(n=n,selected_gain=a.mean([r['selected_gain'] for r in rows]),network_gain=a.mean([r['network_gain'] for r in rows]),difference=a.mean([r['difference'] for r in rows]),win=sum(r['difference']>1e-12 for r in rows),tie=sum(abs(r['difference'])<=1e-12 for r in rows),loss=sum(r['difference']<-1e-12 for r in rows)))
changed=sum(len({r['best_sample_arm'] for r in rows})>1 for rows in a.groups(blocks,['iid','stage']).values())
out=dict(policy_expert=7,blocks=blocks,crossfit=agg,crossfit_rows=cross,states_with_different_sample_winners_across_preferences=changed,total_states=9,limit='Sample-winner variability is descriptive with only8 repetitions; repair also conditioned on preference; no significant causal preference claim')
a.write('fixed_preference_summary.json',out)
text=['','## 固定同一专家的偏好控制','', '固定专家7，在相同路线和真实阶段下改变偏好0/7/15，排除切换专家的影响。沿用相同模式粒度、修复配对、随机种子与两折选择规则。偏好同时进入修复及标量评价，这是偏好条件下整个破坏—修复动作价值的诊断。','', '|N|选定动作留出正增益|网络留出正增益|差值|胜/平/负（18折）|','|---|---:|---:|---:|---|']
for r in agg:text.append(f'|{r["n"]}|{r["selected_gain"]:.6f}|{r["network_gain"]:.6f}|{r["difference"]:+.6f}|{r["win"]}/{r["tie"]}/{r["loss"]}|')
text +=['',f'九个状态中，{changed}个状态在三种偏好下的样本内最佳动作不同。这是候选动作价值随偏好变化的探索性线索，少量重复下的最佳动作不稳定，未当作可靠动作标签。','', '### 可执行判断','', '1. 当前证据支持针对状态/偏好区分模式粒度的研究价值；不支持把整个网络判定为无效。N150留出结果尤其不支持所有状态都存在容易替代的固定粒度优势。','2. 接受前已能看到部分动作选择收益差距，接受规则不是解释这些差距的必要条件。是否改温度仍需相同候选轨迹的接受分支实验，当前不能把拒绝次数当根因。','3. 五步后仍有收益，但相应增加动作和计算；延长轨迹并不等于同墙钟整套算法更好，也尚未证明收益由首个大粒度动作导致。','4. 新解对不同真实档案子集的补充价值会不同，当前标量奖励未直接区分该价值；先用偏好相关覆盖特征与候选价值预测进行表示验证，再决定奖励修改。','5. 下一版修改应一次验证一个因素：优先状态/偏好条件的动作价值预测与完整输入的价值网络，再以独立对照验证表示增益；本次没有修改架构、奖励或重训。','']
with (O/'REPORT.md').open('a',encoding='utf-8') as f:f.write('\n'.join(text))
checks=json.loads((O/'AUDIT.json').read_text());fix=json.loads((O/'FIXED_PREFERENCE_AUDIT.json').read_text())
a.write('FINAL_RECEIPT.json',dict(passed=True,primary=checks,fixed_preference=fix,total_repair_calls=checks['repair_calls']+fix['new_candidate_calls'],candidate_calls=2160+1440,continuation_calls=1215,repeatability_calls=9,training_updates=0,report=str(O/'REPORT.md')))
print(json.dumps(dict(crossfit=agg,states_changed=changed,total_repair_calls=checks['repair_calls']+fix['new_candidate_calls']),indent=2))
