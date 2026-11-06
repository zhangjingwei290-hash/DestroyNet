from pathlib import Path
import csv,json,collections,statistics,math,hashlib
O=Path(__file__).resolve().parent
def load(name):
 with (O/name).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def write(name,x):(O/name).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def mean(x):return statistics.mean(x) if x else None
def groups(rows,keys):
 d=collections.defaultdict(list)
 for r in rows:d[tuple(r[k] for k in keys)].append(r)
 return d
audit=json.loads((O/'AUDIT.json').read_text());assert audit['passed']
rows=load('candidate_panel.csv');tr=load('continuations.csv')
assert len(rows)==2160 and len(tr)==1215
for r in rows:
 for key in ('n','expert','rep','q'):r[key]=int(r[key])
 for key in ('positive_gain','q_ratio','repair_seconds'):r[key]=float(r[key])
 r['gain']=float(r['gain']) if r['gain'] else None
 for key in ('feasible','timeout','nd_add_full','nd_add_near','nd_add_far'):r[key]=r[key]=='True'
valid=sum(r['feasible'] and not r['timeout'] for r in rows)
macro=[]
for (n,arm),rr in groups(rows,['n','arm']).items():
 macro.append(dict(n=n,arm=arm,mean_positive_gain=mean([r['positive_gain'] for r in rr]),mean_signed_gain=mean([r['gain'] for r in rr if r['gain'] is not None]),improvement_rate=mean([r['positive_gain']>0 for r in rr]),q_ratio=mean([r['q_ratio'] for r in rr]),repair_ms=1000*mean([r['repair_seconds'] for r in rr]),nd_add_rate=mean([r['nd_add_full'] for r in rr])))
# Two-fold cross-fit: each selection and evaluation half contains each repair once.
cf=[];blocks=[]
for key,rr in groups(rows,['iid','stage','expert']).items():
 ag=groups(rr,['arm']);means={a[0]:mean([r['positive_gain'] for r in v]) for a,v in ag.items()};best=max(means,key=means.get)
 blocks.append(dict(iid=key[0],stage=key[1],expert=key[2],best_in_sample=best,means=means,network_rank=1+sum(v>means['network']+1e-12 for v in means.values())))
 for fold in (0,1):
  choose=lambda r:(r['rep']<4)==(fold==0)
  train={a[0]:mean([r['positive_gain'] for r in v if choose(r)]) for a,v in ag.items() if a[0]!='network'}
  winner=max(train,key=lambda a:(train[a],a))
  def test(a):return mean([r['positive_gain'] for r in ag[(a,)] if not choose(r)])
  cf.append(dict(iid=key[0],n=int(key[0].split('_')[2]),stage=key[1],expert=key[2],fold=fold,selected=winner,selected_train_gain=train[winner],selected_test_gain=test(winner),network_test_gain=test('network'),test_difference=test(winner)-test('network')))
cf_summary=[]
for (n,),rr in groups(cf,['n']).items():cf_summary.append(dict(n=n,selected_gain=mean([r['selected_test_gain'] for r in rr]),network_gain=mean([r['network_test_gain'] for r in rr]),difference=mean([r['test_difference'] for r in rr]),win=sum(r['test_difference']>1e-12 for r in rr),tie=sum(abs(r['test_difference'])<=1e-12 for r in rr),loss=sum(r['test_difference']<-1e-12 for r in rr)))
cts=[]
for key,rr in groups(tr,['iid','stage','rep','arm']).items():
 rr=sorted(rr,key=lambda r:int(r['step']));assert len(rr)==15
 gains=[float(r['best_gain']) for r in rr]
 assert all(b+1e-12>=a for a,b in zip(gains,gains[1:]))
 cts.append(dict(iid=key[0],n=int(rr[0]['n']),stage=key[1],rep=int(key[2]),arm=key[3],gain5=gains[4],gain10=gains[9],gain15=gains[14],extra_after5=gains[14]-gains[4],late_only=gains[4]<=1e-12 and gains[14]>1e-12))
contmacro=[]
for (n,arm),rr in groups(cts,['n','arm']).items():
 contmacro.append(dict(n=n,arm=arm,gain5=mean([r['gain5'] for r in rr]),gain10=mean([r['gain10'] for r in rr]),gain15=mean([r['gain15'] for r in rr]),extra_after5=mean([r['extra_after5'] for r in rr]),trajectories=len(rr),improved_after5=sum(r['extra_after5']>1e-12 for r in rr),late_only=sum(r['late_only'] for r in rr)))
accept=[]
for (phase,),rr in groups([dict(r,phase='first5' if int(r['step'])<=5 else 'extension6to15') for r in tr],['phase']).items():
 bad=[r for r in rr if r['gain_before_accept'] and float(r['gain_before_accept'])<0]
 accept.append(dict(phase=phase,total=len(rr),deteriorating=len(bad),accepted_deteriorating=sum(r['accepted']=='True' for r in bad),mean_deteriorating_probability=mean([float(r['accept_probability']) for r in bad])))
archive_disagreement=sum(r['nd_add_near']!=r['nd_add_far'] for r in rows)
out=dict(valid_candidates=valid,total_candidates=len(rows),crossfit=cf_summary,candidate_macro=macro,block_details=blocks,continuation_macro=contmacro,acceptance=accept,archive_near_far_nd_disagreements=archive_disagreement,limits=['9 fit snapshots from only 3 IIDs, not independent generalization evidence','8 repetitions per arm; within-state winner estimates noisy','expert and preference change together in first panel; not isolated preference intervention','candidate modes all use learned customer scores; no oracle or independent destroy baseline','15-step continuation is out of trained horizon; final temperature held','CPU diagnostic inference and cache timing; no deployment throughput comparison'])
write('summary.json',out);write('crossfit.json',cf);write('continuation_summary.json',cts)
lines=['# 冻结策略动作价值与时域诊断','',f'物理与来源审计通过。候选{valid}/{len(rows)}有效；总修复调用{audit["repair_calls"]}；无训练更新。三个fit实例×早中晚九状态，当前权重冻结。','', '## 1. 同状态动作比较','', '每个状态/偏好下比较网络采样与三模式×三合法粒度；四原修复器各重复两次。使用两折交叉选择：一半重复选动作，另一半评价，再交换，避免直接把样本内最高值当真实收益上限。','', '|N|选定动作留出平均正增益|网络留出平均正增益|差值|胜/平/负（18折）|','|---|---:|---:|---:|---|']
for r in cf_summary:lines.append(f'|{r["n"]}|{r["selected_gain"]:.6f}|{r["network_gain"]:.6f}|{r["difference"]:+.6f}|{r["win"]}/{r["tie"]}/{r["loss"]}|')
lines+=['','正增益指本方向标量改善max(0,before-candidate)，不等于HV。候选有符号增益、物理五目标、实际q/耗时/移除集合见candidate_panel.csv；全量模式粒度均值见summary.json。单状态最佳动作只作为探索性结果，不称最优动作。','', '## 2. 五步后是否还有收益','', '同一轨迹比较第5/10/15步最好标量改善；三分支仅第一动作分别为网络、局部小粒度、整路线大粒度，后续均网络。第6步起保持终温、剩余步数特征0，连续历史。这是超出训练时域的反事实续搜，未重新训练，也未证明应采用15步。','', '|N|首动作分支|5步增益|10步增益|15步增益|5步后改善轨迹/9|','|---|---|---:|---:|---:|---:|']
for r in contmacro:lines.append(f'|{r["n"]}|{r["arm"]}|{r["gain5"]:.6f}|{r["gain10"]:.6f}|{r["gain15"]:.6f}|{r["improved_after5"]}/9|')
lines+=['','## 3. 接受与解集','']
for r in accept:lines.append(f'- {r["phase"]}：劣化候选{r["deteriorating"]}，SA接受{r["accepted_deteriorating"]}。这是轨迹描述，不等于被拒候选没有后续价值。')
lines +=[f'- 相同候选对near16/far16两个真实档案子集的ND插入判断不同：{archive_disagreement}/{len(rows)}。这是覆盖信息不同的直接现象；不把插入次数当HV贡献，也不据此认定新解集编码一定更好。','', '## 解释边界','',*['- '+x for x in out['limits']],'','完整原始候选、轨迹、来源SHA与AUDIT保留。当前正式对照的代码和权重没有修改。']
(O/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
write('ANALYSIS_AUDIT.json',dict(passed=True,candidate_rows=len(rows),trajectories=len(cts),crossfit_rows=len(cf),monotonic_best_checked=True,no_significance_claim=True))
print(json.dumps(dict(crossfit=cf_summary,continuation=contmacro,acceptance=accept,archive_disagreement=archive_disagreement),indent=2))
