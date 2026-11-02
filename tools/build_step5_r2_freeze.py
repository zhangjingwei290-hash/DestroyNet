"""Archive already-completed R2 artifacts; no model, solver or training imports."""
from pathlib import Path
import json,hashlib,shutil,gzip,csv,collections,os,sys,datetime
import psutil
psutil.Process().cpu_affinity([14,15])
PUB=Path('D:/workspace/destroy-repair-V0.01-publication')
BASE=Path('D:/workspace/fuxian/DPR/runs/experiments');P=BASE/'v5_step5_multiscale_20261031';A=P/'stage2_archive_r2';R=P/'stage2_r3_full12pass_co_adapted';S=R/'supplement_three_methods'
NAME='v5_step5_r2_frozen_20261102';V=PUB/'versions'/NAME
assert not (V/'FREEZE_DECLARATION.json').exists(),'Completed freeze must not be overwritten'
V.mkdir(exist_ok=True);COPIES=[];TRANS=[];STAMPS={}
SKIP={'__pycache__','.git','.pytest_cache'}
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def copy(src,rel):
 assert src.is_file(),str(src)
 dst=V/rel;dst.parent.mkdir(parents=True,exist_ok=True)
 st=src.stat();digest=sha(src)
 if dst.exists():assert sha(dst)==digest,rel
 else:shutil.copy2(src,dst)
 assert sha(dst)==digest
 assert (st.st_size,st.st_mtime_ns)==(src.stat().st_size,src.stat().st_mtime_ns)
 STAMPS[str(src)]=(st.st_size,st.st_mtime_ns)
 COPIES.append(dict(path=dst.relative_to(PUB).as_posix(),source=str(src),bytes=st.st_size,sha256=digest))
 return digest

def tree(src,rel,allowed=None):
 for f in sorted(src.rglob('*')):
  if f.is_file() and not SKIP.intersection(f.relative_to(src).parts) and f.suffix not in ('.pyc','.pyo') and (allowed is None or f.suffix in allowed):copy(f,str(Path(rel)/f.relative_to(src)))
# Bind all 17 R2 final models before collecting anything else.
receipt=read(A/'ARCHIVE_RECEIPT.json');saved={x['path']:x for x in receipt['files']}
weights={}
for f in [A/'stage2_training/meta_retrained.pt']+sorted((A/'stage2_training/checkpoints').glob('direction_*.pt')):
 rel=f.relative_to(A).as_posix();assert sha(f)==saved[rel]['sha256']
 key='meta_retrained.pt' if f.parent.name!='checkpoints' else f.name
 weights[key]=copy(f,'checkpoints/'+key)
assert len(weights)==17 and weights['meta_retrained.pt']==read(R/'PARENT_R2.json')['r2_meta_sha256']
for f in [A/'stage2_training/complete.json',A/'stage2_training/initialization.json',A/'stage2_training/Q_GATE.json',A/'stage2_training/q_behavior.csv',A/'stage2_training/training_report.json',A/'stage2_training/train.log']:
 copy(f,'training/'+f.name)
copy(P/'pretrained/meta_complete.pt','initialization/meta_complete.pt')
assert sha(V/'initialization/meta_complete.pt')==read(P/'FIT_MANIFEST.json')['source_meta_sha256']
# Exact original R2 training/source lock, and effective source for the 864 later evaluations.
for rel,digest in read(A/'stage2_coordinator/SOURCE_MANIFEST.json').items():
 src=A/rel
 if not src.exists():src=P/rel
 assert sha(src)==digest,rel;copy(src,'training_source/'+rel)
for f in A.glob('*.py'):
 if not (V/'training_source'/f.name).exists():copy(f,'training_source/'+f.name)
for f in ['common.py','stats.py','step3_diagnostics.py','diagnostics.py','stage1_mechanisms.py','eval_config.json','plan.json','boxes.json','FIT_MANIFEST.json','budgets.csv','DECISIONS_PENDING.json']:
 src=P/f
 if src.exists() and not (V/'training_source'/f).exists():copy(src,'training_source/'+f)
tree(P/'repo','training_source/repo',{'.py','.json','.md'})
copy(A/'stage2_coordinator/SOURCE_MANIFEST.json','evidence/R2_ORIGINAL_SOURCE_MANIFEST.json')
for rel,digest in read(R/'stage2_coordinator/EFFECTIVE_SOURCE_MANIFEST.json').items():
 src=R/rel;assert sha(src)==digest,rel;copy(src,'evaluation_source/'+rel)
for f in ['eval_config.json','plan.json','boxes.json','FIT_MANIFEST.json','DECISIONS_PENDING.json','PARENT_R2.json','ENGINEERING_AMENDMENT.json']:
 if not (V/'evaluation_source'/f).exists():copy(R/f,'evaluation_source/'+f)
copy(R/'stage2_coordinator/EFFECTIVE_SOURCE_MANIFEST.json','evidence/EFFECTIVE_EVALUATION_SOURCE_MANIFEST.json')
# Original physical input bytes and their roles.
fit=read(P/'FIT_MANIFEST.json');assert len(fit['fit_ids'])==180
for iid in fit['fit_ids']:
 src=P/'fit_inputs'/f'{iid}.json';assert sha(src)==fit['input_sha256'][iid];copy(src,'data/fit/'+src.name)
for f in sorted((P/'inputs').glob('*.json')):copy(f,'data/test/'+f.name)
for iid in fit['excluded_validation_ids']:
 src=PUB/'data/v5/instances'/f'{iid}.json';assert src.exists();copy(src,'data/validation/'+src.name)
copy(P/'FIT_MANIFEST.json','data/FIT_MANIFEST.json');copy(P/'boxes.json','data/test_boxes.json');copy(P/'budgets.csv','data/budgets.csv')
tree(P/'physical','physical',{'.py','.json','.md'})
# Baseline implementations actually used for the latest comparison, not generic substitutes.
for folder,key in [('v5_realworld45_frozen_20seed_20261026','MOEAD_ALNS'),('v5_desktop_alnstuned45_20seed_20261028','MOEAD_ALNS_T')]:
 b=BASE/folder;tree(b/'repo','baseline_sources/'+key+'/repo',{'.py','.json','.md'})
 for f in ['baseline_search.py','config.json','plan.json','tuning/selected.json']:
  if (b/f).exists():copy(b/f,'baseline_sources/'+key+'/'+f)
# Completed evidence only. Process identity JSON files and raw traces stay local.
for src,rel in [(A/'ARCHIVE_RECEIPT.json','evidence/R2_ORIGINAL_ARCHIVE_RECEIPT.json'),(A/'stage2_analysis/AUDIT.json','evidence/R2_TRAINING_AND_CORE_AUDIT.json'),(R/'MODEL_SELECTION.json','evidence/MODEL_SELECTION.json'),(R/'PARENT_R2.json','evidence/PARENT_R2.json'),(R/'AUDIT.json','evidence/FINAL_CAMPAIGN_AUDIT.json'),(R/'COMPLETE.json','evidence/FINAL_CAMPAIGN_COMPLETE.json'),(R/'DELIVERY_RECEIPT.json','evidence/ORIGINAL_DELIVERY_RECEIPT.json'),(R/'ENGINEERING_AMENDMENT.json','evidence/ENGINEERING_AMENDMENT.json'),(R/'STAGE2_R3_12IID_GATE.md','evidence/STAGE2_R3_12IID_GATE.md'),(P/'STAGE2_CHANGES.md','evidence/STAGE2_CHANGES.md'),(P/'STAGE2_R2_README.md','evidence/STAGE2_R2_README.md'),(P/'STAGE2_R2_DELIVERY.json','evidence/STAGE2_R2_DELIVERY.json')]:copy(src,rel)
for f in ['USER_REQUEST.txt','USER_STAGE2_R2_STRUCTURED_REQUEST.txt','USER_EMERGENCY_THREE_FIXES.txt']:
 if (P/'provenance'/f).exists():copy(P/'provenance'/f,'evidence/requests/'+f)
for f in (R/'audits').glob('*.json'):copy(f,'evidence/raw_audits/'+f.name)
for f in (R/'engineering').rglob('REPAIR.json'):copy(f,'evidence/engineering/'+f.parent.name+'/REPAIR.json')
# Latest requested three-method analysis plus its new paper tables, byte copies.
tree(S,'analysis_three_methods',{'.json','.csv','.md','.txt','.tex','.py','.pdf','.png','.svg'})
# Keep the originally completed four-method analysis explicitly as historical evidence.
tree(R/'stage2_analysis_45iid_20seed','historical_analysis_four_methods',{'.json','.csv','.pdf','.png','.py'})
for f in ['README.md','STAGE2_FULL45_20SEED_REPORT.md']:
 copy(R/f,'historical_analysis_four_methods/'+f)
# A compressed compact report per method and IID; all final routes/objectives and initials retained.
plan=read(R/'plan.json');binding=read(S/'SOURCE_RECEIPT.json');methods=['heuristic','alns_tuned','step5_stage2_r2'];ids=plan['test_ids'];seeds=plan['search_seeds']
assert len(binding)==2700 and len(ids)==45 and len(seeds)==20
r2audits={}
for f in (R/'audits').glob('selected_*.json'):
 d=read(f);assert d['passed']
 for row in d['files']:r2audits[row['iid'],row['seed']]=row
assert len(r2audits)==900
r2sha=[weights[f'direction_{i:02d}.pt'] for i in range(16)]
excluded={'records','variation_events','point_events','worker_pid','worker_created','worker_affinity','resource_snapshot','execution'}
by={};allmeta=[]
for row in binding:
 q=Path(row['path']);stem=q.name;matched=next(m for m in methods if (m=='heuristic' and stem.endswith('_heuristic.json')) or (m=='alns_tuned' and stem.endswith('_alns_tuned.json')) or (m=='step5_stage2_r2' and ('_step5_stage2_' in stem)))
 iid=stem.split('_seed')[0];by.setdefault((matched,iid),[]).append(row)
for ni,iid in enumerate(ids):
 initials={};budget=None;scale=None
 for method in methods:
  entries=sorted(by[method,iid],key=lambda x:x['path']);assert len(entries)==20
  out=V/'results/compact_reports'/method/(iid+'.jsonl.gz');out.parent.mkdir(parents=True,exist_ok=True)
  count=0
  with out.open('wb') as rawout,gzip.GzipFile(filename='',mode='wb',compresslevel=6,fileobj=rawout,mtime=0) as gz:
   for entry in entries:
    src=Path(entry['path']);st=src.stat();assert (st.st_size,st.st_mtime_ns)==(entry['size'],entry['mtime_ns'])
    content=src.read_bytes();digest=hashlib.sha256(content).hexdigest();d=json.loads(content);del content
    seed=d['seed'];assert d['iid']==iid and seed in seeds
    if seed not in initials:initials[seed]=d['initial_population']
    assert initials[seed]==d['initial_population']
    if budget is None:budget=d['budget_seconds'];scale=d['physical_scale']
    assert budget==d['budget_seconds'] and scale==d['physical_scale']
    assert len(d['initial_population']['routes'])==len(d['initial_population']['objectives'])==16
    if method=='step5_stage2_r2':
     proof=r2audits[iid,seed];assert digest==proof['compact_sha256'] and d['checkpoint_sha256']==r2sha and d['frozen_tensors_verified']
    removed=sorted(set(d)&excluded);compact={k:v for k,v in d.items() if k not in excluded}
    assert compact['archive'] and all('routes' in x and len(x['objectives'])==5 for x in compact['archive'])
    gz.write((json.dumps(dict(method_key=method,report=compact),ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode())
    assert (st.st_size,st.st_mtime_ns)==(src.stat().st_size,src.stat().st_mtime_ns)
    STAMPS[str(src)]=(st.st_size,st.st_mtime_ns)
    meta=dict(iid=iid,seed=seed,method_key=method,source=str(src),source_bytes=st.st_size,source_sha256=digest,output=out.relative_to(V).as_posix(),omitted_keys=removed)
    if method=='step5_stage2_r2':meta.update(raw_source=proof['raw_path'],raw_sha256=proof['raw_sha256'],raw_hash_scope='copied from completed independent audit; full step log not repackaged')
    allmeta.append(meta);count+=1
  assert count==20
 if (ni+1)%5==0:print('compact reports',ni+1,'/45',flush=True)
assert len(allmeta)==2700
TRANS=allmeta
dump(V/'results/REPORT_PROVENANCE.json',allmeta)
training=read(V/'training/training_report.json')
assert training['meta_batches']==60 and training['meta_merges']==15 and training['optimized_transitions']==223330 and training['all_direction_targets_met']
for key,val in STAMPS.items():
 st=Path(key).stat();assert (st.st_size,st.st_mtime_ns)==val,key
# Freeze metadata reflects actual selected weights, never the stale top-level R3 plan checkpoint list.
dump(V/'FREEZE_DECLARATION.json',dict(version=NAME,frozen_date='2026-11-02',timezone='Asia/Shanghai',selected_model='r2',scientific_revision='stage2_r2_structured_modes_unified_scalar',immutable_by_policy=True,model_files=17,direction_files=16,weights_sha256=weights,source_archive=str(A),evaluation_campaign=str(R),formal_primary_runs=900,core_reused=36,new_primary=864,comparators=['MOEA/D–ALNS','MOEA/D–ALNS-T'],comparison_runs=2700,training_seed=20261024,search_seeds=seeds,fit_ids=180,validation_ids=45,test_ids=45,meta_batches=60,meta_merges=15,direction_batches=20,optimized_transitions=223330,active_training_wall_seconds=training['active_training_wall_seconds'],no_training_search_or_model_inference_executed=True,full_raw_logs_retained_locally=True,original_r3_plan_checkpoint_list_is_not_selected_r2_binding=True,selected_checkpoint_authority='checkpoints/ plus results REPORT_PROVENANCE and completed raw audit checkpoint SHA lists'))
# Actual original sources remain untouched; consumers relocate a separate working copy.
readme='''# MOEA/D–DPR：Step5 r2 冻结版（2026-11-02）

本目录冻结最终45实例×20搜索seed实际使用的整套r2模型：共享meta权重1份、方向权重16份。不是r3 full12权重，也不拼接方向。36个已有r2 core结果复用、864个新r2结果，构成900个primary；r3的36次门控结果只作为模型选择证据。

## 最新三方法比较

各方法900次运行。每IID以三方法×20seed最终档案形成共同经验非支配参考，确定共同ideal/span；HV参考(1.1)^5，不裁剪。各seed自己的档案独立计分，再取均值/样本SD，最后IID等权。此表是单次运行指标的汇总，不是把20seed合成一个档案后的累计HV。

|方法|经验HV ↑|IGD+ ↓|
|---|---:|---:|
|MOEA/D–ALNS|0.64350326|0.14801332|
|MOEA/D–ALNS-T|0.67935885|0.12803591|
|MOEA/D–DPR（本文方法，r2）|0.71940902|0.10095978|

对45个IID各自的20-seed均值作双侧精确配对Wilcoxon符号秩检验，每scope三对方法×两个指标作Holm6校正。本文方法相对ALNS的HV/IGD+校正p分别1.99646e-7/1.82290e-7，相对ALNS-T分别0.00571114/0.00246130。该三方法事后补充分析与原四/六方法不同经验参考下的数值分开保留。

- [45实例均值、样本SD和显著性](analysis_three_methods/instance_paper_tables/论文表_逐实例与Wilcoxon.md)
- [论文PDF：统计表和逐实例结果](analysis_three_methods/instance_paper_tables/45实例_HV_IGD与Wilcoxon_论文表.pdf)
- [逐seed指标](analysis_three_methods/per_run_metrics.csv)、[共同经验参考几何](analysis_three_methods/summary.json)、[统计核验](analysis_three_methods/AUDIT.json)
- [网络与训练机制](MECHANISMS.md)、[复用和文件范围](ARTIFACTS.md)、[冻结声明](FREEZE_DECLARATION.json)、[核验](VERIFICATION.json)

## 真实训练与选择关系

r2只用180合成FIT、训练seed20261024：60meta批/15次Reptile合并/16方向各20完整PPO批/223330优化转移/4460.9478429活跃秒；encoder、context、q_history_norm冻结。r3 full12是另一次训练，144meta和各方向90批不属于本目录选定权重的训练量。核心12IID×3seed共同六方法经验HV门控选择r2（0.70691758 > 0.68851261），不得拿不同参考下的旧HV当门控阈值。

20个seed是搜索随机种子，不是20个训练重复；180FIT/45验证/45正式实例角色分开。12核心IID用于模型选择，45IID曾被检查，不能称盲测或未见规模外推。两种ALNS都包含轮盘赌自适应选择；ALNS-T是在开发集选参数后固定配置，运行中仍更新算子权重。当前比较支持整体实现的性能结论，动态标量化、邻域替换与动作参数化也有差异，不能单独归因于学习q或选点。

## 文件与执行边界

按仓库既有发布方式保留最终路线、五目标、16个共同初始个体、逐方向数据、预算、计时及诊断；2700报告按方法/IID压缩为135份jsonl.gz，每行一个seed。省略逐步records、variation_events、point_events及运行进程身份；完整原始日志/训练trace/优化器恢复状态仍在原本地目录，不随本次发布。SHA与省略字段逐份记录在results/REPORT_PROVENANCE.json。

training_source为原r2训练与core评测源码；evaluation_source为后续864评测实际有效源码。两份分别保留原锁。r2在后续评测明确使用legacy FP32非学习统计，r3归约修复不冒充r2科学变更。原训练回放trigger未证明和缺diagnostics技术smoke事件在evidence/engineering中分别披露；不覆盖原报告。

源码、配置、输入字节按原件复制，绝对路径保留溯源。运行前应在独立副本重定位数据、物理依赖、父模型和基线缓存；原入口不是便携的一键重训命令，不应直接在冻结目录启动。原plan.json可能含继承/过时字段（尤其r3 checkpoint列表），本版部署权重的权威绑定是FREEZE_DECLARATION.json和900份已审计结果中的SHA。不宣称已完成跨平台运行验证。

验证本发布文件与表格：仓库根执行 `python tools/verify_archive.py`，本版额外执行 `python tools/verify_step5_r2_archive.py`。仅做文件/语法/统计核验，不加载Torch、权重或调用求解器。

冻结后如需算法、配置或权重变更，另建版本；不覆写本目录。本次归档没有训练、搜索、模型前向或新增seed。
'''
(V/'README.md').write_text(readme,encoding='utf-8')
(V/'MECHANISMS.md').write_text('''# r2机制和实际执行来源

正式名称为MOEA/D–DPR，研究版本标识stage2_r2_structured_modes_unified_scalar，是上游DPR的五目标迁移实现，不是原论文原样复现。

## 网络和动作

GAT/GRU编码、context与q_history_norm从原V5 meta热载并冻结；头部以结构化三mode categorical + Beta输出，微动作包含已选有序prefix的PL条件密度。mode0锚点局部簇、mode1全局客户、mode2整路线。mode2首条完整路线优先，允许首条超cap后停止追加。四种经典修复器仍以ALNS自适应选择。固定lo/hi=.04/.25拓扑缓存基分按当前规模插值，加当前physical残差；无外部pulse、旧温度或q_state_mu。物理通道[18,17,19,12,13]。

三处评价（修复候选、环境接受、MOEA/D分解）统一S=max(wz)+.05sum(wz)+.001sum(z)+越界惩罚，z由档案动态ideal/span确定，span至少.05physical_scale。邻域替换独立处理self、以前置邻居快照判重，标量改进取gainTop2；严格支配不占该配额。这些并非与两ALNS实现逐字节相同的外层/评价规则。

## r2实际训练

180FIT，单训练seed20261024。每组4同起点2.5s轨迹，primary回写真实ProgressiveStatePool；Mid/Late由真实车辆下降、停滞或完成窗口接受率标记。优势=.6轨迹项+.4步项；轨迹项含Sstart-Sfinal和新增ND奖励；步项8步折扣.85及已批准停滞/消车奖励，不按dt除。macro与micro分别clip，2 Adam步骤/完整PPO批，四inner后Reptile合并。实际60meta批/15merges，16方向各20批，优化转移223330。r2历史上限4500秒是原版训练事实，不是对r3无总时限契约的恢复。

原请求、预检、initialization、training_report、Q_GATE、原SOURCE_MANIFEST及最终17权重均保留。完整trace与恢复optimizer留在本地；发布不承诺可以从任一训练批继续恢复。

## r2与r3隔离

共享meta和16方向均从stage2_archive_r2复制；900个结果的方向SHA逐份核对。r3模型未部署。本版保存r2原训练源码和后续评测有效源码，因为工程续接入口不同。后续CpuHead以model_source=r2选择legacy_struct_state，保持原r2 FP32统计。r3非学习归约改FP64后castFP32的修复、回放trigger未确认及缺diagnostics事件只作为选择与工程证据保留。
''',encoding='utf-8')
(V/'ARTIFACTS.md').write_text('''# 冻结范围和复用说明

- checkpoints/: r2共享meta1份、方向0..15共16份，逐字节验证原ARCHIVE_RECEIPT和900结果中的部署SHA。
- initialization/: 原V5 meta热启动权重；其SHA是来源绑定，不把它称最终r2方向权重。
- training/: 原r2完整训练报告、初始化、FIT诊断和训练日志，不含完整optimizer/pool/trace恢复包。
- training_source/: 原r2锁定源码及显式补充的父目录支持文件，原锁范围见evidence；补充文件的来源和SHA见仓库ARCHIVE_MANIFEST。
- evaluation_source/: 后续864正式搜索有效锁的完整源文件，保留原锁和工程修复证据；包含历史训练脚本仅为源码溯源，本归档不运行它们。
- data/: 180FIT、45validation、45test原输入，角色分开；另有boxes和114–407秒逐IID预算。
- baseline_sources/: 两种实际ALNS实现和开发集调参凭据。旧V5与Step4不作为最新三方法主表对照。
- results/compact_reports/: 3方法各900份已完成报告；135个gzip JSONL，一行一个seed。只去除逐步大事件和进程身份，保留所有最终archive路线/目标、完整16初始个体和聚合诊断。每份source SHA、转换和省略字段见REPORT_PROVENANCE。
- analysis_three_methods/: 最新共同经验参考HV/IGD+、逐seed指标、均值SD、精确符号秩/Holm6、论文图表和逐实例显著性。逐实例6项校正只在各实例内部，未对全部45IID统一校正。
- historical_analysis_four_methods/: 完成时原报告原样保留，方法集合、参考几何和Holm9均与最新三方法不同，不拿其数值代替最新主表。
- evidence/: 原始审核、r2/r3整套模型选择、SHA锁、父模型关系和工程修复说明。raw_audits中的原始大日志SHA是当时独立审计证据，本次未重新读取全量约50GB逐步日志核SHA。

原process/worker JSON、SSH/凭据、完整逐步日志、3GB训练trace、失败分支模型和r3整套训练资产不发布。它们仍保留本地；本目录不是全磁盘镜像，也不声明遗漏资产已上云。
''',encoding='utf-8')
# Source copies and transformations are independently classified.
files=[dict(path=f.relative_to(V).as_posix(),bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(V.rglob('*')) if f.is_file()]
dump(V/'PACKAGE_CONTENTS.json',dict(version=NAME,selected_model='r2',deployed_direction_models=16,meta_models=1,compact_reports=2700,compressed_report_files=135,original_byte_copies=len(COPIES),report_format='gzip JSONL; one report per seed; full final routes/objectives/16initials; omitted step event and process keys recorded',local_only=['full step records','training traces','optimizer/pool recovery boundaries','r3 weights'],files=files))
arch=read(PUB/'ARCHIVE_MANIFEST.json');known={x['path'] for x in arch['files']};assert not known.intersection(x['path'] for x in COPIES)
arch['files']+=COPIES;arch['latest_addition']=NAME;arch['archive_date']='2026-11-02'
dump(PUB/'ARCHIVE_MANIFEST.json',arch)
# Preserve root's old version descriptions; append the new immutable entry and overview.
root=(PUB/'README.md').read_text();root=root.replace('# 冻结神经破坏网络：V2、V4、V4.3、V5及V5两步Pilot','# 冻结神经破坏网络：V2、V4、V4.3、V5、Pilot与Step5 r2',1)
anchor='| [V5两步Pilot](versions/v5_fix_pilot_20261030/README.md)'
pos=root.index('\n',root.index(anchor))
root=root[:pos]+f'\n| [Step5 r2（最终45×20）](versions/{NAME}/README.md) | 2026-11-02冻结，最新三方法比较各900次 | 实际选定r2共享meta及16方向，非r3 |'+root[pos:]
root=root.replace('归档日期：2026-10-29；V5两步Pilot追加日期：2026-10-30。','归档日期：2026-10-29；V5两步Pilot追加日期：2026-10-30；Step5 r2追加日期：2026-11-02。',1)
root+=f'''\n## 最新冻结：Step5 r2（2026-11-02）\n\n[完整入口](versions/{NAME}/README.md)：整套r2共享meta和16方向权重、180FIT/45验证/45test输入、原r2及后续评测源码锁、2700份紧凑报告、最新三方法共同经验参考统计及逐实例论文表。MOEA/D–ALNS、MOEA/D–ALNS-T、MOEA/D–DPR总体经验HV分别0.64350326/0.67935885/0.71940902，IGD+分别0.14801332/0.12803591/0.10095978；20seed为搜索重复。原四/六方法参考报告单独保留，不能混用绝对值。发布前文件/统计核验不加载模型、不执行新训练/搜索。\n'''
(PUB/'README.md').write_text(root,encoding='utf-8')
for fn,title,body in [('ARCHITECTURE.md','Step5 r2','结构化三mode categorical/Beta/选中prefix PL，统一动态标量化与前置邻居快照判重；共享encoder/context/history归一化冻结。'),('TRAINING.md','Step5 r2','实际60meta批、15merges、16方向各20PPO批、223330优化转移、4460.9478429活跃秒；单训练seed20261024。r3 full12训练量不属于部署r2。'),('RESULTS.md','Step5 r2','最新三方法各900次，主指标共同经验前沿、逐seed独立计分。详细逐IID均值SD与总体/逐IID精确Wilcoxon/Holm6均归档，不与其他方法集合的旧参考直接比较。'),('VERSION_CHANGES.md','Step5 r2','增加正式900次选定r2评测归档；相对旧V5含方向适配、结构化三mode、统一动态标量与邻居替换变化，整体性能差异不能仅归因于学习q。'),('ARTIFACTS.md','Step5 r2','新增17份实际r2最终模型、270输入、135个gzip JSONL共2700紧凑报告，原始逐步日志与恢复断点保留本地。')]:
 path=PUB/'docs'/fn;old=path.read_text();old+=f'\n## 2026-11-02追加：{title}\n\n{body} [版本目录](../versions/{NAME}/README.md)与[文件范围](../versions/{NAME}/ARTIFACTS.md)。\n';path.write_text(old,encoding='utf-8')
dump(V/'VERIFICATION.json',dict(status='pending_independent_archive_verifier',no_models_loaded=True))
print(json.dumps(dict(version=str(V),copied_files=len(COPIES),total_files=len(files),deployed_r2_models=len(weights),report_count=len(allmeta),source_stamps_unchanged=len(STAMPS),compressed_MB=sum(f.stat().st_size for f in (V/'results/compact_reports').rglob('*.gz'))/1e6),ensure_ascii=False),flush=True)
