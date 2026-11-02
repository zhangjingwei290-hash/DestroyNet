# Step5最终交付：r3训练与整套模型选择后的45×20评测

全部900个primary结果已完成并通过最终独立核验。选择模型为整套r2：36个既有core结果复用、864个新增primary；另保留36个r3 gate结果，合计900次新增正式搜索。ALNS与旧V5各900个既有结果复用，Step4仅673个既有结果，227缺失不补跑。

训练完成144个meta批次、720组、每FIT4访、36次元合并，以及16方向各90完整PPO批。优化转移824323，活跃训练39099.280109秒；诊断与暂停不混入训练时间。12工作量等效遍数不等于每FIT实际12访。最终encoder冻结、选中前缀density、调度、权重初始化、原始档案可行性/五目标/16初始群/标量/缓存/权重绑定及统计重算均见AUDIT.json和audits/。

## 最终质量

|规模|方法|经验HV|IGD+|固定物理盒HV|
|---|---|---:|---:|---:|
|overall|alns_tuned|0.72551858|0.12400301|0.24408640|
|overall|v5_meta_frozen|0.71453711|0.12234765|0.26145514|
|overall|step4_k4_adapted|0.71521524|0.12050410|0.26273574|
|overall|step5_stage2_r2|0.74166586|0.11538604|0.21718871|
|50|alns_tuned|0.95098529|0.06098837|0.25206747|
|50|v5_meta_frozen|0.93529454|0.06295718|0.25688168|
|50|step4_k4_adapted|0.93316625|0.06357116|0.25858370|
|50|step5_stage2_r2|0.97099753|0.05794877|0.23760104|
|150|alns_tuned|0.62133972|0.13917057|0.25359122|
|150|v5_meta_frozen|0.59037633|0.14271122|0.27237839|
|150|step4_k4_adapted|0.58827013|0.14141839|0.27697149|
|150|step5_stage2_r2|0.64102453|0.13083422|0.22560696|
|250|alns_tuned|0.60423072|0.17185008|0.22660049|
|250|v5_meta_frozen|0.61794047|0.16137456|0.25510536|
|250|step4_k4_adapted|0.62420933|0.15652274|0.25265202|
|250|step5_stage2_r2|0.61297552|0.15737514|0.18835814|

r2经验HV/IGD+总体均值优于ALNS，但45IID配对符号秩检验及Holm9校正未显著（HV p=.31289238，Holm=1；IGD+ p=.27151946，Holm=1）。固定物理盒HV显著较差（对ALNS Holm=4.6584e-7），两种尺度结果均保留。250规模r2经验HV高于ALNS、低于旧V5及Step4，IGD+低于ALNS/旧V5但略高于Step4。Step4表均值覆盖673次，与r2对照检验仅用双方共同673对，不能把其不完整宏观均值当900次完整配对。

## 机制与资源

|N|平均步数/run|端到端策略秒/run|修复秒/run|GNN刷新秒/run|缓存命中率|
|---|---:|---:|---:|---:|---:|
|50|23905.75|23.327|148.896|7.086|86.15%|
|150|11953.94|15.704|221.234|6.793|79.94%|
|250|8937.43|15.117|278.920|8.070|73.74%|

策略耗时是含CPU抽样、刷新/传输/同步的端到端服务墙钟，非纯CUDA时间。三mode使用比例与接受/候选及实际消车、首车超cap计数见mode_distribution.csv。实际q/N精确11区间见q_distribution.csv；状态0..8与9+见state_awareness.csv。原始report的q_diagnostics保留10最近中心槽/精确区间及phase信息，r2旧core缺少新字段的部分不补造。方向行为见direction_behavior.csv、16方向实际点见per_direction.csv；车辆档位/对应距离时长见vehicle_knee.csv，不拼接不同解的目标。

## 评价口径与限制

逐IID把当前方法及全部指定seed档案联合形成经验ND参考和共同尺度；每seed自己的档案独立计算HV/IGD+，之后seed均值/样本SD，最后IID等权。HV参考1.1^5，不裁剪；近零跨度统一置零保留5D。宏观±为完整seed的宏观均值之间SD，另列between-IID SD；逐IID±跨20seed。经验参考并非真实最优前沿。

12核心IID用于整套r2/r3模型选择，45IID也曾检查，结果属于选择条件下的描述性基准，不称盲测，也不能归因于单项训练机制。旧基线来自更早的主机负载时期；同IID/seed是统计配对，不是同时同worker执行。完整45包含9个TW0，其中3个为50客户，不能套用旧9IID pilot无50TW0的描述。

## 工程事件与证据

1. engineering/replay_guard_direction05_batch003/：原训练回放guard触发，原确切transition未保存，触发根因未证明。独立复现近常量特征FP32归约差异，非学习统计改用FP64归约再转FP32，网络/梯度仍FP32，阈值不放宽。保留144meta及dir15更新，dir05前两批从原trace重建，loss差0；恢复已有状态不增加批数。r2部署保留原FP32统计。
2. engineering/smoke_missing_diagnostics/：训练和FIT gate完成后，初次技术短测在Search构造前因漏带diagnostics模块失败，0正式搜索。保留原source/log/manifest/process，原字节复制父模块，Windows spawn导入验证通过；只续接评测，不重训、不重跑FIT gate。17个完成权重SHA不变。

原SOURCE_MANIFEST保留，工程差异由ENGINEERING_AMENDMENT.json串接历史和当前EFFECTIVE_SOURCE_MANIFEST.json。旧来源目录共12035文件只读检查通过，changed=[]。原报告和原始搜索结果未覆盖；本README补充工程交付说明。

## 文件索引

- STAGE2_R3_12IID_GATE.md：六方法、12IID×3seed整套模型选择。
- STAGE2_FULL45_3SEED_REPORT.md：保留首批45×3报告。
- STAGE2_FULL45_20SEED_REPORT.md：完整45×20质量、WTL、检验和逐IID均值SD。
- stage2_analysis_45iid_20seed/summary.json、per_run_metrics.csv、main_table.csv、macro.csv、comparisons.csv、missing.csv：机器可读结果。
- 同目录mode_distribution/q_distribution/state_awareness/direction_behavior/per_direction/vehicle_knee/throughput.csv以及benchmark.png/pdf、modes.png/pdf：诊断与图。
- AUDIT.json、COMPLETE.json、audits/：最终独立核验与逐批原始结果凭据。

本任务至此结束，不追加训练、seed、阶段或发布。
