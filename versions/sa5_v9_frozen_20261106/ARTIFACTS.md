# 工件范围、来源与执行

## 发布目录

|目录|包含内容|
|---|---|
|source/|实际V9 coordinator与repo源码，按原字节复制|
|checkpoints/experts/|16份与上一R2版本对应的原方向专家|
|checkpoints/controller/|initial、cycle1..6、v9_best_gate，共8份state_dict文件；best是cycle3的字节副本|
|data/INPUT_ROLES.json|引用仓库上一版本已冻结的180fit/45validation/45test输入；列出实际18fit和本次9test|
|training/rollouts/|18场训练的compact PPO行、会话收据与排除计数；gzip JSON|
|training/COLLECTION_COUNTS.json|每场实际seed、预算、在线时间、优化样本和排除动作计数|
|results/test/|roulette/V8/V9各27份，81份最终档案与初始种群|
|results/validation/|cycle3/6各45份，90份最终档案|
|results/training/|18场训练搜索最终档案|
|results/smoke/|3场16秒未训练Controller技术搜索|
|results/historical_dbmt/|15份已有DBMT原生最终档案和原始物理审核字段|
|results/REPORT_PROVENANCE.json|每个转换报告的原路径、源SHA、发布SHA和省略字段|
|analysis/main/|81指标、27实例方法均值SD、macro、exact/Holm4、胜平负与吞吐|
|analysis/historical/|R2、DBMT两套独立参考的脚本、几何、指标、审计与报告|
|diagnostics/history/|SA5开发阶段已有探测的机器可读摘要、表格和脚本|
|evidence/|原计划、来源锁、训练/选模/完成记录、CORE/PPO/SMOKE检查、各场AUDIT|
|PACKAGE_CONTENTS.json|本版本文件内容摘要|
|FREEZE_DECLARATION.json|原GitHub父版本、Controller选择、专家绑定、训练与评测计数|

报告转换只省略records、batches、point_events、archive_snapshots四类大事件字段；若原报告不存在某字段，省略列表不会补造该字段。最终archive的路线/五目标、population（原报告存在时）、initial_population、逐方向统计和源预算等保留。缺失batches后不能直接以len(batches)重算吞吐；原始批次/尝试数在REPORT_PROVENANCE与发布per_run.csv中保留。

训练session_rollout.pt转换保留compact rows和sessions等字段，只省略full_replay_proof里的节点/图大张量。发布样本含原base模式/Beta参数、采样mode/u、behavior密度、22维输入、stage和实际一步训练目标，足以重放Controller的changing-factor PPO项。没有为这次发布重新计算动作或收益。

## 依赖与路径

共享原solver、物理模型、原专家模型定义、数据和eval_config在同仓库上一版本 `v5_step5_r2_frozen_20261102`。`evidence/SHARED_DEPENDENCIES.json`列出本次有效源码锁中引用该版本的路径和SHA。原锁135项及本次本地路径按原件保留。

原始入口保留D:/workspace等绝对路径，用于来源追踪。运行时应在独立副本重定位R、F、CASE_ROOT、GATE_PATH、STAGE_MANIFEST和基线报告缓存；不要直接在冻结目录启动coordinator。原入口带claim及输出目录已存在检查，发布目录没有复制旧活动claim或PID状态。这里没有新增便携训练器，也未执行跨平台运行验证。

`training=True`是evaluate_case原报告的继承字段，在训练、验证和测试结果中都存在。实际是否更新由coordinator的阶段与gate_ppo调用位置决定；测试/验证只加载冻结Controller，未执行optimizer.step。本包按stage目录标识这些角色。

## 本地保留与未落盘项

完整逐步记录、训练full_replay_proof、PID/worker/monitor状态及所有大stdout日志留在原本地实验目录；本次不发布运行身份或本机凭据。Critic与optimizer状态在原coordinator中未保存，发布过程没有补造它们。历史诊断的原REPORT可能带当时的解释性结论，本包发布机器摘要与代码，并在DIAGNOSTICS.md以实验范围和数字重述。

R2历史比较的180份原始紧凑报告已在上一冻结版本results/compact_reports/step5_stage2_r2；DBMT15份原生紧凑报告新增在本版。主表V8和roulette工件只读复用，没有为发布重跑。SHA用于绑定实际来源和冻结权重，不代表对所有原始逐步物理过程重新执行搜索。

## 发布核验

`python tools/verify_v9_archive.py`执行本版本文件内容核对、Python语法解析、各阶段计数、专家和Controller绑定，以及已发布指标的均值、SD、exact signedrank与Holm4复算。核验不加载神经模型，不执行训练、推理或局搜。包含moocore环境时可对紧凑档案另作离线指标重算；这不增加搜索。

## 继承来源锁中的额外条目

原SOURCE_MANIFEST除V9与R2依赖外，还核对SA5 V2方向权重、V6 Gate、旧aligned计划和probe.py等继承条目。它们被保存在evidence/inherited_lock_assets/，路径映射见INHERITED_LOCK_ASSETS.json；这些继承条目不等于V9实际部署权重。V9实际加载的16专家和Controller仅以FREEZE_DECLARATION.json为准。split_v5.json继续引用仓库已提交的原文件。
