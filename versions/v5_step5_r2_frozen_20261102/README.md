# MOEA/D–DPR：Step5 r2 冻结版（2026-11-02）

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
