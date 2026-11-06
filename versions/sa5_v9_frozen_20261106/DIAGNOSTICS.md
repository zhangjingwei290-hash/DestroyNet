# 已运行的探测、技术检查与开发试验

下表按本地目录记录列出。SA5 V2–V8为V9之前的本地开发编号，和仓库早期独立V2版本不是同一个编号空间。表中的历史实验没有在本次发布中重跑，历史实验的权重也不作为本次V9选模候选。JSON/CSV原记录和可取得的脚本在 `diagnostics/history/`，本页只转述范围、次数、改动和数值。

## V9本次检查

|检查|实际操作|记录|
|---|---|---|
|Controller数值一致性|4096个合成上下文，FP32 NumPy与GPU输出比较|最大绝对误差2.2351741790771484e−8；阈值1e−6|
|通用轻MLP数值一致性|Linear/Tanh/LayerNorm/ELU组合前向比较|最大绝对误差5.960464477539062e−7|
|四输出梯度|用真实Categorical+Beta PPO terms构造损失，反传Controller|输出层四行梯度范数约0.09858、0.09627、0.08154、0.61915|
|专家来源|读取direction00..15，比较物理/路线矩阵和encoder|16个不同physical矩阵、16个不同route矩阵、encoder张量相同|
|q与整车前缀|检查统一范围、线性映射单调性、整车前缀结果|`evidence/CORE_CHECK.json` passed=true|
|三规模真实冒烟|N50/150/250的fit d0tw0实例，各16秒、seed20261107，未训练Controller|分别2209/688/432次尝试；28/9/6批；三种模式均出现|
|冒烟实际粒度|统计上项所有实际移除客户数|均值q分别10.6804、27.6366、33.3542；含Mode2整车超目标|
|真实轨迹PPO重放|三规模冒烟的2160/640/400条完整会话动作，共3200条，联合2个epoch|更新前最大abs(logratio)=9.5367431640625e−6；各规模总权重约1/3|
|正式每轮覆盖|6轮实际采样检查3规模×3阶段×16方向|每轮144组合非空，最小组合计数87/90/119/105/109/95|
|正式物理审计|每场覆盖、容量、时间窗延误上限、返仓与五目标独立重算|18训练、90验证、27 V9测试的AUDIT passed=true|

冒烟16秒是技术检查的整场预算；它没有替代正式180秒训练或117–338秒测试。`check_rollout.py`中的更新只用于检查，更新后的临时Controller没有作为正式训练初始权重；正式训练加载保存的initial.pt。

本次没有针对训练后V9另外运行冻结状态的偏好/规模/停滞反事实响应面板、模式粒度动作价值枚举或多训练seed实验。上述梯度与数值检查记录的是实现行为，主表记录的是完整搜索指标。

## V9之前的本地探测

|目录标签|被检查对象和处理|已记录范围/数量|本包证据|
|---|---|---|---|
|frozen_response|固定真实解，分别改变偏好、档案摘要、步数/温度和全局阶段；记录q、mode、客户分数|各条件96专家×状态块|range_summary、responses汇总、诊断脚本|
|coverage_v2_response|同面板比较原SA5策略与coverage_v2|旧/新条件响应、客户top10重合率等|old_new_comparison.json、comparisons.csv|
|action_value|同状态下网络动作与模式/粒度候选，跨修复重复；档案敏感性与5→15步续搜|2160候选，2158有效；另有crossfit、continuation与固定偏好记录|candidate_panel.csv、summary.json、continuation_summary.json、fixed_preference_summary.json|
|richstate_v3|增加解集特征、Critic状态和FiLM等组合修改；同权重比较5/10步|训练/冻结验证合计934.173秒；12场60秒开发搜索；16方向门控中12个已验证，8个保留meta|COMPLETE、training/COMPLETE、summary、response汇总|
|rloo_stallq_v4 quick1|停滞M=3后Beta中心+0.10；模式概率/浓度不变|3IID×2组×1seed，6场60秒|summary.json中的quick1|
|rloo_stallq_v4 quick2|同起点4轨迹的PPO继续训练与RLOO继续训练|288次更新、5760转移、307.852秒；3IID×3组=9场60秒|training/COMPLETE和summary|
|cross_session_v5|跨会话停滞输入；共享粒度分支；原专家冻结|288更新/5760转移；收集3场；3IID×3seed×5组=45场60秒|COMPLETE、AUDIT、summary、training/COMPLETE|
|state_action_v6|固定状态下原始、缩小/扩大粒度及状态条件分支的检查|原目录的阶段记录和脚本|diagnostics/history/state_action_v6/|
|state_action_v6_stage2|1089参数状态分支，固定留出面板比较五步标量改善|288更新/5760步；每规模12状态×4重复×2组；总训练/留出470.330秒|training/COMPLETE、heldout_summary（若原记录存在）|
|time_objective|核对原训练终止、全局剩余时间、局搜耗时与标量收益|已有单进程五步配对数据，粒度中心偏移−.10/0/+.10|summary.json和audit.py|
|global_return_v7|将完整宿主回报用于共享分支学习|6场训练、6场留出；留出每场30秒；总528.874秒|COMPLETE及结果summary|
|opportunity_v8技术链路|5步会话净收益、34维FiLM Gate+Critic、三规模联合梯度|每规模6秒，176完整会话/880动作，2个优化步|technical_check_corrected/COMPLETE与IMPLEMENTATION_AUDIT|
|v8_joint18|18周期三规模联合训练与独立选模|36优化步，两个Pass，各训练实例180秒；两个检查点各3个验证IID|TRAINING_COMPLETE、training_history、VALIDATION_SELECTION|
|v8_numpy|同一V8 Gate从GPU逐步调用改为CPU NumPy|5122上下文，其中768真实；误差5.587935447692871e−9；随后27场测试|NUMERICAL_EQUIVALENCE、COMPLETE；27份档案作为主表V8对照|

## 历史快检中已记录的数值

V4 quick1的停滞中心偏移组，相对该阶段fresh V2的HV变化，在50/150/250三个60秒开发实例上为+1.19%、+8.26%、+9.70%。同一目录quick2中的RLOO继续训练，相对其自身阶段fresh V2为+0.30%、−12.30%、+0.06%。两阶段分别重建参考集，绝对HV分开记录。

V3组合修改的五步组，相对其同阶段V2五步组，在三个开发实例的HV变化为−4.93%、−9.20%、−9.29%；十步组相对同一V2组为−6.08%、−14.41%、−7.03%。V3训练/冻结验证934.173秒，原900秒组合上限检查字段为false；更新上限600秒，额外34.173秒为冻结验证，原记录中精确meta/adapt更新数为null。

V5五组比较中，训练分支相对同阶段V2的三个规模HV变化为+0.33%、−4.12%、+3.58%。实际训练是均值目标PPO和原固定Critic；没有执行停滞期Best-of-K赢家轨迹目标。

V6第二阶段留出面板的关闭分支/训练分支五步标量改善均值分别为：N50 0.114849/0.115381，N150 0.041337/0.042376，N250 0.022713/0.023457。这些是固定面板的五步标量值，不是整场HV。

V8 Gate的1024次模块调用记录为GPU 0.573201秒、NumPy 0.0205595秒；该计时只覆盖Gate模块，不等于完整求解器加速比。V8 NumPy阶段的27份结果和V9主表在共同参考下重新计分。

## 本次历史离线重算

- V9与原R2长会话：9IID，原R2每IID20seed、V9每IID3seed，共207份已有档案；没有新增搜索。
- V9与DBMT原生实现：DBMT共15份，N50/150每IID2seed，N250每IID1seed；V9共27份，共42份已有档案；没有新增搜索。
- 两次重算的共同参考独立构造，seed不重合，CPU调度配置不同。所用档案、指标与限定条件见 `analysis/historical/`。
