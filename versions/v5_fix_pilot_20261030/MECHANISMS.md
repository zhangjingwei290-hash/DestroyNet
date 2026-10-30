# V5 两步Pilot：网络、训练和部署变更

归档范围：2026-10-30完成的9真实实例×3搜索seed验证。原V5起点权重为 `checkpoints/meta_complete.pt`，SHA256为 `1a3493e34e4c23f7eae7c465ba6c48a69d49884a01ec80d599eec6cd9c48187f`。这里记录实际代码和运行数据，不将Pilot权重另称正式V6。

## 1. 网络与求解接口

五目标依次为车辆数、总距离、最长路线完成时间、总等待、总迟到，均最小化。外层MOEA/D维护16个方向及对应个体：5个纯轴、10个两轴中点、1个均匀中心。当前方向的后代进入局部搜索；当前方向的物理归一化增强Tchebycheff目标严格改善时接受局部解。非支配档案另外维护。修复器仍为greedy、regret2、regret3、noisy_greedy，由ALNS轮盘赌选择。

保留的网络结构：20维节点输入、45维上下文、隐藏维128；图注意力处理近邻及路线边，GRU式历史更新得到h，节点表示按 `g*(1+h)` 调制。q支路汇集历史、上下文、27维物理结构/偏好动量状态，经过q_body及q_film。选点支路用偏好、q/N和停滞状态进行FiLM调制，结合节点MLP和5×5物理双线性项生成分数。

连续动作由Beta潜变量u与无放回节点顺序组成：

```text
kappa = 8 + 10*sigmoid(q_kappa(q_hidden))
alpha = 1 + mu_prop*kappa
beta  = 1 + (1-mu_prop)*kappa
u ~ Beta(alpha,beta)
q = clamp(round((0.01+0.59*u)*N),1,N)
selected = Gumbel无放回顺序中的前q个客户
```

PPO保留潜变量u的连续密度项及按1/q平均的Plackett–Luce节点log-score；没有改为取整后q的离散边际概率。部署仍随机采样，不使用argmax。CUDA Graph宿主及参数刷新机制保留，首次捕获计入正式搜索时间。

## 2. 第1步：只增加16方向短时适配

`repo/solver/model.py`和`repo/solver/training.py`沿用原V5机制。encoder冻结并逐张量核验；其余可训练参数经Adam更新。树状继承顺序为 `[15]+list(range(5,15))+list(range(5))`：15从原meta初始化，5–14继承15，各纯轴0–4继承首个包含该轴的两轴方向。

每方向保存一份actor，部署时将16个actor作为policies列表传入Search，并按子方向k调用policies[k]。第1步正式适配1744.411秒，计入游标错误、短测和方向3/4补齐后累计2204.629秒。16方向均至少1次有效更新，8965个已优化交互，8个FIT输入，encoder逐张量保持不变。

## 3. 第2步：q参数化、初始化和轨迹评价共同变更

### q输出

原V5根据上一步接受/停滞划分initial/refine/stalled，使用regime锚点 `[0,-1.20,0.85]`；base、regime偏移和indicator分别带0.25/0.30/0.35的前向tanh限幅及直通梯度。

本次删除reg_ids、q_regime_mu、regime_anchors及上述输出限幅，改为：

```text
base_raw = q_mu(q_hidden)
ind_raw = sum(indicators*q_state_mu)
mu_prop = 0.06 + 0.84*sigmoid(base_raw+ind_raw)
```

q_hidden内部的tanh表示层保留；Beta浓度、u到整数q的映射、节点选择和物理约束保留。45维上下文仍包含连续停滞、上次接受、剩余时间和修复器统计。

加载原meta时，仅去掉已删除的q_regime_mu字段，并显式初始化最终q位置支路：q_mu.weight置零、q_state_mu置零、q_mu.bias设为 `-1.1476635056254638`。该标量只用FIT预检校准到随机动作平均q/N约19%；它是训练初值，训练后不强制维持目标比例。其他继承张量保留起点值，第2步encoder也参与PPO更新。

### 同起点四复制、10秒轨迹总收益

每个任务起点复制4条轨迹，路线、节点/边张量和历史状态相同，各有独立动作与修复随机数。每条训练局搜窗口固定10秒求解器预算；并行训练的记账包括该lane的CPU求解工作及计量推理，不把其他lane串行工作或优化器耗时记入该lane求解窗口。正式评测的外层真实墙钟预算及local/repair上限仍沿用原基准。

收益及优势为：

```text
R_m = sum_t max(0,r_mt)
A_m = (R_m - mean(R_1,...,R_4))/(std_population(R_1,...,R_4)+1e-8)
每步优势 = A_m
每步权重 = 1/(4*len(trajectory_m))
```

四条R的标准差不超过1e-12时，优势统一为零；原熵项保留。删去initial/refine/stalled分桶标准化和手工时间惩罚，未使用额外折扣G_t。每条轨迹的总权重为1/4。原学习率0.003、PPO两轮、clip=0.2、entropy=0.001、微批32保留，无Critic。

### 元训练与方向适配

元训练使用5个循环对称偏好任务头、4个同起点复制；配置为4次inner更新后元合并，5头参数算术平均进入共享heads.0。最终恢复段末尾剩余1次inner更新也合并，写入meta_pilot.pt的final_partial_round_updates字段。元参数随后按同一树状顺序适配16方向。

最终有效meta为17批，方向为16批各1次有效更新，合计58834个优化交互。meta跨恢复累计1174.215秒；包含预检、已弃用技术分支和恢复的总成本2590.657秒，上限2700秒。仅5个FIT输入发生有效更新，规模为150或250。训练seed为20261024。

源尾部判定工程错误曾使首段meta在4批后提前结束并过早进入方向适配；该分支保留。最终从4-inner更新的完整元合并检查点恢复，Adam/RNG因未被旧入口序列化而重建，旧方向分支未作为最终权重。详细数据见training_report.json及README。

## 4. 对比设置和实际结果

评测五方法：alns_tuned、v5_meta_frozen、step1_adapted、step2_meta_only、step2_adapted。ALNS和旧V5复用原45实例×20seed中的同IID/seed档案，未新增搜索。新增81次搜索=第1步27次+第2步共享单头27次+第2步方向模型27次。

共享单头是第2步元训练产出的meta_pilot.pt，各方向共享heads.0。方向模型分别加载direction_00.pt至direction_15.pt。所有搜索沿用完整16个初始个体、逐实例原预算、五目标/约束及随机采样。复用基准与新模型运行时期和主机负载不同。

|方法|HV均值|IGD+均值|冻结评测q/N均值|
|---|---:|---:|---:|
|alns_tuned（复用）|0.814569|0.082043|20.234%|
|v5_meta_frozen（复用）|0.803593|0.076046|29.311%|
|step1_adapted|0.781318|0.080379|28.821%|
|step2_meta_only|0.800145|0.069689|16.110%|
|step2_adapted|0.793160|0.073218|15.877%|

同一实例由五方法×3seed共同经验ND确定归一化尺度；每seed自己的档案分别算HV/IGD+，再求3seed均值/样本SD，跨9IID等权。完整表、10槽分布、吞吐量、16方向标量及条件q分布见README和step2_analysis。8项相对ALNS的9实例精确配对符号秩检验采用同报告Holm校正。

本次对比同时改变q参数化、最终q位置支路初始化和训练评价目标，数据对应共同变更。冻结评测中stalled的q均值低于refine；N=150和250步数超过ALNS，N=50仍低于ALNS。TW0仅150-2-0与250-1-0，没有50/TW0。单训练seed、实例间相关及中途查看均在README列明。

## 5. 文件与执行

- `repo/`：第1步网络和原训练目标；`step2_repo/`：第2步修改后的网络、训练器、求解器。
- `checkpoints/meta_complete.pt`：原V5起点；`step1_training/checkpoints/`：第1步16方向权重。
- `step2_training/meta_pilot.pt`：第2步共享单头；`step2_training/checkpoints/`：第2步16方向权重。
- `train.py`、`stage2_train.py`、`resume_stage2_bounded.py`：实际训练包装器；`eval.py`、`eval_step2_both.py`、`resume_eval_step2.py`：评测和缺失项续跑；`analyze.py`：逐seed指标和表格计算。
- `physical/`、`data/`、`inputs/`、config.json、boxes.json：物理源码、输入/划分、9例、参数与物理归一化盒。
- `baseline_cache/`与两阶段`metric_inputs/`：135份紧凑报告，保留最终档案的路线/目标、完整初始群、方向结果、q诊断及计时；不含逐步records、variation_events、point_events。完整逐步原始报告和已弃用方向模型保留在本地实验目录。
- `step2_analysis/VERIFICATION.json`：独立重算全部135个指标及新增29442档案解物理核验；`PACKAGE_CONTENTS.json`与仓库manifest提供文件SHA256。

源代码和配置按原字节归档，保留Windows/本地绝对路径。实际包装器以自身目录作为ROOT；正式运行前应在独立副本重定位config/study_root/split_manifest/project_root/baseline_root，并为评测入口提供所需原基准文件。本发布保留紧凑结果，完整逐步日志由本地实验目录保存。归档验证只检查文件、语法和已有结果。
