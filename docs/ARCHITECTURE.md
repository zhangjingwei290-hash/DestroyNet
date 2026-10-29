# 网络架构与动作

## 共用求解对象

五目标依次为车辆数、总距离、最长路线完成时间、总等待、总迟到，均最小化。外层MOEA/D使用16方向：5个纯轴、10个两轴中点和均匀中心。局搜使用greedy、regret2、regret3、noisy_greedy四个经典修复器；网络负责破坏动作。客户覆盖、容量、允许迟到上限、回仓期限由执行与完整核验处理。满足当前方向标量严格改进才局部接受；非支配档案与局部接受不是同一判断。

## V2

17维节点、45维上下文，隐藏128。HRGCN/GAT处理近邻和路线关系，GRU历史更新后使用g*h逐元素乘法门控。context经Linear+Tanh得到128维；节点均值与context拼接为256维全局状态。

q头输出分类logits；比例列表为1/2/5/10/20/40/60/80/100%，按ceil(N×比例)去重。位置头依次采样至多2个不重复锚点，再采Beta系数，调用DPR路线片段删除函数。训练动作包含q、锚点和Beta项。元训练共享主体、5套任务头；PPO另有每任务Critic。部署保存16个适配actor。

源码：versions/v2/source/moea_emnh_v2/model.py、versions/v2/source/lib/ppo_1.py、versions/v2/source/mo_dpr/environment.py。

## V4 Run C

节点增为20维，新增detour、路线客户比例、makespan gap。GAT/GRU隐藏128，历史调制g*(1+h)。偏好、q/N、停滞经FiLM调制节点读出，另有5×5物理双线性项。按节点分数Gumbel无放回选择q个客户。q为1/2/5/10/20/30/40/60/80/100%十个分类槽；训练中的节点log-score按1/q平均。PPO使用同起点轨迹组基线，没有Critic。

源码：versions/v4/source/solver/{model,network,environment,training}.py。

## V4.3 Run F

20维节点、45维context、128维GAT/GRU、FiLM及物理双线性选点保留。q改为连续潜变量u~Beta(alpha,beta)，rho=.01+.59*u，整数q=round(rho*N)，并受合法范围约束。节点按Gumbel无放回采样。

q模块汇集归一化历史、context、15维物理结构统计和12维动量/偏好状态；FiLM调制。结构统计来自物理通道[18,17,19,12,13]的变异系数、峰值、top10占比。

三状态由决策前历史确定：上一步接受为refine；否则stagnation>0为stalled；其余initial。regime锚点[0,-1.20,.85]和7个状态系数是实现常量/参数初始化的一部分。前向有界、反向直通。mu_prop=.06+.84*sigmoid(regime+base+indicator)，kappa=8+10*sigmoid(head)，alpha=1+mu_prop*kappa，beta=1+(1-mu_prop)*kappa。mu_prop是分布模式相关参数，不等于实际采样均值。

Beta训练项使用保存u的连续密度；节点项按1/q平均。因此组合PPO score并非舍入后离散联合动作的精确概率比。CUDA Graph按device/N/batch/greedy共享宿主，动态刷新输入及参数；首次capture与参数搬运计入正式评测。

源码：versions/v4_3/source/solver/{model,network,cuda_actor,training,environment}.py。

## V5

沿用F的网络与动作结构，元训练改为8个IID调度、7个CPU修复进程、GPU集中微批32；训练数据与时间预算见TRAINING.md。最终真实45例评测只加载一个冻结元任务头，16个偏好方向共享；没有测试时适配或重训。

节点通道（0-based）：0/1中心化坐标，2需求/容量，3/4时间窗起止，5服务时间，6窗口宽度，7前缀载重，8前缀距离，9路线载重，10路线距离，11到达，12等待，13迟到，14硬时间余量，15回仓余量，16仓库标记，17绕路距离，18路线客户数/N，19距最长路线完成时间之差。具体缩放见各版base_environment.py。

context45维：偏好5、归一化目标5、全局剩余秒/已用比例/局搜剩余秒/剩余比例4、停滞/上次接受/上次奖励/档案比例4、允许迟到比/N比例/最大q比例3、当前修复器one-hot4，以及四修复器各自概率/近期耗时/成功率/改善/计数20。

源码：versions/v5/source/solver/{config,model,network,environment}.py。实际权重SHA256见versions/v5/README.md和ARCHIVE_MANIFEST.json。
