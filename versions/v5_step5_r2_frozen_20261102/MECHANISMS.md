# r2机制和实际执行来源

正式名称为MOEA/D–DPR，研究版本标识stage2_r2_structured_modes_unified_scalar，是上游DPR的五目标迁移实现，不是原论文原样复现。

## 网络和动作

GAT/GRU编码、context与q_history_norm从原V5 meta热载并冻结；头部以结构化三mode categorical + Beta输出，微动作包含已选有序prefix的PL条件密度。mode0锚点局部簇、mode1全局客户、mode2整路线。mode2首条完整路线优先，允许首条超cap后停止追加。四种经典修复器仍以ALNS自适应选择。固定lo/hi=.04/.25拓扑缓存基分按当前规模插值，加当前physical残差；无外部pulse、旧温度或q_state_mu。物理通道[18,17,19,12,13]。

三处评价（修复候选、环境接受、MOEA/D分解）统一S=max(wz)+.05sum(wz)+.001sum(z)+越界惩罚，z由档案动态ideal/span确定，span至少.05physical_scale。邻域替换独立处理self、以前置邻居快照判重，标量改进取gainTop2；严格支配不占该配额。这些并非与两ALNS实现逐字节相同的外层/评价规则。

## r2实际训练

180FIT，单训练seed20261024。每组4同起点2.5s轨迹，primary回写真实ProgressiveStatePool；Mid/Late由真实车辆下降、停滞或完成窗口接受率标记。优势=.6轨迹项+.4步项；轨迹项含Sstart-Sfinal和新增ND奖励；步项8步折扣.85及已批准停滞/消车奖励，不按dt除。macro与micro分别clip，2 Adam步骤/完整PPO批，四inner后Reptile合并。实际60meta批/15merges，16方向各20批，优化转移223330。r2历史上限4500秒是原版训练事实，不是对r3无总时限契约的恢复。

原请求、预检、initialization、training_report、Q_GATE、原SOURCE_MANIFEST及最终17权重均保留。完整trace与恢复optimizer留在本地；发布不承诺可以从任一训练批继续恢复。

## r2与r3隔离

共享meta和16方向均从stage2_archive_r2复制；900个结果的方向SHA逐份核对。r3模型未部署。本版保存r2原训练源码和后续评测有效源码，因为工程续接入口不同。后续CpuHead以model_source=r2选择legacy_struct_state，保持原r2 FP32统计。r3非学习归约改FP64后castFP32的修复、回放trigger未确认及缺diagnostics事件只作为选择与工程证据保留。
