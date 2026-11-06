# 与上一 GitHub 冻结版本的逐项差异

对照基准是提交 `b16ca2318add176e8f9a81101f141f4f8d90f65e` 中的 `versions/v5_step5_r2_frozen_20261102`。中间SA5 V2–V8属于本地开发序列，不把它们冒充上一GitHub主版本。

|项目|上一冻结R2|本次SA5 V9|
|---|---|---|
|原方向专家|16份适配权重|相同16份原权重，逐方向对应加载并冻结|
|附加部署网络|无本次Controller|新增共享22→32→4 Controller，868参数|
|新增训练对象|原结构化策略元训练与方向适配|只训练共享Controller和769参数Critic|
|训练组织|4同起点2.5s轨迹；60meta批、15Reptile合并、16方向各20批|fit180池抽18IID；三规模各180s后联合更新，共6轮/12优化步|
|训练输入覆盖|原R2训练报告记录的180FIT及动态状态池|实际18个不同fit IID，早中晚按在线时间三等分，144组合均衡|
|奖励|轨迹与步级复合项，含ND/停滞等原训练项|一步相对标量改善减EMA机会成本×同步波时间|
|HV使用|离线评测与历史模型选择|validation45选Controller检查点和离线评测；不进训练奖励|
|模式控制|专家三模式logits|专家logits加Controller前三输出|
|Beta粒度控制|专家各模式Beta参数|保持各模式浓度，对中心logit加Controller第四输出|
|Mode0/1 q|不同模式的指数映射，下界2/3|统一4%–35%带绝对边界的线性客户数映射|
|Mode2 q|先采样整路线数量，再受cap约束，可截末车客户|先采样客户数目标，PL整车前缀累计达到目标停止，整车可超目标|
|客户/路线打分|原物理双线性与图分数|保持各方向原矩阵；无额外手工偏好加分|
|单次局搜结束|min(10秒,T/16,剩余整场时间)|每方向最多5次破坏—修复，整场截止仍生效|
|当前解接受|原严格标量改进|五步SA，等分/改进接受，较差按温度概率接受|
|返回外层的解|原session执行结果|该五步会话最好解|
|方向调度|每个run内方向局搜按原MOEA/D流程串行|一批16方向，GPU编码成批，修复16进程同步波|
|缓存刷新|repair_amortization_or_clean_age8|每新会话一次GPU编码，其余步复用；已接受解更新物理统计|
|GRU|原refresh时递推|会话入口零hidden，只编码一次；会话后续步不再递推|
|轻头推理|旧R2已含CPU头，legacy FP32统计|CPU NumPy头+Controller；物理统计float64归约后cast FP32|
|跨会话方向状态|原局搜/ALNS状态|额外方向停滞计数、上次修复秒数和EMA历史|
|外层替换|R2动态评分、self/邻居快照、支配优先、gainTop2|仍调用同一原_replace；批内子代来自同一父快照，批末依采样方向顺序回写|
|主评测|45IID×20搜索seed，各方法900次|9IID×3搜索seed，各方法27次|
|对照|ALNS、ALNS-T|同SA5框架roulette和V8 NumPy Gate|
|本机数值并发|历史3个run并发，各固定CPU对，单次局搜串行|单run内16修复进程绑定6逻辑CPU，主进程2逻辑CPU|

## 对应源码

- 新Controller、特征和q：`source/repo/v9_policy.py`。
- 原专家加载和Beta中心变换：`source/repo/network.py`。
- CPU动作采样和整车前缀：`source/repo/aligned_cpu.py`。
- 联合PPO与144组合权重：`source/repo/gate_ppo.py`。
- 五步SA/批次执行和动作奖励：`source/repo/evaluate_case.py`。
- EMA：`source/repo/opportunity.py`。
- 采样、检查点、验证选模和27次测试：`source/coordinator.py`。
- 原R2 q映射和route_removal：上一版本 `evaluation_source/stage2_structured.py`。
- 原评分/缓存刷新：上一版本 `evaluation_source/stage2_repo/stage1_mechanisms.py`。

16原专家的继承不包含“用direction15初始化全部方向”的SA5 V2初始化路径。V9不加载V8 Gate权重，Controller从本次initial.pt起点训练；V8只是本次复用对照之一。
