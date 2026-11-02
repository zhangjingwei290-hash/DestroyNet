# 冻结神经破坏网络：V2、V4、V4.3、V5、Pilot与Step5 r2

归档日期：2026-10-29；V5两步Pilot追加日期：2026-10-30；Step5 r2追加日期：2026-11-02。该仓库存放V2、V4、V4.3、V5、两步Pilot和Step5 r2的冻结源码、权重、训练配置及已完成实验数据。V3、V4.1、V4.2列入演进关系，没有在本仓库另立冻结权重版本。这里的网络是**破坏网络**；经典修复器与HEMT神经插入修复网络是不同组件。

| 目录 | 冻结对象 | 部署权重 |
|---|---|---|
| [V2](versions/v2/README.md) | 2026-10-23乘法GRU门控版，PPO与REINFORCE | 各最终元权重及16方向适配权重 |
| [V4](versions/v4/README.md) | Run C，全粒度分类动作 | 元权重及16方向适配权重 |
| [V4.3](versions/v4_3/README.md) | Run F，连续Beta粒度、CUDA Graph | 元权重及16方向适配权重 |
| [V5](versions/v5/README.md) | 三轮180合成FIT元训练、真实45例冻结评测 | 900次V5搜索实际使用的meta_complete.pt |
| [V5两步Pilot](versions/v5_fix_pilot_20261030/README.md) | 9真实实例×3seed；方向适配、q输出和轨迹总收益目标变更 | 第1步16方向；第2步共享meta与16方向 |
| [Step5 r2（最终45×20）](versions/v5_step5_r2_frozen_20261102/README.md) | 2026-11-02冻结，最新三方法比较各900次 | 实际选定r2共享meta及16方向，非r3 |

V5两步Pilot的[机制与训练方式](versions/v5_fix_pilot_20261030/MECHANISMS.md)、[五方法逐seed结果](versions/v5_fix_pilot_20261030/step2_analysis/main_table.csv)、[完整三张表](versions/v5_fix_pilot_20261030/step2_analysis/REPORT.md)及[独立核验](versions/v5_fix_pilot_20261030/step2_analysis/VERIFICATION.json)已归档。调参ALNS与旧V5分别复用原45实例×20seed中的27份报告；本次新增搜索81次，总比较135份报告。调参ALNS和旧V5均未重跑。五方法本次共同参考尺度下的HV依次为0.814569、0.803593、0.781318、0.800145、0.793160；IGD+依次为0.082043、0.076046、0.080379、0.069689、0.073218。

- [网络架构与输入](docs/ARCHITECTURE.md)
- [训练方式与实测训练量](docs/TRAINING.md)
- [相对上一版的代码与配置改动](docs/VERSION_CHANGES.md)
- [已完成实验结果及统计口径](docs/RESULTS.md)
- [文件范围、路径与校验](docs/ARTIFACTS.md)

结果逐seed计算后汇总；历史V5主表为每实例20seed均值及样本SD，V5两步Pilot为每实例3seed均值及样本SD。不同数据集、预算或参考尺度分别呈现。Pilot五方法使用其共同经验参考，不与历史45×20表混用绝对值。

发布整理过程未执行新的训练、推理、搜索或重跑。归档源文件按字节复制；ARCHIVE_MANIFEST.json记录本地来源、大小和SHA256。历史配置中的绝对路径保留用于溯源，执行前需要在独立运行副本中重定位；原冻结配置不应直接覆盖。目录不是预先验证的跨平台容器。

上游DPR代码：[water-mirror/DPR](https://github.com/water-mirror/DPR)。原许可见[third_party/DPR_LICENSE](third_party/DPR_LICENSE)。本仓库中的五目标MOEA/D迁移与版本改动通过本仓库源码和文档列明。

## 最新冻结：Step5 r2（2026-11-02）

[完整入口](versions/v5_step5_r2_frozen_20261102/README.md)：整套r2共享meta和16方向权重、180FIT/45验证/45test输入、原r2及后续评测源码锁、2700份紧凑报告、最新三方法共同经验参考统计及逐实例论文表。MOEA/D–ALNS、MOEA/D–ALNS-T、MOEA/D–DPR总体经验HV分别0.64350326/0.67935885/0.71940902，IGD+分别0.14801332/0.12803591/0.10095978；20seed为搜索重复。原四/六方法参考报告单独保留，不能混用绝对值。发布前文件/统计核验不加载模型、不执行新训练/搜索。
