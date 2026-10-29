# 冻结神经破坏网络：V2、V4、V4.3、V5

归档日期：2026-10-29。该仓库存放四个已登记冻结节点的源码、权重、训练配置及已完成实验数据。V3、V4.1、V4.2列入演进关系，没有在本仓库另立冻结权重版本。这里的网络是**破坏网络**；经典修复器与HEMT神经插入修复网络是不同组件。

| 目录 | 冻结对象 | 部署权重 |
|---|---|---|
| [V2](versions/v2/README.md) | 2026-10-23乘法GRU门控版，PPO与REINFORCE | 各最终元权重及16方向适配权重 |
| [V4](versions/v4/README.md) | Run C，全粒度分类动作 | 元权重及16方向适配权重 |
| [V4.3](versions/v4_3/README.md) | Run F，连续Beta粒度、CUDA Graph | 元权重及16方向适配权重 |
| [V5](versions/v5/README.md) | 三轮180合成FIT元训练、真实45例冻结评测 | 900次V5搜索实际使用的meta_complete.pt |

- [网络架构与输入](docs/ARCHITECTURE.md)
- [训练方式与实测训练量](docs/TRAINING.md)
- [相对上一版的代码与配置改动](docs/VERSION_CHANGES.md)
- [已完成实验结果及统计口径](docs/RESULTS.md)
- [文件范围、路径与校验](docs/ARTIFACTS.md)

结果逐seed计算后汇总；当前V5主表为每实例20seed均值及样本SD。不同数据集、预算或参考尺度分别呈现。尚在运行的调参ALNS补跑不属于本次发布数据。

本仓库未执行新的训练、推理、搜索或重跑。归档源文件按字节复制；ARCHIVE_MANIFEST.json记录本地来源、大小和SHA256。历史配置中的绝对路径保留用于溯源，执行前需要在独立运行副本中重定位；原冻结配置不应直接覆盖。目录不是预先验证的跨平台容器。

上游DPR代码：[water-mirror/DPR](https://github.com/water-mirror/DPR)。原许可见[third_party/DPR_LICENSE](third_party/DPR_LICENSE)。本仓库中的五目标MOEA/D迁移与版本改动通过本仓库源码和文档列明。
