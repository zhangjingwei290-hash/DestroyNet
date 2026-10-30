# 资产与执行边界

2026-10-30新增目录[versions/v5_fix_pilot_20261030](../versions/v5_fix_pilot_20261030/README.md)：包含第1步/第2步网络源码、实际训练包装器、原起点权重、第1步16方向权重、第2步共享meta及16方向权重、物理依赖、270输入与划分、135份含最终路线/目标的紧凑报告、训练日志和三张结果表。完整逐步records/variation_events/point_events及弃用分支的方向权重仍留在本地。新目录PACKAGE_CONTENTS.json列出具体范围，旧版本范围保留如下。

- versions/*/source：从冻结快照复制的源文件，保持字节不变。V2 source中也保留其运行目录下的元/方向权重；V4/F权重在checkpoints；V5只有实际部署meta权重。
- dependencies：V2外部Python物理定义/约束/评价源码，以及V5同主机补跑保存的物理模块副本。版本目录下的source仍使用原绝对路径约定。
- data：原冻结输入字节和划分；JSON中的本地路径是溯源字段，不是远程下载地址。
- results：已完成运行的配置、逐seed表、summary、训练与审核证据。V4/F小队列原始报告压缩为jsonl.gz，一行保留source_file和完整report。
- ARCHIVE_MANIFEST.json：复制来源与SHA256。历史manifest保留其原路径和原范围，不代表本仓库包含其中列出的所有大体积历史日志。
- MANIFEST_SHA256.json：仓库发布文件的相对路径与SHA256；运行python tools/verify_archive.py检查。

本次不发布完整训练恢复latest_checkpoint、进程身份/SSH/凭据、其他项目、正在运行的实验或桌面完整原始日志包。保留meta和所有已部署方向权重；V5未进行16方向适配。meta_complete可能含训练状态，加载使用原代码所需的payload键。

历史源码中的Windows绝对路径需在独立运行副本中重新配置。V4/F入口source/full_validate.py、V5入口evaluation/evaluate.py；原声明的输入/源码哈希必须在新配置中重新建立，不能删除校验后冒充同一次冻结执行。本次只做文件、数据和代码静态校验，未运行求解器/训练来宣称跨平台复现通过。

原DPR许可原文保留；外部物理模块的来源记录在ARCHIVE_MANIFEST.json及源码。没有为他人代码重新授予许可。
