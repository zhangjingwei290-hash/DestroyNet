# 冻结范围和复用说明

- checkpoints/: r2共享meta1份、方向0..15共16份，逐字节验证原ARCHIVE_RECEIPT和900结果中的部署SHA。
- initialization/: 原V5 meta热启动权重；其SHA是来源绑定，不把它称最终r2方向权重。
- training/: 原r2完整训练报告、初始化、FIT诊断和训练日志，不含完整optimizer/pool/trace恢复包。
- training_source/: 原r2锁定源码及显式补充的父目录支持文件，原锁范围见evidence；补充文件的来源和SHA见仓库ARCHIVE_MANIFEST。
- evaluation_source/: 后续864正式搜索有效锁的完整源文件，保留原锁和工程修复证据；包含历史训练脚本仅为源码溯源，本归档不运行它们。
- data/: 180FIT、45validation、45test原输入，角色分开；另有boxes和114–407秒逐IID预算。
- baseline_sources/: 两种实际ALNS实现和开发集调参凭据。旧V5与Step4不作为最新三方法主表对照。
- results/compact_reports/: 3方法各900份已完成报告；135个gzip JSONL，一行一个seed。只去除逐步大事件和进程身份，保留所有最终archive路线/目标、完整16初始个体和聚合诊断。每份source SHA、转换和省略字段见REPORT_PROVENANCE。
- analysis_three_methods/: 最新共同经验参考HV/IGD+、逐seed指标、均值SD、精确符号秩/Holm6、论文图表和逐实例显著性。逐实例6项校正只在各实例内部，未对全部45IID统一校正。
- historical_analysis_four_methods/: 完成时原报告原样保留，方法集合、参考几何和Holm9均与最新三方法不同，不拿其数值代替最新主表。
- evidence/: 原始审核、r2/r3整套模型选择、SHA锁、父模型关系和工程修复说明。raw_audits中的原始大日志SHA是当时独立审计证据，本次未重新读取全量约50GB逐步日志核SHA。

原process/worker JSON、SSH/凭据、完整逐步日志、3GB训练trace、失败分支模型和r3整套训练资产不发布。它们仍保留本地；本目录不是全磁盘镜像，也不声明遗漏资产已上云。
