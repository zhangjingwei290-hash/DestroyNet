# 普通ALNS、alns_tuned与r2：45实例×20seed补充比较

普通ALNS为已有heuristic方法，仍含轮盘赌自适应；固定版本不等于非自适应算法。只读复用900×3结果，逐条核验IID/seed/预算/完整初始群/物理尺度相同。只重算指标，不执行训练或搜索。共同参考仅包含这三方法，因此不能直接沿用此前四/六方法参考下的绝对值。

|规模|方法|HV↑|IGD+↓|
|---|---|---:|---:|
|overall|heuristic|0.64350326|0.14801332|
|overall|alns_tuned|0.67935885|0.12803591|
|overall|step5_stage2_r2|0.71940902|0.10095978|
|50|heuristic|0.92483587|0.06942451|
|50|alns_tuned|0.93991756|0.06140407|
|50|step5_stage2_r2|0.96700366|0.05387173|
|150|heuristic|0.51380961|0.16655674|
|150|alns_tuned|0.54973649|0.14353480|
|150|step5_stage2_r2|0.59756217|0.11817730|
|250|heuristic|0.49186430|0.20805871|
|250|alns_tuned|0.54842250|0.17916885|
|250|step5_stage2_r2|0.59366124|0.13083030|

|规模|对照|方法|指标|IID胜平负|W|原始p|补充Holm6 p|
|---|---|---|---|---|---:|---:|---:|
|overall|heuristic|alns_tuned|empirical_hv|[45, 0, 0]|0.0|5.684342e-14|3.410605e-13|
|overall|heuristic|alns_tuned|igd_plus|[45, 0, 0]|0.0|5.684342e-14|3.410605e-13|
|overall|heuristic|step5_stage2_r2|empirical_hv|[40, 0, 5]|84.0|6.654852e-08|1.996456e-07|
|overall|heuristic|step5_stage2_r2|igd_plus|[39, 0, 6]|80.0|4.557262e-08|1.822905e-07|
|overall|alns_tuned|step5_stage2_r2|empirical_hv|[30, 0, 15]|276.0|0.005711138|0.005711138|
|overall|alns_tuned|step5_stage2_r2|igd_plus|[32, 0, 13]|238.0|0.001230652|0.002461304|
|50|heuristic|alns_tuned|empirical_hv|[15, 0, 0]|0.0|6.103516e-05|0.0003662109|
|50|heuristic|alns_tuned|igd_plus|[15, 0, 0]|0.0|6.103516e-05|0.0003662109|
|50|heuristic|step5_stage2_r2|empirical_hv|[13, 0, 2]|9.0|0.00201416|0.008056641|
|50|heuristic|step5_stage2_r2|igd_plus|[12, 0, 3]|14.0|0.006713867|0.0201416|
|50|alns_tuned|step5_stage2_r2|empirical_hv|[13, 0, 2]|21.0|0.02557373|0.05114746|
|50|alns_tuned|step5_stage2_r2|igd_plus|[11, 0, 4]|31.0|0.1069946|0.1069946|
|150|heuristic|alns_tuned|empirical_hv|[15, 0, 0]|0.0|6.103516e-05|0.0003662109|
|150|heuristic|alns_tuned|igd_plus|[15, 0, 0]|0.0|6.103516e-05|0.0003662109|
|150|heuristic|step5_stage2_r2|empirical_hv|[13, 0, 2]|15.0|0.008361816|0.03344727|
|150|heuristic|step5_stage2_r2|igd_plus|[12, 0, 3]|15.0|0.008361816|0.03344727|
|150|alns_tuned|step5_stage2_r2|empirical_hv|[9, 0, 6]|40.0|0.2768555|0.3376465|
|150|alns_tuned|step5_stage2_r2|igd_plus|[10, 0, 5]|35.0|0.1688232|0.3376465|
|250|heuristic|alns_tuned|empirical_hv|[15, 0, 0]|0.0|6.103516e-05|0.0003662109|
|250|heuristic|alns_tuned|igd_plus|[15, 0, 0]|0.0|6.103516e-05|0.0003662109|
|250|heuristic|step5_stage2_r2|empirical_hv|[14, 0, 1]|3.0|0.0003051758|0.0009155273|
|250|heuristic|step5_stage2_r2|igd_plus|[15, 0, 0]|0.0|6.103516e-05|0.0003662109|
|250|alns_tuned|step5_stage2_r2|empirical_hv|[8, 0, 7]|36.0|0.1876221|0.1876221|
|250|alns_tuned|step5_stage2_r2|igd_plus|[11, 0, 4]|13.0|0.005371094|0.01074219|

此为用户指定方法集合的事后补充分析，Holm6并未替代原计划Holm9检验家族；两套结论单独保留。45例已检查且12例用于整套模型选择，不能称盲测。旧方法来源主机负载/时期不同，IID/seed配对不等于同步同worker配对。宏观SD为每seed的IID均值之间样本SD；逐IIDSD跨20seed。
