# V5 two-step pilot: completed through stage 1

9 fixed real instances x3 fixed seeds; per-seed HV/IGD+, shared empirical references for all methods in this report.

## Overall paired comparison

|Method|HV mean|IGD+ mean|HV W/T/L (9 IID)|HV W/T/L (27 runs)|
|---|---:|---:|---|---|
|alns_tuned|0.82409682|0.06218198|[0, 9, 0]|[0, 27, 0]|
|v5_meta_frozen|0.81123542|0.06720691|[4, 0, 5]|[11, 0, 16]|
|step1_adapted|0.78672527|0.07579436|[0, 0, 9]|[5, 0, 22]|

## Table 1: granularity diagnostics

Evaluation rows are equal-IID means of conditional per-run proportions; training rows pool optimized transitions. No state is filled with a synthetic zero.

|Domain|Method|Condition|Actions|Mean q/N %|q/N<=15% %|1%|2%|5%|10%|15%|20%|30%|40%|50%|60%|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|evaluation|alns_tuned|overall|262795|20.234|52.243|8.145|16.585|13.969|13.545|0.000|12.563|12.153|11.529|0.000|11.513|
|evaluation|alns_tuned|initial|705|20.188|50.717|7.106|17.922|12.440|13.249|0.000|15.344|11.764|10.814|0.000|11.360|
|evaluation|alns_tuned|refine|11364|19.067|54.185|8.346|16.665|15.138|14.036|0.000|13.444|12.322|10.107|0.000|9.942|
|evaluation|alns_tuned|stalled|250726|20.347|52.005|8.123|16.533|13.842|13.507|0.000|12.556|12.146|11.643|0.000|11.649|
|evaluation|alns_tuned|early|86120|20.457|51.581|8.280|16.296|13.524|13.482|0.000|12.649|12.422|11.654|0.000|11.694|
|evaluation|alns_tuned|middle|87617|20.425|51.832|8.167|16.378|13.906|13.382|0.000|12.658|12.008|11.721|0.000|11.780|
|evaluation|alns_tuned|late|89058|19.897|53.090|7.931|17.012|14.409|13.739|0.000|12.436|12.084|11.271|0.000|11.118|
|evaluation|v5_meta_frozen|overall|179947|29.311|6.976|0.000|0.082|1.242|3.543|4.744|18.490|45.875|23.654|2.326|0.043|
|evaluation|v5_meta_frozen|initial|705|35.870|0.519|0.000|0.000|0.000|0.000|1.590|12.509|29.450|37.996|18.038|0.417|
|evaluation|v5_meta_frozen|refine|8568|15.054|56.611|0.000|0.671|10.413|30.810|26.595|22.056|8.226|1.158|0.071|0.000|
|evaluation|v5_meta_frozen|stalled|170674|31.059|0.701|0.000|0.000|0.002|0.220|1.883|18.263|50.780|26.331|2.472|0.049|
|evaluation|v5_meta_frozen|early|58894|29.556|8.003|0.000|0.087|1.436|4.033|5.106|17.051|43.692|25.435|3.070|0.089|
|evaluation|v5_meta_frozen|middle|60119|29.628|6.017|0.000|0.052|1.060|2.990|4.434|18.206|46.540|24.368|2.333|0.018|
|evaluation|v5_meta_frozen|late|60934|28.803|6.812|0.000|0.106|1.207|3.554|4.653|20.111|47.409|21.336|1.602|0.022|
|evaluation|step1_adapted|overall|167014|28.821|8.178|0.000|0.079|1.539|4.132|5.349|18.621|45.657|22.728|1.882|0.014|
|evaluation|step1_adapted|initial|705|35.485|0.511|0.000|0.000|0.000|0.123|0.952|10.123|38.809|36.965|12.895|0.132|
|evaluation|step1_adapted|refine|8569|14.307|60.060|0.000|0.536|11.224|32.129|28.883|21.371|5.078|0.770|0.010|0.000|
|evaluation|step1_adapted|stalled|157740|30.886|0.667|0.000|0.000|0.003|0.174|1.890|18.449|51.646|25.768|2.053|0.017|
|evaluation|step1_adapted|early|54436|28.924|9.721|0.000|0.086|1.809|4.894|6.103|17.263|43.013|24.189|2.619|0.023|
|evaluation|step1_adapted|middle|55870|29.248|6.711|0.000|0.075|1.134|3.427|4.748|18.397|46.638|23.795|1.772|0.013|
|evaluation|step1_adapted|late|56708|28.373|7.909|0.000|0.072|1.628|3.972|5.101|20.150|47.339|20.443|1.292|0.003|

## Table 2: search throughput

|N|Method|Runs|Steps|Inference s|Repair s|Search s|
|---|---|---:|---:|---:|---:|---:|
|50|alns_tuned|9|22335.33|0.043|215.501|234.118|
|50|v5_meta_frozen|9|15183.56|30.051|189.087|234.081|
|50|step1_adapted|9|14232.00|24.650|193.261|234.084|
|150|alns_tuned|9|4924.00|0.010|229.526|236.385|
|150|v5_meta_frozen|9|3353.44|11.062|220.036|236.363|
|150|step1_adapted|9|3029.11|7.275|223.494|236.358|
|250|alns_tuned|9|1940.11|0.004|300.151|304.013|
|250|v5_meta_frozen|9|1457.11|9.254|291.123|304.007|
|250|step1_adapted|9|1296.00|3.555|296.462|304.007|

## Table 3: nine-instance per-seed metrics (mean +/- sample SD)

|IID|Method|HV|IGD+|
|---|---|---:|---:|
|50-0-2|alns_tuned|0.732348 +/- 0.004386|0.018157 +/- 0.002418|
|50-0-2|v5_meta_frozen|0.714467 +/- 0.011783|0.026031 +/- 0.007485|
|50-0-2|step1_adapted|0.717005 +/- 0.007360|0.021146 +/- 0.002598|
|50-1-3|alns_tuned|1.008167 +/- 0.009308|0.008632 +/- 0.002831|
|50-1-3|v5_meta_frozen|1.009241 +/- 0.008496|0.009124 +/- 0.003962|
|50-1-3|step1_adapted|1.006127 +/- 0.008739|0.010197 +/- 0.004089|
|50-2-1|alns_tuned|1.048695 +/- 0.004888|0.027743 +/- 0.004903|
|50-2-1|v5_meta_frozen|1.022986 +/- 0.023495|0.030612 +/- 0.007999|
|50-2-1|step1_adapted|1.015000 +/- 0.019617|0.035404 +/- 0.004151|
|150-0-2|alns_tuned|0.584065 +/- 0.048397|0.080759 +/- 0.038580|
|150-0-2|v5_meta_frozen|0.586349 +/- 0.010528|0.069881 +/- 0.008113|
|150-0-2|step1_adapted|0.554261 +/- 0.046334|0.070720 +/- 0.010186|
|150-1-1|alns_tuned|0.733052 +/- 0.022295|0.097027 +/- 0.029381|
|150-1-1|v5_meta_frozen|0.725560 +/- 0.023115|0.089477 +/- 0.022201|
|150-1-1|step1_adapted|0.682568 +/- 0.028647|0.097572 +/- 0.013226|
|150-2-0|alns_tuned|0.679943 +/- 0.105395|0.134019 +/- 0.074530|
|150-2-0|v5_meta_frozen|0.666097 +/- 0.209532|0.154710 +/- 0.091952|
|150-2-0|step1_adapted|0.660964 +/- 0.090084|0.179848 +/- 0.090757|
|250-0-4|alns_tuned|1.009434 +/- 0.092300|0.064864 +/- 0.025069|
|250-0-4|v5_meta_frozen|1.045753 +/- 0.065369|0.090416 +/- 0.034479|
|250-0-4|step1_adapted|1.005363 +/- 0.058892|0.104761 +/- 0.005607|
|250-1-0|alns_tuned|0.640971 +/- 0.052023|0.100571 +/- 0.032577|
|250-1-0|v5_meta_frozen|0.523059 +/- 0.074511|0.115494 +/- 0.032399|
|250-1-0|step1_adapted|0.484504 +/- 0.100157|0.131607 +/- 0.040601|
|250-2-3|alns_tuned|0.980196 +/- 0.010547|0.027865 +/- 0.006703|
|250-2-3|v5_meta_frozen|1.007606 +/- 0.001264|0.019117 +/- 0.005473|
|250-2-3|step1_adapted|0.954735 +/- 0.048012|0.030894 +/- 0.012919|

## Direction-specific scalar objectives

Fixed physical augmented Tchebycheff scores, separately from the empirical metric geometry. Full five objectives and conditional q distributions are in CSV.

|Direction|Method|Mean scalar|Mean q/N %|Mean steps|
|---:|---|---:|---:|---:|
|0|alns_tuned|0.18578519|21.350|641.67|
|0|v5_meta_frozen|0.18542222|31.034|511.22|
|0|step1_adapted|0.18573333|30.840|480.70|
|1|alns_tuned|0.08082386|20.094|684.11|
|1|v5_meta_frozen|0.08100959|27.808|445.22|
|1|step1_adapted|0.08159524|27.452|427.78|
|2|alns_tuned|0.92563657|21.672|507.00|
|2|v5_meta_frozen|0.90894676|30.915|325.81|
|2|step1_adapted|0.90449074|30.688|288.33|
|3|alns_tuned|0.06197085|20.290|627.19|
|3|v5_meta_frozen|0.06211149|29.904|451.74|
|3|step1_adapted|0.06220822|29.141|389.59|
|4|alns_tuned|0.00000000|21.241|576.22|
|4|v5_meta_frozen|0.00000000|31.441|375.67|
|4|step1_adapted|0.00000000|31.013|358.26|
|5|alns_tuned|0.09453725|20.308|595.33|
|5|v5_meta_frozen|0.09465700|29.147|460.85|
|5|step1_adapted|0.09484392|28.214|428.07|
|6|alns_tuned|0.46941373|22.037|522.89|
|6|v5_meta_frozen|0.46507299|30.897|351.70|
|6|step1_adapted|0.46205802|30.672|301.22|
|7|alns_tuned|0.09474374|20.171|593.48|
|7|v5_meta_frozen|0.09579964|29.992|437.33|
|7|step1_adapted|0.09517802|29.246|397.81|
|8|alns_tuned|0.09342679|21.644|624.41|
|8|v5_meta_frozen|0.09445399|30.826|456.89|
|8|step1_adapted|0.09442222|30.625|414.85|
|9|alns_tuned|0.46935609|19.365|545.67|
|9|v5_meta_frozen|0.45887374|27.607|411.41|
|9|step1_adapted|0.45563191|27.189|352.63|
|10|alns_tuned|0.05599080|17.776|665.48|
|10|v5_meta_frozen|0.05728081|29.082|382.15|
|10|step1_adapted|0.05714671|28.159|399.15|
|11|alns_tuned|0.04063311|20.150|627.11|
|11|v5_meta_frozen|0.04070567|27.436|413.26|
|11|step1_adapted|0.04103206|26.684|384.56|
|12|alns_tuned|0.47162797|20.275|698.67|
|12|v5_meta_frozen|0.46579173|28.485|482.48|
|12|step1_adapted|0.46322254|28.575|452.74|
|13|alns_tuned|0.46783693|21.252|592.19|
|13|v5_meta_frozen|0.46344859|30.558|377.15|
|13|step1_adapted|0.46150885|30.655|323.93|
|14|alns_tuned|0.03121440|20.358|620.85|
|14|v5_meta_frozen|0.03126328|29.923|395.89|
|14|step1_adapted|0.03131672|29.278|405.15|
|15|alns_tuned|0.18976927|19.234|610.89|
|15|v5_meta_frozen|0.18665285|26.723|385.93|
|15|step1_adapted|0.18586158|26.356|380.93|

## Conditions

- Only FIT synthetic inputs are used for optimization. One training seed; this is the authorized bounded pilot.
- The original9-instance list includes TW=0 only at150-2-0 and250-1-0; it has no50-customer TW=0 case.
- All new runs retain the original per-instance outer budget, local<=min(10,T/16,remaining), stochastic sampling, five objectives and hard constraints.
- Original baseline files remain read-only. The original45x20 reference geometry and numbers are not overwritten.
- Stage1 freezes the encoder; stage2 changes granularity parameterization, explicitly reinitializes its final q location branch, and changes the trajectory objective. The stage2 contrast is this combined change.
- Exact signed-rank tests on9 instance means are exploratory, with Holm adjustment within this report. Related instances and repeated inspection are disclosed.
- The1% reporting center is structurally unreachable for N=50 (minimum q/N=2%). No threshold is used to suppress or rerun unfavorable results.
- No formal full training or additional algorithm stage is launched.
