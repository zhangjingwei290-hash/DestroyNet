# V5 two-step pilot: completed through stage 2

9 fixed real instances x3 fixed seeds; per-seed HV/IGD+, shared empirical references for all methods in this report.

## Overall paired comparison

|Method|HV mean|IGD+ mean|HV W/T/L (9 IID)|HV W/T/L (27 runs)|
|---|---:|---:|---|---|
|alns_tuned|0.81456949|0.08204315|[0, 9, 0]|[0, 27, 0]|
|v5_meta_frozen|0.80359284|0.07604637|[4, 0, 5]|[11, 0, 16]|
|step1_adapted|0.78131765|0.08037933|[1, 0, 8]|[6, 0, 21]|
|step2_meta_only|0.80014486|0.06968889|[2, 0, 7]|[11, 0, 16]|
|step2_adapted|0.79316009|0.07321789|[4, 0, 5]|[9, 0, 18]|

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
|evaluation|step2_meta_only|overall|264115|16.110|47.217|0.000|0.171|5.163|26.717|29.423|29.661|8.287|0.569|0.009|0.000|
|evaluation|step2_meta_only|initial|705|20.768|24.059|0.000|0.000|1.499|14.758|19.011|36.324|24.859|3.278|0.272|0.000|
|evaluation|step2_meta_only|refine|12456|17.605|40.001|0.000|0.092|4.344|22.321|26.914|30.760|13.734|1.801|0.034|0.000|
|evaluation|step2_meta_only|stalled|250954|15.979|47.816|0.000|0.181|5.231|27.033|29.727|29.570|7.806|0.447|0.005|0.000|
|evaluation|step2_meta_only|early|85994|16.274|46.498|0.000|0.161|4.992|26.443|28.862|29.948|8.889|0.691|0.014|0.000|
|evaluation|step2_meta_only|middle|89500|16.120|47.251|0.000|0.158|5.046|26.752|29.483|29.697|8.297|0.560|0.007|0.000|
|evaluation|step2_meta_only|late|88621|15.949|47.830|0.000|0.194|5.420|26.946|29.886|29.369|7.715|0.463|0.006|0.000|
|evaluation|step2_adapted|overall|265369|15.877|49.454|0.000|0.258|6.663|27.790|28.241|27.775|8.555|0.706|0.012|0.000|
|evaluation|step2_adapted|initial|705|21.103|26.442|0.000|0.000|1.655|14.985|18.448|33.248|26.108|5.555|0.000|0.000|
|evaluation|step2_adapted|refine|12679|17.157|43.451|0.000|0.156|5.781|23.570|26.679|29.428|12.640|1.662|0.082|0.000|
|evaluation|step2_adapted|stalled|251985|15.761|49.922|0.000|0.271|6.739|28.127|28.383|27.694|8.195|0.589|0.003|0.000|
|evaluation|step2_adapted|early|87985|15.991|48.853|0.000|0.246|6.549|27.622|27.907|28.003|8.863|0.782|0.028|0.000|
|evaluation|step2_adapted|middle|88069|16.029|48.596|0.000|0.253|6.517|27.318|27.969|28.150|8.997|0.792|0.004|0.000|
|evaluation|step2_adapted|late|89315|15.666|50.612|0.000|0.265|6.842|28.269|28.777|27.326|7.954|0.563|0.005|0.000|
|meta|step2_meta_training|overall|52204|17.219|39.231|0.000|0.103|4.057|19.131|32.268|33.019|10.857|0.563|0.002|0.000|
|meta|step2_meta_training|initial|340|20.295|26.176|0.000|0.000|2.353|13.529|22.941|36.765|21.765|2.647|0.000|0.000|
|meta|step2_meta_training|refine|7835|16.934|41.481|0.000|0.217|4.429|21.149|31.244|31.857|10.517|0.587|0.000|0.000|
|meta|step2_meta_training|stalled|44029|17.246|38.931|0.000|0.084|4.004|18.815|32.522|33.196|10.834|0.543|0.002|0.000|
|meta|step2_meta_training|early|22455|17.331|38.789|0.000|0.125|4.030|18.953|31.882|33.155|11.151|0.699|0.004|0.000|
|meta|step2_meta_training|middle|20479|17.212|38.928|0.000|0.083|4.033|18.819|32.682|33.005|10.884|0.493|0.000|0.000|
|meta|step2_meta_training|late|9270|16.963|40.971|0.000|0.097|4.175|20.248|32.287|32.718|10.086|0.388|0.000|0.000|
|adaptation|step2_adaptation_training|overall|6630|15.716|50.664|0.000|0.377|7.014|28.522|28.688|26.305|8.431|0.618|0.045|0.000|
|adaptation|step2_adaptation_training|initial|64|21.119|21.875|0.000|0.000|0.000|14.062|23.438|34.375|26.562|1.562|0.000|0.000|
|adaptation|step2_adaptation_training|refine|1259|16.813|44.400|0.000|0.556|6.354|24.623|26.926|28.356|11.597|1.509|0.079|0.000|
|adaptation|step2_adaptation_training|stalled|5307|15.390|52.497|0.000|0.339|7.255|29.621|29.169|25.721|7.462|0.396|0.038|0.000|
|adaptation|step2_adaptation_training|early|4319|16.007|49.155|0.000|0.394|7.108|26.812|28.178|27.344|9.308|0.810|0.046|0.000|
|adaptation|step2_adaptation_training|middle|2311|15.171|53.483|0.000|0.346|6.837|31.718|29.641|24.362|6.794|0.260|0.043|0.000|
|adaptation|step2_adaptation_training|late|0|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|
|all_training|step2_all_training_training|overall|58834|17.050|40.519|0.000|0.134|4.390|20.189|31.864|32.262|10.584|0.569|0.007|0.000|
|all_training|step2_all_training_training|initial|404|20.425|25.495|0.000|0.000|1.980|13.614|23.020|36.386|22.525|2.475|0.000|0.000|
|all_training|step2_all_training_training|refine|9094|16.917|41.885|0.000|0.264|4.695|21.630|30.647|31.372|10.666|0.715|0.011|0.000|
|all_training|step2_all_training_training|stalled|49336|17.046|40.390|0.000|0.111|4.354|19.977|32.161|32.392|10.471|0.527|0.006|0.000|
|all_training|step2_all_training_training|early|26774|17.117|40.461|0.000|0.168|4.527|20.221|31.284|32.218|10.854|0.717|0.011|0.000|
|all_training|step2_all_training_training|middle|22790|17.005|40.404|0.000|0.110|4.318|20.127|32.374|32.128|10.470|0.470|0.004|0.000|
|all_training|step2_all_training_training|late|9270|16.963|40.971|0.000|0.097|4.175|20.248|32.287|32.718|10.086|0.388|0.000|0.000|

## Table 2: search throughput

|N|Method|Runs|Steps|Inference s|Repair s|Search s|
|---|---|---:|---:|---:|---:|---:|
|50|alns_tuned|9|22335.33|0.043|215.501|234.118|
|50|v5_meta_frozen|9|15183.56|30.051|189.087|234.081|
|50|step1_adapted|9|14232.00|24.650|193.261|234.084|
|50|step2_meta_only|9|20805.89|35.749|176.326|234.115|
|50|step2_adapted|9|21029.00|35.876|176.177|234.102|
|150|alns_tuned|9|4924.00|0.010|229.526|236.385|
|150|v5_meta_frozen|9|3353.44|11.062|220.036|236.363|
|150|step1_adapted|9|3029.11|7.275|223.494|236.358|
|150|step2_meta_only|9|5788.44|11.066|214.882|236.413|
|150|step2_adapted|9|5762.67|11.123|214.835|236.395|
|250|alns_tuned|9|1940.11|0.004|300.151|304.013|
|250|v5_meta_frozen|9|1457.11|9.254|291.123|304.007|
|250|step1_adapted|9|1296.00|3.555|296.462|304.007|
|250|step2_meta_only|9|2751.78|5.778|291.014|304.014|
|250|step2_adapted|9|2693.78|5.613|291.242|304.013|

## Table 3: nine-instance per-seed metrics (mean +/- sample SD)

|IID|Method|HV|IGD+|
|---|---|---:|---:|
|50-0-2|alns_tuned|0.704655 +/- 0.004214|0.022889 +/- 0.002390|
|50-0-2|v5_meta_frozen|0.688086 +/- 0.011688|0.031869 +/- 0.007918|
|50-0-2|step1_adapted|0.690069 +/- 0.007850|0.027151 +/- 0.001735|
|50-0-2|step2_meta_only|0.669576 +/- 0.012951|0.028684 +/- 0.007310|
|50-0-2|step2_adapted|0.670010 +/- 0.021392|0.032550 +/- 0.007227|
|50-1-3|alns_tuned|1.010480 +/- 0.009278|0.012091 +/- 0.002949|
|50-1-3|v5_meta_frozen|1.011548 +/- 0.008464|0.012697 +/- 0.004221|
|50-1-3|step1_adapted|1.008445 +/- 0.008704|0.013913 +/- 0.004273|
|50-1-3|step2_meta_only|1.016791 +/- 0.008477|0.009856 +/- 0.004754|
|50-1-3|step2_adapted|1.013711 +/- 0.002602|0.009878 +/- 0.001200|
|50-2-1|alns_tuned|1.106002 +/- 0.004640|0.030868 +/- 0.005279|
|50-2-1|v5_meta_frozen|1.084596 +/- 0.022938|0.031966 +/- 0.007782|
|50-2-1|step1_adapted|1.073243 +/- 0.015832|0.037197 +/- 0.003344|
|50-2-1|step2_meta_only|1.084582 +/- 0.023358|0.031782 +/- 0.004244|
|50-2-1|step2_adapted|1.086108 +/- 0.015527|0.030757 +/- 0.006336|
|150-0-2|alns_tuned|0.661914 +/- 0.056732|0.084480 +/- 0.038662|
|150-0-2|v5_meta_frozen|0.669619 +/- 0.011438|0.071973 +/- 0.004576|
|150-0-2|step1_adapted|0.643018 +/- 0.040002|0.072075 +/- 0.007730|
|150-0-2|step2_meta_only|0.659631 +/- 0.047294|0.079997 +/- 0.012488|
|150-0-2|step2_adapted|0.606958 +/- 0.052099|0.087284 +/- 0.012420|
|150-1-1|alns_tuned|0.738438 +/- 0.023556|0.103558 +/- 0.031800|
|150-1-1|v5_meta_frozen|0.731793 +/- 0.023750|0.093809 +/- 0.023704|
|150-1-1|step1_adapted|0.689175 +/- 0.026691|0.101967 +/- 0.013785|
|150-1-1|step2_meta_only|0.710246 +/- 0.009921|0.105912 +/- 0.020631|
|150-1-1|step2_adapted|0.641120 +/- 0.040823|0.137088 +/- 0.002966|
|150-2-0|alns_tuned|0.566422 +/- 0.102160|0.237049 +/- 0.104095|
|150-2-0|v5_meta_frozen|0.564641 +/- 0.162572|0.195729 +/- 0.035522|
|150-2-0|step1_adapted|0.582889 +/- 0.056182|0.173475 +/- 0.029898|
|150-2-0|step2_meta_only|0.524219 +/- 0.146709|0.165650 +/- 0.047930|
|150-2-0|step2_adapted|0.620946 +/- 0.199949|0.141398 +/- 0.054535|
|250-0-4|alns_tuned|0.948929 +/- 0.089566|0.082703 +/- 0.029510|
|250-0-4|v5_meta_frozen|0.985922 +/- 0.062261|0.091088 +/- 0.028052|
|250-0-4|step1_adapted|0.946551 +/- 0.056347|0.105892 +/- 0.012082|
|250-0-4|step2_meta_only|1.012117 +/- 0.047271|0.077496 +/- 0.006742|
|250-0-4|step2_adapted|0.996305 +/- 0.090056|0.074299 +/- 0.025677|
|250-1-0|alns_tuned|0.612220 +/- 0.056261|0.129379 +/- 0.049711|
|250-1-0|v5_meta_frozen|0.487497 +/- 0.072976|0.134422 +/- 0.044288|
|250-1-0|step1_adapted|0.441662 +/- 0.105438|0.155925 +/- 0.053526|
|250-1-0|step2_meta_only|0.548922 +/- 0.098588|0.094884 +/- 0.038981|
|250-1-0|step2_adapted|0.515637 +/- 0.048916|0.118719 +/- 0.015340|
|250-2-3|alns_tuned|0.982065 +/- 0.010512|0.035371 +/- 0.007183|
|250-2-3|v5_meta_frozen|1.008634 +/- 0.001474|0.020864 +/- 0.005663|
|250-2-3|step1_adapted|0.956807 +/- 0.046679|0.035820 +/- 0.015322|
|250-2-3|step2_meta_only|0.975218 +/- 0.031769|0.032937 +/- 0.014077|
|250-2-3|step2_adapted|0.987645 +/- 0.027734|0.026988 +/- 0.005872|

## Direction-specific scalar objectives

Fixed physical augmented Tchebycheff scores, separately from the empirical metric geometry. Full five objectives and conditional q distributions are in CSV.

|Direction|Method|Mean scalar|Mean q/N %|Mean steps|
|---:|---|---:|---:|---:|
|0|alns_tuned|0.18578519|21.350|641.67|
|0|v5_meta_frozen|0.18542222|31.034|511.22|
|0|step1_adapted|0.18573333|30.840|480.70|
|0|step2_meta_only|0.18568148|16.174|681.07|
|0|step2_adapted|0.18526667|13.720|792.44|
|1|alns_tuned|0.08082386|20.094|684.11|
|1|v5_meta_frozen|0.08100959|27.808|445.22|
|1|step1_adapted|0.08159524|27.452|427.78|
|1|step2_meta_only|0.08137617|16.335|673.04|
|1|step2_adapted|0.08101784|14.573|684.30|
|2|alns_tuned|0.92563657|21.672|507.00|
|2|v5_meta_frozen|0.90894676|30.915|325.81|
|2|step1_adapted|0.90449074|30.688|288.33|
|2|step2_meta_only|0.90165509|15.813|498.37|
|2|step2_adapted|0.90295139|21.929|380.70|
|3|alns_tuned|0.06197085|20.290|627.19|
|3|v5_meta_frozen|0.06211149|29.904|451.74|
|3|step1_adapted|0.06220822|29.141|389.59|
|3|step2_meta_only|0.06183921|16.061|635.30|
|3|step2_adapted|0.06188411|16.608|622.67|
|4|alns_tuned|0.00000000|21.241|576.22|
|4|v5_meta_frozen|0.00000000|31.441|375.67|
|4|step1_adapted|0.00000000|31.013|358.26|
|4|step2_meta_only|0.00000000|16.061|573.15|
|4|step2_adapted|0.00000000|21.021|510.44|
|5|alns_tuned|0.09453725|20.308|595.33|
|5|v5_meta_frozen|0.09465700|29.147|460.85|
|5|step1_adapted|0.09484392|28.214|428.07|
|5|step2_meta_only|0.09479492|16.945|674.93|
|5|step2_adapted|0.09457702|15.097|698.30|
|6|alns_tuned|0.46941373|22.037|522.89|
|6|v5_meta_frozen|0.46507299|30.897|351.70|
|6|step1_adapted|0.46205802|30.672|301.22|
|6|step2_meta_only|0.46075185|15.834|549.19|
|6|step2_adapted|0.45927523|15.461|503.56|
|7|alns_tuned|0.09474374|20.171|593.48|
|7|v5_meta_frozen|0.09579964|29.992|437.33|
|7|step1_adapted|0.09517802|29.246|397.81|
|7|step2_meta_only|0.09505749|16.313|636.33|
|7|step2_adapted|0.09498282|16.269|591.15|
|8|alns_tuned|0.09342679|21.644|624.41|
|8|v5_meta_frozen|0.09445399|30.826|456.89|
|8|step1_adapted|0.09442222|30.625|414.85|
|8|step2_meta_only|0.09424074|15.851|686.11|
|8|step2_adapted|0.09358551|15.525|689.26|
|9|alns_tuned|0.46935609|19.365|545.67|
|9|v5_meta_frozen|0.45887374|27.607|411.41|
|9|step1_adapted|0.45563191|27.189|352.63|
|9|step2_meta_only|0.45384637|16.007|598.93|
|9|step2_adapted|0.45566062|15.639|571.85|
|10|alns_tuned|0.05599080|17.776|665.48|
|10|v5_meta_frozen|0.05728081|29.082|382.15|
|10|step1_adapted|0.05714671|28.159|399.15|
|10|step2_meta_only|0.05639130|16.462|566.04|
|10|step2_adapted|0.05631445|16.171|599.74|
|11|alns_tuned|0.04063311|20.150|627.11|
|11|v5_meta_frozen|0.04070567|27.436|413.26|
|11|step1_adapted|0.04103206|26.684|384.56|
|11|step2_meta_only|0.04094597|16.409|582.07|
|11|step2_adapted|0.04075502|16.529|607.07|
|12|alns_tuned|0.47162797|20.275|698.67|
|12|v5_meta_frozen|0.46579173|28.485|482.48|
|12|step1_adapted|0.46322254|28.575|452.74|
|12|step2_meta_only|0.46251240|15.811|650.56|
|12|step2_adapted|0.46298086|14.432|718.67|
|13|alns_tuned|0.46783693|21.252|592.19|
|13|v5_meta_frozen|0.46344859|30.558|377.15|
|13|step1_adapted|0.46150885|30.655|323.93|
|13|step2_meta_only|0.46023272|15.753|546.63|
|13|step2_adapted|0.46129231|15.286|597.81|
|14|alns_tuned|0.03121440|20.358|620.85|
|14|v5_meta_frozen|0.03126328|29.923|395.89|
|14|step1_adapted|0.03131672|29.278|405.15|
|14|step2_meta_only|0.03110055|16.197|654.22|
|14|step2_adapted|0.03105091|15.760|651.93|
|15|alns_tuned|0.18976927|19.234|610.89|
|15|v5_meta_frozen|0.18665285|26.723|385.93|
|15|step1_adapted|0.18586158|26.356|380.93|
|15|step2_meta_only|0.18509106|16.349|576.11|
|15|step2_adapted|0.18529599|15.721|608.59|

## Conditions

- Only FIT synthetic inputs are used for optimization. One training seed; this is the authorized bounded pilot.
- The original9-instance list includes TW=0 only at150-2-0 and250-1-0; it has no50-customer TW=0 case.
- All new runs retain the original per-instance outer budget, local<=min(10,T/16,remaining), stochastic sampling, five objectives and hard constraints.
- Original baseline files remain read-only. The original45x20 reference geometry and numbers are not overwritten.
- Stage1 freezes the encoder; stage2 changes granularity parameterization, explicitly reinitializes its final q location branch, and changes the trajectory objective. The stage2 contrast is this combined change.
- Exact signed-rank tests on9 instance means are exploratory, with Holm adjustment within this report. Related instances and repeated inspection are disclosed.
- The1% reporting center is structurally unreachable for N=50 (minimum q/N=2%). No threshold is used to suppress or rerun unfavorable results.
- No formal full training or additional algorithm stage is launched.


Evaluation interruption: 34 complete reports retained unchanged; 20 missing searches resumed with original seed and full per-instance budget, without training. Two partial pairs (250-0-4 and250-2-3, seed20261025) span different worker processes across the interruption; the other25 pairs used the same worker within each pair. Exit cause remains undetermined.
