# V9 versus original R2 long sessions

Each seed archive scored separately under two-method per-IID common empirical ND from old20+new3; HV ref1.1^5; zero relative physical scale threshold1e-12; IID equally weighted

|N|R2 HV|V9 HV|HV change|R2 IGD+|V9 IGD+|
|---|---:|---:|---:|---:|---:|
|all|0.704919|0.850576|+20.66%|0.118201|0.052451|
|50|0.936076|1.010451|+7.95%|0.060497|0.023149|
|150|0.692782|0.882817|+27.43%|0.140802|0.055236|
|250|0.485900|0.658460|+35.51%|0.153304|0.078968|

- Different non-overlapping search seeds and unequal sample sizes:20 old vs3 new; not paired-seed comparison
- Old R2 long sessions use min(10,T/16,remaining); V9 uses five-step SA and parallel direction repairs
- Model/controller/q mapping/acceptance/cache/scheduling also differ: full-system historical comparison, not isolated local-session-budget effect
- Different historical CPU resource allocation/load epochs; same wall budget does not ensure same CPU core seconds
- Nine IID are repeatedly inspected development instances
