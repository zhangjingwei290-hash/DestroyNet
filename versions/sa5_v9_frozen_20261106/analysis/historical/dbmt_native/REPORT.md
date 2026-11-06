# V9 versus native DBMT historical comparison

Pairwise common per-IID empirical ND over all available archives; own-seed scoring, then IID equal mean; ref1.1^5,zero span relative physical scale1e-12

|N|DBMT HV|V9 HV|HV change|DBMT IGD+|V9 IGD+|
|---|---:|---:|---:|---:|---:|
|all|0.855445|0.854312|-0.13%|0.019360|0.061623|
|50|1.030769|1.016400|-1.39%|0.009114|0.026989|
|150|0.822885|0.864357|+5.04%|0.021869|0.059461|
|250|0.712681|0.682180|-4.28%|0.027098|0.098418|

- DBMT native whole solver; excludes modified window baseline and all neural adapters
- Different search seeds, DBMT2seed at N50/150 and1seed at N250, V9 3seed each
- Native original DBMT rmp integer zero retained; describes actual source implementation
- CPU/GPU allocation and historical loads differ, same wall budgets117-338s are not equal resource budgets
- Development9IID; historical descriptive comparison only; no paired-seed significance claim
