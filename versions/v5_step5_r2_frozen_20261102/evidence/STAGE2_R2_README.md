# Stage2 r2 final delivery

COMPLETE and independent AUDIT passed. Training4460.948s,60 meta batches,180FIT,16directions each20 complete batches/40Adam steps.36 new searches and144 reused yield180 metrics /60meanSD rows.

|Method|EmpHV|FixedBoxHV|IGD+|
|---|---:|---:|---:|
|alns_tuned|0.73267559|0.37539782|0.10662535|
|v5_meta_frozen|0.67635752|0.41570248|0.11544941|
|step4_k4_adapted|0.68763871|0.41747989|0.10723397|
|step5_stage1_zero_shot|0.67036284|0.40249114|0.12114997|
|step5_stage2_retrained|0.72743590|0.30416690|0.13241612|

r2 EmpHV .72743590 versus ALNS .73267559; r2 IGD+ .13241612 versus ALNS .10662535. Retain all outcomes. Current empirical reference differs from Stage1. Several mechanisms changed together; no single-component causal conclusion.
Archive: stage2_archive_r2 contains source, complete checkpoints, raw searches, analysis and receipts with verified byte hashes. r3 is isolated in stage2_r3_full_convergence_45cell.
