# V5 two-step bounded verification, 2026-10-30

The original desktop45x20 benchmark is READ ONLY. New code, data, weights and statistics live only here. No old report or metric is overwritten. Stop after both pilots; do not start formal training.

## Fixed evaluation

Nine instances:50-0-2,50-1-3,50-2-1,150-0-2,150-1-1,150-2-0,250-0-4,250-1-0,250-2-3. Three seeds20261024..26. Reuse27 ALNS-Tuned and27 oldV5 complete desktop reports. Each new model adds27 full-budget runs. Only150-2-0 and250-1-0 are TW=0; no N=50/TW=0 instance is in the original list.

## Step1

Original checkpoint SHA256:1a3493e34e4c23f7eae7c465ba6c48a69d49884a01ec80d599eec6cd9c48187f. Freeze encoder tensors exactly; adapt remaining context/q/FiLM/head parameters under the original PPO objective and real MOEA/D collector. Tree order15,5..14,0..4; original parent inheritance. Each direction needs a completed optimizer update. Formal adaptation1800seconds. Retain the FIT-cursor technical run (~445seconds) and15second smoke; combined work remains below2700seconds. The wrapper must advance next_iid at admission because ParallelRuntime sets active_iid before collection.

## Step2

Warm start independently from the original checkpoint. Remove q_regime_mu/regime_anchors, regime routing and forward output-logit tanh boxes. Preserve Beta concentration, integer_q, stochastic node decoder, five physical objectives, hard constraints and all repairs. Location is .06+.84*sigmoid(base_raw+indicator_raw). Zero only the final q-location weights and state weights; calibrate one global bias on three FIT states to19% stochastic mean q/N, accounting for the Beta mean rather than mode. Other weights transfer unchanged.

Meta up to1200seconds plus original tree adaptation within one1800second absolute training ceiling. Five cyclic symmetric tasks, four replicas, four inner updates per meta merge, PPO2epochs, original learning rate/entropy, CUDA FP32. No validation/test training. Report actual short FIT coverage; do not claim complete FIT epochs.

Fork four replicas of a genuine MOEA/D offspring with identical physical/history initial state. Every admitted training window has exactly10 active seconds, without phase/T/16 shortening. Only the primary updates the population. Queued inactive wait is excluded under the existing runtime convention; active inference/repair wait is counted. Unfinished tails are unoptimized. Phase labels use actual solver clocks.

Return R=sum(max(0,reward)). Normalize the four trajectory returns together; give every trajectory step that trajectory advantage. Each trajectory gets total loss weight1/4, distributed uniformly over its own steps. No state buckets or manual duration coefficients. Identical returns yield zero actor advantage; preserve entropy.

## Deployment and reporting

Use16 real direction-specific actors in Search. Preserve per-IID114..407second budgets, local/repair<=min(10,T/16,remaining), stochastic sampling, complete initial populations, objectives and constraints. Three workers pinned[0,1]/[2,3]/[4,5], numerical threads1, CUDA0, FP32. Checkpoint loading precedes search; first capture/inference/weight binding/CPU transfer inside search is timed. Three-size16second technical deployment smoke is excluded from results. Baseline and new searches share the desktop but occur at different load epochs.

Each IID joint ND over participating methods and selected3seeds defines common empirical ideal/nadir and IGD+ reference. HV reference(1.1)^5, no clipping. Zero span/physical scale<=1e-12 is uniformly zeroed, retaining5coordinates. Score every seed's own archive, then3seed mean/sample SD and equal-IID macro. Never pooled-archive HV. Recompute all compared methods consistently when the method pool changes between steps. Nine-instance signed-rank is exploratory; Holm adjusts the comparisons within each report.

Diagnostics use nearest reporting centers[1,2,5,10,15,20,30,40,50,60]%; this does not change continuous action support or original internal bin logging. Also report exact q/N<=15%. N=50 cannot attain1% (minimum2%). Conditional evaluation means first average available seeds within IID, then equal IIDs; absent states remain missing. Optimized training transition statistics are separate.

## Reporting gate and stop

Present step1 measured tables before releasing stage1_report_presented.json. Then finish the authorized step2, report all three requested tables and per-direction results, verify baseline unchanged, and stop irrespective of outcome. No formal training, K-step inference, new anchors/operators or seed sweep follows automatically.

## Additional user request

Evaluate the step2 meta_pilot.pt shared single head separately from the16direction policies, same9x3cohort. Final five-method reference geometry and tables are recomputed uniformly. Step2 methods run in same-worker paired blocks with alternating order;27blocks,54new runs. Entire two-step pilot has81new+54reused=135runs. Training budgets remain unchanged. At the stage1 reporting gate switch from the original waiting coordinator to coordinator_v2.py under the PID/creation-time/command checks in LATEST_STEERING.md.

## Step2 source-tail engineering incident

The first meta loop ended after4complete batches (~296seconds, plus an unoptimized source tail) because an incomplete five-task source window was incorrectly treated as global budget exhaustion. That trial and its early direction adaptation are retained under incidents/step2_source_tail. Prior total process+preflight cost1528.153949seconds is charged to the user-authorized2700second inclusive ceiling. resume_stage2_bounded.py restores the completed meta merge, retains its four optimized batches, explicitly reinitializes Adam/RNG at that boundary because they were not serialized, advances actual FIT sources whose remaining solver time cannot fit all five10second groups, and continues meta within its remaining20minute quota before one true update per direction. The old direction branch is not used as the final model; its costs/logs/weights remain retained. coordinator_v3.py waits for this bounded continuation, then evaluates the shared-meta and direction models exactly as authorized. No original benchmark file or loss/architecture/physical budget is changed by this engineering correction.
