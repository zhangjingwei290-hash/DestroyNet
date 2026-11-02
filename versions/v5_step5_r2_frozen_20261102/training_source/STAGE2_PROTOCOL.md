# Stage2 r2: structured modes and unified scalarization

Revision: `stage2_r2_structured_modes_unified_scalar`. The latest user attachment is preserved verbatim in `provenance/USER_STAGE2_R2_STRUCTURED_REQUEST.txt`; accepted boundary choices are in `STAGE2_PLAN.json`.

Only the independent `stage2_repo` and Stage2 entrypoints are changed. All three baseline directories (9716 files), completed Stage1 sources/reports/results, Stage1 aborted partial (99 files), and Stage2 r1 archive (125 files) are read-only. The prior 45x20 evaluation remains paused. No subagents, uploads, extra training seeds, tuning by test HV, or additional stages.

## Preserved r1 evidence

`stage2_aborted_r1/ABORT_RECEIPT.json` records the observed replay guard failure, 39 completed meta batches, actual elapsed training 891.6214782 s, all 180 FIT optimized and zero adaptation. User-stated scientific concerns are recorded separately from the observed exception. No r1 weights/traces/results enter r2 training or evaluation.

## Unified mechanism

All repair insertion candidates (including separate repair processes), environmental acceptance/reward and MOEA/D decomposition use float64 arithmetic with identical context:

`z=(F-ideal)/span; S=max(w*z)+.05*sum(w*z)+.001*sum(z)+sum(max((F-reference)/physical_scales,0))`.

Archive-only dynamic coordinates prefer at least two in-box points; span lower bound is `.05*physical_scales`. The physical reference is `offset+physical_scales*box_reference`. Cross-entry scalar agreement is checked below 1e-9. No physical feature-index changes: `[18,17,19,12,13]`.

Replacement processes self independently, then uses the PRE-replacement neighborhood snapshot for all neighbor clone checks (atol1e-7/rtol0). Strict dominance has no quota; remaining positive scalar improvements are sorted by improvement, largest two first with deterministic index ties.

## Policy and likelihood

Frozen warm start: `pretrained/meta_complete.pt`, SHA256 `1a3493e34e4c23f7eae7c465ba6c48a69d49884a01ec80d599eec6cd9c48187f`. Encoder, context and q_history_norm are frozen and checked with torch.equal after every Adam step. Fast lr.003: q_body/q_film/mode_logits/q_mu/q_kappa. Slow lr.0003: film/node/physical_bilinear/route_bilinear/route_score_scale. Meta interpolation .5 applies to shared trainable layers and averaged heads.

GPU refresh encodes GNN once and caches node basis at fixed q/N .04 and .25. Actual q conditions interpolate these bases, then add live normalized physical residuals. No stagnation temperature division. Real raw14 remains continuous; four scale inputs are log(N/50),10/N,raw19,raw14. Two current dynamic-coordinate summaries enter only trainable q inputs. Route pooling uses the six user-specified features and standardizes across current nonempty routes. Network FP32; feature reductions and scalar/density arithmetic use stable precision separately.

Mode0 samples categorical anchor then exact PL selected neighbor prefix; Mode1 samples global PL selected prefix; Mode2 samples route PL Top-K and removes the entire first selected route, then deterministic cap truncation on subsequent routes. The accepted exception gives FIRST WHOLE ROUTE priority over q_cap, stops additional routes and records each exception. Exact densities cover the selected ordered auxiliary action (anchor/prefix or route prefix), never an unselected tail permutation. Deterministic removal mapping may be many-to-one; no claim of a closed-form marginal probability over unordered removal sets.

Independent macro and effective-step-normalized micro PPO ratios are clipped separately. Initial epoch errors must each be <=1e-4; CPU/GPU logits/probabilities <5e-5. Mode and Beta conditional entropies plus mean conditional micro entropy are used as specified. Replay retains actual frozen GNN cache/history and current physical/routes. No cross-refresh BPTT.

## Persistent states and credit

Only 180 copied FIT IDs, seed20261024. Persistent IID/subproblem elite slots store routes/archive/stagnation, elapsed budget, hidden and ALNS state. Four real CPU lanes advance representatives; no zero-step neighboring crossover. Mid means vehicle count <=90% of original source. Late means acceptance rate <15% in a completed2.5s window OR actual stagnation>=4. A new preference resets scalar-specific history and must undergo real advancement before mature classification. Meta batches31+ force at least3/5 groups mature; adaptation forces at least every second group mature. Directions3/4/14 adaptation use tw>0 FIT only; observed gradient norms are reported without promising that every stochastic batch has nonzero reward.

Four same-start2.5s trajectories per group; primary only writes back. Trajectory return S_start-S_final +.15*N_newND*group_mean_positive_reward. Step return8-step discount .85, group-standardized, +.75 for stagnation>=4 with positive reward OR fewer candidate vehicles. Advantage .60 trajectory +.40 step. No time division or large-q waste penalty; trajectory mass1/4, group weights equal.

## Budget and pipeline

Finite preflight updates are discarded. One fresh r2 run: hard4500s including initialization/worker preparation, with20s finalization reserve. Stop meta immediately after at least60 complete PPO batches AND all180 FIT actually optimized, then remaining time for16 adaptation directions, target20 full PPO batches each (2 Adam steps/batch). Preserve the four-inner meta merge and uniform→pair→single adaptation tree. Deadline partial optimizer batches roll back; no extra time/retraining. All directions must have a real update for evaluation entry. Incomplete targets are reported, never labelled met.

Ordered coordinator: train → FIT-only q diagnostic → one16s technical smoke (excluded) →36 frozen searches → five-method report → independent audit → COMPLETE. No automatic retries after failure. Engineering failure retains trace/source/budget before a bounded concrete repair.

## Evaluation and final audit

12 approved test IIDs ×3 seeds20261024..26, new36 only;108 old three methods +36 repaired Stage1 +36 r2 =180,60 IID-method rows. Original per-IID114..407s,16 common initial individuals, five objectives/hard constraints/physical boxes/refdirs and local/repair deadlines unchanged. Three workers0/1,2/3,4/5, GPU0 FP32, numerical single thread. Older baselines are earlier host-load epochs, not simultaneous paired workers; IIDs already inspected and mechanisms selected, not blind testing.

Each IID's five methods ×3 seed archives jointly define empirical ND reference and ideal/nadir; each seed archive independently computes HV/IGD+, followed by3seed mean/sample SD, equal IID macro. HV ref1.1^5, no clipping; negligible spans uniformly zero retaining5D. FixedBox HV separately. Macro SD means SD between IID means; individual IID SD means SD among3 seeds. Old reference systems/reports remain unchanged.

Report includes per-scale mode share/acceptance/candidate and accepted vehicle reductions/first-route-cap exceptions, actual q bins and timing, IID/run WTL and vehicle knees linked to the same real point. Route-coherent expected-q diagnostic is explicitly Monte Carlo with standard error; mode0/1 expected q are exact Beta integrals. No test-result-driven checkpoint selection. CPU latency is measured: current route-feature construction and q head have not met requested microsecond targets.

Completion audit checks unique metrics/means, all new archives and objectives,576 initial individuals, scalar/weight/cache/action binding, selected-prefix trace densities, frozen tensors/checkpoint/source/input SHA, training budget/coverage/mature proportions and all protected receipts. Finish README and delivery receipt, remove hourly automation, then stop without extra stages.
