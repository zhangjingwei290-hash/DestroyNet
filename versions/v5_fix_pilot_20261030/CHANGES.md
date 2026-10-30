# Changes

- Independent copy of the designated frozen repo/physical helpers,270 declared inputs and original SHA-bound checkpoint.
- train.py: bounded16-direction adaptation, encoder frozen and checked, original parent tree, corrected wrapper FIT cursor. Technical incident and weights retained under incidents/fit_cursor.
- step2_repo/solver/model.py: remove regime q location parameters/anchors and forward logit tanh boxes; keep continuous Beta and exact-q stochastic node selection.
- step2_repo/solver/training.py: genuine MOEA/D offspring collector with four identical10second windows; normalized trajectory-total returns with equal trajectory/step weighting. Legacy collector remains named collect_closed_legacy for traceability and is not called by step2.
- stage2_train.py: original four-inner-update meta merging, original checkpoint warm start, explicit FIT-only neutral q initialization and bounded meta/adaptation.
- eval.py: load16 separate actors; verify initial populations, physical scales and budgets against both baseline methods. Recompute objectives and feasibility for every new archive.
- cache_baselines.py: read54 original raw reports and write compact local summaries with SHA256/size/mtime receipts.
- analyze.py: per-seed metrics, shared references, paired exploratory tests, q distributions, throughput and direction scalar/five-objective results.
- validate_step2.py: stochastic q calibration, broad output response, replay density, unequal-length trajectory weighting, state/duration invariance, actual four-replica10second optimizer check. Discard its diagnostic weights.
- coordinator.py: explicit stage1 reporting gate and hard stop after stage2; watch_light.py imports only stdlib/psutil.

No original benchmark or shared installed package is modified. See PROTOCOL.md and plan.json for conditions, budgets and cohort.

- Direction-admission estimate stopped the first pass with directions3/4 unupdated. Preserved all traces and those unadapted checkpoints; completed one original-objective batch per missing direction from its prescribed parent, preserving the14 already-trained checkpoint hashes. Formal adaptation total remains below1800seconds; total including both technical runs remains below2700seconds. See incidents/direction_admission.

- User-requested shared-meta arm: eval_step2_both.py evaluates meta_pilot.pt and16direction actors in27same-worker alternating-order blocks. coordinator_v2.py continues only after the completed step1 report; five-method analysis uses135reports. Training caps unchanged.

## Step2 source-tail engineering incident

The first meta loop ended after4complete batches (~296seconds, plus an unoptimized source tail) because an incomplete five-task source window was incorrectly treated as global budget exhaustion. That trial and its early direction adaptation are retained under incidents/step2_source_tail. Prior total process+preflight cost1528.153949seconds is charged to the user-authorized2700second inclusive ceiling. resume_stage2_bounded.py restores the completed meta merge, retains its four optimized batches, explicitly reinitializes Adam/RNG at that boundary because they were not serialized, advances actual FIT sources whose remaining solver time cannot fit all five10second groups, and continues meta within its remaining20minute quota before one true update per direction. The old direction branch is not used as the final model; its costs/logs/weights remain retained. coordinator_v3.py waits for this bounded continuation, then evaluates the shared-meta and direction models exactly as authorized. No original benchmark file or loss/architecture/physical budget is changed by this engineering correction.
