"""Content-addressed development panels with explicit completion coverage."""
from collections import defaultdict
from functools import lru_cache
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch

from .artifacts import file_id, model_id, runtime_contract, save_json
from .data import fingerprint, manifest
from .evaluation import ARMS, evaluate_episode


def aggregate(rows, full_episodes=False, metric="final_hv"):
    indexed = {(r["iid"],tuple(r["preference"]),r["search_seed"],r["arm"]):r for r in rows}
    summary={}
    for arm in ARMS[1:]:
        differences=defaultdict(list)
        pairs=0
        for key,row in indexed.items():
            if row["arm"]!="learned":
                continue
            other=indexed.get((*key[:3],arm))
            if other is None:
                continue
            if full_episodes and not (row.get("episode_terminal",False) and other.get("episode_terminal",False)):
                continue
            differences[row["iid"]].append(row[metric]-other[metric])
            pairs+=1
        values=np.asarray([np.mean(v) for _,v in sorted(differences.items())])
        if len(values):
            bootstrap=np.random.default_rng(1926).choice(values,(2000,len(values)),replace=True).mean(axis=1)
            mean_key="mean_paired_delta_hv" if metric=="final_hv" else "mean_paired_delta_scalar_improvement"
            summary[arm]=dict(**{mean_key:float(values.mean())},metric=metric,
                iid_bootstrap_95_ci=[float(x) for x in np.quantile(bootstrap,[.025,.975])],
                development_iids=len(values),paired_panels=pairs,
                mean_delta_by_iid={i:float(np.mean(v)) for i,v in differences.items()})
    return summary


@lru_cache(maxsize=4)
def evaluator_identity(project_root):
    return dict(evaluation_source=file_id(Path(__file__).with_name("evaluation.py")),
        independent_audit_source=file_id(Path(project_root)/"hemt_destroy_feature_labels_v1/audit.py"),
        hv_library=version("moocore"))


def panel_identity(cfg,source,split_id,iid,arm,pref,seed,seconds,weights_id,hardware):
    obj=dict(version="dpr_validation_cache_v2",runtime=runtime_contract(cfg),source=source,
        split_manifest_id=split_id,iid=iid,arm=arm,preference=list(pref),search_seed=seed,
        repair_seed_rule="same_seeded_exogenous_schedule_per_action_v1",seconds=seconds,
        max_actions=cfg.episode_steps,metric="guarded_single_episode; fixed_physical_reference_HV_offline",
        evaluator=evaluator_identity(cfg.project_root),hardware=hardware)
    if arm=="learned":
        obj["model_id"]=weights_id
    return obj


def validate(root,pool,plan,cfg,policy,actions,checkpoint_path,budget_seconds=None):
    split=manifest(cfg,pool)
    diagnostic = plan.get("evaluation_scope")=="fit_learnability_v1"
    if diagnostic:
        from .learnability import check_plan_alignment
        check_plan_alignment(cfg,split,plan)
    elif not set(plan["validation_ids"])<=set(split["development_ids"]) or set(plan["fit_ids"]) & set(plan["validation_ids"]):
        raise ValueError("Development validation must use the explicit disjoint manifest")
    weights_id=model_id(policy.state_dict())  # once per validation invocation
    payload=torch.load(checkpoint_path,map_location="cpu",weights_only=True)
    if payload.get("model_id")!=weights_id:
        raise ValueError("Validation checkpoint and live policy do not match")
    hardware=dict(host=platform.node(),machine=platform.machine(),processor=platform.processor(),
                  device=cfg.device,torch=str(torch.__version__),threads=torch.get_num_threads(),
                  numeric_thread_environment={k:os.environ.get(k) for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS")})
    root=Path(root)
    folder=root/"validation_v2"/f"step_{actions:07d}_{weights_id[:16]}"
    folder.mkdir(parents=True,exist_ok=True)
    cached=root/"validation_baselines_v2"
    cached.mkdir(exist_ok=True)
    started=time.perf_counter()
    deadline=started+max(0.,plan["validation_seconds"] if budget_seconds is None else budget_seconds)
    rows=[]
    unavailable=[]
    inputs={}
    # Source identity includes initial routes, physical box, q levels and input
    # bytes; calculated once per IID/panel sweep, never per environment action.
    for iid in plan["validation_ids"]:
        pool.load(iid)  # verifies bytes once on first access; subsequent sweeps reuse identity
        inputs[iid]=dict(input_sha256=pool.input_content_ids[iid],
            box=pool.cfg["boxes_by_instance"][iid],q_values=pool.cfg["q_values_by_instance"][iid])
    for iid in plan["validation_ids"]:
        for pref in plan["validation_preferences"]:
            for seed in plan["validation_seeds"]:
                for arm in (*ARMS[1:],"learned"):
                    identity=panel_identity(cfg,inputs[iid],split["id"],iid,arm,pref,seed,
                        plan["validation_search_seconds"],weights_id,hardware)
                    key=fingerprint(identity)
                    path=(folder if arm=="learned" else cached)/(key+".json")
                    if path.exists():
                        obj=json.loads(path.read_text(encoding="utf-8"))
                        if obj.get("identity")!=identity or obj.get("key")!=key:
                            raise ValueError("Cache identity mismatch; refusing unverified panel reuse")
                        result=obj["result"]
                    else:
                        if time.perf_counter()+plan["validation_search_seconds"]+.05>deadline:
                            unavailable.append(dict(iid=iid,preference=list(pref),seed=seed,arm=arm,reason="validation_budget"))
                            continue
                        result=evaluate_episode(pool,iid,cfg,policy,arm,pref,seed,plan["validation_search_seconds"])
                        save_json(path,dict(key=key,identity=identity,result=result))
                    rows.append(result)
                    save_json(root/"health.json",dict(phase="fit_diagnostic_evaluation" if diagnostic else "development_evaluation",
                        updated_at=time.time(),pid=os.getpid(),checkpoint_actions=actions,panels_done=len(rows),panels_total=len(plan["validation_ids"])*len(plan["validation_preferences"])*len(plan["validation_seeds"])*4,
                        iid=iid,arm=arm,episode_steps=result["steps"],episode_terminal=result["episode_terminal"]))
                    print(f"evaluation step={actions}: {len(rows)} panels; {iid} {arm} actions={result['steps']}",flush=True)
    total=len(plan["validation_ids"])*len(plan["validation_preferences"])*len(plan["validation_seeds"])*4
    finished=sum(r["episode_terminal"] for r in rows)
    complete=len(rows)==total
    full=aggregate(rows,True)
    guarded=aggregate(rows,False)
    report=dict(version="dpr_development_report_v2",actions=actions,checkpoint=str(checkpoint_path),
        model_id=weights_id,panel_coverage_complete=complete,complete=complete,
        planned_panels=total,executed_panels=len(rows),completed_episodes=finished,
        all_episodes_complete=complete and finished==total,episode_completion_fraction=finished/total if total else 0.,
        full_episode_comparisons=full,comparisons=full,guarded_budget_comparisons=guarded,
        guarded_budget_scope="same time cap AND same episode action cap; not unrestricted equal-wall-time search",
        full_episode_scope="paired completed episodes only; conditional on completion, check coverage before any claim",
        incomplete_episodes=[dict(iid=r["iid"],arm=r["arm"],preference=r["preference"],seed=r["search_seed"],
                                  steps=r["steps"],reason=r["stop_reason"]) for r in rows if not r["episode_terminal"]],
        missing_panels=unavailable,seconds=time.perf_counter()-started,runs=len(rows),
        evaluation_scope=plan.get("evaluation_scope","development"),development_only=not diagnostic,
        fit_diagnostic_only=diagnostic,single_training_seed=True,training_iids=len(plan["fit_ids"]),
        validation_iids=len(plan["validation_ids"]),quality_validated=False)
    if diagnostic:
        from .learnability import attach_diagnostic_metrics
        attach_diagnostic_metrics(root,report,rows,plan,cfg)
    save_json(folder/"summary.json",report)
    return report
