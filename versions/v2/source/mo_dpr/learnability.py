"""Explicit TRAIN-fit diagnosis, separate from held-out development validation.

Only configuration, accounting and reports live here. Sampling, PPO, repair,
features, reward and feasibility use the existing shared implementation.
"""
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics

from .artifacts import runtime_contract, save_json
from .config import Config
from .data import fingerprint


IDS=("hemtv2_train_50_C_4","hemtv2_train_50_R_4","hemtv2_train_50_RC_4")
SCOPE="fit_learnability_v1"


def build_plan(output_root):
    cfg=Config(mode="train",train_steps=6000,train_seconds=7200,train_instance_ids=IDS,
               preference=(.2,)*5,device="cpu",output_root=str(Path(output_root).resolve()))
    validation=dict(evaluation_scope=SCOPE,fit_ids=list(IDS),validation_ids=list(IDS),
        validate_every=3000,validation_seconds=2400.,validation_search_seconds=60.,
        validation_seeds=[73,109],validation_preferences=[[.2]*5],
        checkpoints=[0,3000,6000],comparison="same_150_actions; wall_guard_only; report_all_truncations",
        scope_note="same three fit IIDs for learnability only; not development/generalization")
    return json.loads(json.dumps(dict(version=SCOPE,config=cfg.to_dict(),validation=validation,
        total_seconds_including_validation=7200,final_validation_reserve_seconds=600,
        automatic_next_stage=False)))


def check_plan_alignment(cfg,split,plan):
    if (plan.get("evaluation_scope")!=SCOPE or tuple(cfg.train_instance_ids)!=IDS
            or plan.get("fit_ids")!=list(IDS) or plan.get("validation_ids")!=list(IDS)
            or not set(IDS)<=set(split["fit_ids"])):
        raise ValueError("Fit diagnostic must use exactly the approved three TRAIN-fit IIDs")
    original=build_plan(cfg.output_root)
    reference=Config(**original["config"])
    # Output paths are administrative. Every training/inference setting must
    # match the approved plan, including all defaults, not just tensor shapes.
    if cfg.to_dict()!=reference.to_dict() and json.loads(json.dumps(cfg.to_dict()))!=original["config"]:
        raise ValueError("Fit diagnostic training configuration/contract changed")
    if plan!=original["validation"]:
        raise ValueError("Fit diagnostic evaluation protocol changed")
    if tuple(cfg.preference)!=tuple(plan["validation_preferences"][0]):
        raise ValueError("Train/evaluation preference mismatch")


def panel_key(row):
    return row["iid"],tuple(row["preference"]),row["search_seed"]


def paired_initial_gain(current,initial):
    reference={panel_key(r):r for r in initial if r["arm"]=="learned"}
    pairs=[]
    for row in current:
        previous=reference.get(panel_key(row))
        if row["arm"]!="learned" or previous is None:
            continue
        if not (row["episode_terminal"] and previous["episode_terminal"]):
            continue
        pairs.append(dict(iid=row["iid"],seed=row["search_seed"],
            scalar_gain=row["scalar_improvement"]-previous["scalar_improvement"],
            hv_gain=row["final_hv"]-previous["final_hv"],
            initial_network_final_objectives=previous["final_objectives"],current_final_objectives=row["final_objectives"]))
    groups=defaultdict(list)
    for pair in pairs:
        groups[pair["iid"]].append(pair)
    by_iid={i:dict(scalar_gain=statistics.mean(r["scalar_gain"] for r in rs),
                   hv_gain=statistics.mean(r["hv_gain"] for r in rs)) for i,rs in groups.items()}
    return dict(paired_panels=len(pairs),paired_iids=len(groups),by_iid=by_iid,panels=pairs,
        mean_paired_scalar_improvement_gain=statistics.mean(r["scalar_gain"] for r in by_iid.values()) if by_iid else None,
        mean_paired_hv_gain=statistics.mean(r["hv_gain"] for r in by_iid.values()) if by_iid else None,
        scope="fitted instances, paired independent search seeds; exploratory, not generalization")


def attach_diagnostic_metrics(root,report,rows,plan,cfg):
    from .validation import aggregate
    fields=("iid","arm","preference","search_seed","steps","episode_terminal","truncated","stop_reason",
            "scalar_improvement","initial_score","final_score","final_hv","initial_objectives","final_objectives",
            "actual_search_seconds","feature_seconds","inference_seconds","repair_seconds","successes")
    panels=[{k:r[k] for k in fields} for r in rows]
    report["panel_metrics"]=panels
    report["full_episode_scalar_comparisons"]=aggregate(rows,True,metric="scalar_improvement")
    report["bootstrap_scope"]="three fitted IIDs only; no test/generalization significance claim"
    identity=fingerprint(dict(runtime=runtime_contract(cfg),plan=plan))
    path=Path(root)/"diagnostic_initial_reference.json"
    if report["actions"]==0:
        if path.exists():
            previous=json.loads(path.read_text(encoding="utf-8"))
            if previous["protocol_id"]!=identity or previous["model_id"]!=report["model_id"]:
                raise ValueError("Initial diagnostic reference identity mismatch")
        save_json(path,dict(version=SCOPE,protocol_id=identity,model_id=report["model_id"],panels=panels))
    if path.exists():
        initial=json.loads(path.read_text(encoding="utf-8"))
        if initial["protocol_id"]!=identity:
            raise ValueError("Initial/current evaluation contract mismatch")
        report["versus_untrained_network"]=paired_initial_gain(panels,initial["panels"])
    else:
        report["versus_untrained_network"]=dict(paired_panels=0,reason="initial_reference_missing")
    learned=[r for r in rows if r["arm"]=="learned"]
    records=[a for r in learned for a in r["records"]]
    report["learned_actions"]=dict(actions=len(records),q_counts=dict(Counter(r["q"] for r in records)),
        repair_counts=dict(Counter(r["repair"] for r in records)),
        status_counts=dict(Counter(r["status"] for r in records)),
        accepted=sum(bool(r["accepted"]) for r in records),
        unique_removed_sets=len({tuple(sorted(r["removed"])) for r in records}))


def finish_report(root):
    root=Path(root)
    summary=json.loads((root/"summary.json").read_text(encoding="utf-8"))
    validation=summary.get("final_validation",{})
    gain=validation.get("versus_untrained_network",{})
    curve=[json.loads(line) for line in (root/"training_curve.jsonl").read_text(encoding="utf-8").splitlines()] if (root/"training_curve.jsonl").exists() else []
    signal=gain.get("mean_paired_scalar_improvement_gain")
    result=dict(version=SCOPE,training_actions=summary["actions"],optimized_actions=summary["optimized_actions"],
        pending_actions=summary["pending_actions"],total_seconds=summary["elapsed_seconds"],
        training_updates=summary["updates"],optimizer_steps=summary["optimizer_steps"],
        validation_seconds=summary.get("validation_seconds"),final_validation=validation,
        initial_network_gain=gain,training_curve=curve,
        conclusion=("incomplete_comparison" if not validation.get("all_episodes_complete") or gain.get("paired_panels")!=6
            else ("positive_fit_scalar_signal_not_generalization" if signal is not None and signal>0 else "no_positive_fit_scalar_signal_at_this_budget")),
        generalization_proven=False,automatic_next_stage=False,training_seeds=1)
    save_json(root/"diagnostic_report.json",result)
    lines=["# 三实例学习诊断", "", "只评估三个训练实例上的学习迹象，不是研发留出或泛化证明。", "",
        f"训练采样 {result['training_actions']} 次，已优化 {result['optimized_actions']} 次，待优化 {result['pending_actions']} 次。",
        f"总计 {result['total_seconds']/60:.2f} 分钟（含验证），PPO更新 {result['training_updates']} 轮、优化器小批更新 {result['optimizer_steps']} 次。", "",
        f"结论标签：{result['conclusion']}。", "",
        f"完整配对面板：{gain.get('paired_panels',0)}/6；最终全部150步完成：{validation.get('all_episodes_complete',False)}。",
        f"相比未训练网络：偏好评分改善增量 {gain.get('mean_paired_scalar_improvement_gain')}；HV差 {gain.get('mean_paired_hv_gain')}。", "",
        "偏好评分增量为正表示当前网络改善更多；HV差为正表示档案质量更好，二者分开报告。", "",
        "| IID | 相比未训练网络的评分改善增量 | HV差 |", "|---|---:|---:|"]
    for iid,row in gain.get("by_iid",{}).items():
        lines.append(f"| {iid} | {row['scalar_gain']:+.8g} | {row['hv_gain']:+.8g} |")
    lines += ["", "对照基线、奖励分布、各动作头梯度、KL/裁剪比例与采样/推理/修复成本见 diagnostic_report.json 和 training_curve.jsonl。",
        "", "同150动作比较同时报告耗时；未进行无限动作的纯同墙钟对比。单种子、三例拟合结果不能证明收敛或泛化；未自动扩大实验。"]
    (root/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return result
