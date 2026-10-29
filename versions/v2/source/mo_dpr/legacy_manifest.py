"""Verify the known completed v1 study; emit a separate inference-only sidecar.

Never edits the original plan, checkpoint or results. It does not convert old
training into v2 training and cannot be used for v2 optimizer resume.
"""
import json
from pathlib import Path

import torch

from .artifacts import file_id, runtime_contract, save_json
from .config import Config
from .data import fingerprint
from .environment import NODE_FEATURES, CONTEXT_FEATURES
from .model import SCHEMA


def verify(checkpoint, study, destination):
    checkpoint,study,destination=Path(checkpoint).resolve(),Path(study).resolve(),Path(destination).resolve()
    repo=Path(__file__).resolve().parents[1]
    if study!=repo/"runs_stage12"/"20261015_stage12_v1" or checkpoint.parent!=study:
        raise ValueError("This verifier is scoped to the known completed v1 study root checkpoint")
    if not (study/"complete.json").exists() or destination.is_relative_to(study):
        raise ValueError("Completed study required; sidecar must be outside the immutable study")
    plan=json.loads((study/"plan.json").read_text(encoding="utf-8"))
    # v1 used default JSON separators for its frozen plan SHA.
    import hashlib
    raw={k:v for k,v in plan.items() if k!="plan_sha256"}
    if hashlib.sha256(json.dumps(raw,sort_keys=True,ensure_ascii=False).encode()).hexdigest()!=plan["plan_sha256"]:
        raise ValueError("Original frozen plan hash mismatch")
    cfg=Config(**plan["config"])
    payload=torch.load(checkpoint,map_location="cpu",weights_only=True)
    if (payload.get("schema")!=SCHEMA or payload.get("plan_sha256")!=plan["plan_sha256"]
            or payload["node_features"]!=list(NODE_FEATURES) or payload["context_features"]!=list(CONTEXT_FEATURES)
            or payload["model_config"]!={k:getattr(cfg,k) for k in ("hidden_dim","critic_dim","anchors")}):
        raise ValueError("Original weight metadata mismatch")
    if cfg.episode_steps!=150 or cfg.near_neighbors!=10 or cfg.fixed_q is not None:
        raise ValueError("Unreviewed original execution configuration")
    binding=json.loads((study/"source_binding.json").read_text(encoding="utf-8"))
    snapshot=repo/"diagnostics"/"training_contract_v2"/"previous_source"/"mo_dpr"
    for name,divider in (("model.py","class PPOTrainer:"),("environment.py","class Environment:")):
        old=snapshot/name
        current=repo/"mo_dpr"/name
        if file_id(old)!=binding[str(current)]:
            raise ValueError("Pre-edit source snapshot does not match original binding")
        previous_text=old.read_text(encoding="utf-8")
        current_text=current.read_text(encoding="utf-8")
        # Model policy before PPO is unchanged; environment behavior after the
        # input-loader integrity check is unchanged. These checks are explicit.
        old_part=previous_text.split(divider)[0 if name=="model.py" else 1]
        new_part=current_text.split(divider)[0 if name=="model.py" else 1]
        if old_part!=new_part:
            raise ValueError("Policy/environment behavior changed; legacy compatibility needs fresh review")
    for key,sha in binding.items():
        if "\\lib\\" in key or "\\versions\\version0.7\\" in key:
            if file_id(key)!=sha:
                raise ValueError("Original dependency source binding mismatch")
    info=dict(version="dpr_verified_legacy_v2",checkpoint=str(checkpoint),checkpoint_sha256=file_id(checkpoint),
        plan=str(study/"plan.json"),plan_sha256=plan["plan_sha256"],original_training_config=plan["config"],
        original_fit_ids=plan["fit_ids"],original_development_ids=plan["validation_ids"],
        runtime_contract=runtime_contract(cfg),scope="inference_only_under_authorized_sampled_150_action_protocol",
        training_protocol="legacy_128_transition_refresh; NOT v2 whole-trajectory training",
        evidence=[str(repo/"diagnostics/alignment_v1/full_episode_check.json"),
                  str(repo/"diagnostics/upstream_consistency_v1/结论.md")],
        verification="original plan/content/names/dimensions/dependency hashes and unchanged actor/environment code")
    save_json(destination,info)
    return info


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--study",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    verify(a.checkpoint,a.study,a.output)
