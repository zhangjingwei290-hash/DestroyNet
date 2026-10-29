"""Bounded shared entrypoint. Formal training always needs explicit budgets."""
from dataclasses import replace
import json
from pathlib import Path
import random
import sys
import tempfile
import time
import traceback

import numpy as np
import torch

from .artifacts import (save_json, save_torch, file_id, checkpoint_payload, load_checkpoint,
                         training_contract, runtime_contract)
from .data import manifest, fit_ids
from .environment import TrainPool
from .execution import contract
from .model import JointPolicy, PPOTrainer
from .training import TrainingSession


def run(cfg, output_dir=None, validation_plan=None, wall_started=None):
    cfg.validate()
    torch.set_num_threads(1)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if cfg.mode != "evaluate" and cfg.checkpoint:
        raise ValueError("Use the same run directory for exact resume; warm-start training needs a separate approved protocol")
    base = Path(cfg.output_root).resolve()
    base.mkdir(parents=True, exist_ok=True)
    out = Path(output_dir).resolve() if output_dir else Path(tempfile.mkdtemp(
        prefix=time.strftime("%Y%m%d_%H%M%S_")+cfg.mode+"_", dir=base))
    out.mkdir(parents=True, exist_ok=True)
    if (out/"complete.json").exists():
        print(f"Already completed: {out}; nothing restarted", flush=True)
        return out
    try:
        # Raw TRAIN pool is read permission, not permission to fit every IID.
        pool = TrainPool(replace(cfg, train_instance_ids=()))
        split = manifest(cfg, pool)
        selected = fit_ids(cfg, pool, split) if cfg.mode != "evaluate" else ()
        if cfg.mode == "smoke":
            selected = tuple(sorted(selected, key=lambda i:(int(i.split("_")[2]),i)))
        expected = training_contract(cfg, split, selected)
        if validation_plan and validation_plan.get("evaluation_scope")=="fit_learnability_v1":
            from .learnability import check_plan_alignment
            check_plan_alignment(cfg,split,validation_plan)
        configuration = cfg.to_dict()
        # JSON roundtrip removes only tuple/list serialization differences.
        configuration = json.loads(json.dumps(configuration))
        config_path = out/"config.json"
        if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != configuration:
            raise ValueError("Existing run configuration changed; no implicit migration or budget extension")
        save_json(config_path, configuration)
        save_json(out/"split_manifest.json", split)
        sources = list(Path(__file__).parent.glob("*.py"))
        binding = {str(p):file_id(p) for p in sources}
        provenance_path = out/"provenance.json"
        provenance = dict(version="dpr_shared_training_v2", python=sys.executable, torch=str(torch.__version__),
            source_sha256=binding, execution_contract=contract(cfg), runtime_contract=runtime_contract(cfg),
            training_contract=expected,validation_plan=validation_plan,
            scope="fit_learnability_v1" if validation_plan and validation_plan.get("evaluation_scope")=="fit_learnability_v1" else ("diagnostic_only" if cfg.mode=="smoke" else "TRAIN_development_only"))
        if provenance_path.exists():
            previous = json.loads(provenance_path.read_text(encoding="utf-8"))
            if previous != provenance:
                raise ValueError("Source/protocol binding changed; refuse automatic resume")
        save_json(provenance_path, provenance)
        policy = JointPolicy(cfg).to(cfg.device)
        trainer = PPOTrainer(policy,cfg)
        print(f"DPR shared v2 | {cfg.mode} | {cfg.device} | {out}", flush=True)
        if cfg.mode == "evaluate":
            load_checkpoint(cfg.checkpoint,cfg,policy,cfg.checkpoint_manifest)
            print(f"Weights: loaded checkpoint {cfg.checkpoint}",flush=True)
            from .evaluation import evaluate_episode
            ids = tuple(cfg.train_instance_ids) or tuple(split["development_ids"])
            if not set(ids) <= set(pool.ids):
                raise ValueError("Evaluation IDs unavailable in this adapter's TRAIN source")
            iid = ids[0]
            result = evaluate_episode(pool,iid,cfg,policy,"learned",cfg.preference or (.2,)*5,
                cfg.seed,cfg.smoke_seconds,max_actions=cfg.evaluate_steps)
            for row in result.pop("records"):
                append(out/"trajectory.jsonl",row)
            result.update(status="completed" if result["requested_actions_completed"] else "time_limit",
                mode="evaluate",optimizer_epochs=0,quality_validated=False,checkpoint=str(cfg.checkpoint))
            save_json(out/"summary.json",result)
            save_json(out/"complete.json",dict(mode="evaluate",optimizer_steps=0,quality_validated=False))
            print(f"Evaluation: actions={result['steps']}; stop={result['stop_reason']}; optimizer steps=0",flush=True)
            return out
        saved = None
        checkpoint = out/"latest.pt"
        if checkpoint.exists():
            payload = load_checkpoint(checkpoint,cfg,policy)
            if payload["training_contract"] != expected or payload["source_binding"] != binding or payload.get("validation_plan") != validation_plan:
                raise ValueError("Training checkpoint contract/source mismatch")
            trainer.optimizer.load_state_dict(payload["optimizer"])
            saved = payload["session"]
            reconcile_logs(out,saved["actions"],saved["updates"])
            if (out/"failure.json").exists():
                recovery=out/"recovery"
                recovery.mkdir(exist_ok=True)
                (out/"failure.json").replace(recovery/f"resumed_failure_{time.time_ns()}.json")
        print("Weights: exact resume" if saved else "Weights: UNTRAINED / random initialization",flush=True)
        session = TrainingSession(cfg,pool,policy,trainer,selected,saved=saved)
        if saved is None and wall_started is not None:
            session.elapsed_seconds=max(0.,time.perf_counter()-wall_started)
        validation_used = float(saved.get("validation_seconds",0.)) if saved else 0.
        next_validation = int(saved.get("next_validation",validation_plan["validate_every"] if validation_plan else 0)) if saved else (validation_plan["validate_every"] if validation_plan else 0)
        before = {k:v.detach().cpu().clone() for k,v in policy.state_dict().items()}
        updates = []

        def snapshot(s):
            state=s.state()
            state.update(validation_seconds=validation_used,next_validation=next_validation)
            save_torch(checkpoint,checkpoint_payload(policy,trainer,cfg,split,selected,
                session=state,source_binding=binding,run_config=configuration,validation_plan=validation_plan,quality_validated=False))
            save_json(out/"progress.json",s.progress())

        def record(row):
            append(out/"trajectory.jsonl",row)
            if row["action_number"]%50==0:
                save_json(out/"health.json",dict(phase="training",updated_at=time.time(),pid=__import__("os").getpid(),**session.progress()))

        def update(metrics):
            nonlocal next_validation
            metrics["update_index"] = session.updates
            updates.append(metrics)
            append(out/"training_curve.jsonl",metrics)
            print(f"actions={session.actions}; optimized={session.optimized_actions}; "
                  f"PPO minibatch steps={session.optimizer_steps}",flush=True)
            if validation_plan:
                if session.actions >= next_validation:
                    development_check()
                    next_validation += validation_plan["validate_every"]

        def development_check():
            nonlocal validation_used
            from .campaign import validate
            immutable = out/"checkpoints"/f"step_{session.actions:07d}.pt"
            snapshot(session)
            if not immutable.exists():
                import shutil
                immutable.parent.mkdir(exist_ok=True)
                shutil.copyfile(checkpoint,immutable)
            elapsed = session.elapsed_seconds
            remaining = max(0.,min(cfg.train_seconds-elapsed,validation_plan["validation_seconds"]-validation_used))
            ts=time.perf_counter()
            result=validate(out,pool,validation_plan,cfg,policy,session.actions,immutable,remaining)
            validation_used += time.perf_counter()-ts
            return result

        # Validation shares the total runtime guard, but never consumes training
        # torch/numpy RNG. No validation occurs within a frozen collection round.
        if validation_plan and (session.actions == 0 or session.actions >= next_validation and session.actions==session.optimized_actions):
            ts = time.perf_counter()
            development_check()
            session.elapsed_seconds += time.perf_counter()-ts
            if session.actions:
                next_validation += validation_plan["validate_every"]
        snapshot(session)
        result = session.run(record,update,snapshot,final_reserve_seconds=min(600.,cfg.train_seconds/4) if validation_plan else 0.)
        if validation_plan:
            ts = time.perf_counter()
            result["final_validation"] = development_check()
            session.elapsed_seconds += time.perf_counter()-ts
            snapshot(session)
            result["elapsed_seconds"] = session.elapsed_seconds
            result["validation_seconds"] = validation_used
        result.update(mode=cfg.mode,status="completed" if result["stop_reason"]=="action_budget" else "time_limit",
            execution_contract=contract(cfg),training_contract=expected,checkpoint=str(checkpoint),updates_this_call=updates,
            changed_parameters={k:float((v.detach().cpu()-before[k]).abs().max())
                                for k,v in policy.state_dict().items() if not torch.equal(v.detach().cpu(),before[k])})
        if not session.actions or (cfg.mode=="smoke" and not session.successes):
            raise RuntimeError("No successful real repair; diagnostic failed")
        save_json(out/"summary.json",result)
        save_json(out/"complete.json",dict(status=result["status"],quality_validated=False,
                                         pending_actions=result["pending_actions"],stage3_started=False))
        print(f"Finished: {result['status']}; sampled={session.actions}; optimized={session.optimized_actions}; "
              f"pending={result['pending_actions']}",flush=True)
        return out
    except BaseException:
        if (out/"failure.json").exists():
            import shutil
            recovery=out/"recovery"
            recovery.mkdir(exist_ok=True)
            shutil.copyfile(out/"failure.json",recovery/f"previous_failure_{time.time_ns()}.json")
        save_json(out/"failure.json",dict(traceback=traceback.format_exc(),at=time.time()))
        raise


def append(path,value):
    with Path(path).open("a",encoding="utf-8") as f:
        f.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+"\n")


def reconcile_logs(out,actions,updates):
    """Preserve an uncommitted log tail separately when resuming a snapshot."""
    for name,key,limit in (("trajectory.jsonl","action_number",actions),("training_curve.jsonl","update_index",updates)):
        path = out/name
        if not path.exists():
            continue
        kept,tail=[],[]
        for line in path.read_text(encoding="utf-8").splitlines(keepends=True):
            (kept if json.loads(line)[key]<=limit else tail).append(line)
        if tail:
            recovery=out/"recovery"
            recovery.mkdir(exist_ok=True)
            (recovery/f"{time.time_ns()}_{name}").write_text("".join(tail),encoding="utf-8")
            path.write_text("".join(kept),encoding="utf-8")
