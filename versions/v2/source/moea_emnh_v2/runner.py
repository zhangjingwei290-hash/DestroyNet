"""Reviewable smoke/calibration, frozen finite campaign and paired dev report."""
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
import torch
import psutil
from mo_dpr.artifacts import file_id, save_json, save_torch
from mo_dpr.data import manifest, fingerprint
from mo_dpr.environment import TrainPool
from .config import Config, runtime_contract, directions, q_values
from .environment import Environment, Adaptive
from .model import Policy
from .outer import Search
from .training import train_method


def setup(cfg):
    cfg.validate()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; CPU fallback forbidden")
    if importlib.metadata.version("pymoo") != "0.6.1.5" or importlib.metadata.version("alns") != "7.0.0":
        raise RuntimeError("Public solver version differs from reviewed implementation")
    torch.set_num_threads(cfg.cpu_threads)
    torch.manual_seed(cfg.seed); torch.cuda.manual_seed_all(cfg.seed)
    pool = TrainPool(cfg)
    split = manifest(cfg, pool)
    return pool, split


def binding(cfg, split):
    repo = Path(__file__).resolve().parents[1]
    files = list(Path(__file__).parent.glob("*.py"))+[Path(__file__).parent/"PROTOCOL.md", repo/"emnh_v2_run.py"]
    files += [repo/p for p in ("lib/ppo_1.py", "lib/vrp_helper_env.py", "mo_dpr/environment.py",
                               "mo_dpr/data.py", "mo_dpr/artifacts.py")]
    files += [Path(cfg.project_root)/p for p in ("src/evaluation/vrptw_eval.py", "src/pilot/engine.py",
                                               "src/core/protocol.py", "src/core/task_protocol.py")]
    return dict(schema=cfg.version, contract=runtime_contract(cfg), split=split,
        # Windows paths are case-insensitive. Normalize identity, still hash
        # actual bytes: launcher capitalization cannot look like code changes.
        sources={os.path.normcase(str(p.resolve())):file_id(p) for p in files},
        dependencies={k:importlib.metadata.version(k) for k in ("torch", "torch-geometric", "numpy", "pymoo", "alns", "moocore")})


def calibration(root, cfg):
    root = Path(root); root.mkdir(parents=True, exist_ok=False)
    pool, split = setup(cfg)
    payload = binding(cfg, split); save_json(root/"binding.json", payload)
    policy = Policy(cfg, tasks=5, critic=True).cuda()
    torch.cuda.reset_peak_memory_stats()
    rows, timings = [], []
    for n in (50, 150, 250):
        iid = f"hemtv2_train_{n}_C_4"
        env = Environment(pool, iid, cfg, np.full(5, .2))
        obs = env.observation(torch.zeros(1, 128), "greedy", Adaptive(len(env.q_values)),
                              float(n*1.2), float(n*1.2), 10., 10.)
        for batch in (1, 8, 20):
            # Warm CUDA kernels before timing; deployment is also warmed once.
            policy.act([obs]*batch, [i % 5 for i in range(batch)])
            times = []
            for _ in range(3):
                t = time.perf_counter()
                action, lp, _, _ = policy.act([obs]*batch, [i % 5 for i in range(batch)])
                times.append(time.perf_counter()-t)
            timings.append(dict(N=n, graphs=batch, inference_seconds=float(np.median(times))))
            if batch == 8:
                optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.learning_rate)
                optimizer.zero_grad()
                t = time.perf_counter()
                _, logp, values, entropy, _ = policy.terms([obs]*batch, [i % 5 for i in range(batch)], action)
                loss = -logp.mean()+values.square().mean()-.001*entropy.mean()
                loss.backward(); torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.)
                optimizer.step(); torch.cuda.synchronize()
                timings.append(dict(N=n, graphs=batch, optimizer_microbatch_seconds=time.perf_counter()-t))
        for r in ("greedy", "regret2", "regret3", "noisy_greedy"):
            for q in sorted({env.q_values[0], env.q_values[len(env.q_values)//2], n}):
                env = Environment(pool, iid, cfg, np.full(5, .2))
                action = dict(q=q, anchors=[1] if q == 1 else [1, 2], coefficients=[.5]*min(2, q))
                # Diagnostic deadline only; formal protocol remains 10 seconds.
                record = env.step(action, r, cfg.seed, time.perf_counter()+1.)
                if record["reason"] == "removal_cap_exceeded":
                    raise RuntimeError("Broad-granularity repair adapter not wired")
                rows.append(dict(iid=iid, N=n, diagnostic_cap_seconds=1., **record))
        save_json(root/"progress.json", dict(N=n, records=len(rows), updated=time.time()))
        print(f"Calibration N={n} complete", flush=True)
    one = Policy(cfg).cuda()
    deployment_params = sum(p.numel() for p in one.parameters())
    report = dict(device=torch.cuda.get_device_name(0), total_vram_bytes=torch.cuda.get_device_properties(0).total_memory,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        deployment_actor_parameters=deployment_params, inner_5task_5critic_parameters=sum(p.numel() for p in policy.parameters()),
        context_dimensions=45, microbatch=cfg.microbatch, logical_trajectories=cfg.task_count*cfg.replicas,
        timings=timings, repair_diagnostics=rows, tests_required="tests_emnh_v2/test_contract.py",
        gate_passed=all(any(r["N"] == n and r["q"] == min(q_values(n)) and r["feasible"] and not r["timeout"]
                           for r in rows) for n in (50, 150, 250)),
        limitations="1s repair diagnostics are not formal performance; no learned-policy benefit asserted")
    save_json(root/"calibration.json", report)
    print(json.dumps({k:v for k,v in report.items() if k not in ("timings", "repair_diagnostics")}, ensure_ascii=False), flush=True)
    return report


def development_json(value):
    """Convert NumPy report containers/scalars without changing solver state or values.

    Unknown types still fail in save_json, and allow_nan=False remains enforced.
    """
    if isinstance(value, np.generic):
        return development_json(value.item())
    if isinstance(value, np.ndarray):
        return development_json(value.tolist())
    if isinstance(value, dict):
        return {development_json(k): development_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [development_json(v) for v in value]
    return value


def save_development_result(path, result):
    save_json(path, development_json(result))


def dev_report(results, root, cfg):
    by_arm = {}
    for row in results:
        by_arm.setdefault(row["arm"], {})[row["iid"]] = row
    ids = sorted(by_arm["heuristic"])
    comparisons = {}
    rng = np.random.default_rng(cfg.seed)
    for arm in ("ppo", "reinforce"):
        diff = np.array([by_arm[arm][i]["hv"]-by_arm["heuristic"][i]["hv"] for i in ids])
        sample = rng.choice(len(ids), size=(10000, len(ids)), replace=True)
        interval = np.quantile(diff[sample].mean(1), [.025, .975]).tolist()
        comparisons[arm] = dict(paired_hv_difference_mean=float(diff.mean()),
            iid_bootstrap_95_ci=interval, differences=dict(zip(ids, diff.tolist())),
            distinct_iids=len(ids), training_seeds=1, search_seeds=1,
            positive_lower_bound=bool(interval[0]>0))
    summary = dict(comparisons=comparisons, scope="9 existing TRAIN development IIDs; one training/search seed",
                   final_generalization_proven=False, automatic_next_stage=False)
    save_json(Path(root)/"development_report.json", summary)
    text = ["# 首轮研发结果", "", "单训练种子、单搜索种子、9个研发IID；不构成最终泛化证明。", ""]
    for arm, row in comparisons.items():
        text.append(f"- {arm}: 配对HV均差 {row['paired_hv_difference_mean']:.8g}，按IID bootstrap 95%区间 {row['iid_bootstrap_95_ci']}。")
    (Path(root)/"REPORT.md").write_text("\n".join(text)+"\n", encoding="utf-8")
    return summary


def load_directions(root, cfg, expected_binding):
    models = []
    for k in range(16):
        payload = torch.load(Path(root)/"directions"/f"direction_{k:02d}.pt", map_location="cuda", weights_only=True)
        if (payload["schema"] != cfg.version or payload["config"]["binding_id"] != expected_binding
                or payload["direction"] != k or payload["weight"] != directions()[k].tolist()):
            raise ValueError("Frozen deployment checkpoint/contract/direction mismatch")
        saved_cfg = Config(**payload["config"])
        if runtime_contract(saved_cfg) != runtime_contract(cfg):
            raise ValueError("Deployment runtime contract changed")
        actor = Policy(cfg).cuda(); actor.load_state_dict(payload["model"]); actor.eval()
        models.append(actor)
    return models


def campaign(root, cfg, calibration_path, resume=False):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    from .recovery import acquire, release
    token = acquire(root, cfg, resume)
    lock = root/"worker.lock"
    def notify(v):
        v = dict(updated=time.time(), pid=os.getpid(), **v)
        save_json(root/"progress.json", v)
        save_json(root/"health.json", dict(updated=v["updated"], phase=v["phase"], pid=os.getpid()))
        if v["phase"] != "collection":
            print(json.dumps(v, ensure_ascii=False, allow_nan=False), flush=True)
    try:
        if (root/"complete.json").exists():
            raise RuntimeError("Completed run cannot resume")
        pool, split = setup(cfg)
        cal = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
        if not cal["gate_passed"]:
            raise RuntimeError("Calibration gate failed")
        cal_binding = json.loads(Path(calibration_path).with_name("binding.json").read_text(encoding="utf-8"))
        current = binding(cfg, split)
        if current["sources"] != cal_binding["sources"] or current["dependencies"] != cal_binding["dependencies"]:
            raise RuntimeError("Source/dependency changed since calibration")
        cfg.binding_id = fingerprint(current)
        if resume:
            old = json.loads((root/"binding.json").read_text(encoding="utf-8"))
            if old["id"] != cfg.binding_id:
                raise RuntimeError("Source/config binding mismatch: diagnosis required before recovery")
        else:
            save_json(root/"binding.json", dict(id=cfg.binding_id, **current))
        save_json(root/"config.json", cfg.to_dict())
        if not resume:
            import zipfile
            with zipfile.ZipFile(root/"source_snapshot.zip", "x", zipfile.ZIP_DEFLATED) as z:
                z.writestr("binding.json", json.dumps(current, ensure_ascii=False, indent=2))
                for index, path in enumerate(current["sources"]):
                    z.write(path, f"sources/{index:03d}_{Path(path).name}")
        notify(dict(phase="initializing", config=cfg.to_dict()))
        # Warm neural CUDA path outside measured deployment solver clocks, same
        # across both neural arms; disclose warm-up separately.
        env = Environment(pool, split["fit_ids"][0], cfg, np.full(5, .2))
        obs = env.observation(torch.zeros(1, 128), "greedy", Adaptive(len(env.q_values)), 60., 60., 10., 10.)
        Policy(cfg).cuda().act([obs])
        policies, training_reports = {}, {}
        def budget_update(method, stage, ms, ads):
            path = root/"budget.json"
            ledger = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            ledger[method] = dict(meta_spent=ms, adapt_spent=ads)
            ledger.update(active_method=method, active_stage=stage, updated=time.time())
            save_json(path, ledger)
        for method in ("ppo", "reinforce"):
            policies[method], training_reports[method] = train_method(pool, split, cfg, method, root/method, notify, budget_update)
        save_json(root/"training_reports.json", training_reports)
        # Evaluate exported frozen actors, not unrecorded in-memory weights.
        for method in ("ppo", "reinforce"):
            policies[method] = load_directions(root/method, cfg, cfg.binding_id)
        ledger = json.loads((root/"budget.json").read_text(encoding="utf-8"))
        ledger.update(active_stage="evaluation", updated=time.time()); save_json(root/"budget.json", ledger)
        results = []
        # Shared problem ingestion outside all timed solver calls; no first-arm
        # input-file/verification tax that later cached arms would avoid.
        for iid in split["development_ids"]:
            pool.load(iid)
        # Rotate arm order by IID, avoiding a systematic first/last thermal bias.
        for index, iid in enumerate(split["development_ids"]):
            arms = ("heuristic", "ppo", "reinforce")
            order = arms[index % 3:]+arms[:index % 3]
            for arm in order:
                cache = root/"development"/f"{iid}_{arm}.json"
                if cache.exists():
                    cached = json.loads(cache.read_text(encoding="utf-8"))
                    if cached.get("binding_id") != cfg.binding_id:
                        raise ValueError("Cached evaluation binding mismatch")
                    results.append(cached); continue
                torch.manual_seed(cfg.seed+7000); torch.cuda.manual_seed_all(cfg.seed+7000)
                search = Search(pool, iid, cfg, cfg.seed+7000, policies.get(arm))
                search.run(progress=lambda s: notify(dict(phase="development", iid=iid, arm=arm,
                    solver_elapsed=time.perf_counter()-s.started, actions=len(s.records))))
                result = dict(arm=arm, binding_id=cfg.binding_id, **search.report())
                results.append(result)
                save_development_result(root/"development"/f"{iid}_{arm}.json", result)
        report = dev_report(results, root, cfg)
        report["training_coverage_passed"] = all(r["training_coverage_passed"] for r in training_reports.values())
        report["training_reports"] = training_reports
        save_json(root/"development_report.json", report)
        notify(dict(phase="complete", report=report))
        save_json(root/"complete.json", dict(at=time.time(), report=report, automatic_next_stage=False))
    except BaseException:
        save_json(root/"failure.json", dict(at=time.time(), traceback=traceback.format_exc()))
        notify(dict(phase="failed"))
        raise
    finally:
        release(root, token)
