import copy
import time
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from mo_dpr.environment import TrainPool
from mo_dpr.data import manifest
from moea_emnh_v2.config import Config, CONTEXT, q_values, directions
from moea_emnh_v2.model import Policy
from moea_emnh_v2.environment import Environment, Adaptive, LocalSession, classical_removal
from moea_emnh_v2.training import update, returns
from moea_emnh_v2.outer import FixedTchebycheff, Search


@pytest.fixture(scope="module")
def cfg():
    torch.set_num_threads(1)
    value = Config(); value.validate()
    assert torch.cuda.is_available(), "GPU tests cannot silently fall back to CPU"
    return value


@pytest.fixture(scope="module")
def pool(cfg):
    p = TrainPool(cfg); manifest(cfg, p)
    return p


@pytest.fixture()
def env(pool, cfg):
    return Environment(pool, "hemtv2_train_50_C_4", cfg, np.full(5, .2))


def observation(env, cfg, previous=None):
    if previous is not None:
        env.last_removed = previous; env.last_accepted = True
    return env.observation(torch.zeros(1, cfg.hidden_dim), "regret2", Adaptive(len(env.q_values)), 50., 60., 5., 10.)


def test_real_cuda_batch_preserves_variable_history(env, cfg):
    torch.manual_seed(42)
    policy = Policy(cfg).cuda()
    first = observation(env, cfg, [1])
    second = observation(env, cfg, [2, 3, 4, 5, 6])
    assert len(first["context"]) == len(CONTEXT) == 45
    assert first["context"][10].item() == pytest.approx(50/300)
    with torch.no_grad():
        batch = policy.encode_batch([first, second])
        for i, obs in enumerate([first, second]):
            single = policy.encode_batch([obs])
            for a, b in zip(batch, single):
                torch.testing.assert_close(a[i], b[0], rtol=2e-5, atol=2e-6)
    assert next(policy.parameters()).is_cuda


def test_joint_likelihood_replay_and_full_removal(env, cfg):
    p = Policy(cfg).cuda()
    obs = observation(env, cfg)
    actions, lp, _, h = p.act([obs]*8)
    _, replay, _, _, _ = p.terms([obs]*8, actions=actions)
    np.testing.assert_allclose(replay.detach().cpu(), lp, atol=2e-5, rtol=0)
    assert h.shape == (8, 128)
    # Broad q support really reaches full reconstruction; exact q and no depot.
    for q in env.q_values:
        a = dict(q=q, anchors=[1] if q == 1 else [1, 2], coefficients=[.5]*min(2, q))
        removed = env.deletion(a)
        assert len(removed) == len(set(removed)) == q and 0 not in removed
    assert env.q_values[-1] == env.instance.n_customers
    bad = copy.deepcopy(actions)
    bad[0]["q"] += 1
    with pytest.raises(ValueError, match="mismatch"):
        p.terms([obs]*8, actions=bad)


def test_meta_body_copy_head_interpolation_not_whole_reptile(cfg):
    meta, inner = Policy(cfg).cuda(), Policy(cfg, tasks=5, critic=True).cuda()
    inner.start_inner(meta)
    old = copy.deepcopy(meta.state_dict())
    with torch.no_grad():
        for n, p in inner.named_parameters():
            p.add_(float(int(n.split('.')[1])+1) if n.startswith("heads.") else 7.)
    critic = copy.deepcopy(inner.critics.state_dict())
    meta.meta_update(inner, .2)
    for n, p in meta.state_dict().items():
        expected = old[n]+(.6 if n.startswith("heads.0.") else 7.)
        torch.testing.assert_close(p, expected)
    for n, p in inner.critics.state_dict().items():
        torch.testing.assert_close(p, critic[n])
    assert not any(n.startswith("critics.") for n in meta.actor_state())


def test_strict_equal_reject_and_late_exclusion(env, cfg):
    original = env.api.repair
    env.api.repair = lambda *a, **k: dict(feasible=True, timeout=False, reason="ok",
        routes=copy.deepcopy(env.routes), objectives=env.objectives.tolist(), seconds=0.)
    action = dict(q=1, anchors=[1], coefficients=[.5])
    result = env.step(action, "greedy", 1, time.perf_counter()+1.)
    assert result["reward"] == 0 and not result["accepted"]
    before = copy.deepcopy(env.archive)
    result = env.step(action, "greedy", 1, time.perf_counter()-1.)
    assert result["timeout"] and result["candidate_objectives"] is None
    assert env.archive == before
    env.api.repair = original


def test_failed_objectives_not_filled(env):
    original = env.api.repair
    env.api.repair = lambda *a, **k: dict(feasible=False, timeout=False, reason="no_legal_insertion", seconds=.1)
    result = env.step(dict(q=1, anchors=[1], coefficients=[.5]), "greedy", 1, time.perf_counter()+1)
    assert result["candidate_objectives"] is None and result["reward"] == 0
    env.api.repair = original


def test_heuristic_does_not_pay_unused_neural_features(env, cfg):
    adaptive = Adaptive(len(env.q_values), heuristic=True)
    session = LocalSession(env, adaptive, 1, 60., 60., 10.)
    assert session.budget == 60/16
    env.observation = lambda *a: (_ for _ in ()).throw(AssertionError("Unused neural features"))
    assert session.prepare(include_observation=False) is None
    assert session.pending is not None


def test_classical_support_and_same_scalar(env):
    rng = np.random.default_rng(1)
    for d in range(4*len(env.q_values)):
        removed = classical_removal(env, d, rng)
        assert len(set(removed)) == len(removed) == env.q_values[d % len(env.q_values)]
    dec = FixedTchebycheff(env.offset, env.scales)
    for w in directions():
        env.preference = w
        assert float(dec.do(env.objectives, w).item()) == pytest.approx(env.score(env.objectives))
    assert directions().shape == (16, 5)
    assert q_values(50) == (1, 3, 5, 10, 20, 30, 40, 50)


@pytest.mark.parametrize("method", ["ppo", "reinforce"])
def test_batched_optimizer_changes_actor_and_rejects_false_terminal(env, cfg, method):
    torch.manual_seed(22)
    p = Policy(cfg, critic=method == "ppo").cuda()
    o = observation(env, cfg)
    actions, lp, values, _ = p.act([o]*4)
    episodes = [[dict(observation=o, action=actions[i], logprob=lp[i], value=values[i],
        task=0, reward=.01*(i+1), terminal=True, next_value=0., diagnostics={})] for i in range(4)]
    before = p.encoder.gat_0.lin.weight.detach().clone()
    result = update(p, torch.optim.Adam(p.parameters(), lr=cfg.learning_rate), episodes, method, cfg)
    assert result["steps"] > 0
    assert torch.isfinite(p.encoder.gat_0.lin.weight).all()
    assert not torch.equal(before, p.encoder.gat_0.lin.weight)
    assert max(result["head_gradients"].values()) > 0
    if method == "ppo":
        assert result["initial_ratio_error"] < 1e-4
    else:
        episodes[0][-1]["terminal"] = False
        with pytest.raises(ValueError, match="complete"):
            update(p, torch.optim.Adam(p.parameters(), lr=cfg.learning_rate), episodes, method, cfg)
    assert returns([dict(reward=1., terminal=False, next_value=2.)]) == [3.]
    assert returns([dict(reward=1., terminal=True, next_value=2.)]) == [1.]


def test_actual_public_moead_physical_population(pool, cfg):
    local = copy.deepcopy(cfg); local.local_seconds = .05; local.repair_seconds = .05
    search = Search(pool, "hemtv2_train_50_C_4", local, 123, seconds=.5)
    search.run()
    assert search.algorithm.__class__.__bases__[0].__name__ == "MOEAD"
    assert len(search.algorithm.pop) == 16
    for p in search.algorithm.pop:
        np.testing.assert_allclose(search.base.evaluate(p.X[0].routes), p.F, atol=1e-7, rtol=0)
    assert search.report()["hv"] >= 0


def test_repair_adapter_matches_old_within_old_cap_and_extends_it(env):
    from hemt_destroy_feature_labels_v1.repair import repair as old
    from moea_emnh_v2.repair import repair as new
    for repairer in ("greedy", "regret2", "regret3", "noisy_greedy"):
        args = (env.instance, env.routes, [1, 2, 3], repairer, env.preference,
                {"mins":env.offset, "ranges":env.scales}, 321)
        before, after = old(*args), new(*args)
        assert before["reason"] == after["reason"]
        assert before["routes"] == after["routes"]
        assert before["objectives"] == after["objectives"]
    result = new(env.instance, env.routes, list(range(1, 51)), "greedy", env.preference,
                 {"mins":env.offset, "ranges":env.scales}, 3, time.perf_counter()+.02)
    assert result["reason"] != "removal_cap_exceeded"


def test_finite_training_and_export_reload(pool, cfg, tmp_path):
    from moea_emnh_v2.training import train_method
    from mo_dpr.data import manifest
    # Diagnostic resource truncation, not a replacement for formal 10s episodes.
    small = copy.deepcopy(cfg)
    small.meta_seconds = 25.; small.adaptation_seconds = 0.; small.legacy_ppo_seconds = 0.
    small.local_seconds = .15; small.repair_seconds = .1; small.replicas = 2
    messages = []
    models, report = train_method(pool, manifest(small, pool), small, "ppo", tmp_path, messages.append)
    assert len(models) == len(report["directions"]) == 16
    assert report["updates"] > 0
    assert all(not d["adapted"] and d["updates"] == 0 for d in report["directions"])
    checkpoint = torch.load(tmp_path/"directions/direction_00.pt", weights_only=True)
    loaded = Policy(small).cuda(); loaded.load_state_dict(checkpoint["model"])
    for key, value in models[0].state_dict().items():
        torch.testing.assert_close(value, loaded.state_dict()[key])
    assert all(not key.startswith("critics.") for key in checkpoint["model"])


def test_windows_binding_path_case_does_not_change_content_identity(pool, cfg, monkeypatch):
    import moea_emnh_v2.runner as runner
    from mo_dpr.data import manifest
    first = runner.binding(cfg, manifest(cfg, pool))
    monkeypatch.setattr(runner, "__file__", runner.__file__.upper())
    assert runner.binding(cfg, manifest(cfg, pool)) == first


def test_existing_run_refused_without_overwriting_receipts(cfg, tmp_path):
    from moea_emnh_v2.runner import campaign
    receipt = tmp_path/"process.json"
    receipt.write_text('{"untouched":true}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="immutable"):
        campaign(tmp_path, cfg, "unused")
    assert receipt.read_text(encoding="utf-8") == '{"untouched":true}'
    assert not (tmp_path/"worker.lock").exists()
    assert not (tmp_path/"failure.json").exists()


def test_actual_failed_skeleton_is_rejected_and_recorded(pool, cfg):
    import json
    from pathlib import Path
    sample = json.loads((Path(__file__).resolve().parents[1]/
        'diagnostics/emnh_repair_incident_20261020/reproduced_failure.json').read_text())
    e = Environment(pool, sample['iid'], cfg, sample['preference'], sample['source_routes'])
    before, archive = e.objectives.copy(), copy.deepcopy(e.archive)
    assert e.api.feasible(e.solution(e.routes), e.instance)
    result = e.step(dict(q=10), sample['repairer'], sample['seed'], time.perf_counter()+10,
                    removal_override=sample['removed'])
    assert result['reason'] == 'final_solution_infeasible'
    assert result['failed'] and not result['accepted'] and result['reward'] == 0
    assert result['candidate_objectives'] is None
    assert e.routes == sample['source_routes'] and e.archive == archive
    np.testing.assert_array_equal(e.objectives, before)


def test_false_success_still_stops_for_physical_violation(pool, cfg):
    import json
    from pathlib import Path
    sample = json.loads((Path(__file__).resolve().parents[1]/
        'diagnostics/emnh_repair_incident_20261020/reproduced_failure.json').read_text())
    e = Environment(pool, sample['iid'], cfg, sample['preference'], sample['source_routes'])
    e.api.repair = lambda *a, **k: dict(feasible=True, failed=False, timeout=False, reason='ok',
        routes=sample['reconstructed_final'], objectives=[0.]*5, seconds=0.)
    with pytest.raises(ValueError):
        e.step(dict(q=10), sample['repairer'], sample['seed'], time.perf_counter()+10,
               removal_override=sample['removed'])


def test_recent_history_changes_without_erasing_cumulative_counts():
    a = Adaptive(8)
    first = dict(accepted=True, feasible=True, timeout=False, reward=.4, environment_seconds=2.)
    later = dict(accepted=False, feasible=False, timeout=False, reward=0., environment_seconds=12.)
    a.update(0, 1, first, 1.)
    a.update(0, 1, later, 1.)
    assert a.count[1] == 2 and a.seconds[1] == 14.
    assert a.recent_seconds[1] == pytest.approx(4.)
    assert a.recent_success[1] == pytest.approx(.8)
    assert a.recent_improvement[1] == pytest.approx(.32)
    assert a.features()[5] == pytest.approx(4/300)
    restored = copy.deepcopy(a)
    assert restored.features() == a.features()
    assert restored.best == a.best
