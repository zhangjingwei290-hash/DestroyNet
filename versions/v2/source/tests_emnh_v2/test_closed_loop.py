import copy
import time
from types import SimpleNamespace
import numpy as np
import torch
import pytest
from moea_emnh_v2.config import Config, directions
from moea_emnh_v2.runner import setup
from moea_emnh_v2.outer import Search, search_state, restore_search, TaskPolicy
from moea_emnh_v2.training import collect_closed, freeze_collection, cyclic_tasks, training_gates
from moea_emnh_v2.model import Policy


@pytest.fixture(scope='module')
def resources():
    cfg = Config(legacy_ppo_seconds=0., local_seconds=.1, repair_seconds=.05)
    pool, split = setup(cfg)
    return cfg, pool, split


def no_local(k, s):
    return s, 0., []


def assert_same_search(a, b):
    assert a.algorithm.order == b.algorithm.order
    assert a.algorithm.cursor == b.algorithm.cursor
    for x, y in zip(a.algorithm.pop, b.algorithm.pop):
        assert x.X[0].routes == y.X[0].routes
        np.testing.assert_array_equal(x.F, y.F)
    for x, y in zip(a.adaptive, b.adaptive):
        assert x.features() == y.features()
    if a.algorithm.state is None or b.algorithm.state is None:
        assert a.algorithm.state is None and b.algorithm.state is None
    else:
        assert a.algorithm.state.X[0].routes == b.algorithm.state.X[0].routes


def test_checkpoint_keeps_pending_child_generation_rng_and_feedback(resources):
    cfg, pool, split = resources
    s = Search(pool, split['fit_ids'][0], cfg, 624, seconds=60.)
    for _ in range(3):
        assert s.step(no_local)
    s.adaptive[4].recent_improvement[2] = .123
    s.adaptive[4].count[2] = 17
    payload = search_state(s)
    restored = restore_search(payload, pool, cfg, None, time.perf_counter()+60)
    assert_same_search(s, restored)
    # Cross a generation boundary: no lost/duplicated pending child or RNG draw.
    for _ in range(22):
        assert s.step(no_local) and restored.step(no_local)
        assert_same_search(s, restored)


def test_pymoo_cursor_adapter_matches_unmodified_public_algorithm(resources):
    from pymoo.algorithms.moo.moead import MOEAD
    from pymoo.core.termination import NoTermination
    from pymoo.operators.mutation.nom import NoMutation
    from moea_emnh_v2.outer import InitialRoutes, RouteExchange, FixedTchebycheff
    cfg, pool, split = resources
    s = Search(pool, split['fit_ids'][0], cfg, 842, seconds=60.)
    original = MOEAD(ref_dirs=directions(), n_neighbors=5, prob_neighbor_mating=.9,
        decomposition=FixedTchebycheff(s.base.offset, s.base.scales), sampling=InitialRoutes(s.base),
        crossover=RouteExchange(), mutation=NoMutation())
    original.setup(s.problem, termination=NoTermination(), seed=842, verbose=False)
    initial = original.ask()
    for off in initial:
        off.set('F', s.base.evaluate(off.X[0].routes))
    original.tell(infills=initial)
    reference_rng = np.random.get_state()
    # Ignore local refinement in BOTH; check actual variation + replacement order.
    for _ in range(22):
        assert s.step(no_local)
        global_rng = np.random.get_state(); np.random.set_state(reference_rng)
        off = original.ask()
        g = off.X[0]
        routes = g.routes if s.base.api.feasible(s.base.solution(g.routes), s.base.instance) else g.fallback
        from moea_emnh_v2.outer import Genome
        off.set('X', np.asarray([Genome(copy.deepcopy(routes), copy.deepcopy(routes))], dtype=object))
        off.set('F', s.base.evaluate(routes)); original.tell(infills=off)
        reference_rng = np.random.get_state(); np.random.set_state(global_rng)
        for a, b in zip(s.algorithm.pop, original.pop):
            assert a.X[0].routes == b.X[0].routes
            np.testing.assert_array_equal(a.F, b.F)


def test_actual_twenty_rollouts_from_neural_search_and_persistent_history(resources):
    cfg, pool, split = resources
    torch.manual_seed(123)
    actor, meta = Policy(cfg, tasks=5, critic=True).cuda(), Policy(cfg).cuda()
    actor.start_inner(meta)
    rng = np.random.default_rng(74)
    weights = cyclic_tasks(rng)
    state = dict(next_iid=0, completed_searches=0)
    episodes, complete, provenance = collect_closed(pool, split, cfg, actor, meta, weights, state, rng,
                                                   time.perf_counter()+20, lambda v: None)
    assert complete and len(episodes) == 20 and len(provenance) == 5
    search = state['object']
    np.testing.assert_allclose(search.ref_dirs[:5], weights)
    assert all(not a.heuristic for a in search.adaptive)
    assert all(p['primary_committed'] for p in provenance)
    assert set(t['task'] for ep in episodes for t in ep) == set(range(5))
    for p in provenance:
        np.testing.assert_allclose(search.base.evaluate(p['source_routes']),
                                   next(ep[0]['diagnostics']['before_objectives'] for ep in episodes
                                        if ep[0]['task'] == p['direction']), atol=1e-7, rtol=0)
        assert search.adaptive[p['direction']].count.sum() > sum(p['source_adaptive_counts'])
    counts = [a.count.copy() for a in search.adaptive]
    payload = freeze_collection(state)
    assert payload['search']['elapsed'] == state['frozen_elapsed']
    # Subsequent sampling continues actual population; history is not copied then cleared.
    _, complete2, provenance2 = collect_closed(pool, split, cfg, actor, meta, weights, state, rng,
                                              time.perf_counter()+20, lambda v: None)
    assert complete2
    assert any(sum(p['source_adaptive_counts']) > 0 for p in provenance2)
    assert all(np.all(a.count >= before) for a,before in zip(search.adaptive, counts))


def test_coverage_is_actual_updates_not_file_count():
    split = dict(fit_ids=['a', 'b'])
    directions_info = [dict(direction=k, adapted=True) for k in range(16)]
    gate = training_gates([dict(steps=1, iids=['a']), dict(steps=0, iids=['b'])], directions_info, split)
    assert not gate['fit_coverage_passed'] and gate['missing_fit_iids'] == ['b']
    directions_info[6]['adapted'] = False
    gate = training_gates([dict(steps=1, iids=['a', 'b'])], directions_info, split)
    assert gate['fit_coverage_passed'] and not gate['training_coverage_passed']
    assert gate['unadapted_directions'] == [6]
    assert not gate['meta_learning_benefit_proven']


def test_training_resume_keeps_optimizer_and_closed_loop_state(resources, tmp_path):
    from moea_emnh_v2.training import train_method
    cfg, pool, split = resources
    cfg = copy.deepcopy(cfg); cfg.meta_seconds = 22.; cfg.adaptation_seconds = 0.
    cfg.local_seconds = .04; cfg.repair_seconds = .02
    def crash(v):
        if v['phase'] == 'meta_training':
            raise RuntimeError('test_interruption_after_committed_checkpoint')
    with pytest.raises(RuntimeError, match='test_interruption'):
        train_method(pool, split, cfg, 'ppo', tmp_path, crash)
    first = torch.load(tmp_path/'resume.pt', weights_only=False)
    assert first['history'][0]['steps'] > 0
    assert first['state']['search']['pending'] is not None
    models, report = train_method(pool, split, cfg, 'ppo', tmp_path, lambda v: None)
    assert report['updates'] > len(first['history'])
    assert report['meta_seconds'] >= first['meta_spent']
    assert not report['training_coverage_passed']
    assert not report['meta_learning_benefit_proven']
    assert len(models) == 16
