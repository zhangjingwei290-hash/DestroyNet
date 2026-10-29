"""Regression for the observed solver-horizon gap and external resource cutoff."""
from types import SimpleNamespace
import copy
import torch
import pytest
from moea_emnh_v2.environment import LocalSession
from moea_emnh_v2.outer import run_session
from moea_emnh_v2.training import returns


def session(monkeypatch, solver=10., resource=20., elapsed=3.6139538997085765):
    clock = SimpleNamespace(now=9.)
    monkeypatch.setattr('moea_emnh_v2.environment.time.perf_counter', lambda: clock.now)
    s = LocalSession.__new__(LocalSession)
    s.elapsed=elapsed; s.budget=3.6142480998532847
    s.solver_task_deadline=solver; s.outer_deadline=resource
    s.hidden=torch.zeros(1,1); s.pending=None; s.records=[]
    def prepare(**kwargs):
        s.pending=(0,0)
        return {'remaining': s.remaining()}
    def finish(*args):
        clock.now=10.; s.pending=None
        result=dict(reward=2., **s.ending()); s.records.append(result)
        return result
    s.prepare=prepare; s.finish=finish
    return s, clock


class Critic:
    critic=True
    def act(self, obs): return [{}],[0.],[0.],[torch.zeros(1)]
    def value(self, obs):
        assert obs['remaining'] > 0
        return 7.


def test_real_solver_horizon_is_terminal_despite_294_microsecond_gap(monkeypatch):
    s,_=session(monkeypatch)
    ep=[]; run_session(s, Critic(), episode=ep)
    assert s.elapsed < s.budget
    assert ep[-1]['terminal'] and not ep[-1]['truncated']
    assert ep[-1]['next_value'] == 0.
    assert ep[-1]['diagnostics']['terminated']
    assert ep[-1]['ending_reason']=='solver_budget'
    assert returns(ep)==[2.]


def test_external_resource_is_truncated_and_bootstraps_ppo(monkeypatch):
    s,_=session(monkeypatch, solver=20., resource=10.)
    ep=[]; run_session(s, Critic(), episode=ep)
    assert not ep[-1]['terminal'] and ep[-1]['truncated']
    assert ep[-1]['next_value']==7. and returns(ep)==[9.]
    assert ep[-1]['diagnostics']['truncated'] and s.outer_deadline==10.


def test_task_horizon_wins_simultaneous_resource_cutoff(monkeypatch):
    s,_=session(monkeypatch, solver=10., resource=10.)
    ep=[]; run_session(s, Critic(), episode=ep)
    assert ep[-1]['terminal'] and not ep[-1]['truncated']
    assert ep[-1]['next_value']==0.


def test_local_horizon_is_true_terminal(monkeypatch):
    s,_=session(monkeypatch, solver=20., resource=20., elapsed=3.62)
    ending=s.ending()
    assert ending['terminated'] and not ending['truncated']
    assert ending['ending_reason']=='local_budget'


def test_feature_preparation_exhausts_solver_horizon(monkeypatch):
    s,clock=session(monkeypatch)
    ep=[dict(terminal=False, next_value=0., reward=1.)]
    def prepare(**kwargs): clock.now=10.; s.pending=(0,0)
    s.prepare=prepare
    run_session(s, Critic(), episode=ep)
    assert len(ep)==1 and ep[-1]['terminal'] and ep[-1]['next_value']==0.
    assert s.pending is None


def test_controller_refuses_early_phase_transition():
    # Guard is invoked on the real meta -> adaptation path below.
    from moea_emnh_v2.training import require_resource_cutoff
    import time
    with pytest.raises(RuntimeError, match='before external'):
        require_resource_cutoff(time.perf_counter()+60., 5.)
    require_resource_cutoff(time.perf_counter()+.1, 5.)


def test_training_controller_rejects_incomplete_group_without_resource_cutoff(monkeypatch, tmp_path):
    from moea_emnh_v2.config import Config
    from moea_emnh_v2.training import train_method
    cfg=Config(meta_seconds=60., adaptation_seconds=0., legacy_ppo_seconds=0.)
    monkeypatch.setattr('moea_emnh_v2.training.collect_closed', lambda *args: ([], False, []))
    with pytest.raises(RuntimeError, match='Incomplete meta collection before external'):
        train_method(None, {'fit_ids': ['diagnostic_only']}, cfg, 'ppo', tmp_path, lambda v: None)
    assert not (tmp_path/'directions').exists()


def test_restarted_adaptation_preserves_debit_and_distributes_only_remainder():
    from moea_emnh_v2.training import adaptation_allocation
    debit=1249.5682869037846
    allocations=[adaptation_allocation(3600., debit, k) for k in range(16)]
    assert allocations[-1]==3600.
    assert allocations[0]-debit == pytest.approx((3600.-debit)/16)
    assert [adaptation_allocation(3600., 0., k) for k in range(16)] == [225.*(k+1) for k in range(16)]
    with pytest.raises(ValueError): adaptation_allocation(3600., -1., 0)
