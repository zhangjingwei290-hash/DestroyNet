"""Replication must preserve pair identities, variance hierarchy and frozen actors."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
import emnh_seed_stability as repeat

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO/'runs_emnh/20261020_gpu_v2_seed20261020'


def original(arm='ppo'):
    return repeat.read(SOURCE/'development'/f'hemtv2_train_50_C_9_{arm}.json')


def plan_for(row):
    return copy.deepcopy(dict(solver_binding_id=row['binding_id'],boxes={row['iid']:dict(
        offset=row['physical_offset'],scale=row['physical_scale'],reference=row['normalized_reference'])}))


@pytest.mark.parametrize('field,value', [('seed',12),('iid','hemtv2_train_50_C_4'),
    ('arm','heuristic'),('budget_seconds',120.),('binding_id','changed')])
def test_wrong_cache_never_reused(field,value):
    row=original(); plan=plan_for(row); expected=(row['iid'],row['arm'],row['seed'])
    row[field]=value
    with pytest.raises(ValueError):repeat.validate_result(row,plan,*expected)


def test_frozen_box_and_finite_budget_checks():
    row=original(); plan=plan_for(row); expected=(row['iid'],row['arm'],row['seed'])
    repeat.validate_result(row,plan,*expected)
    row['physical_scale'][1]*=2
    with pytest.raises(AssertionError):repeat.validate_result(row,plan,*expected)
    row=original();row['search_seconds']=float('nan')
    with pytest.raises(ValueError):repeat.validate_result(row,plan,*expected)


def test_seed_and_iid_uncertainty_are_not_pooled_as_independent_iids():
    # Different IID effects, identical seeds: conditional seed uncertainty is 0,
    # while exploratory IID uncertainty remains. Treating 180 rows as IID fails.
    effect=np.repeat(np.arange(9,dtype=float)[:,None],20,axis=1)
    s=repeat.paired_statistics(effect,effect,samples=2000)
    assert s['distinct_iids']==9 and s['search_seed_count']==20
    assert s['relative_loss_intervals']['fixed_nine_iid_seed_bootstrap_95_ci']==[4.,4.]
    lo,hi=s['relative_loss_intervals']['crossed_iid_seed_bootstrap_95_ci']
    assert lo<4.<hi and s['training_seeds']==1


def test_single_seed_has_no_seed_confidence_interval_or_equivalence_claim():
    s=repeat.paired_statistics([[.01],[-.01]],[[1.],[-1.]])
    assert s['mean_delta_hv']==0.
    assert s['relative_loss_intervals']['fixed_nine_iid_seed_bootstrap_95_ci'] is None
    assert not s['equivalence_or_noninferiority_claim']


def test_nonfinite_pair_cannot_disappear_in_average():
    with pytest.raises(ValueError):repeat.paired_statistics([[float('nan')]],[[0.]])


def test_completed_campaign_cannot_accidentally_restart(tmp_path):
    (tmp_path/'complete.json').write_text('{}',encoding='utf-8')
    with pytest.raises(RuntimeError,match='Complete run'):repeat.run(tmp_path)


def test_entry_has_no_training_invocation_and_keeps_search_seed_separate():
    import ast
    tree=ast.parse(Path(repeat.__file__).read_text('utf-8'))
    calls=[n.func for n in ast.walk(tree) if isinstance(n,ast.Call)]
    assert not any(isinstance(f,ast.Name) and f.id in ('train_method','campaign') for f in calls)
    assert not any(isinstance(f,ast.Attribute) and f.attr in ('backward','start_inner','meta_update') for f in calls)
    assignments=[n for n in ast.walk(tree) if isinstance(n,(ast.Assign,ast.AnnAssign))]
    assert not any(isinstance(t,ast.Attribute) and t.attr=='seed' for n in assignments
                   for t in (n.targets if isinstance(n,ast.Assign) else [n.target]))


def test_real_frozen_policy_is_stochastic_repeatable_and_never_updated():
    from moea_emnh_v2.config import Config
    from moea_emnh_v2.runner import setup,load_directions
    from moea_emnh_v2.environment import Environment,Adaptive
    from moea_emnh_v2.outer import Search
    cfg=Config(**repeat.read(SOURCE/'config.json'));pool,split=setup(cfg)
    actors=load_directions(SOURCE/'ppo',cfg,cfg.binding_id)
    actor=actors[0];before={k:v.detach().clone() for k,v in actor.state_dict().items()}
    env=Environment(pool,split['fit_ids'][0],cfg,np.full(5,.2))
    obs=env.observation(torch.zeros(1,128),'greedy',Adaptive(len(env.q_values)),60.,60.,10.,10.)
    def samples(seed):
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        return [actor.act([obs])[0][0] for _ in range(5)]
    first=samples(20268021)
    assert first==samples(20268021) and first!=samples(20268022)
    # Four seconds of technical smoke on FIT data, not a formal development run.
    starts=[]
    for policy in (None,actors):
        torch.manual_seed(20268021);torch.cuda.manual_seed_all(20268021)
        search=Search(pool,split['fit_ids'][0],cfg,20268021,policy,seconds=2.)
        starts.append([p['objectives'] for p in search.point_events[:16]])
        search.run();row=search.report()
        assert row['seed']==20268021 and cfg.seed==20261020
        repeat.verify_result(row,search.base)
    assert starts[0]==starts[1]
    for key,value in actor.state_dict().items():torch.testing.assert_close(value,before[key],atol=0,rtol=0)
