"""The actual evaluation report must persist while invalid values still fail."""
import json
import numpy as np
import pytest
from moea_emnh_v2.runner import development_json, save_development_result


def test_nested_routes_scalars_arrays_preserve_values_and_original_types(tmp_path):
    data=dict(routes=[[np.int64(1),np.int32(2)]],removed=[np.int64(3)],
              q=np.int64(25),hv=np.float64(.123),flags=np.array([True,False]),
              exact_large_integer=np.uint64(2**63+7),success=np.bool_(True),objectives=None)
    with pytest.raises(TypeError):json.dumps(data,allow_nan=False)
    path=tmp_path/'result.json';save_development_result(path,data)
    decoded=json.loads(path.read_text(encoding='utf-8'))
    assert decoded==dict(routes=[[1,2]],removed=[3],q=25,hv=.123,flags=[True,False],
                         exact_large_integer=2**63+7,success=True,objectives=None)
    assert isinstance(data['routes'][0][0],np.int64)
    assert isinstance(data['flags'],np.ndarray)


@pytest.mark.parametrize('value',[np.float64('nan'),np.float32('inf'),np.float64('-inf')])
def test_nonfinite_remains_fatal_and_previous_file_survives(tmp_path,value):
    path=tmp_path/'result.json';save_development_result(path,dict(hv=1.))
    original=path.read_bytes()
    with pytest.raises(ValueError):save_development_result(path,dict(hv=value))
    assert path.read_bytes()==original


def test_unknown_type_not_silently_stringified_or_dropped(tmp_path):
    with pytest.raises(TypeError):save_development_result(tmp_path/'result.json',dict(routes=object()))
    assert not (tmp_path/'result.json').exists()


def test_real_search_report_exports_all_fields_with_numpy_route_id(tmp_path):
    from moea_emnh_v2.config import Config
    from moea_emnh_v2.runner import setup
    from moea_emnh_v2.outer import Search
    cfg=Config(local_seconds=.04,repair_seconds=.02,legacy_ppo_seconds=0.)
    pool,split=setup(cfg)
    # Use fit data only for this short technical check; no dev performance claim.
    search=Search(pool,split['fit_ids'][0],cfg,20261021,seconds=1.)
    search.run();report=search.report()
    report['archive'][0]['routes'][0][0]=np.int64(report['archive'][0]['routes'][0][0])
    with pytest.raises(TypeError):json.dumps(report,allow_nan=False)
    path=tmp_path/'actual_report.json';save_development_result(path,report)
    decoded=json.loads(path.read_text(encoding='utf-8'))
    assert decoded.keys()==report.keys()
    assert decoded['hv']==report['hv']
    assert decoded['archive'][0]['routes'][0][0]==report['archive'][0]['routes'][0][0]
    assert decoded['budget_seconds']==1.
    assert all(k in decoded for k in ('records','repair_costs','per_direction','hv_time_curve'))
