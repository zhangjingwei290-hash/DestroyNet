"""Finite synthetic archive fixtures exercise aggregation, not the routing solver."""
import json,tempfile
from pathlib import Path
import moocore,numpy as np
import summarize

def main():
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);out=root/'results';out.mkdir();summarize.ROOT=root
        plan=dict(ids=['fixture'],search_seeds=list(range(20)),methods=['heuristic','v5_meta_frozen'],scope='fixture',igd_plus='fixture',limitations=[])
        (root/'plan.json').write_text(json.dumps(plan))
        for seed in range(20):
            for method,x in [('heuristic',.3),('v5_meta_frozen',.2)]:
                r=dict(iid='fixture',seed=seed,method=method,physical_offset=[0]*5,physical_scale=[1]*5,normalized_reference=[1.1]*5,budget_seconds=1,initial_population=dict(routes=[[[1]]],objectives=[[.5]*5]),archive=[dict(objectives=[x]*5)],hv=float(moocore.hypervolume(np.array([[x]*5]),ref=[1.1]*5)),objective_minima=[x]*5,records=[],inference_seconds=0,repair_costs={},search_seconds=1)
                (out/f'fixture_seed{seed}_{method}.json').write_text(json.dumps(r))
        summarize.main();s=json.loads((out/'summary.json').read_text());v=s['instances'][0]
        assert s['pairs']==20 and v['hv_wins']==20 and v['hv_losses']==0
        np.testing.assert_allclose(v['hv_delta_mean'],.9**5-.8**5)
        np.testing.assert_allclose(v['v5_igd_plus_mean'],0.)
        assert v['heuristic_igd_plus_mean']>0
    print('Summary fixture passed: paired deltas, empirical IGD+ reference, SD, bootstrap, CSV')
if __name__=='__main__':main()
