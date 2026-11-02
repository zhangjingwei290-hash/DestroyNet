"""Recompute wall allowances from search horizons; never launches training.

Clone windows are a scheduling envelope, not a measured throughput estimate.
Epoch means one completed early/middle/late search cycle on each FIT IID.
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def training_budget(table, local_seconds=10., replicas=4, meta_epochs=3,
                    train_replicas_per_cell=4, adaptation_cell_passes=1,
                    reserve_fraction=.20):
    if len(table)!=45 or any(not math.isfinite(v) or v<=0 for v in table.values()):
        raise ValueError('Require the 45 positive per-cell search horizons')
    if local_seconds<=0 or replicas!=4 or meta_epochs<1 or train_replicas_per_cell!=4:
        raise ValueError('Invalid training coverage parameters')
    # Three source phases, five tasks in meta or one task in adaptation.
    # The primary trajectory is already charged to T. Charge only 3 clone copies.
    cap_sum=sum(min(local_seconds,t/16.) for t in table.values())
    primary_epoch=train_replicas_per_cell*sum(table.values())
    meta_clone_epoch=train_replicas_per_cell*3*5*(replicas-1)*cap_sum
    adapt_primary=16*adaptation_cell_passes*sum(table.values())
    adapt_clone=16*adaptation_cell_passes*3*(replicas-1)*cap_sum
    meta_envelope=meta_epochs*(primary_epoch+meta_clone_epoch)
    adapt_envelope=adapt_primary+adapt_clone
    return dict(meta_target_epochs=meta_epochs, fit_instances=180,
        adaptation_target_distinct_cells_per_direction=45,
        adaptation_cell_passes=adaptation_cell_passes,
        reserve_fraction=reserve_fraction,
        primary_seconds_per_meta_epoch=primary_epoch,
        clone_window_envelope_per_meta_epoch=meta_clone_epoch,
        meta_seconds_per_epoch_envelope=primary_epoch+meta_clone_epoch,
        adaptation_primary_seconds=adapt_primary,
        adaptation_clone_window_envelope=adapt_clone,
        meta_seconds=math.ceil(meta_envelope*(1+reserve_fraction)),
        adaptation_seconds=math.ceil(adapt_envelope*(1+reserve_fraction)),
        status='planning_allowance_not_measured_or_coverage_guarantee',
        note='3 epochs and 20% reserve are explicit planning choices, not literature convergence thresholds. '
             'No runtime budget extension; report actual complete cycles and per-direction cell coverage. '
             'Fit scheduling/data adapter still pending; no training launched.')

if __name__=='__main__':
    table=json.loads((ROOT/'paper_search_budgets.json').read_text())['seconds_by_instance']
    print(json.dumps(training_budget(table),ensure_ascii=False,indent=2))
