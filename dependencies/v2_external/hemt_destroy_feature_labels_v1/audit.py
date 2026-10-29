"""Independent CPU proof checks for saved generic-ALNS counterfactual labels.

No training, solver launch, GPU or file mutation. Explicit evidence is required.
"""
from __future__ import annotations
import hashlib
import json
import math
import numpy as np


class AuditError(ValueError):
    pass


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _require(condition, message):
    if not condition:
        raise AuditError(message)


def _value(instance, key):
    return instance[key] if isinstance(instance, dict) else getattr(instance, key)


def checked_objectives(routes, instance, claimed=None, tolerance=1e-7):
    """Independent arithmetic, retaining directional matrices and service timing."""
    n = int(_value(instance, 'n_customers'))
    _require(all(isinstance(c, (int, np.integer)) and not isinstance(c, bool) for r in routes for c in r), 'noninteger customer ID')
    visited = [c for r in routes for c in r]
    _require(sorted(visited) == list(range(1, n+1)), 'complete coverage must be exactly once')
    distance = np.asarray(_value(instance, 'distance_matrix'), dtype=np.float64)
    duration = np.asarray(_value(instance, 'duration_matrix'), dtype=np.float64)
    demands = np.asarray(_value(instance, 'customer_demands'), dtype=np.float64)
    windows = np.asarray(_value(instance, 'customer_time_windows'), dtype=np.float64)
    services = np.asarray(_value(instance, 'customer_service_times'), dtype=np.float64)
    delay = float(_value(instance, 'max_delay'))
    close = float(_value(instance, 'depot_time_window')[1])
    capacity = float(_value(instance, 'vehicle_capacity'))
    _require(distance.shape == duration.shape == (n+1, n+1), 'matrix dimensions')
    _require(all(np.isfinite(x).all() for x in (distance, duration, demands, windows, services)), 'nonfinite physical inputs')
    vehicles, total_distance, makespan, waiting, tardiness = 0, 0., 0., 0., 0.
    for route in routes:
        if not route:
            continue
        vehicles += 1
        _require(sum(demands[c-1] for c in route) <= capacity + 1e-6, 'capacity violation')
        clock, previous = 0., 0
        for c in route:
            total_distance += distance[previous, c]
            arrival = clock + duration[previous, c]
            early, late = windows[c-1]
            waiting += max(early-arrival, 0.)
            customer_late = max(arrival-late, 0.)
            tardiness += customer_late
            _require(customer_late <= delay + 1e-6, 'customer hard delay violation')
            clock = max(arrival, early) + services[c-1]
            previous = c
        total_distance += distance[previous, 0]
        returned = clock + duration[previous, 0]
        _require(returned <= close + 1e-6, 'depot return violation')
        makespan = max(makespan, returned)
    values = np.asarray([vehicles, total_distance, makespan, waiting, tardiness], dtype=np.float64)
    if claimed is not None:
        supplied = np.asarray(claimed, dtype=np.float64)
        _require(supplied.shape == (5,) and np.isfinite(supplied).all(), 'claimed objectives must be finite 5-vector')
        _require(np.allclose(values, supplied, rtol=0., atol=tolerance), 'claimed objectives mismatch independently recomputed routes')
    return values.tolist()


def exact_hv(solutions, box):
    import moocore
    reference = np.asarray(box['reference'], dtype=np.float64)
    scale = np.asarray(box.get('scale', [1.]*5), dtype=np.float64)
    offset = np.asarray(box.get('offset', [0.]*5), dtype=np.float64)
    _require(reference.shape == scale.shape == offset.shape == (5,), 'HV requires 5-dimensional box')
    _require(np.isfinite([reference, scale, offset]).all() and (scale > 0).all(), 'invalid fixed HV box')
    if not solutions:
        return 0.
    values = np.asarray([s['objectives'] for s in solutions], dtype=np.float64)
    _require(values.ndim == 2 and values.shape[1] == 5 and np.isfinite(values).all(), 'HV finite complete objectives')
    return float(moocore.hypervolume((values-offset)/scale, ref=reference))


def checked_archive(solutions, instance):
    _require(isinstance(solutions, list), 'archive must be an explicit list')
    for solution in solutions:
        checked_objectives(solution['routes'], instance, solution['objectives'])
    return solutions


def nondominated_objective_keys(solutions):
    values = sorted(set(tuple(float(v) for v in s['objectives']) for s in solutions))
    return {v for v in values if not any(all(a <= b for a, b in zip(other, v)) and any(a < b for a, b in zip(other, v)) for other in values)}


def check_deletion(source_routes, partial_routes, removed, n):
    _require(all(isinstance(c, (int, np.integer)) and not isinstance(c, bool) for c in removed), 'removed IDs must be integers')
    cap = min(15, math.floor(.15*n))
    _require(0 < len(removed) <= cap and len(set(removed)) == len(removed), 'q1/nonempty/unique/deletion cap violation')
    _require(sorted(c for r in source_routes for c in r) == list(range(1, n+1)), 'source coverage mismatch')
    _require(sorted([c for r in partial_routes for c in r] + list(removed)) == list(range(1, n+1)), 'partial+removed must partition customers')
    retained = [[c for c in r if c not in set(removed)] for r in source_routes]
    _require([r for r in retained if r] == [r for r in partial_routes if r], 'deletion reordered retained source routes')
    return cap


def check_feature_inputs(context):
    from hemt_destroy_features_v1.features import CONTEXT_FIELDS
    _require(isinstance(context, dict), 'feature context mapping required')
    _require(not (set(context)-set(CONTEXT_FIELDS)), 'unknown/future label field in feature context')
    _require(all(v is None or (isinstance(v, (int, float)) and math.isfinite(v)) for v in context.values()), 'feature context values must be finite predecision scalars')
    return True


def audit_branch(branch, state, instance, cfg):
    """Strict evidence schema; collector adapter will follow the actual saved schema."""
    required = ('instance_id', 'split', 'source', 'pre_archive', 'box', 'state_hash', 'pre_archive_hash', 'reference_hash', 'feature_context')
    _require(all(k in state for k in required), 'missing explicit predecision state evidence')
    iid = state['instance_id']
    _require(state['split'] == 'train' and iid.startswith('hemtv2_train_'), 'TRAIN-only IID required')
    _require(iid not in set(cfg.get('forbidden_instance_ids', [])), 'official/forbidden IID')
    _require(iid in set(cfg['train_instance_ids']), 'IID outside frozen TRAIN manifest')
    _require(_value(instance, 'instance_id') == iid, 'instance identity mismatch')
    _require(state['stage_scope'] == 'generic_ALNS', 'new captures must be generic ALNS, not native HEMT stage')
    check_feature_inputs(state['feature_context'])
    source = state['source']
    checked_objectives(source['routes'], instance, source['objectives'])
    checked_archive(state['pre_archive'], instance)
    _require(bool(state['pre_archive']), 'Apre must contain at least the initial solution')
    _require(digest(state['pre_archive']) == state['pre_archive_hash'], 'Apre mutation/hash mismatch')
    _require(digest(state['box']) == state['reference_hash'], 'reference mutation/hash mismatch')
    _require(state['box'] == cfg['boxes_by_instance'][iid], 'reference differs from immutable instance-only physical bound')
    payload = dict(instance_id=iid, source=source, pre_archive_hash=state['pre_archive_hash'], reference_hash=state['reference_hash'], feature_context=state['feature_context'])
    _require(digest(payload) == state['state_hash'], 'state hash mismatch')
    for key in ('state_hash', 'pre_archive_hash', 'reference_hash'):
        _require(branch[key] == state[key], 'branch did not restore shared predecision state: ' + key)
    _require(branch['repair_id'] in cfg['repair_ids'], 'unknown repair capability')
    _require(isinstance(branch['repetition'], int) and isinstance(branch['seed'], int), 'repetition and seed required')
    deletion = branch['destroy']
    _require(branch['location_pattern'] in ('random', 'related'), 'unknown deletion location pattern')
    cap = check_deletion(source['routes'], deletion['partial_routes'], deletion['removed_customers'], int(_value(instance, 'n_customers')))
    _require(branch['q'] == len(deletion['removed_customers']), 'claimed q mismatch')
    _require(branch['destroy_hash'] == digest(deletion), 'same-D proof hash mismatch')
    _require(branch['timing']['includes_first_action'] is True, 'horizon omits first action cost')
    h = float(branch['horizon_seconds'])
    times = branch['timing']
    _require(h > 0 and all(math.isfinite(float(times[k])) and float(times[k]) >= 0 for k in ('elapsed_seconds', 'first_action_seconds', 'first_commit_seconds')), 'invalid horizon/timing')
    _require(times['elapsed_seconds'] + 1e-9 >= times['first_action_seconds'], 'elapsed excludes first action')
    _require(times['first_commit_seconds'] + 1e-9 >= times['first_action_seconds'], 'first commit precedes first action finish')
    accepted = []
    for outcome in branch['outcomes']:
        _require(math.isfinite(outcome['commit_elapsed_seconds']) and outcome['commit_elapsed_seconds'] >= 0, 'invalid outcome wall time')
        _require(outcome['commit_elapsed_seconds'] <= times['elapsed_seconds'] + 1e-9, 'outcome occurs after branch finished')
        if outcome.get('routes') is not None:
            _require(outcome['objectives'] is not None, 'returned routes missing full objectives')
            if outcome['feasible']:
                checked_objectives(outcome['routes'], instance, outcome['objectives'])
        if outcome['committed']:
            _require(outcome['status'] == 'success' and outcome['feasible'] is True, 'failed repair manufactured committed success')
            _require(outcome['commit_elapsed_seconds'] <= h + 1e-9, 'late repair admitted after actual deadline')
            checked_objectives(outcome['routes'], instance, outcome['objectives'])
            accepted.append(outcome)
    checked_archive(branch['post_archive'], instance)
    _require(branch['post_archive_hash'] == digest(branch['post_archive']), 'Apost hash mismatch')
    allowed_routes = {digest(dict(routes=s['routes'], objectives=s['objectives'])) for s in state['pre_archive']+accepted}
    _require(all(digest(dict(routes=s['routes'], objectives=s['objectives'])) in allowed_routes for s in branch['post_archive']), 'Apost route not present in actual prearchive/committed outputs')
    _require(nondominated_objective_keys(branch['post_archive']) == nondominated_objective_keys(state['pre_archive']+accepted), 'Apost includes future/uncommitted outcome or omitted declared archive update')
    _require(branch['attempted_count'] == len(branch['outcomes']) and branch['committed_count'] == len(accepted), 'outcome sampling count mismatch')
    pre_hv, post_hv = exact_hv(state['pre_archive'], state['box']), exact_hv(branch['post_archive'], state['box'])
    for claimed, recomputed in ((branch['pre_hv'], pre_hv), (branch['post_hv'], post_hv), (branch['delta_hv'], post_hv-pre_hv)):
        _require(math.isclose(claimed, recomputed, rel_tol=1e-9, abs_tol=1e-10), 'exact 5D HV/delta mismatch')
    if not branch['first_repair_success']:
        _require(branch['label_success'] is False, 'first repair failure cannot be a successful control label')
    return dict(passed=True, instance_id=iid, pre_hv=pre_hv, post_hv=post_hv, delta_hv=post_hv-pre_hv, committed_count=len(accepted), deletion_cap=cap, scope='independent saved generic ALNS label proof; no native HEMT quality claim')


def audit_comparison(branches):
    """All four r share a D/prestate within each explicit repetition/seed block."""
    groups = {}
    for branch in branches:
        key = (branch['state_hash'], branch['q'], branch['location_pattern'], branch['repetition'], branch['seed'], branch['horizon_seconds'])
        groups.setdefault(key, []).append(branch)
    for rows in groups.values():
        _require(len(rows) == 4 and len({r['repair_id'] for r in rows}) == 4, 'comparison must contain four distinct repairs')
        for key in ('destroy_hash', 'pre_archive_hash', 'reference_hash'):
            _require(len({r[key] for r in rows}) == 1, 'repair comparison changes shared D/state/reference: ' + key)
    return dict(passed=True, paired_blocks=len(groups), scope='same-D and prestate pairing; timing measured separately')
