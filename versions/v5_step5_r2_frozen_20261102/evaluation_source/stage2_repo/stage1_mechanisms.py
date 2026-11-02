"""Step5 CPU mechanisms. Experiment constants are supplied by the approved caller.

No torch, model loading, training, or background work at import time.
"""
import math
import numpy as np


def refinement_q(u, n):
    qmax = min(math.floor(.45*n), 100)
    u_calibrated = np.clip((float(u)-.06)/.32, 0., 1.)
    return int(np.clip(np.rint(2.*(qmax/2.)**u_calibrated), 2, n))


def pulse_q(u, n, mapping):
    if mapping != 'linear':
        raise ValueError('Pulse mapping needs an explicit definition')
    lo, hi = max(6, math.floor(.12*n)), min(math.floor(.50*n), 120)
    return int(np.clip(np.rint(lo+(hi-lo)*float(u)), lo, hi))


def select_nodes(scores, q, neighbors):
    """The explicit anchor is always included; nearest candidate set includes it."""
    scores = np.asarray(scores)
    n = len(scores)
    if not 1 <= q <= n or not np.isfinite(scores).all():
        raise ValueError('Invalid scores or q')
    if q > 8:
        return np.argsort(-scores, kind='stable')[:q]+1, None
    anchor = int(np.argmax(scores))+1
    m = min(n, max(2*q, 12))
    candidates = [anchor]
    candidates += [int(c) for c in neighbors[anchor] if c != 0 and c != anchor][:m]
    if len(candidates) != min(n,m+1) or len(set(candidates)) != len(candidates):
        raise ValueError('Nearest-neighbor list must cover all distinct customers')
    others = np.asarray(candidates[1:], dtype=int)
    selected = others[np.argsort(-scores[others-1], kind='stable')[:q-1]]
    return np.concatenate(([anchor], selected)), candidates


def refresh_required(session):
    dirty = bool(getattr(session, 'dirty_since_refresh', False) or session.env.last_accepted)
    repair = float(getattr(session, 'repair_seconds_since_refresh', 0.))
    gnn = float(getattr(session, 'last_gnn_refresh_seconds', 0.))
    return (session.cached_distribution is None or session.step_in_session == 0
            or (dirty and repair >= 5.*max(gnn, 1e-4))
            or (not dirty and session.cache_age >= 8))


def physical_features(instance, routes):
    """Exactly x[:, [18,17,19,12,13]], customers only, without graph construction.

    Cumulative waiting uses the max-plus recurrence for time-window service.
    Features retain the frozen model's original meanings and float32 storage.
    """
    from itertools import chain
    n = instance.n_customers
    horizon = max(float(instance.depot_time_window[1]), 1.)
    distance = max(float(instance.distance_matrix.max()), 1e-9)
    lengths = np.asarray([len(r) for r in routes if len(r)],dtype=int)
    customers = np.fromiter(chain.from_iterable(routes),dtype=int,count=n)
    ends = np.cumsum(lengths)-1
    starts = np.concatenate(([0],ends[:-1]+1))
    ids = customers-1
    prev=np.roll(customers,1);prev[starts]=0
    nxt=np.roll(customers,-1);nxt[ends]=0
    service=instance.customer_service_times[ids]
    previous_service=np.roll(service,1);previous_service[starts]=0.
    global_base=np.cumsum(instance.duration_matrix[prev,customers]+previous_service)
    base=global_base-np.repeat(np.concatenate(([0.],global_base[ends[:-1]])),lengths)
    early,late=instance.customer_time_windows[ids].T
    required_wait=np.maximum(early-base,0.)
    group=np.repeat(np.arange(len(lengths)),lengths)
    # A strictly separated monotone offset implements segmented prefix maxima.
    offset=group*(float(required_wait.max())+1.)
    accumulated_wait=np.maximum.accumulate(required_wait+offset)-offset
    previous_wait=np.roll(accumulated_wait,1);previous_wait[starts]=0.
    arrival=base+previous_wait
    completion=base[ends]+accumulated_wait[ends]+service[ends]+instance.duration_matrix[customers[ends],0]
    phys=np.empty((n,5),dtype=np.float32)
    phys[ids,0]=np.repeat(lengths/n,lengths)
    phys[ids,1]=(instance.distance_matrix[prev,customers]+instance.distance_matrix[customers,nxt]-instance.distance_matrix[prev,nxt])/distance
    phys[ids,2]=np.repeat((completion.max()-completion)/horizon,lengths)
    phys[ids,3]=np.maximum(early-arrival,0.)/horizon
    phys[ids,4]=np.maximum(arrival-late,0.)/horizon
    if not np.isfinite(phys).all():raise ValueError('Nonfinite physical features')
    return phys


def normalize_physical(physical):
    physical=np.asarray(physical,dtype=np.float64)
    return ((physical-physical.mean(0))/np.maximum(physical.std(0),1e-5)).astype(np.float32)


def residual_logits(values, norm_physical, physical_weights, gate):
    return ((values+norm_physical @ physical_weights)/(1.+.75*gate)).astype(np.float32)


def stage2_refinement_q(u, n):
    """Unlocked new head: raw u, no legacy calibration; Stage2 round upper bound."""
    qmax = max(2,min(int(round(.45*n)),100))
    return int(np.clip(np.rint(2.*(qmax/2.)**float(u)),2,n))


def dynamic_coordinates(objectives, physical_scales, physical_reference=None):
    points = np.asarray(objectives, dtype=float)
    if points.ndim!=2 or points.shape[1]!=5 or not len(points) or not np.isfinite(points).all():
        raise ValueError('Require nonempty finite five-objective archive')
    pts=points
    if physical_reference is not None:
        in_box=points[np.all(points<=np.asarray(physical_reference),axis=1)]
        if len(in_box)>=2:pts=in_box
    ideal = pts.min(0)
    span = np.maximum(pts.max(0)-ideal, .05*np.asarray(physical_scales))
    if not np.isfinite(span).all() or np.any(span <= 0):
        raise ValueError('Invalid dynamic span')
    return ideal, span


def dynamic_score(values, weights, ideal, span, physical_reference, physical_scales, penalty=1.,tie_eps=1e-3):
    """One float64 formula for repair insertion, acceptance, and decomposition."""
    z=(np.asarray(values,dtype=np.float64)-np.asarray(ideal,dtype=np.float64))/np.asarray(span,dtype=np.float64)
    weighted=np.asarray(weights,dtype=np.float64)*z
    excess=np.maximum((np.asarray(values,dtype=np.float64)-np.asarray(physical_reference,dtype=np.float64))/np.asarray(physical_scales,dtype=np.float64),0.)
    return weighted.max(-1)+.05*weighted.sum(-1)+float(tie_eps)*z.sum(-1)+float(penalty)*excess.sum(-1)
