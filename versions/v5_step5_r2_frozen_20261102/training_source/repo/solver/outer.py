"""Public pymoo MOEA/D with feasible route objects and timed DPR local search.

MOEAD selection, neighbor replacement and generation lifecycle are pymoo's.
Only expose the active direction (otherwise unavailable through ask/tell), use
fixed physical decomposition, and replace continuous operators with route ops.
"""
import copy
import threading
from dataclasses import dataclass
import time
import numpy as np
import torch
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.decomposition import Decomposition
from pymoo.core.termination import NoTermination
from pymoo.operators.mutation.nom import NoMutation
from .config import directions, search_budget
from stage1_mechanisms import dynamic_coordinates,dynamic_score,refresh_required,refinement_q,pulse_q,select_nodes,normalize_physical,residual_logits
from .environment import Environment, Adaptive, LocalSession


@dataclass
class Genome:
    routes: list
    fallback: list


class FixedTchebycheff(Decomposition):
    def __init__(self, offset, scale):
        super().__init__()
        self.offset, self.scale = np.asarray(offset), np.asarray(scale)

    def _do(self, F, weights, **kwargs):
        if hasattr(self,'dynamic_ideal'):
            return dynamic_score(F,weights,self.dynamic_ideal,self.dynamic_span,self.reference,self.scale,self.penalty)
        z = weights*((F-self.offset)/self.scale)
        return z.max(-1)+.05*z.sum(-1)


class InitialRoutes(Sampling):
    def __init__(self, env):
        super().__init__(); self.env = env

    def _do(self, problem, n_samples, **kwargs):
        result = np.empty((n_samples, 1), dtype=object)
        # Common seeded population: at most four feasible swaps of the same
        # frozen source per member. Invalid mutations retain a checked source.
        for k in range(n_samples):
            routes = copy.deepcopy(self.env.routes)
            for _ in range(4 if k else 0):
                trial = swapped(routes)
                if self.env.api.feasible(self.env.solution(trial), self.env.instance):
                    routes = trial
            result[k, 0] = Genome(routes, copy.deepcopy(routes))
        return result


def swapped(routes):
    result = copy.deepcopy(routes)
    positions = [(r, c) for r, route in enumerate(result) for c in range(len(route))]
    if len(positions) >= 2:
        a, b = np.random.choice(len(positions), 2, replace=False)
        i, j = positions[a]; k, l = positions[b]
        result[i][j], result[k][l] = result[k][l], result[i][j]
    return result


class RouteExchange(Crossover):
    def __init__(self):
        super().__init__(2, 1, prob=1.)

    def _do(self, problem, X, **kwargs):
        result = np.empty((1, X.shape[1], 1), dtype=object)
        for m in range(X.shape[1]):
            recipient, donor = X[0, m, 0], X[1, m, 0]
            selected = list(donor.routes[np.random.randint(len(donor.routes))])
            taken = set(selected)
            child = [[c for c in route if c not in taken] for route in recipient.routes]
            child = [r for r in child if r]+[selected]
            if np.random.random() < .2:
                child = swapped(child)
            result[0, m, 0] = Genome(child, copy.deepcopy(recipient.routes))
        return result


class RoutedMOEAD(MOEAD):
    def _next(self):
        # Public pymoo loop, with explicit permutation/cursor for exact checkpoint
        # restoration. No change to selection, mating or replacement.
        if getattr(self, "resume_pending", None) is not None:
            pending = self.resume_pending; self.resume_pending = None
            off = yield pending
            self._accept(off)
        if getattr(self, "order", None) is None:
            self.order = np.random.permutation(len(self.pop)).tolist(); self.cursor = 0
        while self.cursor < len(self.order):
            k = self.order[self.cursor]; self.active_direction = int(k)
            P = self.selection.do(self.problem, self.pop, 1, self.mating.crossover.n_parents,
                                  neighbors=[self.neighbors[k]])
            off = np.random.choice(self.mating.do(self.problem, self.pop, 1, parents=P, n_max_iterations=1))
            off = yield off
            self._accept(off)
        self.order = None

    def _replace(self, k, off):
        neighbors = self.neighbors[k]
        replaced = 0
        max_nondom_replace = 2
        for j in neighbors:
            old_F = self.pop[j].F
            if np.allclose(off.F, old_F, atol=1e-7, rtol=0.):
                continue
            strictly_dominates = bool(np.all(off.F <= old_F) and np.any(off.F < old_F))
            if strictly_dominates:
                self.pop[j] = off
                continue
            already_in_neighbors = any(
                m != j and np.allclose(off.F, self.pop[m].F, atol=1e-7, rtol=0.)
                for m in neighbors
            )
            if already_in_neighbors and j != k:
                self.clone_rejections = getattr(self, 'clone_rejections', 0) + 1
                continue
            parent = float(self.decomposition.do(old_F[None, :], weights=self.ref_dirs[j:j+1], ideal_point=self.ideal)[0])
            child = float(self.decomposition.do(off.F[None, :], weights=self.ref_dirs[j:j+1], ideal_point=self.ideal)[0])
            if child < parent:
                if j == k or replaced < max_nondom_replace:
                    self.pop[j] = off
                    if j != k:
                        replaced += 1

    def _accept(self, off):
        self.ideal = np.minimum(self.ideal, off.F)
        self._replace(self.active_direction, off)
        self.cursor += 1


def union(archive, points):
    result = list(archive)
    for p in points:
        v = np.asarray(p["objectives"])
        if not any(np.all(np.asarray(other["objectives"]) <= v) for other in result):
            result = [other for other in result if not np.all(v <= np.asarray(other["objectives"]))]
            result.append(copy.deepcopy(p))
    return result


_RNG_LOCK=threading.RLock()

class Search:
    def __init__(self,*args,**kwargs):
        with _RNG_LOCK:self._initialize(*args,**kwargs)

    def _initialize(self, pool, iid, cfg, seed, policies=None, seconds=None, resource_deadline=None, ref_dirs=None):
        global_rng = np.random.get_state()
        self.started = time.perf_counter()
        self.pool, self.iid, self.cfg, self.seed, self.policies = pool, iid, cfg, seed, policies
        inst, row, _, _ = pool.load(iid)
        self.budget = search_budget(inst.n_customers, cfg, iid=iid, row=row) if seconds is None else float(seconds)
        self.deadline = self.started+self.budget
        self.external_deadline = float("inf") if resource_deadline is None else resource_deadline
        self.ref_dirs = directions() if ref_dirs is None else np.asarray(ref_dirs, dtype=float)
        if self.ref_dirs.shape != (16, 5) or not np.allclose(self.ref_dirs.sum(1), 1):
            raise ValueError("MOEAD requires sixteen valid weight vectors")
        self.base = Environment(pool, iid, cfg, directions()[-1])
        self.problem = Problem(n_var=1, n_obj=5, vtype=object)
        self.algorithm = RoutedMOEAD(ref_dirs=self.ref_dirs, n_neighbors=5, prob_neighbor_mating=.9,
            decomposition=FixedTchebycheff(self.base.offset, self.base.scales), sampling=InitialRoutes(self.base),
            crossover=RouteExchange(), mutation=NoMutation())
        self.algorithm.setup(self.problem, termination=NoTermination(), seed=int(seed), verbose=False)
        self.adaptive = [Adaptive(len(self.base.q_values), heuristic=policies is None) for _ in range(16)]
        self.archive, self.records, self.events = [], [], []
        self.point_events = []
        self.snapshots = []
        self.dynamic_scalarization_snapshots=[]
        # Upstream initializes and scores all 16 members before first generation.
        infills = self.algorithm.ask()
        for off in infills:
            values = self.base.evaluate(off.X[0].routes)
            off.set("F", values)
            self.archive = union(self.archive, [dict(routes=off.X[0].routes, objectives=values.tolist())])
            self.point_events.append(dict(seconds=time.perf_counter()-self.started, objectives=values.tolist()))
        self.algorithm.tell(infills=infills)
        self.initial_seconds = time.perf_counter()-self.started
        self.numpy_state = np.random.get_state(); np.random.set_state(global_rng)
        if self.initial_seconds > self.budget:
            raise TimeoutError("Common population initialization exceeded solver budget")

    def set_preferences(self,refs,policies):
        """Rebind current EMNH preferences without replacing the population."""
        from scipy.spatial.distance import cdist
        refs=np.asarray(refs,dtype=float)
        if refs.shape!=(16,5) or not np.allclose(refs.sum(1),1):raise ValueError('Invalid direction vectors')
        changed=np.any(refs!=self.ref_dirs,axis=1)
        self.ref_dirs=refs.copy();self.algorithm.ref_dirs=self.ref_dirs
        self.algorithm.neighbors=np.argsort(cdist(refs,refs),axis=1,kind='quicksort')[:,:self.algorithm.n_neighbors]
        self.policies=policies
        for k in np.flatnonzero(changed):
            # Preserve actual repair counts/EWMA; the scalar best must be scored
            # under the new preference, rather than comparing incompatible scores.
            vals=(self.algorithm.pop[k].F-self.base.offset)/self.base.scales
            self.adaptive[k].best=float((refs[k]*vals).max()+.05*(refs[k]*vals).sum())

    @property
    def resource_deadline(self):
        return min(self.deadline, self.external_deadline)

    def remaining(self):
        return max(0., self.deadline-time.perf_counter())

    def _random_call(self,callback):
        # Only pymoo ask/tell touch the process-global NumPy RNG. Do not hold
        # this lock across GPU or repair waits in other independent searches.
        with _RNG_LOCK:
            global_rng=np.random.get_state();np.random.set_state(self.numpy_state)
            try:
                value=callback();self.numpy_state=np.random.get_state();return value
            finally:np.random.set_state(global_rng)

    def step(self, rollout=None):
        return self._step(rollout)

    def _step(self, rollout=None):
        self.last_episodes = []
        before = time.perf_counter()
        off = self._random_call(self.algorithm.ask)
        if off is None or self.remaining() <= 0 or time.perf_counter() >= self.resource_deadline:
            return False
        k = self.algorithm.active_direction
        genome = off.X[0]
        physical = self.base.api.feasible(self.base.solution(genome.routes), self.base.instance)
        if not physical:
            self.events.append(dict(event="variation_infeasible_parent_retained", direction=k))
            genome = Genome(copy.deepcopy(genome.fallback), copy.deepcopy(genome.fallback))
        env = Environment(self.pool, self.iid, self.cfg, self.ref_dirs[k], genome.routes)
        evaluated_at = time.perf_counter()
        if evaluated_at >= self.resource_deadline:
            return False
        self.archive = union(self.archive, env.archive)
        self.point_events.append(dict(seconds=evaluated_at-self.started, objectives=env.objectives.tolist()))
        values=[p['objectives'] for p in self.archive]
        physical_reference=self.base.offset+self.base.scales*np.asarray(self.base.box['reference'])
        ideal,span=dynamic_coordinates(values,self.base.scales,physical_reference)
        normalization_points=np.asarray(values,dtype=float)
        in_box=normalization_points[np.all(normalization_points<=physical_reference,axis=1)]
        selected_points=in_box if len(in_box)>=2 else normalization_points
        env.dynamic_ideal=ideal.copy();env.dynamic_span=span.copy()
        decomp=self.algorithm.decomposition
        decomp.dynamic_ideal=ideal.copy();decomp.dynamic_span=span.copy()
        decomp.reference=self.base.offset+self.base.scales*np.asarray(self.base.box['reference'])
        decomp.penalty=self.cfg.step5_reference_penalty
        self.adaptive[k].best=env.score(env.objectives)
        self.dynamic_scalarization_snapshots.append(dict(direction=k,ideal=ideal.tolist(),span=span.tolist(),weight=self.ref_dirs[k].tolist(),penalty=decomp.penalty,normalization_source='archive_only_prefer_at_least_two_in_box',archive_points=len(values),in_box_points=len(in_box),selected_points=len(selected_points),selected_min=selected_points.min(0).tolist(),selected_max=selected_points.max(0).tolist(),span_floor_fraction=.15,seconds=time.perf_counter()-self.started))
        self.events.append(dict(event='dynamic_scale_snapshot',snapshot=len(self.dynamic_scalarization_snapshots)-1,direction=k,ideal=ideal.tolist(),span=span.tolist()))
        session = LocalSession(env, self.adaptive[k], int(self.seed+len(self.records)*1009+k),
            self.budget, self.remaining(), self.cfg.local_seconds, self.external_deadline)
        session.solver_task_deadline = self.deadline
        # A collector can fork four SAME-START rollouts. Only primary lane
        # drives this actual population; auxiliary sampling does not age its clock.
        old_started = self.started
        paused = 0.
        if rollout is not None:
            session, paused, self.last_episodes = rollout(k, session)
            env = session.env
        else:
            run_session(session, self.policies[k] if self.policies is not None else None)
        self.started += paused; self.deadline += paused
        for result in session.records:
            if result["candidate_objectives"] is not None:
                self.point_events.append(dict(seconds=result["candidate_finished_perf"]-old_started-result.get("inactive_pause_before_candidate",0.),
                                             objectives=result["candidate_objectives"]))
            self.records.append(dict(direction=k,scalar_snapshot=len(self.dynamic_scalarization_snapshots)-1, outer_elapsed=time.perf_counter()-self.started, **result))
        # A solution accepted before the cutoff is retained even if the final
        # attempted action overran. No late candidate enters archive/population.
        self.archive = union(self.archive, env.archive)
        off.set("X", np.asarray([Genome(copy.deepcopy(env.routes), copy.deepcopy(env.routes))], dtype=object))
        off.set("F", env.objectives.copy())
        self._random_call(lambda:self.algorithm.tell(infills=off))
        self.snapshots.append(dict(direction=k, elapsed=time.perf_counter()-self.started,
            routes=copy.deepcopy(env.routes), adaptive=copy.deepcopy(self.adaptive[k]),
            global_remaining=self.remaining(), global_budget=self.budget,
            outer_step_seconds=time.perf_counter()-before))
        return True

    def run(self, until_fraction=1., progress=None):
        stop = min(self.started+self.budget*until_fraction, self.resource_deadline)
        while time.perf_counter() < stop and self.remaining() > 0:
            if not self.step():
                break
            if progress:
                progress(self)
        self.finished_seconds = time.perf_counter()-self.started
        return self

    def start_state(self):
        # Training snapshots are unfiltered actual states. No winner filtering.
        if self.snapshots:
            return copy.deepcopy(self.snapshots[-1])
        return dict(routes=copy.deepcopy(self.base.routes), global_budget=self.budget,
            global_remaining=self.remaining(), adaptive=Adaptive(len(self.base.q_values)), direction=15)

    def report(self):
        import moocore
        audit_start = time.perf_counter()
        for p in self.archive:
            np.testing.assert_allclose(self.base.evaluate(p["routes"]), p["objectives"], atol=1e-7, rtol=0)
        values = (np.asarray([p["objectives"] for p in self.archive])-self.base.offset)/self.base.scales
        hv = float(moocore.hypervolume(values, ref=np.asarray(self.base.box["reference"])))
        curve = []
        for fraction in (.1, .25, .5, .75, 1.):
            points = [p["objectives"] for p in self.point_events if p["seconds"] <= self.budget*fraction]
            partial = 0. if not points else float(moocore.hypervolume(
                (np.asarray(points)-self.base.offset)/self.base.scales, ref=np.asarray(self.base.box["reference"])))
            curve.append(dict(seconds=self.budget*fraction, hv=partial))
        np.testing.assert_allclose(curve[-1]["hv"], hv, rtol=1e-10, atol=1e-12)
        per_direction = []
        for k, p in enumerate(self.algorithm.pop):
            per_direction.append(dict(direction=k, weight=self.ref_dirs[k].tolist(), objectives=p.F.tolist(),
                scalar=float(self.algorithm.decomposition.do(p.F, self.ref_dirs[k]).item())))
        costs = {}
        for repair in ("greedy", "regret2", "regret3", "noisy_greedy"):
            records = [r for r in self.records if r["repair"] == repair]
            costs[repair] = dict(calls=len(records), seconds=sum(r["repair_seconds"] for r in records),
                timeouts=sum(r["timeout"] for r in records), accepted=sum(r["accepted"] for r in records),
                failed=sum(not r["feasible"] and not r["timeout"] for r in records))
        return dict(iid=self.iid, seed=self.seed, budget_seconds=self.budget, initial_seconds=self.initial_seconds,
            search_seconds=getattr(self, "finished_seconds", time.perf_counter()-self.started),
            hv=hv, normalized_reference=self.base.box["reference"],
            physical_reference=(self.base.offset+self.base.scales*np.asarray(self.base.box["reference"])).tolist(),
            physical_offset=self.base.offset.tolist(), physical_scale=self.base.scales.tolist(), archive=self.archive,
            records=self.records, variation_events=self.events,
            hv_time_curve=curve, per_direction=per_direction, repair_costs=costs,
            q_histogram={str(q):sum(r["q"] == q for r in self.records) for q in sorted(set(self.base.q_values)|{int(r["q"]) for r in self.records})},
            inference_seconds=sum(r["inference_seconds"] for r in self.records),
            point_events=self.point_events,
            offline_audit_seconds=time.perf_counter()-audit_start,
            scope="development_only; fixed_physical_reference; no_optimality_claim")


def cache_refresh_required(session,k=None):
    return refresh_required(session)

def pulse_required(stagnation,k=4):
    return stagnation>=6 and (stagnation-6)%k==0

def cached_sample(cache,n,width,rng,pulse,neighbors):
    stride=4+2*n+5
    start=width+(stride if pulse else 0)
    alpha,beta,mode,anchor_q=cache[start:start+4]
    logits=cache[start+4:start+4+n]
    u=np.float32(rng.beta(float(alpha),float(beta)))
    q=pulse_q(u,n,'linear') if pulse else refinement_q(u,n)
    exponentials=np.maximum(np.asarray(rng.exponential(size=n),dtype=np.float32),np.finfo(np.float32).tiny)
    scores=logits-np.log(exponentials)
    selected,candidates=select_nodes(scores,q,neighbors)
    from .q_statistics import nearest_q_slot
    return dict(u=float(u),q_ratio=q/n,q_index=nearest_q_slot(q,n),q=q,removed=selected.tolist(),
                cached_mode=float(mode),cached_anchor_q=int(anchor_q),anchor_candidates=candidates)

def run_cached_session(session,policy,task=0):
    n=session.env.instance.n_customers;width=policy.cfg.hidden_dim;stride=4+2*n+5
    session.repair_seconds_since_refresh=0.;session.last_gnn_refresh_seconds=0.;session.dirty_since_refresh=False
    while session.remaining()>0:
        service_start=time.perf_counter()
        session.dirty_since_refresh|=bool(session.env.last_accepted)
        refresh=cache_refresh_required(session)
        obs=session.prepare(include_observation=refresh)
        if session.remaining()<=0:
            session.pending=None;break
        # prepare already charges observation/ALNS time; charge only added policy work
        # to the local session clock, but report full service including observation.
        sampled=time.perf_counter();refresh_seconds=0.;physical_seconds=0.
        age_before=session.cache_age;repair_before=session.repair_seconds_since_refresh
        previous_gnn=session.last_gnn_refresh_seconds;dirty_before=session.dirty_since_refresh
        if refresh:
            packed=policy.dual_cache([obs],[task])[0]
            session.cached_distribution=packed.numpy().copy()
            session.cache_hidden=packed[:width].reshape(1,-1).clone()
            session.cache_age=0;refresh_seconds=time.perf_counter()-sampled
            session.last_gnn_refresh_seconds=refresh_seconds
            session.repair_seconds_since_refresh=0.;session.dirty_since_refresh=False
        elif session.env.last_accepted:
            physical_start=time.perf_counter();physical=normalize_physical(session.env.physical_features())
            for gate in (0,1):
                start=width+stride*gate;values=session.cached_distribution[start+4+n:start+4+2*n]
                weights=session.cached_distribution[start+4+2*n:start+stride]
                session.cached_distribution[start+4:start+4+n]=residual_logits(values,physical,weights,float(gate))
            physical_seconds=time.perf_counter()-physical_start
        pulse=pulse_required(session.env.stagnation)
        action=cached_sample(session.cached_distribution,n,width,session.cache_rng,pulse,session.env.neighbors)
        session.cache_age+=1
        service_seconds=time.perf_counter()-service_start
        charged_seconds=time.perf_counter()-sampled
        result=session.finish(action,session.cache_hidden,charged_seconds)
        result['inference_seconds']=charged_seconds
        result['full_policy_service_seconds']=service_seconds
        result['observation_and_prepare_seconds']=max(0.,service_seconds-charged_seconds)
        session.repair_seconds_since_refresh+=result['repair_seconds']
        result.update(cache_hit=not refresh,cache_refresh=refresh,cache_age_before=age_before,
            cache_age_after=session.cache_age,cache_step=session.step_in_session,
            use_pulse=pulse,mode='pulse' if pulse else 'refinement',
            gnn_refresh_seconds=refresh_seconds,physical_refresh_seconds=physical_seconds,
            physical_refresh=bool(not refresh and session.decision_last_accepted),
            repair_seconds_since_refresh_before=repair_before,last_gnn_refresh_seconds_before=previous_gnn,
            dirty_since_refresh_before=dirty_before,
            cached_sampling_seconds=max(0.,charged_seconds-refresh_seconds-physical_seconds),
            logits_anchor_q=action['cached_anchor_q'])
        session.step_in_session+=1
    session.finalize()
    return None

def run_session(session, policy, task=0, episode=None, resource_deadline=None):
    if policy is not None and episode is None:
        return run_cached_session(session,policy,task)
    while session.remaining() > 0:
        obs = session.prepare(include_observation=policy is not None)
        if session.remaining() <= 0:
            session.pending = None
            break
        sampled = time.perf_counter()
        if policy is None:
            action, hidden = {}, session.hidden
        else:
            actions, lp, value, hs = policy.act([obs])
            action, hidden = actions[0], hs[0]
        result = session.finish(action, hidden, time.perf_counter()-sampled)
        if episode is not None:
            episode.append(dict(observation=obs, action=action, logprob=lp[0], value=value[0],
                reward=result["reward"], terminal=result["terminated"], truncated=result["truncated"],
                ending_reason=result["ending_reason"], task=task, next_value=0., diagnostics=result))
    ending = session.finalize()
    if episode:
        episode[-1].update(terminal=ending["terminated"], truncated=ending["truncated"],
                           ending_reason=ending["ending_reason"], next_value=0.)
    if episode and ending["truncated"] and policy.critic:
        original_deadline = session.outer_deadline; session.outer_deadline = None
        try:
            obs = session.prepare(); session.pending = None
            episode[-1]["next_value"] = float(policy.value(obs))
        finally:
            session.outer_deadline = original_deadline
    return episode


class TaskPolicy:
    def __init__(self, policy, task=0):
        self.policy, self.task = policy, task
        self.critic = bool(policy.critics)

    def act(self, obs):
        return self.policy.act(obs, [self.task]*len(obs))

    def value(self, obs):
        with torch.no_grad():
            # Do not draw a second random action just to bootstrap a truncated tail.
            _, _, _, global_state = self.policy.encode_batch([obs])
            return self.policy.critics[self.task](global_state[0]).squeeze()


def _individual_state(off):
    if off is None:
        return None
    g = off.X[0]
    return dict(routes=copy.deepcopy(g.routes), fallback=copy.deepcopy(g.fallback),
                objectives=None if off.F is None or len(off.F) == 0 else off.F.tolist())


def _restore_individual(data):
    if data is None:
        return None
    from pymoo.core.individual import Individual
    off = Individual(X=np.asarray([Genome(data["routes"], data["fallback"])], dtype=object))
    if data["objectives"] is not None:
        off.set("F", np.asarray(data["objectives"]))
    return off


def search_state(search):
    a = search.algorithm
    return dict(iid=search.iid, seed=search.seed, budget=search.budget,
        elapsed=time.perf_counter()-search.started, ref_dirs=search.ref_dirs.tolist(),
        population=[_individual_state(p) for p in a.pop], pending=_individual_state(a.state),
        order=getattr(a, "order", None), cursor=getattr(a, "cursor", 0),
        active_direction=getattr(a, "active_direction", 0), ideal=a.ideal.tolist(), n_iter=a.n_iter,
        adaptive=copy.deepcopy(search.adaptive), archive=copy.deepcopy(search.archive),
        records=copy.deepcopy(search.records), events=copy.deepcopy(search.events),
        points=copy.deepcopy(search.point_events), initial_seconds=search.initial_seconds,
        numpy_state=search.numpy_state, snapshots=copy.deepcopy(search.snapshots))


def _restore_search_unlocked(data, pool, cfg, policies, deadline):
    from pymoo.core.population import Population
    rng = np.random.get_state()
    s = Search(pool, data["iid"], cfg, data["seed"], policies, seconds=data["budget"],
               resource_deadline=deadline, ref_dirs=data["ref_dirs"])
    np.random.set_state(rng)
    s.started = time.perf_counter()-data["elapsed"]; s.deadline = s.started+s.budget
    a = s.algorithm
    a.pop = Population([_restore_individual(p) for p in data["population"]])
    for p in a.pop:
        np.testing.assert_allclose(s.base.evaluate(p.X[0].routes), p.F, atol=1e-7, rtol=0)
    a.state = _restore_individual(data["pending"]); a.generator = None
    a.resume_pending = a.state
    a.order, a.cursor = data["order"], data["cursor"]
    a.active_direction, a.ideal, a.n_iter = data["active_direction"], np.asarray(data["ideal"]), data["n_iter"]
    if a.state is not None:
        a.generator = a._next()
        next(a.generator)
    for name, key in (("adaptive", "adaptive"), ("archive", "archive"), ("records", "records"),
                      ("events", "events"), ("point_events", "points"), ("initial_seconds", "initial_seconds"),
                      ("numpy_state", "numpy_state")):
        setattr(s, name, copy.deepcopy(data[key]))
    s.snapshots=copy.deepcopy(data.get("snapshots",[]))
    return s


def restore_search(data,pool,cfg,policies,deadline):
    with _RNG_LOCK:return _restore_search_unlocked(data,pool,cfg,policies,deadline)
