"""Isolated search adapters. Frozen solver modules are not edited."""
import copy
import time
import numpy as np
from alns.select import RouletteWheel
from solver.environment import Adaptive
from solver.outer import Search, run_session


class SelectedAdaptive(Adaptive):
    """Map an explicitly restricted set of q slots onto unchanged removal code."""
    def __init__(self, q_values, n, parameters, state_rule=False):
        super().__init__(len(q_values), heuristic=True)
        self.q_values = q_values
        self.parameters = parameters
        self.state_rule = state_rule
        self.env = None
        self.ratio = .2
        self.ratios = np.asarray(q_values, dtype=float) / n
        self.allowed = [i for i, q in enumerate(q_values) if q <= int(np.ceil(n*parameters.get('max_ratio', 1.))) ]
        self.full_to_dense = {kind*len(q_values)+qi:kind*len(self.allowed)+j
                              for kind in range(4) for j, qi in enumerate(self.allowed)}
        self.selector = RouletteWheel([5., 3., 0., 0.], parameters['decay'],
                                      4 if state_rule else 4*len(self.allowed), 4)

    def choose(self, rng):
        d, r = (int(x) for x in self.selector(rng, None, None))
        if self.state_rule:
            if self.env.last_accepted:
                self.ratio = max(.05, self.ratio*.7)
            elif self.env.stagnation > 0 and self.env.stagnation % self.parameters['stall_threshold'] == 0:
                self.ratio = min(.6, self.ratio*1.5)
            legal = [i for i, value in enumerate(self.ratios) if .05-1e-12 <= value <= .6+1/len(self.env.instance.customer_demands)]
            qi = min(legal, key=lambda i: (abs(self.ratios[i]-self.ratio), i))
            return d*self.nq+qi, r
        kind, j = divmod(d, len(self.allowed))
        return kind*self.nq+self.allowed[j], r

    def update(self, d, r, result, final_score):
        mapped = d//self.nq if self.state_rule else self.full_to_dense[d]
        super().update(mapped, r, result, final_score)


def moves(routes, name, rng):
    """Exhaustive one-move neighborhoods, shuffled traversal; all customer IDs preserved."""
    positions = [(ri, ci) for ri, row in enumerate(routes) for ci in range(len(row))]
    order = rng.permutation(len(positions))
    if name == 'relocate':
        targets = [(ri, ci) for ri, row in enumerate(routes) for ci in range(len(row)+1)] + [(len(routes), 0)]
        target_order = rng.permutation(len(targets))
        for oi in order:
            ri, ci = positions[oi]
            for ti in target_order:
                rj, cj = targets[ti]
                if ri == rj and cj in (ci, ci+1):
                    continue
                trial = [list(row) for row in routes]
                customer = trial[ri].pop(ci)
                if rj == len(routes):
                    trial.append([customer])
                else:
                    trial[rj].insert(cj-int(ri == rj and cj > ci), customer)
                yield [row for row in trial if row], 1
    elif name == 'swap':
        for i in range(len(order)):
            ri, ci = positions[order[i]]
            for j in range(i+1, len(order)):
                rj, cj = positions[order[j]]
                trial = [list(row) for row in routes]
                trial[ri][ci], trial[rj][cj] = trial[rj][cj], trial[ri][ci]
                yield trial, 2
    elif name == '2opt':
        for ri in rng.permutation(len(routes)):
            row = routes[ri]
            endpoints = [(i, j) for i in range(len(row)) for j in range(i+1, len(row))]
            for index in rng.permutation(len(endpoints)):
                i, j = endpoints[index]
                trial = [list(route) for route in routes]
                trial[ri][i:j+1] = reversed(trial[ri][i:j+1])
                yield trial, j-i+1
    elif name == '2opt_star':
        route_order = rng.permutation(len(routes))
        for a in range(len(route_order)):
            ri = route_order[a]
            for b in range(a+1, len(route_order)):
                rj = route_order[b]
                cuts = [(i, j) for i in range(len(routes[ri])+1) for j in range(len(routes[rj])+1)]
                for index in rng.permutation(len(cuts)):
                    i, j = cuts[index]
                    if (i == 0 and j == 0) or (i == len(routes[ri]) and j == len(routes[rj])):
                        continue
                    trial = [list(row) for row in routes]
                    trial[ri] = routes[ri][:i]+routes[rj][j:]
                    trial[rj] = routes[rj][:j]+routes[ri][i:]
                    yield [row for row in trial if row], max(1, len(routes[ri])-i+len(routes[rj])-j)
    else:
        raise ValueError(name)


def vnd_session(session):
    """First improvement; restart at relocate after strict scalar improvement."""
    env = session.env
    start = time.perf_counter()
    deadline = min(start+session.remaining(), session.solver_task_deadline, session.outer_deadline)
    neighborhoods = ('relocate', 'swap', '2opt', '2opt_star')
    counters = {name:dict(attempts=0, infeasible=0, feasible=0, accepted=0, archive_updates=0, late=0, seconds=0.) for name in neighborhoods}
    k = 0
    while k < 4 and time.perf_counter() < deadline:
        name = neighborhoods[k]
        changed = False
        for trial, q in moves(env.routes, name, session.rng):
            if time.perf_counter() >= deadline:
                break
            before = time.perf_counter()
            stat = counters[name]
            stat['attempts'] += 1
            solution = env.solution(trial)
            if not env.api.feasible(solution, env.instance):
                stat['infeasible'] += 1
                stat['seconds'] += time.perf_counter()-before
                continue
            # Independent complete evaluate() checks the same hard constraints again.
            candidate = env.evaluate(trial)
            checked = time.perf_counter()
            stat['seconds'] += checked-before
            if checked >= deadline:
                stat['late'] += 1
                break
            stat['feasible'] += 1
            old_values = env.objectives.copy()
            gain = env.score(old_values)-env.score(candidate)
            accepted = gain > 0.
            novel = not any(np.all(np.asarray(p['objectives']) <= candidate) for p in env.archive)
            if novel:
                env.archive = [p for p in env.archive if not np.all(candidate <= np.asarray(p['objectives']))]
                env.archive.append(dict(routes=copy.deepcopy(trial), objectives=candidate.tolist()))
                stat['archive_updates'] += 1
            if accepted:
                env.routes = trial
                env.objectives = candidate.copy()
                stat['accepted'] += 1
                changed = True
            env.steps += 1
            env.last_accepted = accepted
            env.last_reward = max(0., gain)
            env.stagnation = 0 if accepted else env.stagnation+1
            # Non-novel rejected moves cannot alter HV. Retain their exact counts, not route copies.
            if accepted or novel:
                session.records.append(dict(repair='vnd_'+name, repair_seed=None, repair_counters={},
                    q=int(q), removed=[], accepted=bool(accepted), reward=max(0., gain), feasible=True,
                    failed=False, timeout=False, reason='ok', candidate_objectives=candidate.tolist(),
                    candidate_finished_perf=checked, before_objectives=old_values.tolist(),
                    current_objectives=env.objectives.tolist(), repair_seconds=0.,
                    environment_seconds=checked-before, inference_seconds=0.,
                    local_elapsed=checked-start, local_budget=session.budget,
                    global_used_fraction_before=1-session.global_start/session.global_budget,
                    stall_fraction_before=0., last_accepted_before=False, stagnation_before=0,
                    terminated=False, truncated=False, ending_reason='continuing', action={'neighborhood':name}))
            if changed:
                break
        k = 0 if changed else k+1
    session.elapsed += time.perf_counter()-start
    return counters


class BenchmarkSearch(Search):
    def __init__(self, *args, method, parameters=None, **kwargs):
        self.method = method
        self.parameters = parameters or {}
        self.vnd_audits = []
        super().__init__(*args, **kwargs)
        if method in ('alns_tuned', 'alns_state_rule'):
            self.adaptive = [SelectedAdaptive(self.base.q_values, self.base.instance.n_customers,
                              self.parameters, method == 'alns_state_rule') for _ in range(16)]

    def _collect(self, direction, session):
        if self.method == 'vnd':
            self.vnd_audits.append(dict(direction=direction, counters=vnd_session(session)))
        else:
            session.adaptive.env = session.env
            run_session(session, None)
            session.adaptive.env = None
        return session, 0., []

    def _step(self, rollout=None):
        if self.method == 'v5_meta_frozen':
            return super()._step(rollout)
        return super()._step(self._collect)

    def report(self):
        result = super().report()
        result.update(method=self.method, parameters=self.parameters, vnd_audits=self.vnd_audits,
            vnd_record_policy='All attempted moves counted; routes/objectives recorded only for accepted or local-archive-novel feasible moves.')
        return result
