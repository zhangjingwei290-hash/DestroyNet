"""Local physical environment bridge. No source hashes or q clamps."""
import copy
import numpy as np
import torch
from .route_ops import build_edge_index_near,build_edge_index_routes
REPAIRERS=('greedy','regret2','regret3','noisy_greedy')

class Environment:
    def __init__(self, api, instance, row, box, q_values, preference, cfg):
        self.api, self.instance, self.cfg = api, instance, cfg
        self.box = copy.deepcopy(box)
        self.scales = np.asarray(box["scale"], dtype=np.float64)
        self.offset = np.asarray(box["offset"], dtype=np.float64)
        self.q_values = tuple(q_values)
        self.preference = np.asarray(preference, dtype=np.float64)
        if self.preference.shape != (5,) or (self.preference < 0).any() or not np.isfinite(self.preference).all() or not np.isclose(self.preference.sum(), 1.):
            raise ValueError("Invalid five-objective preference")
        self.routes = copy.deepcopy(row["source_solution"])
        self.objectives = self.evaluate(self.routes)
        self.archive = [{"routes": copy.deepcopy(self.routes), "objectives": self.objectives.tolist()}]
        self.steps = self.stagnation = 0
        self.last_accepted = False
        self.last_reward = 0.
        self.last_removed = []
        self.neighbors = [list(np.argsort(d, kind="stable")) for d in instance.distance_matrix]
        self.near = torch.as_tensor(build_edge_index_near(
            [instance.distance_matrix], min(cfg.near_neighbors, instance.n_customers))[0], dtype=torch.long)

    def solution(self, routes):
        return self.api.Solution([self.api.Route(list(r)) for r in routes])

    def evaluate(self, routes):
        sol = self.solution(routes)
        if not self.api.feasible(sol, self.instance):
            raise ValueError("Complete solution failed coverage/capacity/max-delay/depot-return checks")
        values = self.api.evaluate(sol, self.instance).to_array()
        if not np.isfinite(values).all():
            raise ValueError("Non-finite objectives")
        return values

    def score(self, values):
        weighted = self.preference * ((np.asarray(values) - self.offset) / self.scales)
        return float(weighted.max() + .05 * weighted.sum())

    def node_features(self):
        inst=self.instance;n=inst.n_customers;horizon=max(float(inst.depot_time_window[1]),1.)
        distance=max(float(inst.distance_matrix.max()),1e-9)
        x=np.zeros((n+1,20),dtype=np.float32)
        center=inst.customer_coords.mean(0)
        coord=max(float(np.sqrt(np.mean(np.sum((inst.customer_coords-center)**2,axis=1)))),1e-9)
        x[1:,:2]=(inst.customer_coords-center)/coord
        x[1:,2]=inst.customer_demands/inst.vehicle_capacity
        x[0,3:5]=np.asarray(inst.depot_time_window)/horizon
        x[1:,3:5]=inst.customer_time_windows/horizon
        x[1:,5]=inst.customer_service_times/horizon
        x[:,6]=x[:,4]-x[:,3];x[0,16]=1.
        completions=[]
        for route in self.routes:
            current=load=dist=0.;previous=0
            for c in route:
                arrival=current+inst.duration_matrix[previous,c]
                early,late=inst.customer_time_windows[c-1];waiting=max(early-arrival,0.);tardy=max(arrival-late,0.)
                current=arrival+waiting+inst.customer_service_times[c-1]
                load+=inst.customer_demands[c-1];dist+=inst.distance_matrix[previous,c]
                x[c,7:9]=load/inst.vehicle_capacity,dist/distance
                x[c,11:15]=arrival/horizon,waiting/horizon,tardy/horizon,(late+inst.max_delay-(arrival+waiting))/horizon
                previous=c
            dist+=inst.distance_matrix[previous,0];returned=float(current+inst.duration_matrix[previous,0]);completions.append(returned)
            sequence=[0]+route+[0]
            for j,c in enumerate(route,1):
                x[c,9:11]=load/inst.vehicle_capacity,dist/distance
                x[c,15]=(horizon-returned)/horizon
                x[c,17]=(inst.distance_matrix[sequence[j-1],c]+inst.distance_matrix[c,sequence[j+1]]-inst.distance_matrix[sequence[j-1],sequence[j+1]])/distance
                x[c,18]=len(route)/n
        makespan=max(completions,default=0.)
        for route,returned in zip(self.routes,completions):
            for c in route:x[c,19]=(makespan-returned)/horizon
        if not np.isfinite(x).all():raise ValueError('Nonfinite node features')
        return x

    def observation(self, hidden, repair_id, remaining_fraction):
        if repair_id not in REPAIRERS:
            raise ValueError("Unknown repairer")
        x = torch.from_numpy(self.node_features())
        previous = x[self.last_removed].clone() if self.last_removed and self.last_accepted else torch.zeros((1, x.shape[1]))
        values = list(self.preference) + list((self.objectives-self.offset)/self.scales)
        values += [max(0., min(1., remaining_fraction)), self.stagnation/self.cfg.episode_steps,
                   float(self.last_accepted), self.last_reward, len(self.archive)/(self.steps+1),
                   self.instance.max_delay/max(self.instance.depot_time_window[1], 1.),
                   self.instance.n_customers / 250., max(self.q_values) / self.instance.n_customers]
        values += [float(repair_id == r) for r in REPAIRERS]
        return {
            "x": x, "edge_index_n": self.near.clone(),
            "edge_index_r0": torch.as_tensor(build_edge_index_routes([self.routes])[0], dtype=torch.long),
            "edge_index_r1": torch.as_tensor(build_edge_index_routes([self.routes], inverse=True)[0], dtype=torch.long),
            "state_": hidden.detach().cpu().reshape(1, -1).clone(), "input_": previous,
            "context": torch.tensor(values, dtype=torch.float32),
            "q_values": self.q_values,
        }

