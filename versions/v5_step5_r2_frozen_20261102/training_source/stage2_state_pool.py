"""Real progressive elite states; no unoptimized-neighbor crossover in training."""
import copy,time
import numpy as np
from stage1_mechanisms import dynamic_coordinates,dynamic_score

class StatePool:
    def __init__(self,pool):
        self.pool=pool;self.slots={};self.archives={};self.clock=0;self.advances=[]
        self.ids=list(pool.manifest['fit_order'])
    def eligible(self,n,tw_positive=False):
        return [i for i in self.ids if int(i.split('_')[2])==n and (not tw_positive or int(i.split('_')[4][2:])>0)]
    def label(self,slot):
        if slot['needs_advance']:return 'Early'
        if slot['env'].stagnation>=4 or (slot['acceptance'] is not None and slot['acceptance']<.15):return 'Late'
        if len(slot['env'].routes)<=.9*slot['initial_vehicles']:return 'Mid'
        return 'Early'
    def available(self,slot):return slot['budget']-slot['used']>=2.6
    def get(self,iid,target,preference):
        from solver.environment import Environment,Adaptive
        from solver.config import search_budget
        key=(iid,int(target));old=self.slots.get(key);pref=np.asarray(preference,dtype=float)
        if old is not None and np.array_equal(old['env'].preference,pref) and self.available(old):return old
        if old is not None and not self.available(old):return None
        inst,row,box,_=self.pool.load(iid)
        archive=copy.deepcopy(self.archives.get(iid,[]))
        if not archive:archive=[dict(routes=copy.deepcopy(row['source_solution']),objectives=self.pool.api.evaluate(self.pool.api.Solution([self.pool.api.Route(v) for v in row['source_solution']]),inst).to_array().tolist())]
        reference=np.asarray(box['offset'])+np.asarray(box['scale'])*np.asarray(box['reference'])
        ideal,span=dynamic_coordinates([p['objectives'] for p in archive],box['scale'],reference)
        scores=dynamic_score(np.asarray([p['objectives'] for p in archive]),pref,ideal,span,reference,box['scale'],1.,1e-3)
        best=archive[int(np.argmin(scores))]
        env=Environment(self.pool,iid,self.pool.cfg,pref,best['routes']);env.archive=archive
        env.dynamic_ideal=ideal;env.dynamic_span=span
        rebound=old is not None or iid in self.archives
        # A new preference is a new scalar task: mature classification must follow real advancement.
        if old is not None:env.steps=old['env'].steps
        slot=dict(iid=iid,target=int(target),env=env,adaptive=copy.deepcopy(old['adaptive']) if old else Adaptive(len(env.q_values)),
            hidden=np.zeros((1,self.pool.cfg.hidden_dim),dtype=np.float32),used=old['used'] if old else 0.,
            budget=search_budget(inst.n_customers,self.pool.cfg,iid=iid,row=row),initial_vehicles=len(row['source_solution']),
            acceptance=None,needs_advance=rebound,last_used=self.clock,preference_rebound=rebound)
        self.slots[key]=slot;return slot
    def candidates(self,iid,target,preference,tw_positive=False,mature=False):
        n=int(iid.split('_')[2]);allowed=self.eligible(n,tw_positive)
        ready=[]
        for key,slot in self.slots.items():
            if key[1]!=target or key[0] not in allowed or not self.available(slot):continue
            if not np.array_equal(slot['env'].preference,np.asarray(preference)) or slot['needs_advance']:continue
            if mature and self.label(slot) not in ('Mid','Late'):continue
            ready.append(slot)
        ready.sort(key=lambda x:(x['last_used'],x['iid']))
        return ready,allowed
    def session(self,slot,seed,deadline):
        import torch
        from solver.environment import LocalSession
        env=copy.deepcopy(slot['env'])
        archive=copy.deepcopy(self.archives.get(slot['iid'],env.archive));env.archive=archive
        ideal,span=dynamic_coordinates([p['objectives'] for p in archive],env.scales,env.offset+env.scales*np.asarray(env.box['reference']))
        env.dynamic_ideal=ideal;env.dynamic_span=span
        session=LocalSession(env,copy.deepcopy(slot['adaptive']),seed,slot['budget'],slot['budget']-slot['used'],2.5,deadline)
        session.budget=2.5;session.hidden=torch.from_numpy(slot['hidden'].copy());session.group_start_routes=copy.deepcopy(env.routes)
        return session
    def commit(self,slot,session,traversal):
        from solver.outer import union
        self.clock+=1;slot['env']=session.env;slot['adaptive']=session.adaptive;slot['hidden']=session.hidden.numpy().copy()
        slot['used']+=session.elapsed;slot['last_used']=self.clock
        slot['acceptance']=sum(bool(r['accepted']) for r in session.records)/max(1,len(session.records))
        slot['needs_advance']=session.elapsed<2.5
        self.archives[slot['iid']]=union(self.archives.get(slot['iid'],[]),session.env.archive)
        if traversal:self.advances.append(dict(iid=slot['iid'],target=slot['target'],steps=len(session.records),accepted_rate=slot['acceptance'],stagnation=slot['env'].stagnation,label=self.label(slot),source_elapsed=slot['used'],source_budget=slot['budget']))
    def snapshot(self):
        rows=[]
        for (iid,k),slot in self.slots.items():
            e=slot['env']
            rows.append(dict(iid=iid,target=k,routes=e.routes,objectives=e.objectives,archive=e.archive,
                preference=e.preference,stagnation=e.stagnation,steps=e.steps,last_accepted=e.last_accepted,
                last_reward=e.last_reward,last_removed=e.last_removed,hidden=slot['hidden'],adaptive=slot['adaptive'],
                used=slot['used'],budget=slot['budget'],acceptance=slot['acceptance'],label=self.label(slot)))
        return dict(slots=rows,archives=self.archives,advances=self.advances)
