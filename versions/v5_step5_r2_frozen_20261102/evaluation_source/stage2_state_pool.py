"""Real progressive elite states; no unoptimized-neighbor crossover in training."""
import copy,time
import numpy as np
from stage1_mechanisms import dynamic_coordinates,dynamic_score

class StatePool:
    def __init__(self,pool):
        self.pool=pool;self.slots={};self.archives={};self.clock=0;self.advances=[];self.resets=[]
        self.ids=list(pool.manifest['fit_order'])
    def restore(self,payload):
        from solver.environment import Environment
        self.archives=copy.deepcopy(payload['archives']);self.clock=0;self.advances=[]
        for row in payload['slots']:
            iid=row['iid'];assert iid in self.pool.ids
            env=Environment(self.pool,iid,self.pool.cfg,row['preference'],row['routes']);env.archive=copy.deepcopy(row['archive'])
            np.testing.assert_allclose(env.objectives,row['objectives'],atol=1e-7,rtol=0.)
            for key in ('stagnation','steps','last_accepted','last_reward','last_removed'):setattr(env,key,copy.deepcopy(row[key]))
            ref=env.offset+env.scales*np.asarray(env.box['reference']);env.dynamic_ideal,env.dynamic_span=dynamic_coordinates([p['objectives'] for p in env.archive],env.scales,ref)
            initial=len(self.pool.load(iid)[1]['source_solution'])
            slot=dict(generation=int(row.get('generation',0)),iid=iid,target=int(row['target']),env=env,adaptive=copy.deepcopy(row['adaptive']),hidden=np.asarray(row['hidden']).copy(),
                used=float(row['used']),budget=float(row['budget']),initial_vehicles=initial,acceptance=row['acceptance'],needs_advance=False,last_used=self.clock,preference_rebound=False)
            self.clock+=1;self.slots[iid,slot['target']]=slot
        self.restored_slots=len(self.slots)
        return dict(slots=len(self.slots),archives=len(self.archives),labels={name:sum(self.label(v)==name for v in self.slots.values()) for name in ('Early','Mid','Late')},preserved_old_elapsed=True)
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
        expired=old is not None and not self.available(old)
        generation=int(old.get('generation',0))+1 if expired else int(old.get('generation',0)) if old else 0
        if expired:
            self.resets.append(dict(iid=iid,target=int(target),generation=generation,previous_elapsed=old['used'],previous_budget=old['budget'],previous_routes=copy.deepcopy(old['env'].routes),reason='horizon_exhausted_fresh_source_trajectory'))
            old=None
        inst,row,box,_=self.pool.load(iid)
        archive=[] if expired else copy.deepcopy(self.archives.get(iid,[]))
        if not archive:archive=[dict(routes=copy.deepcopy(row['source_solution']),objectives=self.pool.api.evaluate(self.pool.api.Solution([self.pool.api.Route(v) for v in row['source_solution']]),inst).to_array().tolist())]
        reference=np.asarray(box['offset'])+np.asarray(box['scale'])*np.asarray(box['reference'])
        ideal,span=dynamic_coordinates([p['objectives'] for p in archive],box['scale'],reference)
        scores=dynamic_score(np.asarray([p['objectives'] for p in archive]),pref,ideal,span,reference,box['scale'],1.,1e-3)
        best=archive[int(np.argmin(scores))]
        env=Environment(self.pool,iid,self.pool.cfg,pref,best['routes']);env.archive=archive
        env.dynamic_ideal=ideal;env.dynamic_span=span
        rebound=(old is not None or iid in self.archives) and not expired
        # A new preference is a new scalar task: mature classification must follow real advancement.
        if old is not None:env.steps=old['env'].steps
        slot=dict(generation=generation,iid=iid,target=int(target),env=env,adaptive=copy.deepcopy(old['adaptive']) if old else Adaptive(len(env.q_values)),
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
        from solver.parallel_context import current_client
        client=current_client()
        if client is not None:env.api.repair=client.bound_repair(slot['iid'])
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
            rows.append(dict(generation=slot.get('generation',0),iid=iid,target=k,routes=e.routes,objectives=e.objectives,archive=e.archive,
                preference=e.preference,stagnation=e.stagnation,steps=e.steps,last_accepted=e.last_accepted,
                last_reward=e.last_reward,last_removed=e.last_removed,hidden=slot['hidden'],adaptive=slot['adaptive'],
                used=slot['used'],budget=slot['budget'],acceptance=slot['acceptance'],label=self.label(slot)))
        return dict(slots=rows,archives=self.archives,advances=self.advances,resets=self.resets)
