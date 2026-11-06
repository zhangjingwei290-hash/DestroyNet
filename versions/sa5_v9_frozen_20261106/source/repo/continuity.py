import hashlib,pickle,numpy as np

def fingerprint(alg,archive,stall,rate,last_repair,adaptive,historical,numpy_state):
 state=dict(historical=historical,numpy_state=numpy_state,pop=[(p.X[0].routes,p.F.tolist()) for p in alg.pop],archive=archive,stall=stall,ema=list(rate.history),last_repair=last_repair,adaptive=[(a.selector.destroy_weights,a.selector.repair_weights,a.count,a.seconds,a.best,a.recent_seconds) for a in adaptive])
 return hashlib.sha256(pickle.dumps(state,protocol=4)).hexdigest()
