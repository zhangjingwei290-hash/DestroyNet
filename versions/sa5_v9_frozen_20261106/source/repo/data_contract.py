import numpy as np

def physical_box(instance):
 n=instance.n_customers;d=float(np.max(instance.distance_matrix));h=max(float(instance.depot_time_window[1]),1.)
 bound=[n,2*n*d,float(instance.depot_time_window[1]),n*max(0.,float(np.max(instance.customer_time_windows[:,0]))),n*instance.max_delay]
 floors=[1.,max(d,1e-9),h,h,h]
 return dict(scale=[max(float(x),float(f)) for x,f in zip(bound,floors)],offset=[0.]*5,reference=[1.1]*5)
