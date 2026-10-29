import copy
import unittest
import numpy as np
from .audit import AuditError, audit_branch, audit_comparison, checked_objectives, check_deletion, digest, exact_hv


def fixture():
    n = 7
    matrix = np.ones((n+1, n+1)); np.fill_diagonal(matrix, 0.)
    instance = dict(instance_id='hemtv2_train_50_R_fixture', n_customers=n, distance_matrix=matrix,
                    duration_matrix=matrix, customer_demands=np.ones(n), customer_time_windows=np.tile([0.,100.],(n,1)),
                    customer_service_times=np.zeros(n), max_delay=0., depot_time_window=[0.,100.], vehicle_capacity=20.)
    source = dict(routes=[list(range(1,n+1))])
    source['objectives'] = checked_objectives(source['routes'], instance)
    box = dict(reference=[10.,100.,100.,100.,100.], offset=[0.]*5, scale=[1.]*5)
    state = dict(instance_id=instance['instance_id'], split='train', source=source, pre_archive=[copy.deepcopy(source)],
                 box=box, feature_context={}, stage_scope='generic_ALNS')
    state['pre_archive_hash']=digest(state['pre_archive']);state['reference_hash']=digest(box)
    state['state_hash']=digest(dict(instance_id=state['instance_id'],source=source,pre_archive_hash=state['pre_archive_hash'],reference_hash=state['reference_hash'],feature_context={}))
    deletion=dict(partial_routes=[list(range(2,n+1))],removed_customers=[1])
    hv=exact_hv(state['pre_archive'],box)
    outcome=dict(source, status='success', feasible=True, committed=True, commit_elapsed_seconds=.2)
    branch=dict(state_hash=state['state_hash'],pre_archive_hash=state['pre_archive_hash'],reference_hash=state['reference_hash'],
                repair_id='greedy',repetition=0,seed=11,q=1,destroy=deletion,destroy_hash=digest(deletion),location_pattern='random',
                timing=dict(includes_first_action=True,elapsed_seconds=5.01,first_action_seconds=.1,first_commit_seconds=.2),
                horizon_seconds=5.,outcomes=[outcome],post_archive=copy.deepcopy(state['pre_archive']),
                post_archive_hash=digest(state['pre_archive']),attempted_count=1,committed_count=1,pre_hv=hv,post_hv=hv,delta_hv=0.,
                first_repair_success=True,label_success=True)
    cfg=dict(train_instance_ids=[state['instance_id']],boxes_by_instance={state['instance_id']:box},repair_ids=['greedy','regret2','regret3','noisy_greedy'])
    return branch,state,instance,cfg


class AuditTests(unittest.TestCase):
    def test_valid_duplicate_has_zero_gain(self):
        b,s,i,c=fixture(); result=audit_branch(b,s,i,c)
        self.assertEqual(result['delta_hv'],0.)

    def reject(self, edit):
        b,s,i,c=fixture();edit(b,s,i,c)
        with self.assertRaises((AuditError,KeyError)):
            audit_branch(b,s,i,c)

    def test_late_commit(self):
        self.reject(lambda b,s,i,c:b['outcomes'][0].update(commit_elapsed_seconds=5.001))

    def test_first_action_cost(self):
        self.reject(lambda b,s,i,c:b['timing'].update(includes_first_action=False))
        self.reject(lambda b,s,i,c:b['timing'].update(elapsed_seconds=.05))

    def test_first_repair_failed_label(self):
        self.reject(lambda b,s,i,c:b.update(first_repair_success=False,label_success=True))

    def test_future_feature(self):
        self.reject(lambda b,s,i,c:s['feature_context'].update(future_archive_gain=1.))

    def test_raw_objective_tampering(self):
        self.reject(lambda b,s,i,c:b['outcomes'][0].update(objectives=[1.,9.,8.,0.,0.]))

    def test_iid_leakage(self):
        self.reject(lambda b,s,i,c:s.update(split='holdout'))
        self.reject(lambda b,s,i,c:c.update(forbidden_instance_ids=[s['instance_id']]))
        self.reject(lambda b,s,i,c:c.update(train_instance_ids=[]))

    def test_reference_mutation_and_archive_mutation(self):
        self.reject(lambda b,s,i,c:s['box']['reference'].__setitem__(0,20.))
        self.reject(lambda b,s,i,c:s['pre_archive'].clear())
        self.reject(lambda b,s,i,c:b['post_archive'].clear())

    def test_failed_candidate_cannot_commit(self):
        self.reject(lambda b,s,i,c:b['outcomes'][0].update(status='timeout',feasible=False))

    def test_q1_and_strict_tiny_cap(self):
        b,s,i,c=fixture();self.assertEqual(check_deletion(s['source']['routes'],b['destroy']['partial_routes'],[1],7),1)
        with self.assertRaises(AuditError):check_deletion([[1,2,3]],[[2,3]],[1],3)
        self.reject(lambda b,s,i,c:b.update(q=2))

    def test_four_repairs_each_pattern(self):
        b,s,i,c=fixture();branches=[]
        for pattern in ('random','related'):
            for name in c['repair_ids']:
                branch=copy.deepcopy(b);branch.update(location_pattern=pattern,repair_id=name);branches.append(branch)
        self.assertEqual(audit_comparison(branches)['paired_blocks'],2)
        branches[1]['destroy_hash']='different'
        with self.assertRaises(AuditError):audit_comparison(branches)

    def test_tradeoff_hv_no_source_dominance_gate(self):
        box=dict(reference=[10.,10.,1.,1.,1.])
        original=[dict(objectives=[5.,5.,0.,0.,0.])]
        candidate=dict(objectives=[3.,7.,0.,0.,0.])
        self.assertEqual(exact_hv(original,box),25.)
        self.assertEqual(exact_hv(original+[candidate],box),31.)

    def test_route_coverage_capacity_and_time(self):
        b,s,i,c=fixture()
        with self.assertRaises(AuditError):checked_objectives([[1,1,2,3,4,5,6]],i)
        i['vehicle_capacity']=1.
        with self.assertRaises(AuditError):checked_objectives(s['source']['routes'],i)
        i['vehicle_capacity']=20.;i['depot_time_window']=[0.,2.]
        with self.assertRaises(AuditError):checked_objectives(s['source']['routes'],i)


if __name__=='__main__':unittest.main()
