"""Summarize finite diagnostic; all comparisons remain conditional/descriptive."""
import json
import statistics as st
from pathlib import Path

R=Path(__file__).resolve().parent
def read(n):return json.loads((R/n).read_text(encoding='utf-8'))
def avg(xs):return st.mean(xs)
def span(xs):return dict(mean=avg(xs),min=min(xs),max=max(xs))
def main():
    d=read('sensitivity.json');models=sorted({x['model'] for x in d});sensitivity={}
    for m in models:
        rows=[x for x in d if x['model']==m];center=[x for x in rows if x['variant']=='preference_15']
        pref=[x for x in rows if x['variant'] in ('preference_0','preference_1','preference_4')]
        sensitivity[m]=dict(q_entropy_ratio=span([x['q_entropy']/x['q_entropy_max'] for x in center]),
            anchor_entropy_ratio=span([x['anchor_normalized_entropy'] for x in center]),
            preference_q_tv=span([x['q_tv_to_center'] for x in pref]),
            preference_anchor_tv=span([x['anchor_tv_to_center'] for x in pref]))
        for prefix in ('global_used_','local_remaining_','stagnation_ratio_'):
            s=[x for x in rows if x['variant'].startswith(prefix)]
            sensitivity[m][prefix]=dict(q_tv=span([x['q_tv_to_center'] for x in s]),anchor_tv=span([x['anchor_tv_to_center'] for x in s]))
    features=read('deletion_features.json');paired={}
    for x in features:paired.setdefault((x['iid'],x['rep']),{})[x['preference']]=x
    preference_removal={}
    for k in (0,1,4):
        ps=list(paired.values());preference_removal[k]=dict(pairs=len(ps),same_set=sum(set(p[k]['removed'])==set(p[15]['removed']) for p in ps),
            dynamic_feature_difference={f:avg([p[k][f]-p[15][f] for p in ps]) for f in ('tardiness','hard_slack','detour')})
    records=read('repair_records.json');cells={}
    for x in records:cells.setdefault(x['cell'],{})[x['arm']]=x
    assert all(len(c)==4 for c in cells.values())
    repair={};paired_repair={}
    for n in ('all',50,150,250):
        for q in ('all','small','large'):
            selected=[x for x in records if (n=='all' or int(x['iid'].split('_')[2])==n) and (q=='all' or (x['q_fraction']<.1)==(q=='small'))]
            if not selected:continue
            k=f'{n}_{q}';repair[k]={}
            for arm in ('trained_center','initial','random','worst_distance'):
                rows=[x for x in selected if x['arm']==arm]
                repair[k][arm]=dict(calls=len(rows),accepted=sum(x['accepted'] for x in rows),
                    timeout=sum(x['timeout'] for x in rows),failed=sum(x['failed'] for x in rows),
                    mean_reward=avg([x['reward'] for x in rows]),total_reward=sum(x['reward'] for x in rows),
                    total_call_seconds=sum(x['call_seconds'] for x in rows))
            cc=[c for c in cells.values() if c['trained_center'] in selected];paired_repair[k]={}
            for arm in ('initial','random','worst_distance'):
                delta=[c['trained_center']['reward']-c[arm]['reward'] for c in cc]
                paired_repair[k][arm]=dict(pairs=len(cc),wins=sum(v>1e-12 for v in delta),ties=sum(abs(v)<=1e-12 for v in delta),losses=sum(v< -1e-12 for v in delta),
                    mean_reward_delta=avg(delta),same_removal_sets=sum(set(c['trained_center']['removed'])==set(c[arm]['removed']) for c in cc))
    hist=read('real_history_check.json');history={}
    for m in models:
        for nonzero in (False,True):
            s=[x for x in hist['rows'] if x['model']==m and (x['hidden_norm']>0)==nonzero]
            if s:history[f'{m}_hidden_{nonzero}']=dict(n=len(s),anchor_tv_uniform=span([x['anchor_tv_uniform'] for x in s]),
                anchor_entropy_ratio=span([x['anchor_entropy_ratio'] for x in s]),q_entropy_ratio=span([x['q_entropy_ratio'] for x in s]),
                preference_anchor_tv=span([x['preference_to_tardy_anchor_tv'] for x in s]))
    balanced={}
    for arm in ('trained_center','initial','random','worst_distance'):
        s=[x for x in records if x['rep']==0 and x['arm']==arm]
        balanced[arm]=dict(calls=len(s),accepted=sum(x['accepted'] for x in s),timeout=sum(x['timeout'] for x in s),
            mean_reward=avg([x['reward'] for x in s]),total_reward=sum(x['reward'] for x in s))
    output=dict(scope='Finite FIT diagnostics; no training, no plots, no new deployment evaluation; 3 physical repair states, 6 controlled sensitivity states. No inferential generalization claims.',
        sensitivity=sensitivity,preference_removal=preference_removal,repair=repair,paired_repair=paired_repair,
        balanced_first_repair_seed=balanced,
        real_history=history,parameter_changes=hist['parameter_changes'],gradient=read('gradient_check.json'),verification=read('verification.json'))
    (R/'summary.json').write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(repair=repair['all_all'],paired=paired_repair['all_all'],history=history.get('deployed_15_hidden_True'),gradient=output['gradient']),indent=2))

if __name__=='__main__':main()
