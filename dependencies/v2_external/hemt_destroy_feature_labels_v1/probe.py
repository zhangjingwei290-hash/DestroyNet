"""Exploratory grouped feature probes; no neural model or DEV/holdout fitting."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mutual_info_score
from .collector import save


def impute_scale(train,test):
    keep=np.isfinite(train).any(axis=0)
    a=train[:,keep];b=test[:,keep]
    med=np.array([np.median(c[np.isfinite(c)]) for c in a.T])
    a=np.where(np.isfinite(a),a,med);b=np.where(np.isfinite(b),b,med)
    scale=np.quantile(a,.75,axis=0)-np.quantile(a,.25,axis=0);scale[scale<1e-10]=1.
    return (a-med)/scale,(b-med)/scale,np.where(keep)[0]


def mrmr(train,target,k):
    # Quantile-discretized plug-in MI, not a claim of conditional-MI optimality.
    bins=[]
    for column in train.T:
        edges=np.unique(np.quantile(column,[.2,.4,.6,.8]))
        bins.append(np.searchsorted(edges,column))
    y=np.searchsorted(np.unique(np.quantile(target,[.2,.4,.6,.8])),target)
    relevance=np.array([mutual_info_score(x,y) for x in bins]);selected=[]
    eligible=[j for j,x in enumerate(bins) if len(np.unique(x))>1]
    cache={}
    for _ in range(min(k,len(eligible))):
        scores=[]
        for j in eligible:
            redundancy=[]
            for s in selected:
                pair=tuple(sorted((j,s)))
                if pair not in cache:cache[pair]=mutual_info_score(bins[j],bins[s])
                redundancy.append(cache[pair])
            scores.append((relevance[j]-(np.mean(redundancy) if redundancy else 0.),-j,j))
        chosen=max(scores)[2];selected.append(chosen);eligible.remove(chosen)
    return selected


def probe(out,cfg):
    out=Path(out);groups=defaultdict(list)
    for path in sorted((out/'branches').glob('*.json')):
        if path.name.endswith('.binding.json'):continue
        b=json.loads(path.read_text(encoding='utf-8'))
        groups[(b['state_hash'],b['q'],b['location_pattern'],b['repair_id'])].append(b)
    rows=[]
    for key,bs in sorted(groups.items()):
        assert len(bs)==cfg['repetitions'] and all(b['feature_names']==bs[0]['feature_names'] for b in bs)
        assert all(b['features']==bs[0]['features'] for b in bs),'Predecision inputs changed across paired repeats'
        rows.append(dict(iid=bs[0]['instance_id'],state=key[0],features=bs[0]['features'],
                         target=float(np.mean([b['delta_hv']/max(b['pre_hv'],1e-12) for b in bs])),
                         observed_success=float(np.mean([b['first_repair_success'] for b in bs]))))
    names=next(iter(groups.values()))[0]['feature_names']
    x=np.array([[np.nan if v is None else v for v in r['features']] for r in rows]);y=np.array([r['target'] for r in rows])
    # Fixed IID order; each scale's six IID are divided equally among three folds.
    fold_by_iid={}
    for n in (50,150,250):
        ids=sorted(i for i in cfg['train_instance_ids'] if f'_train_{n}_' in i)
        assert len(ids)==6
        for j,iid in enumerate(ids):fold_by_iid[iid]=j%3
    folds=np.array([fold_by_iid[r['iid']] for r in rows]);predictions={m:np.full(len(y),np.nan) for m in ('q_only_ridge','controls_tree','all_tree','mrmr_tree')}
    evidence=[]
    for fold in range(3):
        fit=folds!=fold;test=folds==fold
        train,val,indices=impute_scale(x[fit],x[test]);available=[names[j] for j in indices]
        control=[j for j,n in enumerate(available) if n.startswith('control:') or 'preference_' in n or n=='global:log1p_n_customers']
        q=[available.index('control:q_fraction')]
        selected=mrmr(train,y[fit],cfg['probe']['mrmr_k'])
        selectors=dict(q_only_ridge=q,controls_tree=control,all_tree=list(range(train.shape[1])),mrmr_tree=selected)
        for name,cols in selectors.items():
            if not cols:cols=control
            model=Ridge(alpha=1.) if name.endswith('ridge') else ExtraTreesRegressor(n_estimators=100,max_depth=6,min_samples_leaf=5,n_jobs=1,random_state=cfg['seed']+fold)
            model.fit(train[:,cols],y[fit]);predictions[name][test]=model.predict(val[:,cols])
        evidence.append(dict(fold=fold,fit_ids=sorted({r['iid'] for j,r in enumerate(rows) if fit[j]}),
                             test_ids=sorted({r['iid'] for j,r in enumerate(rows) if test[j]}),
                             selected_features=[available[j] for j in selected],
                             preprocessing='TRAIN fold medians/IQR and 5-quantile MI bins only'))
    state_rows=defaultdict(list)
    for j,r in enumerate(rows):state_rows[r['state']].append(j)
    results={}
    for method,pred in predictions.items():
        assert np.isfinite(pred).all()
        losses=[];chosen=[];oracle=[];uniform=[];coverage=[]
        for inds in state_rows.values():
            truth=y[inds];chosen.append(float(truth[int(np.argmax(pred[inds]))]));oracle.append(float(truth.max()));uniform.append(float(truth.mean()))
            spread=float(np.ptp(truth));coverage.append(spread>1e-8)
            if spread>1e-8:losses.append((truth.max()-truth[int(np.argmax(pred[inds]))])/spread)
        corr=spearmanr(y,pred).statistic if np.ptp(y)>0 and np.ptp(pred)>0 else None
        results[method]=dict(mae=float(np.mean(np.abs(y-pred))),spearman=None if corr is None or not np.isfinite(corr) else float(corr),
            mean_selected_relative_hv_gain=float(np.mean(chosen)),mean_uniform_panel_gain=float(np.mean(uniform)),
            mean_panel_oracle_gain=float(np.mean(oracle)),normalized_regret_nonflat=None if not losses else float(np.mean(losses)),
            distinguishable_states=sum(coverage),states=len(coverage),selected_by_state=chosen)
    rng=np.random.default_rng(cfg['seed']);ids=sorted(cfg['train_instance_ids']);state_keys=list(state_rows)
    gains=np.array(results['mrmr_tree']['selected_by_state'])-np.array(results['controls_tree']['selected_by_state'])
    iid_gains=[]
    for iid in ids:
        positions=[j for j,key in enumerate(state_keys) if rows[state_rows[key][0]]['iid']==iid]
        iid_gains.append(float(np.mean(gains[positions])))
    samples=np.mean(np.asarray(iid_gains)[rng.integers(0,len(ids),size=(2000,len(ids)))],axis=1)
    report=dict(scope='Exploratory TRAIN18 grouped-IID out-of-fold classical probes; not independent DEV nor neural/search success',
        instances=len(ids),states=len(state_rows),mean_repeat_rows=len(rows),branches=sum(len(bs) for bs in groups.values()),
        target='mean over 3 repeats of (HVpost-HVpre)/HVpre; timeout observed gain zero explicitly censored',
        timeout_fraction=float(1-np.mean([r['observed_success'] for r in rows])),
        reference='Immutable conservative physical box before outcomes; not final benchmark HV reference',
        mrmr='TRAIN-fold 5-quantile discrete MI relevance minus mean pairwise MI redundancy; k fixed24',
        fold_evidence=evidence,results=results,
        selected_vs_controls_mean_iid_gain=float(np.mean(iid_gains)),
        selected_vs_controls_iid_bootstrap95=[float(v) for v in np.quantile(samples,[.025,.975])],
        final_features=None,conclusion='Feature candidates remain exploratory; confirmation on more IID and actual complete search is required')
    save(out/'PROBE_REPORT.json',report)
    save(out/'out_of_fold_predictions.json',dict(rows=rows,predictions={k:v.tolist() for k,v in predictions.items()}))
