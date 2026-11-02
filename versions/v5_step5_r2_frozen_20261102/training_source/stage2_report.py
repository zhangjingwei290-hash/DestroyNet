"""Stage1 four-method per-seed metrics. No search/model/torch import."""
from stage2_common import ROOT, read, write
from pathlib import Path
from collections import defaultdict
import statistics, math, time


def main():
    import psutil
    psutil.Process().cpu_affinity([14,15])
    import numpy as np
    import moocore
    from stats import geometry, csv_write, wtl
    plan=read(ROOT/'plan.json'); methods=plan['methods']+['step5_stage2_retrained']; seeds=plan['search_seeds']
    assert read(ROOT/'stage2_evaluation/complete.json')['runs']==36
    out=ROOT/'stage2_analysis'; out.mkdir(exist_ok=True)
    runs=[]; means=[]; refs={}; reports=[]; knees=[]
    for iid in plan['test_ids']:
        instance=[]
        for method in methods:
            source=(Path(plan['baseline_root'])/'metric_inputs/results' if method in methods[:2]
                    else Path(plan['stage4_root'])/'evaluation/metric_inputs' if method==methods[2]
                    else ROOT/'stage1_evaluation/metric_inputs' if method=='step5_stage1_zero_shot' else ROOT/'stage2_evaluation/metric_inputs')
            for seed in seeds:
                r=read(source/f'{iid}_seed{seed}_{method}.json')
                assert (r['iid'],r['seed'],r['method'])==(iid,seed,method)
                instance.append(r); reports.append(r)
        transform,pf,refs[iid]=geometry(instance)
        for r in instance:
            values=np.asarray([p['objectives'] for p in r['archive']])
            fixed=(values-np.asarray(r['physical_offset']))/np.asarray(r['physical_scale'])
            offset=np.asarray(r['physical_offset']); scale=np.asarray(r['physical_scale'])
            reference=(np.asarray(r['physical_reference'])-offset)/scale
            row=dict(iid=iid,n=int(iid.split('-')[0]),seed=r['seed'],method=r['method'],
                empirical_hv=float(moocore.hypervolume(transform(values),ref=[1.1]*5)),
                fixed_box_hv=float(moocore.hypervolume(fixed,ref=reference)),
                igd_plus=float(moocore.igd_plus(transform(values),ref=pf)))
            runs.append(row)
            for vehicles in sorted(set(int(v[0]) for v in values)):
                group=values[values[:,0]==vehicles]
                best_d=group[np.argmin(group[:,1])];best_t=group[np.argmin(group[:,2])]
                knees.append(dict(iid=iid,seed=r['seed'],method=r['method'],vehicles=vehicles,
                    best_distance=float(best_d[1]),completion_at_best_distance=float(best_d[2]),
                    best_completion=float(best_t[2]),distance_at_best_completion=float(best_t[1])))
        for method in methods:
            rows=[v for v in runs if v['iid']==iid and v['method']==method]
            assert len(rows)==3
            row=dict(iid=iid,n=int(iid.split('-')[0]),method=method,seeds=3)
            for metric in ('empirical_hv','fixed_box_hv','igd_plus'):
                vals=[v[metric] for v in rows]
                row[metric+'_mean']=statistics.mean(vals);row[metric+'_sample_sd']=statistics.stdev(vals)
            means.append(row)
    assert len(runs)==len({(r['iid'],r['seed'],r['method']) for r in runs})==180
    assert len(means)==60
    macro=[]; comparisons=[]
    for n in ('overall',50,150,250):
        for method in methods:
            rows=[r for r in means if r['method']==method and (n=='overall' or r['n']==n)]
            row=dict(n=n,method=method,iids=len(rows))
            for metric in ('empirical_hv','fixed_box_hv','igd_plus'):
                row[metric]=statistics.mean(r[metric+'_mean'] for r in rows)
                row[metric+'_between_iid_sample_sd']=statistics.stdev(r[metric+'_mean'] for r in rows)
                row[metric+'_mean_within_iid_seed_sd']=statistics.mean(r[metric+'_sample_sd'] for r in rows)
            macro.append(row)
        for baseline in methods[:-1]:
            for metric in ('empirical_hv','fixed_box_hv','igd_plus'):
                sign=-1 if metric=='igd_plus' else 1
                mm={(r['iid'],r['method']):r[metric+'_mean'] for r in means if n=='overall' or r['n']==n}
                rr={(r['iid'],r['seed'],r['method']):r[metric] for r in runs if n=='overall' or r['n']==n}
                ids=[i for i in plan['test_ids'] if n=='overall' or int(i.split('-')[0])==n]
                comparisons.append(dict(n=n,method=methods[-1],baseline=baseline,metric=metric,
                    iid_wtl=wtl([sign*(mm[i,methods[-1]]-mm[i,baseline]) for i in ids]),
                    run_wtl=wtl([sign*(rr[i,s,methods[-1]]-rr[i,s,baseline]) for i in ids for s in seeds])))
    qrows=[]; throughput=[]
    for n in (50,150,250):
        for method in methods:
            group=[r for r in reports if r['method']==method and int(r['iid'].split('-')[0])==n]
            histogram=defaultdict(int)
            for r in group:
                for q,count in r['q_histogram'].items():histogram[int(q)]+=int(count)
            steps=sum(histogram.values());costs=[v for r in group for v in r['repair_costs'].values()]
            assert steps==sum(v['calls'] for v in costs)
            cuts=[sum(c for q,c in histogram.items() if lo<q/n<=hi) for lo,hi in [(-1,.02),(.02,.05),(.05,.10),(.10,.25),(.25,1.)]]
            assert sum(cuts)==steps
            qrows.append(dict(n=n,method=method,steps=steps,mean_q=sum(q*c for q,c in histogram.items())/steps,
                **{name:count/steps for name,count in zip(['le2pct','gt2_le5pct','gt5_le10pct','gt10_le25pct','gt25pct'],cuts)}))
            cache=[r.get('cache_diagnostics') for r in group]
            refreshes=sum(c['refreshes'] for c in cache) if all(c is not None for c in cache) else None
            throughput.append(dict(n=n,method=method,runs=len(group),total_steps=steps,mean_steps_per_run=steps/len(group),
                inference_seconds=sum(r['inference_seconds'] for r in group),
                inf_ms=1000*sum(r['inference_seconds'] for r in group)/steps,
                repair_seconds=sum(v['seconds'] for v in costs),rep_ms=1000*sum(v['seconds'] for v in costs)/steps,
                gnn_refresh_rate=None if refreshes is None else refreshes/steps,
                refresh_count=refreshes))
    mode_rows=[]
    for n in (50,150,250):
        selected=[r for r in reports if r['method']==methods[-1] and int(r['iid'].split('-')[0])==n]
        total=sum(r['steps'] for r in selected)
        for m in range(3):
            items=[r['mode_diagnostics'][m] for r in selected];steps=sum(v['steps'] for v in items);accepted=sum(v['accepted'] for v in items)
            mode_rows.append(dict(n=n,mode=m,steps=steps,proportion=steps/total,accepted=accepted,acceptance_rate=accepted/steps if steps else None,vehicle_reductions=sum(v['vehicle_reductions'] for v in items),accepted_vehicle_reductions=sum(v['accepted_vehicle_reductions'] for v in items),first_route_cap_exceptions=sum(v['first_route_cap_exceptions'] for v in items)))
    csv_write(out/'mode_distribution.csv',mode_rows)
    for name,rows in [('per_run_metrics',runs),('main_table',means),('macro',macro),('q_distribution',qrows),('throughput',throughput),('vehicle_knee_detail',knees)]:
        csv_write(out/f'{name}.csv',rows)
    summary=dict(stage=2,mechanism_revision='stage2_r2_structured_modes_unified_scalar',aborted_partial_runs_excluded=24,aborted_partial_archive=plan['aborted_partial_archive'],complete=True,total_runs=180,new_runs=36,reused_runs=144,
        scope='12 previously inspected IIDs x3 seeds; combined mechanism comparison; single_bounded_retraining',
        seed_aggregation='separate seed archives; mean and sample SD; IID equal-weight macro',
        geometry=refs,macro=macro,mode_distribution=mode_rows,comparisons=comparisons,q_distribution=qrows,throughput=throughput,
        performance_timing='inference excludes observation/prepare for compatibility; new full service recorded separately; old GNN refresh counts may be unavailable',
        limitations=['Several mechanisms changed together; no single-mechanism causal attribution',
                     'Older baselines run at earlier host-load periods',
                     'CPU timing targets and behavioral learning targets require measurements'],
        generated_unix=time.time())
    write(ROOT/'stage2_summary.json',summary);write(out/'summary.json',summary)
    text=['# Stage2 机制升级与跨规模q头重训结果','',
          '五方法共180份逐seed指标；其中Stage2新36次，复用旧三方法108次和修复Stage1的36次。中止版本24次排除。每个seed档案独立计算，再取均值/样本SD，跨IID等权。',
          '经验参考使用本阶段五方法联合ND；固定物理盒HV另列。旧表不覆盖。','',
          '|范围|方法|Empirical HV|Fixed-Box HV|IGD+|', '|---|---|---:|---:|---:|']
    for r in macro:text.append(f"|{r['n']}|{r['method']}|{r['empirical_hv']:.8f} ± {r['empirical_hv_between_iid_sample_sd']:.8f}|{r['fixed_box_hv']:.8f} ± {r['fixed_box_hv_between_iid_sample_sd']:.8f}|{r['igd_plus']:.8f} ± {r['igd_plus_between_iid_sample_sd']:.8f}|")
    text+=['','宏观表±为IID均值之间的样本SD（overall n=12、分规模n=4）；逐IID表±为3seed样本SD，CSV另列seed SD的跨IID均值。','详见stage2_analysis下逐seed、逐IID均值/样本SD、q分箱、推理/修复与车辆档位表。',
           'W/T/L记录在stage2_summary.json，分别列IID均值与逐run配对。',
           '旧方法较早的主机负载条件与本阶段不同；不将组合机制收益归因于单个修改，不把目标时延或拐点改善写成已证实。']
    enrich_report(summary,text,means,knees,out)
    (ROOT/'STAGE2_REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf-8')




def enrich_report(summary,text,means,knees,out):
    from stats import csv_write
    train=read(ROOT/'stage2_training/training_report.json');gate=read(ROOT/'stage2_training/Q_GATE.json')
    training={k:v for k,v in train.items() if k not in ('history','tails')}
    summary['training']=training;summary['q_behavior_gate']=gate
    summary['stage2_vs_stage1_additional_mechanisms']=['snapshot_clone_self_then_sorted_top2','unified_repair_env_decomposition_with_unweighted_tie','span_floor_0.05_instead_of0.15','three_structured_modes_route_pooling','dual_ratio_PPO','genuine_progressive_state_pool','net_return_eight_step_credit']
    summary['scope']='12previously_inspected_IIDs_x3seeds; single_bounded_training_seed; joint_mechanism_and_training_change'
    write(ROOT/'stage2_summary.json',summary);write(out/'summary.json',summary)
    text += ['', 'Stage2 r2同时改变标量化、邻居替换、三模态架构、状态池和训练信用；同表结果是组合机制比较。中止Stage2 r1单独归档，未混入本阶段。',
        f"训练实际墙钟{train['active_training_wall_seconds']:.3f}秒；meta批{train['meta_batches']}，FIT覆盖{len(train['fit_ids_optimized'])}/180。",
        '方向更新计数为完整PPO批，每批2次Adam步；逐方向更新量和实际Adam步另列training_report.json。',
        f"全部方向达到目标：{train['all_direction_targets_met']}；q行为目标通过：{gate['all_behavior_targets_met']}。",
        '', '|范围|比较基线|指标|12/4 IID W/T/L|36/12 run W/T/L|', '|---|---|---|---|---|']
    for row in summary['comparisons']:text.append(f"|{row['n']}|{row['baseline']}|{row['metric']}|{row['iid_wtl']}|{row['run_wtl']}|")
    text += ['', '每个IID的3seed均值±样本SD：', '|IID|方法|Empirical HV|FixedBox HV|IGD+|','|---|---|---:|---:|---:|']
    for row in means:
        values=[f"{row[m+'_mean']:.7f} ± {row[m+'_sample_sd']:.7f}" for m in ('empirical_hv','fixed_box_hv','igd_plus')]
        text.append('|'+row['iid']+'|'+row['method']+'|'+'|'.join(values)+'|')
    text += ['', '动作统计为步骤池化分布（非独立运行性能统计），区间右闭。',
             '|N|方法|平均q|≤2%|2–5%|5–10%|10–25%|>25%|','|---|---|---:|---:|---:|---:|---:|---:|']
    for row in summary['q_distribution']:
        vals=[f"{row['mean_q']:.3f}"]+[f"{100*row[k]:.2f}%" for k in ['le2pct','gt2_le5pct','gt5_le10pct','gt10_le25pct','gt25pct']]
        text.append('|'+str(row['n'])+'|'+row['method']+'|'+'|'.join(vals)+'|')
    text += ['', '|N|方法|平均搜索步数|inf ms/步|rep ms/步|GNN刷新率|','|---|---|---:|---:|---:|---:|']
    for row in summary['throughput']:
        refresh='缺失' if row['gnn_refresh_rate'] is None else f"{100*row['gnn_refresh_rate']:.2f}%"
        text.append(f"|{row['n']}|{row['method']}|{row['mean_steps_per_run']:.2f}|{row['inf_ms']:.6f}|{row['rep_ms']:.6f}|{refresh}|")
    text += ['', '|N|Mode|动作占比|接受率|候选消车次数|接受消车次数|首车超cap次数|','|---|---|---:|---:|---:|---:|---:|']
    for row in summary['mode_distribution']:
        rate='NA' if row['acceptance_rate'] is None else f"{100*row['acceptance_rate']:.2f}%"
        text.append(f"|{row['n']}|{row['mode']}|{100*row['proportion']:.2f}%|{rate}|{row['vehicle_reductions']}|{row['accepted_vehicle_reductions']}|{row['first_route_cap_exceptions']}|")
    text += ['', f"Meta第31批起Mid/Late组比例：{train['late_meta_mature_fraction']:.3f}；适配比例：{train['adaptation_mature_fraction']:.3f}。",'Mode2首条整车超过cap时保留整车并停止添加，次数独立统计。路径期望q的Monte Carlo估计与其标准误见Q_GATE；不把它写成精确离散期望。']
    # Vehicle-count profiles are descriptive cumulative minima; do not splice objectives of different points.
    profile=[];groups={}
    for row in knees:groups.setdefault((row['iid'],row['method'],row['vehicles']),[]).append(row)
    for (iid,method,v),rows in groups.items():
        distance=min(rows,key=lambda r:r['best_distance']);completion=min(rows,key=lambda r:r['best_completion'])
        profile.append(dict(iid=iid,method=method,vehicles=v,seeds_with_this_vehicle_count=len(rows),
            best_distance=distance['best_distance'],completion_at_best_distance=distance['completion_at_best_distance'],
            best_completion=completion['best_completion'],distance_at_best_completion=completion['distance_at_best_completion']))
    csv_write(out/'vehicle_knee_summary.csv',profile)
    text += ['', '痛点车辆档位的三seed累计最小距离及该实际解的最长时长（非逐run期望值）：',
             '|IID|方法|车辆数|有该档位的seed数|最小距离|该点最长时长|','|---|---|---:|---:|---:|---:|']
    for row in profile:
        if (row['iid']=='50-0-0' and 5<=row['vehicles']<=8) or (row['iid']=='150-0-0' and 11<=row['vehicles']<=15) or (row['iid']=='250-0-0' and 16<=row['vehicles']<=27):
            text.append(f"|{row['iid']}|{row['method']}|{row['vehicles']}|{row['seeds_with_this_vehicle_count']}|{row['best_distance']:.4f}|{row['completion_at_best_distance']:.4f}|")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(3,1,figsize=(10,10),constrained_layout=True)
    methods=list(dict.fromkeys(r['method'] for r in summary['q_distribution']));keys=['le2pct','gt2_le5pct','gt5_le10pct','gt10_le25pct','gt25pct']
    for ax,n in zip(axes,(50,150,250)):
        for j,method in enumerate(methods):
            row=next(r for r in summary['q_distribution'] if r['method']==method and r['n']==n)
            ax.bar(np.arange(5)+(j-2)*.15,[100*row[k] for k in keys],width=.15,label=method)
        ax.set_xticks(np.arange(5),['<=2%','(2,5]%','(5,10]%','(10,25]%','>25%']);ax.set_title(f'N={n}');ax.set_ylabel('Action steps (%)')
    axes[0].legend(fontsize=7,ncol=2);fig.savefig(out/'q_distribution.png',dpi=180);fig.savefig(out/'q_distribution.pdf');plt.close(fig)
    text += ['', '![实际q分布](stage2_analysis/q_distribution.png)',
        '全量车辆档位和逐seed明细见CSV；缺失档位不插值、不补造。不同阶段经验参考已分别重算，绝对值不能跨不同参考体系直接横比。']

if __name__=='__main__':main()
