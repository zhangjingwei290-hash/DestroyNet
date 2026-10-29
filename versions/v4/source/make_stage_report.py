"""Offline stage/16-direction q and paired five-search-seed development report."""
import argparse,json,csv
from pathlib import Path
import numpy as np
from solver.config import RATIOS,directions
from solver.training import STAGES,search_stage
from run import write_json


def empty_q():return dict(count=0,q_ratio_sum=0.,q_slot_counts=[0]*10)

def finish_q(block):
    n=block['count']
    return dict(**block,mean_q_over_n=block['q_ratio_sum']/n if n else None,
                q_slot_percent=[100*v/n if n else 0. for v in block['q_slot_counts']])

def combine_training_q(history):
    merged={name:{str(k):empty_q() for k in range(16)} if name=='by_direction' else {s:empty_q() for s in STAGES}
            for name in ('by_step_stage','by_source_stage','by_direction')}
    for h in history:
        if h['steps']<=0:continue
        for name,blocks in h['q_breakdown'].items():
            for label,block in blocks.items():
                target=merged[name][label];target['count']+=block['count'];target['q_ratio_sum']+=block['q_ratio_sum']
                target['q_slot_counts']=[a+b for a,b in zip(target['q_slot_counts'],block['q_slot_counts'])]
    return {name:{label:finish_q(v) for label,v in blocks.items()} for name,blocks in merged.items()}

def main(root):
    root=Path(root);s=json.loads((root/'summary.json').read_text(encoding='utf-8'))
    t=json.loads((root/'training/training_report.json').read_text(encoding='utf-8'))
    analysis=dict(training_q={},validation_q={},collection_stage_counts={},stage_balance={},per_iid=[],per_seed=[],validation_costs={},
        stage_definition='actual decision global_used_fraction: early[0,1/3), middle[1/3,2/3), late[2/3,1)',
        inference='No artificial elapsed offsets; optimizer, other-IID waits and serialization excluded from saved solver elapsed',
        comparison_scope='one trained model seed; five SEARCH seeds; nine fixed development IIDs; no training-seed variance')
    for phase,key in [('meta','history'),('adaptation','adaptation_history')]:
        history=t[key];q=combine_training_q(history);analysis['training_q'][phase]=q
        counts={str(n):{stage:0 for stage in STAGES} for n in (50,150,250)}
        for h in history:
            if h['steps']<=0:continue
            for group in h['provenance']:
                if group['optimized_group']:counts[h['iid'].split('_')[2]][group['source_stage']]+=1
        analysis['collection_stage_counts'][phase]=counts
        total=sum(v for x in counts.values() for v in x.values())
        shares={stage:sum(x[stage] for x in counts.values())/max(1,total) for stage in STAGES}
        steps=q['by_step_stage'];nt=sum(x['count'] for x in steps.values())
        step_shares={stage:steps[stage]['count']/max(1,nt) for stage in STAGES}
        analysis['stage_balance'][phase]=dict(source_group_fractions=shares,actual_step_fractions=step_shares,
            source_group_25_to_40_passed=all(.25<=v<=.4 for v in shares.values()),
            actual_step_25_to_40_passed=all(.25<=v<=.4 for v in step_shares.values()),
            source_groups=total,actual_steps=nt)
    results={}
    for f in sorted(root.glob('hemtv2*_seed*_*.json')):
        row=json.loads(f.read_text(encoding='utf-8'));results[row['iid'],row['search_seed'],row['method']]=row
    expected=len(s['search_seeds'])*len({v['iid'] for v in s['comparison']})*2
    if len(results)!=expected:raise ValueError('Incomplete paired validation result set')
    for pair in s['comparison']:
        h=results[pair['iid'],pair['search_seed'],'heuristic'];p=results[pair['iid'],pair['search_seed'],'ppo']
        if h['initial_population_objectives']!=p['initial_population_objectives']:raise ValueError('Paired initial population mismatch')
    for method in ('heuristic','ppo'):
        rows=[v for k,v in results.items() if k[2]==method];records=[r for row in rows for r in row['records']]
        n=len(records)
        analysis['validation_costs'][method]=dict(actions=n,inference_seconds=sum(v['inference_seconds'] for v in rows),
            mean_inference_ms=1000*sum(v['inference_seconds'] for v in rows)/max(1,n),
            timeouts=sum(r['timeout'] for r in records),timeout_fraction=sum(r['timeout'] for r in records)/max(1,n),
            repair_seconds=sum(r['repair_seconds'] for r in records),
            timeout_at_local_solver_cutoff=sum(r['timeout'] and r['ending_reason'] in ('local_budget','solver_budget') for r in records))
    stages={stage:empty_q() for stage in STAGES};dirs={str(k):empty_q() for k in range(16)}
    cross={str(k):{stage:empty_q() for stage in STAGES} for k in range(16)}
    for (iid,seed,method),row in results.items():
        if method!='ppo':continue
        n=int(iid.split('_')[2])
        for record in row['records']:
            stage=search_stage(record['global_used_fraction_before']);d=str(record['direction'])
            for block in (stages[stage],dirs[d],cross[d][stage]):
                block['count']+=1;block['q_ratio_sum']+=record['q']/n;block['q_slot_counts'][record['action']['q_index']]+=1
    analysis['validation_q']=dict(by_step_stage={k:finish_q(v) for k,v in stages.items()},
        by_direction={k:finish_q(v) for k,v in dirs.items()},by_direction_stage={k:{stage:finish_q(v) for stage,v in block.items()} for k,block in cross.items()})
    for iid in sorted({r['iid'] for r in s['comparison']}):
        pairs=[r for r in s['comparison'] if r['iid']==iid]
        hs=np.asarray([r['heuristic_hv'] for r in pairs]);ps=np.asarray([r['ppo_hv'] for r in pairs]);gaps=np.asarray([r['hv_gap_percent'] for r in pairs])
        analysis['per_iid'].append(dict(iid=iid,seeds=len(pairs),heuristic_hv_mean=float(hs.mean()),heuristic_hv_std=float(hs.std(ddof=1)) if len(hs)>1 else 0.,
            ppo_hv_mean=float(ps.mean()),ppo_hv_std=float(ps.std(ddof=1)) if len(ps)>1 else 0.,gap_mean_percent=float(gaps.mean()),gap_std_percent=float(gaps.std(ddof=1)) if len(gaps)>1 else 0.,
            ppo_wins=int((gaps<0).sum()),ties=int((gaps==0).sum()),ppo_losses=int((gaps>0).sum()),min_gap=float(gaps.min()),max_gap=float(gaps.max())))
    for seed in s['search_seeds']:
        pairs=[r for r in s['comparison'] if r['search_seed']==seed]
        analysis['per_seed'].append(dict(search_seed=seed,mean_gap_percent=float(np.mean([v['hv_gap_percent'] for v in pairs])),
            ppo_wins=sum(v['hv_gap_percent']<0 for v in pairs),ppo_losses=sum(v['hv_gap_percent']>0 for v in pairs)))
    matrix=np.array([[next(v['hv_gap_percent'] for v in s['comparison'] if v['iid']==iid and v['search_seed']==seed)
        for seed in s['search_seeds']] for iid in sorted({r['iid'] for r in s['comparison']})])
    rng=np.random.default_rng(30261024);indices=rng.integers(0,matrix.shape[1],size=(10000,matrix.shape[1]))
    bootstrap=matrix[:,indices].mean(axis=(0,2))
    analysis['paired_hv']=dict(mean_gap_percent=float(matrix.mean()),seed_block_bootstrap_95_percent=np.quantile(bootstrap,[.025,.975]).tolist(),
        ppo_wins=int((matrix<0).sum()),ties=int((matrix==0).sum()),ppo_losses=int((matrix>0).sum()),
        note='Search-seed block bootstrap on fixed nine IIDs; only five seeds, exploratory interval. No claim of equivalence or training-seed robustness.')
    write_json(root/'analysis.json',analysis)
    with (root/'per_seed_results.csv').open('w',encoding='utf-8',newline='') as f:
        fields=['iid','search_seed','initial_hv','heuristic_hv','ppo_hv','hv_gap_percent']
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(s['comparison'])
    gap=analysis['paired_hv'];ci=gap['seed_block_bootstrap_95_percent']
    lines=['# V4 早中晚闭环续接与五搜索种子验证','',
        f"本次总墙钟{s['wall_seconds']/60:.2f}分钟；元训练{t['meta_seconds']:.3f}秒，适配{t['adaptation_seconds']:.3f}秒；一个训练seed={s['training_seed']}，搜索seed={s['search_seeds']}。",
        '同一真实MOEA/D搜索保留种群、档案、ALNS、待评后代、代际游标、NumPy RNG和已用求解时间；跨批继续早中晚周期。主轨迹实际推进种群，辅助轨迹不进入种群。未修改目标、参考点、采样、网络、奖励或总预算。','',
        '## 覆盖与训练','',
        f"元训练：{len(t['optimized_fit_ids'])}/45 FIT，{s['meta_sampling']['batches']}批，{s['meta_sampling']['transitions']}交互，{s['meta_sampling']['optimizer_steps']}优化器更新。",
        f"适配：{len(s['adaptation_sampling']['optimized_iids'])}/45 FIT，{s['adaptation_sampling']['batches']}批，{s['adaptation_sampling']['transitions']}交互；{sum(v>0 for v in t['direction_updates'].values())}/16方向更新。",
        f"实际完成求解周期：元训练{t['meta_completed_search_cycles']}，适配{t['adapt_completed_search_cycles']}；只推进主搜索、未进入优化的阶段过渡动作分别{t['meta_traversal_only_steps']} / {t['adapt_traversal_only_steps']}，均计入实际训练墙钟。",
        '阶段按实际求解已用时间定义，优化/排队/序列化等待不伪造为搜索进度。源组比例和逐步比例分开报告：组数均衡不代表交互数自动均衡。','',
        '| 阶段 | early源组% | middle源组% | late源组% | early交互% | middle交互% | late交互% |','|---|---:|---:|---:|---:|---:|---:|']
    for phase in ('meta','adaptation'):
        b=analysis['stage_balance'][phase]
        vals=[100*b[key][st] for key in ('source_group_fractions','actual_step_fractions') for st in STAGES]
        lines.append('| '+phase+' | '+' | '.join(f'{v:.2f}' for v in vals)+' |')
    lines+=['','## 按搜索阶段的 q 粒度','',
        '| 数据 | 阶段 | 动作数 | 平均q/N | 1% | 2% | 5% | 10% | 20% | 30% | 40% | 60% | 80% | 100% |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    def qrow(label,stage,v):
        mean='—' if v['mean_q_over_n'] is None else f"{100*v['mean_q_over_n']:.2f}%"
        return f"| {label} | {stage} | {v['count']} | {mean} | "+' | '.join(f'{x:.2f}' for x in v['q_slot_percent'])+' |'
    for phase in ('meta','adaptation'):
        for stage,v in analysis['training_q'][phase]['by_step_stage'].items():lines.append(qrow(phase,stage,v))
    for stage,v in analysis['validation_q']['by_step_stage'].items():lines.append(qrow('validation',stage,v))
    lines+=['','## 按16偏好方向的 q 分布','',
        '元训练前5槽是当轮连续循环偏好任务，不等同于固定纯目标轴；完整实际权重保存在每批provenance。下表使用固定16方向的适配与验证动作。','',
        '| 数据 | 方向/权重 | 动作数 | 平均q/N | 1% | 2% | 5% | 10% | 20% | 30% | 40% | 60% | 80% | 100% |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for label,blocks in [('adaptation',analysis['training_q']['adaptation']['by_direction']),('validation',analysis['validation_q']['by_direction'])]:
        for k,w in enumerate(directions()):lines.append(qrow(label,f'{k}: {w.tolist()}',blocks[str(k)]))
    lines+=['','## 九IID×五搜索种子 HV 配对','',
        f"逐IID及seed等权平均gap={gap['mean_gap_percent']:+.4f}%（正=PPO较差）；PPO胜/平/负={gap['ppo_wins']}/{gap['ties']}/{gap['ppo_losses']}。",
        f"按搜索seed整块bootstrap的探索性95%区间：[{ci[0]:+.4f}%, {ci[1]:+.4f}%]，只适用于固定九开发IID；五seed不能代表训练种子方差。",
        '每对初始种群逐项一致；同IID采用固定物理归一化/参考点，见原始JSON；各seed档案独立，不合并为best-of-5。','',
        '| IID | 启发式HV均值±SD | PPO HV均值±SD | 平均gap% | PPO胜/负 |','|---|---:|---:|---:|---:|']
    for v in analysis['per_iid']:
        lines.append(f"| {v['iid'].removeprefix('hemtv2_train_')} | {v['heuristic_hv_mean']:.6f} ± {v['heuristic_hv_std']:.6f} | {v['ppo_hv_mean']:.6f} ± {v['ppo_hv_std']:.6f} | {v['gap_mean_percent']:+.4f} | {v['ppo_wins']}/{v['ppo_losses']} |")
    lines+=['','| 搜索seed | 平均gap% | PPO胜/负 |','|---|---:|---:|']
    for v in analysis['per_seed']:lines.append(f"| {v['search_seed']} | {v['mean_gap_percent']:+.4f} | {v['ppo_wins']}/{v['ppo_losses']} |")
    lines+=['','## 五目标与成本','', '| 目标 | 启发式平均改善% | PPO平均改善% |','|---|---:|---:|']
    for j,name in enumerate(('车辆数','总距离','makespan','等待','迟到')):
        vals=[]
        for method in ('heuristic','ppo'):
            improvements=[row['objective_component_minima']['improvement_percent'][j] for k,row in results.items() if k[2]==method and j not in row['objective_component_minima']['zero_initial_objectives']]
            vals.append('—' if not improvements else f'{np.mean(improvements):.3f}')
        lines.append(f'| {name} | '+ ' | '.join(vals)+' |')
    lines+=['','上述是档案逐目标最小值对共同初始种群逐目标最小值的改善，可能来自不同解；初始为零的目标不计算百分比。','']
    for method,cost in analysis['validation_costs'].items():
        lines.append(f"- {method}：{cost['actions']}动作，超时{100*cost['timeout_fraction']:.2f}%；推理{cost['inference_seconds']:.3f}秒，平均{cost['mean_inference_ms']:.3f}ms/动作。局搜/求解尾步超时{cost['timeout_at_local_solver_cutoff']}次。")
    lines+=['','## 结论边界与文件','',
        '阶段q均值下降或方向间差异只是行为证据；需要对应收益/消融才能归因于学会阶段与偏好。若阶段比例未达到25%～40%，或FIT/方向覆盖不足，按实际结果标记，绝不增加预算或更改阶段标签。',
        '本次只一个训练seed，五个独立搜索seed；45个IID×seed配对不是45个独立测试IID。没有使用DEV18/holdout/正式45，没有best-seed挑选。',
        '`analysis.json`：源组阶段数、逐步阶段/16方向q分布、均值q/N、验证方向×阶段交叉分布、HV逐IID/seed均值SD和配对区间；`summary.json`、`per_seed_results.csv`：原始配对汇总。',
        '`training/training_report.json`：实际覆盖、每批阶段与续接边界、方向更新；`training/latest_checkpoint.pt`：实际最新模型、Adam、RNG、求解池及预算状态；`source/`：本次源码与参数。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);main(parser.parse_args().root)
