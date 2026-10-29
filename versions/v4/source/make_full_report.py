"""Summarize the authorized full-budget run from saved results; no search or training."""
import argparse,json,statistics
from pathlib import Path
from run import write_json

def main(root):
    root=Path(root);summary=json.loads((root/'summary.json').read_text(encoding='utf-8'))
    train=json.loads((root/'training/training_report.json').read_text(encoding='utf-8'))
    objectives=['车辆数','总距离','makespan','等待','迟到'];ratios=['1%','2%','5%','10%','20%','30%','40%','60%','80%','100%']
    extended={'training_by_size':{},'validation':{},'initial_populations_match':True}
    for phase,key in [('meta','history'),('adaptation','adaptation_history')]:
        extended['training_by_size'][phase]={}
        for n in (50,150,250):
            rows=[h for h in train[key] if int(h['iid'].split('_')[2])==n]
            count=sum(h['transitions'] for h in rows)
            extended['training_by_size'][phase][n]=dict(batches=len(rows),transitions=count,
                timeout_fraction=sum(h['timeout_fraction']*h['transitions'] for h in rows)/max(1,count),
                positive_reward_fraction=sum(h['positive_reward_fraction']*h['transitions'] for h in rows)/max(1,count))
    for method in ('heuristic','ppo'):
        rows=[json.loads(p.read_text(encoding='utf-8')) for p in root.glob(f'hemtv2*_{method}.json')]
        records=[r for row in rows for r in row['records']]
        timed=[r for r in records if r['timeout']]
        costs=[r['repair_seconds'] for r in records if r['feasible'] and not r['timeout']]
        extended['validation'][method]=dict(actions=len(records),timeouts=len(timed),
            timeout_fraction=len(timed)/max(1,len(records)),
            timeouts_at_local_or_solver_cutoff=sum(r.get('ending_reason') in ('local_budget','solver_budget') for r in timed),
            timeout_without_insertion=sum(r['repair_counters']['inserted_customers']==0 for r in timed),
            completed_repair_median_seconds=statistics.median(costs) if costs else None,
            completed_repair_max_seconds=max(costs) if costs else None,
            inference_seconds=sum(row['inference_seconds'] for row in rows),
            objective_improvement_percent_mean=[statistics.mean([row['objective_component_minima']['improvement_percent'][j] for row in rows if j not in row['objective_component_minima']['zero_initial_objectives']]) if any(j not in row['objective_component_minima']['zero_initial_objectives'] for row in rows) else None for j in range(5)])
    for pair in summary['comparison']:
        iid=pair['iid']
        h=json.loads((root/f'{iid}_heuristic.json').read_text(encoding='utf-8'))
        p=json.loads((root/f'{iid}_ppo.json').read_text(encoding='utf-8'))
        extended['initial_populations_match'] &= h['initial_population_objectives']==p['initial_population_objectives']
    write_json(root/'analysis.json',extended)
    lines=['# V4 完整预算闭环训练与开发集比较（2026-10-24）','',
        f"运行 {summary['wall_seconds']/60:.2f} 分钟；PPO 单训练/搜索种子；9 个既有开发 IID；18 份配对结果。",
        '30 项单测通过；训练2100秒，适配上限1500秒，搜索按50/150/250客户分别10/15/20秒。未启动额外训练或追加预算。','',
        '## 覆盖与更新','',
        f"元训练：{len(train['optimized_fit_ids'])}/45 FIT，{summary['meta_sampling']['batches']} 批、{summary['meta_sampling']['transitions']} 次交互、{summary['meta_sampling']['optimizer_steps']} 次优化器更新。",
        f"适配：{len(summary['adaptation_sampling']['optimized_iids'])}/45 FIT，{summary['adaptation_sampling']['batches']} 批、{summary['adaptation_sampling']['transitions']} 次交互、{summary['adaptation_sampling']['optimizer_steps']} 次优化器更新。",
        f"有效适配方向：{sum(v>0 for v in train['direction_updates'].values())}/16；零更新方向：{[k for k,v in train['direction_updates'].items() if not v]}。",
        f"元训练实际 {train['meta_seconds']:.3f} 秒；适配实际 {train['adaptation_seconds']:.3f} 秒。方向配额按剩余时间/剩余方向动态分配，接近配额时主动让位，batch只受适配全局截止约束。未返还/追加训练预算。",
        f"元训练覆盖通过：{train['fit_coverage_passed']}；16方向有效适配通过：{train['adaptation_coverage_passed']}。覆盖通过不等于收敛或元学习贡献已证明。",'',
        '| 阶段 | N | 已优化交互 | 超时率 | 正收益步占比 |','|---|---:|---:|---:|---:|']
    for phase,label in [('meta','元训练'),('adaptation','适配')]:
        for n,vals in extended['training_by_size'][phase].items():
            lines.append(f"| {label} | {n} | {vals['transitions']} | {100*vals['timeout_fraction']:.2f}% | {100*vals['positive_reward_fraction']:.2f}% |")
    lines+=['','## 十档 q 分布','', '| 比例 | 元训练 | 适配 |','|---|---:|---:|']
    for i,r in enumerate(ratios):lines.append(f"| {r} | {summary['meta_sampling']['q_slot_percent'][i]:.2f}% | {summary['adaptation_sampling']['q_slot_percent'][i]:.2f}% |")
    lines+=['','十档分布按实际已优化交互统计。全局分布更分散本身不能证明随偏好/阶段/停滞产生有效切换，需结合逐方向与阶段统计。','',
        '## 开发集 HV','',
        f"逐IID等权平均 gap = {summary['mean_paired_hv_gap_percent']:.4f}%；正值表示 PPO 较差，负值表示 PPO 较好。初始种群逐项一致：{extended['initial_populations_match']}。",
        '采用现有每IID固定物理 affine 归一化及预先配置的参考点；未按算法分别拟合参考点。原始参考点/尺度/档案在每例JSON中。','',
        '| IID | 初始HV | 启发式HV | PPO HV | gap | PPO相对初始HV提升 |','|---|---:|---:|---:|---:|---:|']
    for row in summary['comparison']:
        lines.append(f"| {row['iid'].removeprefix('hemtv2_train_')} | {row['initial_hv']:.6f} | {row['heuristic_hv']:.6f} | {row['ppo_hv']:.6f} | {row['hv_gap_percent']:+.3f}% | {row['ppo_hv_gain_percent']:.3f}% |")
    lines+=['','## 五目标改进','',
        '以下是最终档案的逐目标最小值相对初始种群逐目标最小值的改善率，在初始值非零的IID上等权平均。五个最小值可能来自不同解，并非某一解同时达到五项；初始为零的目标保留绝对值，不混入百分比平均。','',
        '| 目标 | 启发式改善 | PPO改善 |','|---|---:|---:|']
    def fmt(v):return '—' if v is None else f'{v:.3f}%'
    for i,name in enumerate(objectives):lines.append(f"| {name} | {fmt(extended['validation']['heuristic']['objective_improvement_percent_mean'][i])} | {fmt(extended['validation']['ppo']['objective_improvement_percent_mean'][i])} |")
    lines+=['','每IID五目标原始前后值、改善绝对量和百分比见 summary.json 的 comparison；最终每方向目标见各搜索JSON的 per_direction。','',
        '## 超时与修复性能','',
        '配置 local_seconds=2，但沿用方向共享预算后的实际局搜段上限为0.625/0.9375/1.25秒。末次动作撞上局搜截止也计超时，不能把所有超时说成触发2秒修复上限。','']
    for method in ('heuristic','ppo'):
        d=extended['validation'][method]
        lines.append(f"- {method}：{d['actions']}动作，{d['timeouts']}超时（{100*d['timeout_fraction']:.2f}%）；其中{d['timeouts_at_local_or_solver_cutoff']}发生在局搜/求解截止。完成且可行的修复中位耗时{d['completed_repair_median_seconds']:.6f}秒；推理合计{d['inference_seconds']:.3f}秒。")
    lines+=['','250_C_4随机删除100客户的独立微测：向量化后四修复器约1.0–1.25秒，尚未达到0.3秒目标；不能宣称大粒度超时已消除。','',
        '## 文件','',
        '- quick_config.json：已授权参数；training/training_report.json：逐批训练与适配指标、方向继承/更新。',
        '- summary.json / analysis.json：配对结果和分规模、超时、目标统计。',
        '- 每个 hemtv2_*_ppo.json / *_heuristic.json：HV、完整档案、逐步动作、修复成本、每方向目标。',
        '- training/meta_complete.pt、direction_00.pt..direction_15.pt：实际权重；meta_unoptimized_tail.pt：资源截止未优化尾部。','',
        '单训练种子、单搜索种子、九例开发集；未设等效/非劣阈值，不把接近或胜出解释为统计证明。未读DEV18/holdout/正式45。']
    lines+=['','## 方向更新及自适应分布','', '| 方向 | 权重 | 更新批数 | 配额秒 | 实耗秒 |','|---|---|---:|---:|---:|']
    from solver.config import directions
    for k,w in enumerate(directions()):
        lines.append(f"| {k} | {w.tolist()} | {train['direction_updates'][str(k)]} | {train['direction_quota_seconds'][str(k)]:.3f} | {train['direction_spent'][str(k)]:.3f} |")
    previous=root.parent/'quick_closedloop_20261024_a'/'summary.json'
    if previous.exists():
        old=json.loads(previous.read_text(encoding='utf-8'))
        oldrows=[json.loads(f.read_text(encoding='utf-8')) for f in previous.parent.glob('hemtv2*_ppo.json')]
        oldcount=sum(len(r['records']) for r in oldrows);oldinference=sum(r['inference_seconds'] for r in oldrows)
        new=extended['validation']['ppo']
        old_ms=1000*oldinference/max(1,oldcount);new_ms=1000*new['inference_seconds']/max(1,new['actions'])
        lines+=['','## 与前次快检的效率对照','',
            f"相同九例搜索预算下，PPO总推理耗时：{oldinference:.3f} → {new['inference_seconds']:.3f}秒；交互数：{oldcount} → {new['actions']}；每动作平均推理：{old_ms:.3f} → {new_ms:.3f}毫秒（{old_ms/max(new_ms,1e-12):.2f}倍）。",
            '该数值是实际整次前向与采样耗时，包含编码器；训练、权重、q分布均变化，不把它当纯采样核的严格测速，也不预设达到39倍。']
    stage_data={}
    for phase,key in [('meta','history'),('adaptation','adaptation_history')]:
        bins={str(n):{'early':[],'middle':[],'late':[]} for n in (50,150,250)}
        for h in train[key]:
            if h['steps']<=0:continue
            for prov in h['provenance']:
                if not prov.get('optimized_group'):continue
                used=1-prov['global_remaining']/prov['global_budget']
                name='early' if used<1/3 else 'middle' if used<2/3 else 'late'
                bins[h['iid'].split('_')[2]][name].append(prov['weight'])
        stage_data[phase]={n:{stage:len(v) for stage,v in d.items()} for n,d in bins.items()}
    extended['collection_stage_counts']=stage_data
    write_json(root/'analysis.json',extended)
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(mean_gap=summary['mean_paired_hv_gap_percent'],validation=extended['validation'],training=extended['training_by_size']),ensure_ascii=False,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);main(parser.parse_args().root)
