"""Final human-readable empirical report; preserve the original physical-scale tables."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import csv
import json
import shutil
from pathlib import Path
import numpy as np
import psutil
psutil.Process().cpu_affinity([12,13]);psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent


def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def head(p):
    data=b'';marker=b',"records":'
    with p.open('rb') as f:
        while marker not in data:
            x=f.read(65536)
            if not x:raise ValueError(p)
            data+=x
    return json.loads(data.split(marker,1)[0]+b'}')


def main():
    summary=json.loads((OUT/'summary.json').read_text());plan=json.loads((ROOT/'plan.json').read_text())
    single=rows(OUT/'instances.csv');pooled=rows(OUT/'pooled_instances.csv');pmap={r['iid']:r for r in pooled}
    merged=[]
    for row in single:
        row=dict(row)
        for k,v in pmap[row['iid']].items():
            if k not in ('iid','n'):row['pooled_'+k]=v
        merged.append(row)
    with (OUT/'main_table.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(merged[0]));w.writeheader();w.writerows(merged)
    shutil.copy2(OUT/'pairs.csv',OUT/'paired_results.csv')
    rng=np.random.default_rng(20261028)
    for kind,data,suffix in [('single_run_mean',single,'_mean'),('pooled_archive',pooled,'')]:
        g=summary['groups']['all'][kind]
        for metric,sign in [('hv',1),('igd_plus',-1)]:
            delta=np.array([(float(r['v5_'+metric+suffix])-float(r['heuristic_'+metric+suffix]))*sign for r in data])
            boot=delta[rng.integers(0,len(delta),(5000,len(delta)))].mean(1)
            key='hv_test' if metric=='hv' else 'igd_test'
            g[key].update(mean_advantage=float(delta.mean()),iid_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),
                bootstrap_note='Equal-instance paired percentile bootstrap, 5000 resamples; conditional on one trained model; related variants limit IID independence.')
    zero_diagnostics=[]
    for iid in plan['ids']:
        ref=json.loads((OUT/f'{iid}_reference.json').read_text());zero=ref['zero_span_objectives']
        if not zero:continue
        reports=[head(ROOT/'results'/f'{iid}_seed{s}_{m}.json') for s in plan['search_seeds'] for m in plan['methods']]
        points=np.concatenate([np.asarray([p['objectives'] for p in r['archive']]) for r in reports])
        physical_scale=np.asarray(reports[0]['physical_scale']);raw_span=points.max(0)-points.min(0)
        varying=[int(j+1) for j in zero if raw_span[j]/physical_scale[j]>1e-12]
        zero_diagnostics.append(dict(iid=iid,zero_reference_objectives_1based=[int(j+1) for j in zero],
            raw_archive_variation_in_zero_reference_coordinates_1based=varying,
            raw_min=points.min(0).tolist(),raw_max=points.max(0).tolist()))
    summary['zero_span_diagnostics']=dict(instances_with_zero_reference_span=len(zero_diagnostics),
        instances_with_raw_variation_on_dropped_coordinates=sum(bool(r['raw_archive_variation_in_zero_reference_coordinates_1based']) for r in zero_diagnostics),
        details=zero_diagnostics,note='The agreed convention fixes zero-reference-span coordinates to zero. No raw archive variation was found on these coordinates in this dataset, so no observed objective differences are suppressed here. Original full physical-scale reports are retained separately.')
    summary['original_fixed_physical_macro']=json.loads((ROOT/'results/summary.json').read_text())['macro_equal_instance_means']
    summary['statistics_note']='Two-sided Wilcoxon signed-rank on 45 instance summaries (not 900 independent pairs), SciPy 1.10.1 method=auto, zero_method=wilcox. Four overall endpoints corrected by Holm. Paired-difference symmetry is assumed, not independently established; related variants limit independence. Subgroup p-values are descriptive and not covered by this four-endpoint correction. Repeated interim looks were not corrected; benchmark was previously examined. Mean bootstrap intervals and signed-rank tests target different summaries and may differ. Not an equivalence test or independent-training-seed inference.'
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    g=summary['groups']['all'];a=g['pooled_archive'];b=g['single_run_mean']
    lines=['# V5冻结元权重：45真实实例×20搜索seed最终补充报告','',
        '本次搜索完整结束：1800次运行、900个配对，两个方法共用每对的完整初始种群、预算、五目标及物理约束；单个冻结任务头共享16个偏好方向，无适配或重训。','',
        '## 用户指定主口径：20个seed合并档案','',
        '每实例分别汇集每方法20个最终档案，合并、去重并取非支配集。两方法共同经验非支配前沿提供ideal/nadir及IGD+参考；统一HV参考点(1.1)^5、不裁剪。逐实例计算，再对45实例等权平均。','',
        '| 指标 | Heuristic-ALNS | V5 | 相对变化 | 双侧配对p | Holm校正p | V5胜/平/负 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    lines.append(f"| 合并HV ↑ | {a['heuristic_hv']:.8f} | {a['v5_hv']:.8f} | +{a['hv_gain_percent']:.4f}% | {a['hv_test']['p']:.8g} | {a['hv_test']['holm_p_four_overall_endpoints']:.8g} | {'/'.join(map(str,a['hv_test']['wins_ties_losses']))} |")
    lines.append(f"| 合并IGD+ ↓ | {a['heuristic_igd_plus']:.8f} | {a['v5_igd_plus']:.8f} | 降低{a['igd_reduction_percent']:.4f}% | {a['igd_test']['p']:.8g} | {a['igd_test']['holm_p_four_overall_endpoints']:.8g} | {'/'.join(map(str,a['igd_test']['wins_ties_losses']))} |")
    lines+=['','合并HV的优势未达到0.05显著性水平；合并IGD+通过本次四项总体终点的Holm校正。未显著不代表两方法等效。以上推断受单训练seed、相关实例变体及既往中期查看限制，不作为未经查看的确证性结论。','',
        '## 补充口径：单次运行指标的20seed均值','',
        f"HV：{b['heuristic_hv']:.8f} → {b['v5_hv']:.8f}，提升{b['hv_gain_percent']:.4f}%；校正p={b['hv_test']['holm_p_four_overall_endpoints']:.8g}。",
        f"IGD+：{b['heuristic_igd_plus']:.8f} → {b['v5_igd_plus']:.8f}，降低{b['igd_reduction_percent']:.4f}%；校正p={b['igd_test']['holm_p_four_overall_endpoints']:.8g}。",
        '','这是每次运行质量的另一种汇总，与合并档案质量分开解释。每实例的均值/样本SD见main_table.csv，900配对见paired_results.csv。','',
        '## 规模分组（描述性）','',
        '| 客户数 | 合并HV相对变化 | 合并IGD+相对降低 |','|---|---:|---:|']
    for n in (50,150,250):
        x=summary['groups'][f'N{n}']['pooled_archive'];lines.append(f"| {n} | {x['hv_gain_percent']:+.4f}% | {x['igd_reduction_percent']:.4f}% |")
    lines+=['','优势主要体现在N250；本表不作经多重校正的分组显著性声明。','',
        '## 零跨度和参考集局限','',
        f"共同经验前沿存在零跨度目标的实例数：{len(zero_diagnostics)}/45；其中{summary['zero_span_diagnostics']['instances_with_raw_variation_on_dropped_coordinates']}个实例的原始档案在相应坐标仍有变化。按事先约定，这些坐标两方法统一置零、保留五维参考点。本次核对未发现置零坐标上的原始档案变化；这些目标在相应实例的全部最终档案中恒定，不提供区分度。不能声称每个实例都有五个变化目标。详情见summary.json的zero_span_diagnostics及逐实例reference.json。",
        '','经验参考来自被比较方法的最终档案，并非已知真实Pareto前沿。新增方法/seed需要统一重算尺度与参考集，不能直接与17seed的绝对数值比较。原固定物理五目标尺度主表完整保留在../results/，没有覆盖。','',
        '## 统计口径与运行记录','',summary['statistics_note'],'',
        '原始搜索2026-10-28出现进程退出，未保存Python traceback；已保存现场并经用户明确授权续跑。保留完整落盘运行，未完成的单次搜索按原seed/完整预算重做。退出根因未确认，不能把故障等待或丢失计算当作有效搜索。恢复记录见../monitor_incidents/。','',
        '真实统计参与合成生成器拟合；50-0-0已有V4.3试跑；不称盲测。一个训练seed，不测训练方差，未做部件消融，因此不能单凭本对比归因到元训练、结构特征或粒度分支。','',
        '## 完整性核验与文件','',
        'AUDIT.json记录独立逐档案完整可行性/目标重算、成对初始路线及目标、冻结源码/输入/权重文件绑定、1800原始文件/900配对/45实例×20seed及原始CSV行数核对。核验未通过时不交付为已验证结果。','',
        '- 本目录：main_table.csv、paired_results.csv、summary.json、pooled_instances.csv、REPORT.md、AUDIT.json、可复现统计/审计脚本。',
        '- ../results/：原固定物理归一化主表、原始1800份JSON、经验IGD+参考、逐运行时间曲线及完整记录。']
    (OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(overall=g,zero_span_instances=len(zero_diagnostics),zero_span_with_raw_variation=summary['zero_span_diagnostics']['instances_with_raw_variation_on_dropped_coordinates'])),flush=True)


if __name__=='__main__':main()
