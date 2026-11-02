"""Teacher briefing from audited supplementary metrics; no experiments executed."""
from pathlib import Path
import json,shutil
import psutil
psutil.Process().cpu_affinity([14,15])
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Patch
import numpy as np
HERE=Path(__file__).resolve().parent
DATA=json.loads((HERE.parent/'summary.json').read_text())
font=Path('C:/Windows/Fonts/msyh.ttc')
fm.fontManager.addfont(str(font));family=fm.FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'font.family':family,'font.size':12,'axes.unicode_minus':False,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#9BA4AE','axes.labelcolor':'#334155','xtick.color':'#334155','ytick.color':'#526170','savefig.facecolor':'white'})
METHODS=['heuristic','alns_tuned','step5_stage2_r2'];NAMES=['MOEA/D–ALNS','MOEA/D–ALNS-T','MOEA/D–DPR（本文方法）'];COLORS=['#56B4E9','#E69F00','#009E73'];SCOPES=[50,150,250,'overall'];LABELS=['50客户','150客户','250客户','总体（45实例）']
IDX={(x['n'],x['method']):x for x in DATA['macro']}
fig=plt.figure(figsize=(16,9),facecolor='white')
fig.text(.055,.946,'MOEA/D框架下三种方法的性能比较',fontsize=24,weight='bold',color='#152C41')
fig.text(.055,.902,'45个实例 × 20个搜索种子  |  每种规模15个实例  |  各方法900次运行',fontsize=13,color='#536575')
fig.legend(handles=[Patch(facecolor=c,label=n) for c,n in zip(COLORS,NAMES)],loc='upper center',bbox_to_anchor=(.52,.875),ncol=3,frameon=False,fontsize=13,handlelength=1.5,columnspacing=2.8)
axes=[fig.add_axes([.075,.36,.405,.405]),fig.add_axes([.56,.36,.405,.405])]
for ax,metric,title,lim,ticks in [(axes[0],'empirical_hv','(a) 经验HV ↑',1.12,np.arange(0,1.01,.2)),(axes[1],'igd_plus','(b) IGD+ ↓',.25,np.arange(0,.251,.05))]:
 ax.set_title(title,loc='left',fontsize=16,pad=17,weight='bold',color='#152C41');ax.set_axisbelow(True);ax.grid(axis='y',color='#E6EBEF',linewidth=.8)
 ax.axvspan(2.53,3.48,color='#F1F5F8',zorder=0)
 for j,(method,color) in enumerate(zip(METHODS,COLORS)):
  ys=np.array([IDX[s,method][metric] for s in SCOPES]);sd=np.array([IDX[s,method][metric+'_macro_seed_sd'] for s in SCOPES]);xx=np.arange(4)+(j-1)*.235
  ax.bar(xx,ys,width=.215,color=color,edgecolor='white',linewidth=.5,zorder=3)
  ax.errorbar(xx,ys,yerr=sd,fmt='none',ecolor='#314151',elinewidth=.9,capsize=2.7,zorder=4)
  for x,y,e in zip(xx,ys,sd):ax.text(x,y+e+lim*.018,f'{y:.3f}',ha='center',va='bottom',fontsize=9.4,weight='bold' if j==2 else 'normal',color='#163B33' if j==2 else '#334155')
 ax.set_xticks(np.arange(4),LABELS,fontsize=11);ax.set_ylim(0,lim);ax.set_yticks(ticks);ax.set_xlim(-.55,3.5);ax.tick_params(length=0,pad=8)
fig.text(.075,.304,'总体配对检验：本文方法 vs 两种基线',fontsize=14,weight='bold',color='#152C41')
ax=fig.add_axes([.075,.155,.89,.125]);ax.axis('off')
body=[['MOEA/D–ALNS',r'$2.00 \times 10^{-7}$',r'$1.82 \times 10^{-7}$','HV与IGD+均显著改善'],['MOEA/D–ALNS-T','0.00571','0.00246','HV与IGD+均显著改善']]
tab=ax.table(cellText=body,colLabels=['对照方法','HV：Holm校正p','IGD+：Holm校正p','总体结论（α = 0.05）'],colWidths=[.25,.23,.23,.29],cellLoc='center',bbox=[0,0,1,1]);tab.auto_set_font_size(False);tab.set_fontsize(11)
for (rr,cc),cell in tab.get_celld().items():
 cell.set_edgecolor('white');cell.set_linewidth(2);cell.set_facecolor('#EAF0F4' if rr==0 else ('#F5F8FA' if rr==1 else '#FFFFFF'));cell.set_text_props(color='#234133' if rr>0 and cc==3 else '#334155',weight='bold' if rr==0 or cc==3 else 'normal')
fig.text(.075,.115,'误差线：20个seed各自跨实例宏观均值的样本SD；逐seed档案独立计分，实例等权汇总。',fontsize=10,color='#526170')
fig.text(.075,.083,'检验单位为45个实例的20seed均值；双侧精确配对Wilcoxon符号秩检验，Holm校正覆盖三组方法对×两指标（6项）。',fontsize=9.5,color='#526170')
fig.text(.075,.050,'口径：三方法共同经验非支配参考的补充分析；12个核心实例曾用于模型选择。固定物理盒HV较差的结果保留于完整报告。',fontsize=9.5,color='#526170')
for ext in ['png','pdf','svg']:fig.savefig(HERE/f'三方法性能对比_汇报版.{ext}',dpi=300)
# A clean paper panel without the presentation title and statistics table.
plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(10.4,4.2));fig.subplots_adjust(left=.07,right=.985,bottom=.21,top=.82,wspace=.23)
for ax,metric,title,lim in [(axs[0],'empirical_hv','(a) Empirical HV ↑',1.1),(axs[1],'igd_plus','(b) IGD+ ↓',.25)]:
 for j,(method,col) in enumerate(zip(METHODS,COLORS)):
  ax.bar(np.arange(4)+(j-1)*.24,[IDX[s,method][metric] for s in SCOPES],width=.22,yerr=[IDX[s,method][metric+'_macro_seed_sd'] for s in SCOPES],color=col,error_kw=dict(elinewidth=.8,capsize=2,ecolor='#334155'),label=NAMES[j])
 ax.set_title(title,loc='left',fontsize=13);ax.set_xticks(range(4),['50','150','250','Overall']);ax.set_ylim(0,lim);ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True);ax.set_xlabel('Number of customers / aggregate')
fig.legend(handles=[Patch(facecolor=c,label=n.replace('（本文方法）','')) for c,n in zip(COLORS,NAMES)],loc='upper center',ncol=3,frameon=False)
fig.text(.07,.035,'Mean ± SD of 20 seed-level macro averages. Shared three-method empirical reference; per-seed archives scored separately.',fontsize=8,color='#526170')
for ext in ['png','pdf','svg']:fig.savefig(HERE/f'三方法性能对比_论文版.{ext}',dpi=300)
plt.close(fig)
desk=Path('C:/Users/awe/Desktop/DPR_三方法对比_老师汇报_20261102');desk.mkdir(exist_ok=True)
for f in HERE.iterdir():
 if f.suffix in ['.png','.pdf','.svg','.py']:shutil.copy2(f,desk/f.name)
for n in ['PAPER_TABLES.md','PAPER_TABLES.tex','PAPER_RESULTS.txt','summary.json','comparisons.csv','AUDIT.json']:shutil.copy2(HERE.parent/n,desk/n)
caption='图1 三种MOEA/D方法在45个实例、20个搜索种子上的经验HV和IGD+比较。HV越大越好，IGD+越小越好。误差线表示20个seed各自跨实例宏观均值的样本标准差。每实例使用三种方法全部seed最终档案形成共同经验非支配参考，各seed独立计算指标，再对实例等权汇总。'
(desk/'汇报说明.txt').write_text(caption+'\n\nMOEA/D–ALNS：原始参数配置基线；MOEA/D–ALNS-T：开发集调参后的基线；MOEA/D–DPR：本文学习增强方法。两版ALNS均自适应选择破坏选项和修复算子。\n\n总体对比经配对符号秩及Holm6校正显著，不表示所有分规模比较均显著。本图属于三方法经验参考下的补充分析，12核心实例曾用于模型选择。固定物理盒HV口径的负向结果未覆盖，见完整原报告。\n',encoding='utf-8')
print('Saved',desk,flush=True)
