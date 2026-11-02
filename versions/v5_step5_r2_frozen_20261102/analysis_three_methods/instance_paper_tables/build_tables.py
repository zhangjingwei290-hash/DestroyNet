from pathlib import Path
import csv,json,hashlib,itertools,statistics,math,shutil
import psutil
psutil.Process().cpu_affinity([14,15])
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import wilcoxon
P=Path('D:/workspace/fuxian/DPR/runs/experiments/v5_step5_multiscale_20261031/stage2_r3_full12pass_co_adapted/supplement_three_methods')
O=P/'instance_paper_tables';O.mkdir(exist_ok=True)
D=Path('C:/Users/awe/Desktop/DPR_三方法对比_老师汇报_20261102/逐实例与Wilcoxon表');D.mkdir(parents=True,exist_ok=True)
M=['heuristic','alns_tuned','step5_stage2_r2'];N=dict(zip(M,['MOEA/D–ALNS','MOEA/D–ALNS-T','MOEA/D–DPR']))
K=['empirical_hv','igd_plus'];KN={'empirical_hv':'HV','igd_plus':'IGD+'}
pairs=list(itertools.combinations(M,2))
source={f:hashlib.sha256((P/f).read_bytes()).hexdigest() for f in ['main_table.csv','per_run_metrics.csv','comparisons.csv']}
def read(f):
 with (P/f).open(encoding='utf-8-sig',newline='') as h:return list(csv.DictReader(h))
means=read('main_table.csv');runs=read('per_run_metrics.csv');old=read('comparisons.csv')
iids=sorted({r['iid'] for r in means},key=lambda x:tuple(map(int,x.split('-'))))
mi={(r['iid'],r['method']):r for r in means};ri={(r['iid'],r['method'],int(r['seed'])):r for r in runs};seeds=sorted({int(r['seed']) for r in runs})
assert len(iids)==45 and len(seeds)==20 and len(ri)==len(runs)==2700
for iid in iids:
 for m in M:
  for k in K:
   v=[float(ri[iid,m,s][k]) for s in seeds]
   assert math.isclose(statistics.mean(v),float(mi[iid,m][k+'_mean']),abs_tol=1e-12)
   assert math.isclose(statistics.stdev(v),float(mi[iid,m][k+'_sample_sd']),abs_tol=1e-12)
def signed(ds):
 nz=sorted([(abs(d),d>0) for d in ds if d!=0],key=lambda x:x[0]);n=len(nz)
 if not n:return dict(R_plus=0.,R_minus=0.,W=0.,p=1.,nonzero=0,tied_abs=False)
 ranks=[];i=0;tied=False
 while i<n:
  j=i+1
  while j<n and nz[j][0]==nz[i][0]:j+=1
  tied|=j-i>1
  r2=i+1+j
  ranks.extend([(r2,nz[t][1]) for t in range(i,j)]);i=j
 total=sum(r for r,_ in ranks);pos=sum(r for r,s in ranks if s);low=min(pos,total-pos)
 dp=[0]*(total+1);dp[0]=1;reach=0
 for r,_ in ranks:
  for t in range(reach,-1,-1):dp[t+r]+=dp[t]
  reach+=r
 p=min(1.,2*sum(dp[:low+1])/2**n)
 if not tied:
  chk=wilcoxon([d for d in ds if d!=0],alternative='two-sided',method='exact')
  assert math.isclose(chk.pvalue,p,abs_tol=1e-12)
 return dict(R_plus=pos/2,R_minus=(total-pos)/2,W=low/2,p=p,nonzero=n,tied_abs=tied)
assert signed([1,1])['p']==.5 and signed([1,-1])['p']==1 and signed([0,0])['p']==1
assert signed([1]*20)['p']==2/2**20
# One independent tied-rank enumeration check.
ds=[1,1,2,3,-3];z=signed(ds)
r2=[3,3,6,9,9];target=9
brute=sum(min(sum(r for r,b in zip(r2,bits) if b),sum(r for r,b in zip(r2,bits) if not b))<=target for bits in itertools.product([0,1],repeat=5))/32
assert z['p']==brute

def holm(rows):
 last=0.
 for j,r in enumerate(sorted(rows,key=lambda x:x['p'])):
  last=max(last,min(1.,(len(rows)-j)*r['p']));r['p_holm6']=last

def compare(scope,iid,a,b,k,ds,rawdiff):
 # Orient signed ranks so positive always means method b improves over a.
 better=ds if k=='empirical_hv' else [-d for d in ds]
 z=signed(better);z.update(scope=scope,iid=iid,baseline=N[a],target=N[b],baseline_key=a,target_key=b,metric=KN[k],mean_delta=statistics.mean(rawdiff),wins=sum(d>0 for d in better),ties=sum(d==0 for d in better),losses=sum(d<0 for d in better))
 return z
allrows=[];instance=[]
for scope in ['overall',50,150,250]:
 ids=[i for i in iids if scope=='overall' or int(i.split('-')[0])==scope];group=[]
 for a,b in pairs:
  for k in K:
   ds=[float(mi[i,b][k+'_mean'])-float(mi[i,a][k+'_mean']) for i in ids]
   r=compare(scope,'ALL',a,b,k,ds,ds);group.append(r)
   c=next(x for x in old if str(x['n'])==str(scope) and x['baseline']==a and x['target']==b and x['metric']==k)
   assert math.isclose(r['p'],float(c['p']),abs_tol=1e-12) and math.isclose(r['W'],float(c['W']),abs_tol=1e-12)
 holm(group)
 for r in group:
  c=next(x for x in old if str(x['n'])==str(scope) and x['baseline']==r['baseline_key'] and x['target']==r['target_key'] and KN[x['metric']]==r['metric'])
  assert math.isclose(r['p_holm6'],float(c['p_holm6']),abs_tol=1e-12)
 allrows+=group
for iid in iids:
 group=[]
 for a,b in pairs:
  for k in K:
   ds=[float(ri[iid,b,s][k])-float(ri[iid,a,s][k]) for s in seeds]
   group.append(compare(int(iid.split('-')[0]),iid,a,b,k,ds,ds))
 holm(group);instance+=group
for r in allrows+instance:
 r['significance']='+' if r['p_holm6']<.05 and r['R_plus']>r['R_minus'] else '-' if r['p_holm6']<.05 else '='
 r['significant_0_05']='YES' if r['p_holm6']<.05 else 'NO';r['significant_0_01']='YES' if r['p_holm6']<.01 else 'NO'
 # Significant signs should also match mean quality in this dataset.
 if r['significance']!='=':
  improvement=r['mean_delta'] if r['metric']=='HV' else -r['mean_delta']
  assert (improvement>0)==(r['significance']=='+')

def writecsv(name,rows):
 with (O/name).open('w',encoding='utf-8-sig',newline='') as h:
  w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
writecsv('Wilcoxon_总体与分规模.csv',allrows);writecsv('Wilcoxon_逐实例_完整p值.csv',instance)
ix={(r['iid'],r['target_key'],r['baseline_key'],r['metric']):r for r in instance}
wide=[]
for iid in iids:
 for k in K:
  row={'实例':iid,'客户数':int(iid.split('-')[0]),'指标':KN[k],'seed数':20}
  for m in M:
   row[N[m]+'_mean']=float(mi[iid,m][k+'_mean']);row[N[m]+'_sample_SD']=float(mi[iid,m][k+'_sample_sd'])
  for a in M[:2]:
   c=ix[iid,M[2],a,KN[k]];row['本文vs'+N[a]+'_符号']=c['significance'];row['本文vs'+N[a]+'_p']=c['p'];row['本文vs'+N[a]+'_Holm6_p']=c['p_holm6']
  wide.append(row)
writecsv('45实例_HV_IGD_均值标准差及显著性.csv',wide)

def pf(x):return f'{x:.3e}' if x<.0001 else f'{x:.6f}'
def num(x):return f'{x:g}'
def cells(r):return [r['target']+' vs '+r['baseline'],r['metric'],f"{r['wins']}/{r['ties']}/{r['losses']}",num(r['R_plus']),num(r['R_minus']),pf(r['p']),pf(r['p_holm6']),r['significant_0_05'],r['significant_0_01']]
headers=['比较（前者相对后者）','指标','胜/平/负','R+','R−','原始p','Holm6 p','α=.05','α=.01']
notes=[
'只包含最新三方法各900次运行；MOEA/D–DPR使用最终选定r2整套权重。',
'HV越大越好，IGD+越小越好。各实例使用三方法全部20 seed档案构建共同经验非支配参考，逐seed独立计分。',
'总体和分规模检验：每个实例先对20 seed取均值，再跨实例配对；总体n=45，分规模n=15。',
'逐实例检验：同一实例内按相同seed配对，n=20；均值±SD的SD是这20次运行的样本标准差。',
'采用双侧精确Wilcoxon符号秩检验，零差剔除，绝对差相同取平均秩；按非零差的符号翻转分布精确计算p。',
'差值统一按改进方向定义：HV=前者−后者；IGD+=后者−前者。因此R+表示前者更好，R−表示前者更差的秩和。',
'每一scope分别对三方法对×两个指标共6项作Holm校正；每个实例的6项也是一个独立校正家族，未对全部45实例共同校正。',
'α列根据Holm6校正p判断是否存在差异；方向看R+/R−。+/=/−表示本文方法显著更好/无显著差异/显著更差，阈值.05。',
'粗体仅表示同一实例同一指标的最优均值，不等同于显著优于所有方法。胜/平/负仅按数值方向计数，与显著+/=/−不同。',
'总体显著性不代表每个实例显著。当前属于三方法事后补充分析，12核心实例曾用于模型选择。'
]
lines=['# 最新三方法：论文统计表与45实例明细','']+[f'- {s}' for s in notes]+['']
for scope in ['overall',50,150,250]:
 lines += [f'## Wilcoxon汇总：{scope}','','|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']
 lines += ['|'+'|'.join(cells(r))+'|' for r in allrows if r['scope']==scope];lines+=['']
summary=[]
for scale in [50,150,250,'overall']:
 for k in K:
  for a in M[:2]:
   rr=[r for r in instance if r['target_key']==M[2] and r['baseline_key']==a and r['metric']==KN[k] and (scale=='overall' or r['scope']==scale)]
   summary.append(dict(scale=scale,metric=KN[k],baseline=N[a],significant_wins=sum(r['significance']=='+' for r in rr),nonsignificant=sum(r['significance']=='=' for r in rr),significant_losses=sum(r['significance']=='-' for r in rr)))
writecsv('逐实例显著胜平负汇总.csv',summary)
lines+=['## 逐实例显著胜/无显著差异/显著负','', '|规模|指标|本文方法相对|+/=/−|','|---|---|---|---|']
for r in summary:lines.append(f"|{r['scale']}|{r['metric']}|{r['baseline']}|{r['significant_wins']}/{r['nonsignificant']}/{r['significant_losses']}|")
page_data=[]
for k in K:
 for scale in [50,150,250]:
  hd=['实例']+[N[m] for m in M]+['vs ALNS','vs ALNS-T'];body=[];bold=[]
  lines += ['',f'## {scale}客户：{KN[k]} '+('↑' if k=='empirical_hv' else '↓'),'','|'+'|'.join(hd)+'|','|'+'|'.join(['---']*6)+'|']
  for iid in [i for i in iids if int(i.split('-')[0])==scale]:
   vals=[float(mi[iid,m][k+'_mean']) for m in M];best=(max(vals) if k=='empirical_hv' else min(vals));row=[iid];md=[iid]
   for j,m in enumerate(M):
    s=f"{vals[j]:.4f} ± {float(mi[iid,m][k+'_sample_sd']):.4f}";row.append(s);md.append('**'+s+'**' if vals[j]==best else s)
   signs=[ix[iid,M[2],a,KN[k]]['significance'] for a in M[:2]];row+=signs;md+=signs;body.append(row);bold.append([j+1 for j,v in enumerate(vals) if v==best]);lines.append('|'+'|'.join(md)+'|')
  page_data.append((scale,KN[k],hd,body,bold))
(O/'论文表_逐实例与Wilcoxon.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
# LaTeX tabulars with booktabs, ready to embed in a manuscript.
tex=['% Requires booktabs, graphicx, longtable, and a Unicode-capable compiler for Chinese comments.','% Directions and significance conventions: see accompanying Markdown notes.']
for scope in ['overall',50,150,250]:
 tex += ['\\begin{table*}[t]','\\centering',f'\\caption{{Pairwise Wilcoxon signed-rank tests ({scope}).}}','\\resizebox{\\textwidth}{!}{%','\\begin{tabular}{llrrrrrcc}','\\toprule',r'Comparison & Metric & $w/t/l$ & $R^+$ & $R^-$ & $p$ & $p_{\mathrm{Holm}}$ & $\alpha=.05$ & $\alpha=.01$ \\',r'\midrule']
 for r in allrows:
  if r['scope']==scope:tex.append(' & '.join(cells(r)).replace('–','--').replace('−','-')+r' \\')
 tex += [r'\bottomrule',r'\end{tabular}}',r'\end{table*}','']
for scale,k,hd,body,bold in page_data:
 tex += [r'\begin{table*}[t]',r'\centering',f'\\caption{{Per-instance {k} results on {scale}-customer instances (mean $\\pm$ sample SD over 20 search seeds).}}',r'\begin{tabular}{lccccc}',r'\toprule',r'Instance & MOEA/D--ALNS & MOEA/D--ALNS-T & MOEA/D--DPR & vs ALNS & vs ALNS-T \\',r'\midrule']
 for row,bc in zip(body,bold):
  out=[row[0]]
  for j in range(1,4):
   s=row[j].replace(' ± ',r' \pm ');out.append('$'+(r'\mathbf{'+s+'}' if j in bc else s)+'$')
  out+=['$'+s+'$' for s in row[4:]];tex.append(' & '.join(out)+r' \\')
 tex += [r'\bottomrule',r'\end{tabular}',r'\end{table*}','']
(O/'论文表_逐实例与Wilcoxon.tex').write_text('\n'.join(tex)+'\n',encoding='utf-8')
# Render standalone paper-style tables, not new experimental plots.
font=Path('C:/Windows/Fonts/msyh.ttc');fm.fontManager.addfont(str(font));family=fm.FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'font.family':family,'axes.unicode_minus':False,'pdf.fonttype':42,'savefig.facecolor':'white'})
def style(tab,fontsize):
 tab.auto_set_font_size(False);tab.set_fontsize(fontsize)
 for (rr,cc),cell in tab.get_celld().items():
  cell.set_linewidth(0);cell.set_facecolor('white');cell.set_text_props(color='#18212A',weight='bold' if rr==0 else 'normal')
  if rr==0:cell.visible_edges='BT';cell.set_edgecolor('#18212A');cell.set_linewidth(1.)
  if cc==0:cell.set_text_props(ha='left')
 last=max(rr for rr,cc in tab.get_celld())
 for cc in range(len(tab.get_celld())//(last+1)):
  cell=tab[last,cc];cell.visible_edges='B';cell.set_edgecolor('#18212A');cell.set_linewidth(1.)
fig=plt.figure(figsize=(16,5.6));fig.text(.045,.94,'三种MOEA/D方法的Wilcoxon符号秩检验',fontsize=18,weight='bold');fig.text(.045,.88,'45实例 × 20搜索seed；检验单位为45个实例各自的20-seed均值',fontsize=11)
ax=fig.add_axes([.04,.28,.92,.52]);ax.axis('off');body=[cells(r) for r in allrows if r['scope']=='overall']
tab=ax.table(cellText=body,colLabels=headers,cellLoc='center',colWidths=[.28,.06,.10,.065,.065,.10,.10,.065,.065],bbox=[0,0,1,1]);style(tab,10)
fig.text(.045,.21,r'$R^+$表示前者更好的秩和，$R^-$表示前者更差的秩和；IGD+差值方向已反转，使两指标方向一致。',fontsize=10)
fig.text(.045,.155,'双侧精确配对检验；Holm校正覆盖3组方法对 × 2指标；α列依据校正p值判断差异。',fontsize=10)
fig.text(.045,.10,'指标均基于三方法共同经验非支配参考；MOEA/D–DPR使用最终r2权重。p值保留非零精度。',fontsize=10)
fig.savefig(O/'Wilcoxon_论文汇总表.png',dpi=240,bbox_inches='tight');fig.savefig(O/'Wilcoxon_论文汇总表.pdf',bbox_inches='tight')
with PdfPages(O/'45实例_HV_IGD与Wilcoxon_论文表.pdf') as pdf:
 pdf.savefig(fig);plt.close(fig)
 for scale,k,hd,body,bold in page_data:
  fig=plt.figure(figsize=(11.69,8.27));fig.text(.055,.94,f'{scale}客户实例：{k} '+('↑' if k=='HV' else '↓'),fontsize=18,weight='bold');fig.text(.055,.895,'20个搜索seed：均值 ± 样本标准差；粗体表示最优均值',fontsize=11)
  ax=fig.add_axes([.05,.23,.90,.61]);ax.axis('off');tab=ax.table(cellText=body,colLabels=hd,cellLoc='center',colWidths=[.13,.23,.23,.23,.09,.09],bbox=[0,0,1,1]);style(tab,10)
  for rr,cols in enumerate(bold,1):
   for cc in cols:tab[rr,cc].set_text_props(weight='bold')
  fig.text(.055,.16,'最后两列：本文方法相对基线的 +（显著更好）、=（无显著差异）、−（显著更差）。',fontsize=10)
  fig.text(.055,.115,'同seed配对精确Wilcoxon检验；每实例6项Holm校正，α=.05。未跨45实例统一校正。',fontsize=10)
  fig.text(.055,.07,'三方法共同经验非支配参考；完整原始p、校正p与秩和见配套CSV。',fontsize=10)
  pdf.savefig(fig)
  fig.savefig(O/f'{scale}客户_{k.replace("+","plus")}_逐实例表.png',dpi=180,bbox_inches='tight');plt.close(fig)
receipt=dict(passed=True,scope='derived statistics and tables only; no training or solver evaluation',instances=45,seeds=20,metrics=2700,overall_and_scale_tests=24,per_instance_tests=270,existing_24_exact_p_and_W_and_holm_matched=True,all_135_mean_sd_matched=True,tied_rank_enumeration_check_passed=True,R_definition='positive means target better: target-baseline for HV; baseline-target for IGD+',holm_family='6 per scope and 6 per instance; not global correction across 45 instances',inputs_sha256=source,significant_instance_summary=summary)
assert all(hashlib.sha256((P/f).read_bytes()).hexdigest()==h for f,h in source.items())
(O/'TABLE_AUDIT.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
(O/'表注与检验口径.txt').write_text('\n'.join(notes),encoding='utf-8')
for f in O.iterdir():
 if f.is_file():shutil.copy2(f,D/f.name)
shutil.copy2(Path(__file__),O/'build_tables.py');shutil.copy2(Path(__file__),D/'build_tables.py')
print(json.dumps({'output':str(O),'desktop':str(D),'overall':[dict(comparison=r['target']+' vs '+r['baseline'],metric=r['metric'],R_plus=r['R_plus'],R_minus=r['R_minus'],p=r['p'],holm=r['p_holm6']) for r in allrows if r['scope']=='overall'],'instance_counts':[r for r in summary if r['scale']=='overall']},ensure_ascii=False,indent=2))
