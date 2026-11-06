from pathlib import Path
import json,hashlib,shutil
P=Path(__file__).resolve().parents[1];V=P/'versions/sa5_v9_frozen_20261106';BASE=Path('D:/workspace/fuxian/DPR/runs/experiments');R=BASE/'sa5_v9_scale_preference_20261105';F=P/'versions/v5_step5_r2_frozen_20261102'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
copies=json.loads((V/'evidence/COPY_PROVENANCE.json').read_text());mapped=[]
for x in json.loads((R/'SOURCE_MANIFEST.json').read_text())['files']:
 p=Path(x['windows_path'])
 if str(p).lower().startswith(str(R).lower()) or str(p).lower().startswith(str(F).lower()):continue
 if str(p).lower().startswith(str(P).lower()):mapped.append(dict(source=str(p),repository_path=p.relative_to(P).as_posix(),sha256=x['sha256']));continue
 label=p.as_posix().split('/workspace/',1)[-1];dest=V/'evidence/inherited_lock_assets'/label
 dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);assert sha(dest)==x['sha256']
 copies.append(dict(path=dest.relative_to(P).as_posix(),source=str(p),bytes=p.stat().st_size,sha256=sha(p)));mapped.append(dict(source=str(p),repository_path=dest.relative_to(P).as_posix(),sha256=x['sha256']))
dump(V/'evidence/INHERITED_LOCK_ASSETS.json',mapped);dump(V/'evidence/COPY_PROVENANCE.json',copies)
a=V/'ARTIFACTS.md';s=a.read_text();s+='\n## 继承来源锁中的额外条目\n\n原SOURCE_MANIFEST除V9与R2依赖外，还核对SA5 V2方向权重、V6 Gate、旧aligned计划和probe.py等继承条目。它们被保存在evidence/inherited_lock_assets/，路径映射见INHERITED_LOCK_ASSETS.json；这些继承条目不等于V9实际部署权重。V9实际加载的16专家和Controller仅以FREEZE_DECLARATION.json为准。split_v5.json继续引用仓库已提交的原文件。\n';a.write_text(s,encoding='utf-8')
# Append this immutable version to the repository's existing copy manifest.
manifest=json.loads((P/'ARCHIVE_MANIFEST.json').read_text());existing={x['path'] for x in manifest['files']};assert not any(x['path'] in existing for x in copies);manifest['files']+=copies;manifest.update(archive_date='2026-11-06',latest_addition=V.name);dump(P/'ARCHIVE_MANIFEST.json',manifest)
p=P/'README.md';s=p.read_text().replace('# 冻结神经破坏网络：V2、V4、V4.3、V5、Pilot与Step5 r2','# 冻结神经破坏网络：V2、V4、V4.3、V5、Pilot、Step5 R2与SA5 V9',1).replace('Step5 r2追加日期：2026-11-02。','Step5 r2追加日期：2026-11-02；SA5 V9追加日期：2026-11-06。',1)
anchor='| [Step5 r2（最终45×20）]';i=s.index('\n',s.index(anchor));s=s[:i]+'\n| [SA5 V9](versions/sa5_v9_frozen_20261106/README.md) | 六轮共享Controller联合PPO，validation45选模，9IID×3seed | 原R2 16专家＋第3轮868参数Controller |'+s[i:];s=s.replace('## 最新冻结：Step5 r2（2026-11-02）','## Step5 r2冻结（2026-11-02）')
s+='''\n## 最新冻结：SA5 V9（2026-11-06）

[版本入口](versions/sa5_v9_frozen_20261106/README.md) · [架构](versions/sa5_v9_frozen_20261106/ARCHITECTURE.md) · [训练](versions/sa5_v9_frozen_20261106/TRAINING.md) · [与上一GitHub R2差异](versions/sa5_v9_frozen_20261106/CHANGES_FROM_R2.md) · [探测实验清单](versions/sa5_v9_frozen_20261106/DIAGNOSTICS.md) · [结果与口径](versions/sa5_v9_frozen_20261106/RESULTS.md)。

V9加载原R2全部16个对应方向专家并冻结，新增22维输入、三项模式偏移＋一项Beta中心logit偏移的868参数共享Controller。fit180池实际抽18IID，名义54分钟采样，6轮/12个PPO优化步，181360条优化动作；validation45比较第3/6轮共90次运行，选第3轮。每会话最多五步SA，GPU首步批编码、CPU NumPy轻头、16修复进程同步执行。

9开发IID×3seed主表采用roulette、V8 NumPy和V9共同经验参考、逐seed独立评分。三者总体HV分别0.8427661252/0.8508122903/0.8739028170，IGD+分别0.0537265755/0.0451862889/0.0347458662。与旧R2、原生DBMT的两套历史离线比较另列参考与seed/硬件差异。训练、验证、测试、技术检查及历史探测的范围分别记录；本次归档没有新增求解运行。
''';p.write_text(s,encoding='utf-8')
for fn,detail in [('ARCHITECTURE.md','16冻结原专家，新增22→32→4共享Controller；五步SA与首步GPU缓存；详见架构字段和动作密度。'),('TRAINING.md','fit180池抽18IID，54分钟名义采样，6联合轮/12优化步/181360动作，validation45选cycle3；详见实际回报与144组合权重。'),('VERSION_CHANGES.md','上一GitHub提交b16ca23是R2。V9的q映射、模式控制、训练目标、缓存、SA及方向调度变化在逐项差异表列明。'),('RESULTS.md','9IID×3seed的三方主表；R2/DBMT历史比较各自重建参考，不跨参考混用绝对HV。'),('ARTIFACTS.md','原字节源码、16原专家与8个Controller文件、18场compact PPO样本、81测试/90验证/18训练/3冒烟/15历史DBMT紧凑报告。')]:
 p=P/'docs'/fn;s=p.read_text();s+='\n## 2026-11-06追加：SA5 V9\n\n'+detail+' [版本目录](../versions/sa5_v9_frozen_20261106/README.md)。\n';p.write_text(s,encoding='utf-8')
print('Finalized source provenance and repository entry points')
