"""Windows default GBK must not reinterpret UTF-8 artifact bytes."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
from mo_dpr.artifacts import save_json


def test_completed_training_utf8_report_loads_without_any_optimization(tmp_path,monkeypatch):
    import moea_emnh_v2.training as training
    import moea_emnh_v2.runner as runner
    expected=dict(method='ppo',note='经批准的恢复：只加载冻结权重，不再训练',training_coverage_passed=False)
    save_json(tmp_path/'training_report.json',expected)
    sentinel=object()
    monkeypatch.setattr(runner,'load_directions',lambda *args:(sentinel,))
    def forbidden(*args,**kwargs):raise AssertionError('Completed training must not create/update policy')
    monkeypatch.setattr(training,'Policy',forbidden)
    models,report=training.train_method(None,None,SimpleNamespace(binding_id='test'),'ppo',tmp_path,forbidden)
    assert models==(sentinel,) and report==expected


def test_all_v2_text_readers_declare_utf8():
    package=Path(__file__).resolve().parents[1]/'moea_emnh_v2'
    for source in package.glob('*.py'):
        for call in ast.walk(ast.parse(source.read_text(encoding='utf-8'))):
            if isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute) and call.func.attr=='read_text':
                assert any(k.arg=='encoding' and isinstance(k.value,ast.Constant) and k.value.value=='utf-8' for k in call.keywords),str(source)


def test_utf8_failure_can_resume_without_charging_evaluation_to_training(tmp_path):
    from moea_emnh_v2.recovery import acquire,release
    from moea_emnh_v2.config import Config
    save_json(tmp_path/'binding.json',dict(id='test'))
    save_json(tmp_path/'failure.json',dict(at=100.,traceback='保存结果失败，等待受控恢复'))
    save_json(tmp_path/'budget.json',dict(active_stage='evaluation',active_method='reinforce',updated=90.,
        reinforce=dict(meta_spent=10797.,adapt_spent=3598.)))
    token=acquire(tmp_path,Config(),True)
    ledger=json.loads((tmp_path/'budget.json').read_text(encoding='utf-8'))
    assert ledger['reinforce']==dict(meta_spent=10797.,adapt_spent=3598.)
    failures=list((tmp_path/'recovery').glob('*_failure.json'))
    assert json.loads(failures[0].read_text(encoding='utf-8'))['traceback']=='保存结果失败，等待受控恢复'
    release(tmp_path,token)
