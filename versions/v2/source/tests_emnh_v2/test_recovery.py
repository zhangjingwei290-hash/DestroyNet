import json
import time
import psutil
import pytest
from moea_emnh_v2.config import Config
from moea_emnh_v2.recovery import acquire, release


def dump(p, data):
    p.write_text(json.dumps(data), encoding='utf-8')


def test_recovery_debits_lost_work_and_preserves_failure(tmp_path):
    dump(tmp_path/'binding.json', {'id':'test'})
    dump(tmp_path/'process.json', dict(pid=99999998, created=0., command=[]))
    dump(tmp_path/'worker.lock', dict(pid=99999998, created=0., nonce='old'))
    dump(tmp_path/'failure.json', dict(at=100., traceback='saved failure'))
    dump(tmp_path/'budget.json', dict(active_method='ppo', active_stage='meta', updated=90.,
        ppo=dict(meta_spent=350., adapt_spent=0.)))
    token = acquire(tmp_path, Config(), True)
    ledger = json.loads((tmp_path/'budget.json').read_text())
    assert ledger['ppo']['meta_spent'] == 360.
    assert not (tmp_path/'failure.json').exists()
    archived = list((tmp_path/'recovery').glob('*_failure.json'))
    assert len(archived) == 1 and json.loads(archived[0].read_text())['traceback'] == 'saved failure'
    assert len(list(tmp_path.glob('stale_lock_*.json'))) == 1
    release(tmp_path, token)
    assert not (tmp_path/'worker.lock').exists()


def test_live_lock_owner_refuses_duplicate_and_preserves_files(tmp_path):
    p = psutil.Process()
    dump(tmp_path/'binding.json', {'id':'test'})
    lock = dict(pid=p.pid, created=p.create_time(), nonce='must_stay')
    dump(tmp_path/'worker.lock', lock)
    with pytest.raises(RuntimeError, match='Lock owner still alive'):
        acquire(tmp_path, Config(), True)
    assert json.loads((tmp_path/'worker.lock').read_text()) == lock
    assert not (tmp_path/'process.json').exists()


def test_evaluation_failure_does_not_charge_extra_training_budget(tmp_path):
    dump(tmp_path/'binding.json', {'id':'test'})
    dump(tmp_path/'failure.json', dict(at=100., traceback='eval'))
    dump(tmp_path/'budget.json', dict(active_method='reinforce', active_stage='evaluation', updated=80.,
        reinforce=dict(meta_spent=10800., adapt_spent=3600.)))
    token = acquire(tmp_path, Config(), True)
    ledger = json.loads((tmp_path/'budget.json').read_text())
    assert ledger['reinforce'] == dict(meta_spent=10800., adapt_spent=3600.)
    release(tmp_path, token)
