"""Read-only physical inputs: explicit FIT/development split, no source hash locks."""
import copy,json,sys
from pathlib import Path
from types import SimpleNamespace
from .config import q_values

def project_api(root):
    root=Path(root).resolve()
    if not (root/'src/evaluation/vrptw_eval.py').is_file():raise FileNotFoundError(root)
    if str(root) not in sys.path:sys.path.insert(0,str(root))
    from src.core.protocol import Route,Solution
    from src.data_generation.sharded import _instance_from_row
    from src.evaluation.vrptw_eval import evaluate_solution,is_solution_feasible
    return SimpleNamespace(Route=Route,Solution=Solution,instance=_instance_from_row,evaluate=evaluate_solution,feasible=is_solution_feasible)

class TrainPool:
    def __init__(self,cfg):
        self.api=project_api(cfg.project_root)
        v5=cfg.version.startswith('v5')
        self.cfg=json.loads((Path(cfg.study_root)/('v5_config.json' if v5 else 'config.json')).read_text(encoding='utf-8'))
        self.split=json.loads(Path(cfg.split_manifest).read_text(encoding='utf-8'))
        if v5:
            fit,dev,test=(self.split[k] for k in ('fit_ids','validation_ids','test_ids'))
            if (len(fit),len(dev),len(test))!=(180,45,45) or len(set(fit+dev+test))!=270:
                raise ValueError('Require disjoint180FIT/45validation/45realtest')
            for key,values in (('train',fit),('validation',dev),('test',test)):
                if set(values)!=set(self.cfg['allowed_ids_by_split'][key]):raise ValueError('Split declaration mismatch')
            receipt=json.loads(Path(self.cfg['distribution_receipt']).read_text())
            if receipt['unaccepted_failures']:raise ValueError('Unaccepted distribution failure')
            self.expected_split={i:'train' for i in fit}|{i:'validation' for i in dev}
        else:
            fit,dev=self.split['fit_ids'],self.split['development_ids']
            if len(fit)!=45 or len(dev)!=9 or len(set(fit+dev))!=54 or set(fit+dev)!=set(self.cfg['allowed_ids_by_split']['train']):raise ValueError('Require disjoint original45FIT/9development')
            self.expected_split={i:'train' for i in fit+dev}
        self.ids=tuple(fit+dev);self.cache={}
    def load(self,iid):
        if iid not in self.ids:raise ValueError('IID outside FIT/development pool')
        if iid not in self.cache:
            row=json.loads(Path(self.cfg['inputs'][iid]['path']).read_text(encoding='utf-8'))
            if row['instance_id']!=iid or row.get('split')!=self.expected_split[iid]:raise ValueError('Input identity/split mismatch')
            inst=self.api.instance(row)
            source=self.api.Solution([self.api.Route(list(r)) for r in row['source_solution']])
            if not self.api.feasible(source,inst):raise ValueError('Infeasible input source')
            self.cache[iid]=(inst,row,copy.deepcopy(self.cfg['boxes_by_instance'][iid]),q_values(inst.n_customers))
        return self.cache[iid]
