from dataclasses import asdict
import numpy as np
import torch
from src.core.protocol import RepairAction
from src.data_generation.sharded import _solution
from src.evaluation.vrptw_eval import is_solution_feasible, evaluate_solution
from src.pilot.config import RUN
from src.pilot.data import PilotDataset, save_json, canonical
from src.pilot.engine import enumerate_actions, apply_action
from src.pilot.model import RepairNet, tensor_features
from src.pilot.experiment import trajectory_backward, cuda_check


def validate(run=RUN):
    ds = PilotDataset(run)
    checked, steps = 0, 0
    for tid in sorted(ds.items):
        inst, state, task = ds.get(tid)
        for candidate in task['teacher_candidates']:
            partial = _solution(state['partial_solution'])
            remaining = state['removed_customers'][:]
            for target in candidate['actions']:
                action = RepairAction(**target)
                options = enumerate_actions(inst, partial, remaining)
                if not options.mask[options.actions.index(action)]:
                    raise ValueError(f'Masked teacher: {tid} {target}')
                partial, remaining = apply_action(partial, remaining, action)
                steps += 1
            if remaining or not is_solution_feasible(partial, inst):
                raise ValueError(f'Invalid final solution: {tid}')
            if canonical([r.customers for r in partial.routes]) != canonical(candidate['final_solution']):
                raise ValueError(f'Replay mismatch: {tid}')
            if not np.allclose(evaluate_solution(partial, inst).to_array(), candidate['objectives'], atol=1e-6, rtol=1e-8):
                raise ValueError(f'Objective mismatch: {tid}')
            checked += 1
        if checked % 300 == 0:
            print(f'Validated {checked} trajectories / {steps} actions', flush=True)
    cuda_check()
    model = RepairNet().cuda()
    inst, state, task = ds.get(ds.train[0])
    trajectory_backward(model, inst, state, task, 'cuda', 1)
    if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
        raise ValueError('Invalid gradients')
    save_json(ds.run/'stage_a.json', dict(status='passed', trajectories=checked, actions=steps, cuda=torch.cuda.get_device_name(0)))
    print('Stage A passed', flush=True)
