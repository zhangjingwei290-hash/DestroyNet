"""Independent resumable architecture BC jobs; evaluation runs without GPU competitors."""
import argparse
import json
import random
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from src.core.protocol import RepairAction
from src.data_generation.sharded import _solution
from src.pilot.architecture_models import build_model
from src.pilot.config import CONFIG
from src.pilot.data import PilotDataset, save_json, sha
from src.pilot.engine import enumerate_actions, apply_action
from src.pilot.model import tensor_features, repair
from src.pilot.experiment import seed_all, evaluate_tasks, summary, ranking


def backward(model, inst, state, task, batch):
    if hasattr(model, 'reset_context'):
        model.reset_context()
    partial = _solution(state['partial_solution'])
    remaining = list(state['removed_customers'])
    targets = task['label']['actions']
    if len(targets) != len(remaining) or not targets:
        raise ValueError('Invalid teacher length')
    weights = torch.tensor(task['preference_weights'], device='cuda', dtype=torch.float32)
    losses = []
    for raw in targets:
        options = enumerate_actions(inst, partial, remaining)
        action = RepairAction(**raw)
        index = options.actions.index(action)
        if not options.mask[index]:
            raise ValueError(f"Masked label {task['task_id']}")
        logits = model(tensor_features(inst, partial, remaining, options, 'cuda'), weights)
        losses.append(F.cross_entropy(logits[None], torch.tensor([index], device='cuda')))
        partial, remaining = apply_action(partial, remaining, action)
    loss = torch.stack(losses).mean()
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite loss')
    (loss / batch).backward()
    value = loss.item()
    if hasattr(model, 'reset_context'):
        model.reset_context()
    return value


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=('train', 'evaluate'), required=True)
    p.add_argument('--architecture', choices=('transformer', 'mlp', 'encode_once'), required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--memory-fraction', type=float, default=.28)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    seed_all(args.seed)
    ds = PilotDataset(args.source)
    out = args.output / args.architecture / str(args.seed)
    out.mkdir(parents=True, exist_ok=True)
    binding = dict(architecture=args.architecture, seed=args.seed, epochs=args.epochs,
                   manifest_hash=sha(ds.run/'pilot_manifest.json'), config=CONFIG,
                   implementation={name: sha(Path(__file__).parent/name) for name in
                                   ('architecture_models.py', 'architecture_run.py', 'model.py', 'engine.py')})
    model = build_model(args.architecture).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    checkpoint = out / ('last.pt' if args.stage == 'train' else 'best.pt')
    epoch, best, stale = 0, None, 0
    if checkpoint.exists():
        ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if ck['binding'] != binding:
            raise ValueError('Checkpoint configuration/source changed; choose a new output name')
        model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer'])
        epoch, best, stale = ck['epoch'], ck['best'], ck['stale']
        torch.set_rng_state(ck['torch_rng']); torch.cuda.set_rng_state_all(ck['cuda_rng'])
        random.setstate(ck['random_rng']); np.random.set_state(ck['numpy_rng'])
    elif args.stage == 'evaluate':
        raise FileNotFoundError(checkpoint)

    def save(name):
        temp = out/(name+'.tmp')
        torch.save(dict(binding=binding, model=model.state_dict(), optimizer=optimizer.state_dict(),
                        epoch=epoch, best=best, stale=stale, torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all(), random_rng=random.getstate(),
                        numpy_rng=np.random.get_state()), temp)
        temp.replace(out/name)

    if args.stage == 'evaluate':
        # All three architectures finish training before the launcher enters this stage.
        model.chunk_size = 4096
        warm_id = ds.dev[0]
        inst, state, task = ds.get(warm_id)
        for _ in range(3):
            repair(model, inst, state, task['preference_weights'])
        rows = evaluate_tasks(model, ds, ds.test, out/'test.jsonl')
        shuffled = evaluate_tasks(model, ds, ds.test, out/'test_shuffled.jsonl', shuffle=True)
        save_json(out/'evaluation.json', dict(normal=summary(rows), shuffled=summary(shuffled),
                                            binding=binding, isolated_gpu=True))
        print(args.architecture, args.seed, summary(rows), flush=True)
        return
    if (out/'complete.json').exists():
        print('Already trained:', out, flush=True)
        return
    save_json(out/'binding.json', binding)
    if not (out/'initial.pt').exists():
        save('initial.pt')
    while epoch < args.epochs and stale < CONFIG['patience']:
        started = time.perf_counter()
        order = random.sample(ds.train, len(ds.train))
        total = 0.
        for start in range(0, len(order), CONFIG['batch_size']):
            batch = order[start:start+CONFIG['batch_size']]
            model.train(); optimizer.zero_grad(set_to_none=True)
            for tid in batch:
                total += backward(model, *ds.get(tid), len(batch))
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['gradient_clip'])
            optimizer.step()
            if start % (CONFIG['batch_size']*20) == 0:
                print(args.architecture, args.seed, 'epoch', epoch+1, 'tasks', start+len(batch), '/', len(order), flush=True)
        epoch += 1
        rows = evaluate_tasks(model, ds, ds.dev)
        metrics = summary(rows)
        improved = best is None or ranking(metrics) < ranking(best)
        if improved:
            best, stale = metrics, 0
            save('best.pt')
        else:
            stale += 1
        record = dict(epoch=epoch, loss=total/len(order), validation=metrics,
                      seconds=time.perf_counter()-started, timing_contended=True)
        with (out/'training.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(record)+'\n')
        save('last.pt')
        save_json(out/'progress.json', record)
        print(args.architecture, args.seed, record, flush=True)
    save_json(out/'complete.json', dict(epoch=epoch, best=best, stale=stale))


if __name__ == '__main__':
    main()
