"""Explicit IID split; input access and permission to fit are separate."""
import hashlib
import json
from pathlib import Path

import numpy as np


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                      allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def manifest(cfg, pool):
    path = Path(cfg.split_manifest)
    if not path.is_file():
        raise ValueError("An explicit 45/9 split manifest is required")
    obj = json.loads(path.read_text(encoding="utf-8"))
    fit, dev = obj["fit_ids"], obj["development_ids"]
    allowed = set(pool.cfg["allowed_ids_by_split"]["train"])
    if (len(fit) != 45 or len(set(fit)) != 45 or len(dev) != 9 or len(set(dev)) != 9
            or set(fit) & set(dev) or set(fit+dev) != allowed):
        raise ValueError("Expected the frozen disjoint 45 fit / 9 development split of TRAIN54")
    # Check the declared manifest itself, not a subset-specific pool.
    for size in (50, 150, 250):
        for dist in ("C", "R", "RC"):
            cell = lambda ids: [i for i in ids if i.split("_")[2:4] == [str(size), dist]]
            if len(cell(fit)) != 5 or len(cell(dev)) != 1:
                raise ValueError("Split must retain five fit / one development IID per cell")
            if cell(dev)!=[sorted(cell(list(allowed)))[-1]]:
                raise ValueError("Development IID differs from the frozen original 45/9 partition")
    source_binding={"config_version":pool.cfg.get("version"),
        "inputs":{i:pool.cfg["inputs"][i] for i in fit+dev},
        "boxes":{i:pool.cfg["boxes_by_instance"][i] for i in fit+dev},
        "q_values":{i:pool.cfg["q_values_by_instance"][i] for i in fit+dev}}
    return {**obj, "id": fingerprint(obj), "input_configuration_id":fingerprint(source_binding)}


def fit_ids(cfg, pool, split):
    ids = tuple(cfg.train_instance_ids) or tuple(split["fit_ids"])
    if not ids or len(set(ids)) != len(ids) or not set(ids) <= set(split["fit_ids"]):
        raise ValueError("Training may select FIT IIDs only; development IIDs cannot be fitted")
    if not set(ids) <= set(pool.ids):
        raise ValueError("FIT IDs unavailable in this input pool")
    return ids


def preference_for_episode(cfg, episode_index, rng):
    if cfg.preference is not None:
        return tuple(cfg.preference)
    return tuple([.2]*5) if episode_index % 2 == 0 else tuple(rng.dirichlet(np.ones(5)))
