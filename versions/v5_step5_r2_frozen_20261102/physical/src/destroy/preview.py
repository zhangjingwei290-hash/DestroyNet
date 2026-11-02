"""Small training-source preview. Produces states only, never teacher labels."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

from src.core.protocol import Route, Solution
from src.data_generation.sharded import _instance_from_row
from src.pilot.data import read_rows, sha
from .unified import POLICY_VERSION, RATIOS, TYPES, destroy


ROOT = Path(__file__).resolve().parents[2]


def preview(pilot_run, output, instances=1, seed=17081):
    if instances < 1:
        raise ValueError("instances must be positive")
    manifest_path = Path(pilot_run) / "pilot_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sources = manifest["splits"]["train"][:instances]
    if len(sources) != instances:
        raise ValueError("Not enough training instances")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    records, files = [], {}
    for relative in sources:
        path = Path(manifest["source"]) / relative
        digest = sha(path)
        if digest != manifest["source_hashes"][relative]:
            raise ValueError(f"Source changed: {relative}")
        files[relative] = digest
        rows = read_rows(path)
        try:
            instance_row = next(rows)
            state_row = next(row for row in rows if row["kind"] == "state")
        finally:
            rows.close()
        instance = _instance_from_row(instance_row)
        source = Solution([Route(list(r)) for r in state_row["source_solution"]])
        for kind in TYPES:
            for ratio in RATIOS:
                token = f"{seed}:{instance.instance_id}:{kind.value}:{ratio}"
                child_seed = int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "little")
                result = destroy(source, instance, kind, ratio, seed=child_seed)
                repeat = destroy(source, instance, kind, ratio, seed=child_seed)
                if result.state != repeat.state:
                    raise RuntimeError("Non-reproducible destroy")
                retained = result.state.partial_solution.get_all_customers()
                if sorted(retained + result.state.removed_customers) != list(range(1, instance.n_customers + 1)):
                    raise RuntimeError("Customer partition mismatch")
                records.append(dict(instance_id=instance.instance_id, split="train", source_file=relative,
                                    source_solution=state_row["source_solution"],
                                    partial_solution=[r.customers for r in result.state.partial_solution.routes],
                                    removed_customers=result.state.removed_customers, **result.metadata()))
    with (output / "states.jsonl").open("x", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row) + "\n")
    report = dict(policy_version=POLICY_VERSION, seed=seed, instances=instances,
                  states=len(records), split="train", contains_teacher_labels=False,
                  pilot_manifest_sha256=sha(manifest_path), source_hashes=files,
                  code_sha256=sha(Path(__file__).with_name("unified.py")),
                  closed_routes=sum(row["closed_routes"] for row in records),
                  checks="seed replay and complete customer partition passed")
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Destroy preview passed: {len(records)} states / {instances} train instances; {output}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-run", type=Path, default=ROOT / "artifacts/repairnet_pilot/pilot_50_v2")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--instances", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17081)
    args = parser.parse_args()
    output = args.output or ROOT / "artifacts/destroy_preview" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    preview(args.pilot_run, output, args.instances, args.seed)


if __name__ == "__main__":
    main()
