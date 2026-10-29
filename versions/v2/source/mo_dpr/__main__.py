import argparse
import os

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key]="1"

from .config import Config
from .runner import run

parser = argparse.ArgumentParser(description="DPR five-objective adapter (default: bounded CPU smoke)")
parser.add_argument("--mode", choices=["smoke", "train", "evaluate"], default="smoke")
parser.add_argument("--device", default="cpu")
parser.add_argument("--smoke-steps", type=int, default=8)
parser.add_argument("--smoke-seconds", type=float, default=120)
parser.add_argument("--evaluate-steps", type=int, help="short diagnostic cap; default is the full 150-action episode")
parser.add_argument("--train-steps", type=int)
parser.add_argument("--train-seconds", type=float)
parser.add_argument("--checkpoint")
parser.add_argument("--checkpoint-manifest")
parser.add_argument("--split-manifest", default=Config().split_manifest)
parser.add_argument("--resume-dir", help="same config/source and unchanged budget only")
parser.add_argument("--fixed-q", type=int)
parser.add_argument("--seed", type=int, default=20261015)
args = parser.parse_args()
options=vars(args)
resume=options.pop("resume_dir")
run(Config(**options),output_dir=resume)
