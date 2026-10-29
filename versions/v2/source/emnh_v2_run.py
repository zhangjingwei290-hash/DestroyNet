"""PyCharm / background entry for the corrected five-task experiment only."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('smoke', 'campaign'), default='smoke')
    p.add_argument('--root', required=True)
    p.add_argument('--calibration')
    p.add_argument('--meta-seconds', type=float, default=10800.)
    p.add_argument('--adaptation-seconds', type=float, default=3600.)
    p.add_argument('--legacy-ppo-seconds', type=float, default=335.0391342)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--log', action='store_true')
    args = p.parse_args()
    from moea_emnh_v2.config import Config
    from moea_emnh_v2.runner import calibration, campaign
    cfg = Config(meta_seconds=args.meta_seconds, adaptation_seconds=args.adaptation_seconds,
                 legacy_ppo_seconds=args.legacy_ppo_seconds)
    if args.mode == 'smoke':
        calibration(args.root, cfg)
    else:
        if not args.calibration:
            p.error('--calibration required for campaign')
        root = Path(args.root); root.mkdir(parents=True, exist_ok=True)
        if args.log:
            with (root/'worker.log').open('a', encoding='utf-8', buffering=1) as log:
                sys.stdout = sys.stderr = log
                campaign(root, cfg, args.calibration, args.resume)
        else:
            campaign(root, cfg, args.calibration, args.resume)


if __name__ == '__main__':
    main()
