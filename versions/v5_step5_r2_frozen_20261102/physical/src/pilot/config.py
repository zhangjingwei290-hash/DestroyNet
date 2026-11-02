from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'data/destroy-repair-v017/destroy-repair-v017/artifacts/trainval_selected'
RUN = ROOT / 'artifacts/repairnet_pilot/pilot_50_v1'
SEEDS = (17, 29, 43)
SPLIT_SEED = 17071
PREFERENCE_SEED = 17072
CONFIG = dict(d_model=128, layers=2, heads=4, ff=256, dropout=0.0,
              learning_rate=3e-4, weight_decay=1e-4, batch_size=8,
              gradient_clip=1.0, max_epochs=30, patience=6,
              overfit_updates=3000, eval_every=100, budget_hours=24,
              rho=0.05, bootstrap_replicates=2000)
