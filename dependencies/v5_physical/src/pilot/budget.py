import time
from src.pilot.data import read_json, save_json
from src.pilot.config import CONFIG


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    """Cumulative wall time of manual training/evaluation commands, excluding idle time."""
    def __init__(self, run):
        self.file = run/'compute_budget.json'
        self.used = read_json(self.file)['seconds'] if self.file.exists() else 0.
        self.start = time.monotonic()
        self.limit = CONFIG['budget_hours']*3600

    def check(self):
        if self.used + time.monotonic()-self.start >= self.limit:
            raise BudgetExceeded('24-hour cumulative computation budget exhausted')

    def close(self):
        save_json(self.file, dict(seconds=self.used+time.monotonic()-self.start, budget_seconds=self.limit))
