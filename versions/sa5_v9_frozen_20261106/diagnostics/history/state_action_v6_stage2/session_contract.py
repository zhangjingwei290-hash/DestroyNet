"""Shared training/evaluation SA transition. No solver import or dispatch."""
from dataclasses import dataclass
from math import exp, isfinite, log

STATE_FIELDS = ('attempts_remaining_fraction', 'temperature_fraction',
                'current_best_gap', 'best_stagnation_fraction',
                'last_accepted', 'last_best_gain', 'global_remaining_fraction')

def calibrated_temperature(positive_delta, probability):
    if not isfinite(positive_delta) or positive_delta <= 0 or not 0 < probability < 1:
        raise ValueError('Need positive finite deterioration and probability in (0,1)')
    return -positive_delta / log(probability)

@dataclass
class Session:
    current: float
    best: float
    temperature_start: float
    temperature_end: float
    cap: int = 5
    attempts: int = 0
    stagnation: int = 0
    last_accepted: bool = False
    last_gain: float = 0.0
    closed: bool = False

    def __post_init__(self):
        if self.cap < 1 or not all(isfinite(x) for x in (self.current,self.best,self.temperature_start,self.temperature_end)):
            raise ValueError('Invalid session')
        if self.best > self.current or not 0 < self.temperature_end <= self.temperature_start:
            raise ValueError('Invalid best/temperature ordering')

    @property
    def temperature(self):
        return self.temperature_start * (self.temperature_end/self.temperature_start)**(min(self.attempts,self.cap-1)/max(1,self.cap-1))

    def state(self, remaining_fraction):
        if not 0 <= remaining_fraction <= 1: raise ValueError('Global fraction out of range')
        return [max(0,self.cap-self.attempts)/self.cap,self.temperature/self.temperature_start,
                self.current-self.best,min(1,self.stagnation/self.cap),float(self.last_accepted),
                self.last_gain,remaining_fraction]

    def transition(self, candidate, uniform, *, feasible=True, on_time=True):
        if self.closed or self.attempts >= self.cap: raise RuntimeError('Session already ended')
        if not 0 <= uniform < 1: raise ValueError('Invalid random variate')
        if feasible and on_time and (candidate is None or not isfinite(candidate)):
            raise ValueError('Invalid feasible score')
        before=self.current; best_before=self.best; temperature=self.temperature
        probability=0.0
        if feasible and on_time:
            probability=1.0 if candidate<=before else exp(-(candidate-before)/temperature)
        accepted=feasible and on_time and uniform<probability
        if accepted: self.current=candidate
        # Archive admissibility is independent of Metropolis acceptance.
        if feasible and on_time: self.best=min(self.best,candidate)
        self.last_gain=best_before-self.best; self.last_accepted=accepted
        self.stagnation=0 if self.last_gain>0 else self.stagnation+1
        self.attempts+=1
        # An externally expired deadline is truncation, not a completed K-step episode.
        terminated=self.attempts>=self.cap and on_time
        truncated=not on_time
        self.closed=terminated or truncated
        return dict(accepted=bool(accepted),archive_eligible=bool(feasible and on_time),
                    probability=probability,temperature=temperature,before=before,current=self.current,
                    best_before=best_before,best=self.best,reward=self.last_gain,
                    terminated=terminated,truncated=truncated,attempts=self.attempts)
