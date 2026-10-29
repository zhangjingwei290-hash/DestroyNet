"""Core data structures and protocols for v0.17."""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
import numpy as np


class PreferenceCategory(Enum):
    """Preference distribution categories."""
    EXTREME = "extreme"  # One-hot, 30% of dataset
    BIASED_MIX = "biased_mix"  # Main weight U(0.5,0.8), 30%
    GENERAL_MIX = "general_mix"  # 0.30 <= max(w) < 0.50, 25%
    NEAR_UNIFORM = "near_uniform"  # max(w) < 0.30, 15%


class DestroyType(Enum):
    """Destroy operator types."""
    ROUTE_REPACK = "route_repack"
    WORST_EDGE_RELATED = "worst_edge_related"
    MAKESPAN_SEGMENT = "makespan_segment"
    WAITING_SEGMENT = "waiting_segment"
    TARDINESS_SEGMENT = "tardiness_segment"
    GENERIC_RANDOM = "generic_random"
    GENERIC_MIXED = "generic_mixed"


@dataclass
class VRPTWInstance:
    """VRPTW instance specification."""
    instance_id: str
    n_customers: int
    vehicle_capacity: float
    max_delay: float  # Maximum allowed customer tardiness (soft constraint)
    depot_time_window: Tuple[float, float]
    customer_coords: np.ndarray  # (n_customers, 2)
    customer_demands: np.ndarray  # (n_customers,)
    customer_time_windows: np.ndarray  # (n_customers, 2)
    customer_service_times: np.ndarray  # (n_customers,)
    distance_matrix: np.ndarray  # (n_customers+1, n_customers+1), includes depot at 0
    duration_matrix: np.ndarray  # (n_customers+1, n_customers+1)

    def __post_init__(self):
        """Validate instance data."""
        if self.n_customers <= 0:
            raise ValueError("Must have at least one customer")
        if self.vehicle_capacity <= 0:
            raise ValueError("Vehicle capacity must be positive")
        if self.max_delay < 0:
            raise ValueError("max_delay must be non-negative")
        if self.customer_coords.shape != (self.n_customers, 2):
            raise ValueError(f"customer_coords shape mismatch: expected ({self.n_customers}, 2), got {self.customer_coords.shape}")
        if self.customer_demands.shape != (self.n_customers,):
            raise ValueError(f"customer_demands shape mismatch: expected ({self.n_customers},), got {self.customer_demands.shape}")
        if self.customer_time_windows.shape != (self.n_customers, 2):
            raise ValueError(f"customer_time_windows shape mismatch: expected ({self.n_customers}, 2), got {self.customer_time_windows.shape}")
        if self.customer_service_times.shape != (self.n_customers,):
            raise ValueError(f"customer_service_times shape mismatch: expected ({self.n_customers},), got {self.customer_service_times.shape}")
        if self.distance_matrix.shape != (self.n_customers + 1, self.n_customers + 1):
            raise ValueError(f"distance_matrix shape mismatch: expected ({self.n_customers + 1}, {self.n_customers + 1}), got {self.distance_matrix.shape}")
        if self.duration_matrix.shape != (self.n_customers + 1, self.n_customers + 1):
            raise ValueError(f"duration_matrix shape mismatch: expected ({self.n_customers + 1}, {self.n_customers + 1}), got {self.duration_matrix.shape}")
        if not np.all(self.customer_demands >= 0):
            raise ValueError("Demands must be non-negative")
        if not np.all(self.customer_service_times >= 0):
            raise ValueError("Service times must be non-negative")


@dataclass
class Route:
    """A single vehicle route."""
    customers: List[int]  # Customer IDs (0 is depot, excluded from route)

    def __post_init__(self):
        """Validate route."""
        if 0 in self.customers:
            raise ValueError("Route should not contain depot (0)")
        if len(self.customers) != len(set(self.customers)):
            raise ValueError("Route contains duplicates")


@dataclass
class Solution:
    """A complete VRPTW solution."""
    routes: List[Route]

    def get_all_customers(self) -> List[int]:
        """Get all customers in solution (flattened)."""
        return [c for route in self.routes for c in route.customers]

    def __post_init__(self):
        """Validate solution."""
        all_customers = self.get_all_customers()
        if len(all_customers) != len(set(all_customers)):
            raise ValueError("Solution has duplicate customers")


@dataclass
class FiveObjectives:
    """
    Five-objective evaluation for VRPTW.

    Objective order (minimize all):
        f1: n_vehicles - Number of vehicles used
        f2: total_distance - Total travel distance across all routes
        f3: makespan - Maximum route duration (parallel completion time)
        f4: total_waiting - Total waiting time across all customers
        f5: total_tardiness - Total tardiness (lateness) across all customers
    """
    n_vehicles: int
    total_distance: float
    makespan: float
    total_waiting: float
    total_tardiness: float

    def to_array(self) -> np.ndarray:
        """Convert to numpy array [n_vehicles, total_distance, makespan, total_waiting, total_tardiness]."""
        return np.array([
            float(self.n_vehicles),
            self.total_distance,
            self.makespan,
            self.total_waiting,
            self.total_tardiness
        ], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'FiveObjectives':
        """Create from numpy array."""
        if arr.shape != (5,):
            raise ValueError(f"Expected shape (5,), got {arr.shape}")
        return cls(
            n_vehicles=int(arr[0]),
            total_distance=float(arr[1]),
            makespan=float(arr[2]),
            total_waiting=float(arr[3]),
            total_tardiness=float(arr[4])
        )


@dataclass
class DestroyedState:
    """State after destruction operator."""
    partial_solution: Solution  # Routes with some customers
    removed_customers: List[int]  # Customers to be reinserted

    def __post_init__(self):
        """Validate destroyed state."""
        if len(self.removed_customers) == 0:
            raise ValueError("Must remove at least one customer")
        if len(self.removed_customers) != len(set(self.removed_customers)):
            raise ValueError("Duplicate removals")
        partial_customers = set(self.partial_solution.get_all_customers())
        removed_set = set(self.removed_customers)
        if not partial_customers.isdisjoint(removed_set):
            raise ValueError("Customer appears in both partial and removed")


@dataclass
class RepairAction:
    """A single repair action with complete replay information."""
    customer_id: int  # Customer being inserted
    route_index: int  # Index of route to insert into (-1 for new route)
    insert_position: int  # Position within route (0 = after depot, before first customer)
    opens_new_route: bool  # True if this action creates a new route

    def __post_init__(self):
        """Validate action."""
        if self.customer_id <= 0:
            raise ValueError(f"Invalid customer_id: {self.customer_id}")
        if self.route_index < -1:
            raise ValueError(f"Invalid route_index: {self.route_index}")
        if self.insert_position < 0:
            raise ValueError(f"Invalid insert_position: {self.insert_position}")
        if self.opens_new_route and self.route_index != -1:
            raise ValueError("opens_new_route=True requires route_index=-1")


@dataclass
class ActionTrajectory:
    """Repair action sequence with full replay information."""
    actions: List[RepairAction]

    def __post_init__(self):
        """Validate trajectory."""
        if len(self.actions) == 0:
            raise ValueError("Trajectory must have at least one action")


@dataclass
class TeacherCandidate:
    """A repair candidate from one teacher."""
    teacher_name: str
    trajectory: ActionTrajectory
    final_solution: Solution
    objectives: FiveObjectives
    is_feasible: bool  # Meets hard constraints
    runtime_seconds: float

    def __post_init__(self):
        """Validate candidate."""
        if self.runtime_seconds < 0:
            raise ValueError("Runtime must be non-negative")
        if len(self.teacher_name) == 0:
            raise ValueError("Teacher name required")


@dataclass
class StateRecord:
    """Record of a destroyed state (without preference)."""
    state_id: str  # Unique identifier
    instance: VRPTWInstance
    source_solution: Solution
    source_objectives: FiveObjectives
    destroyed_state: DestroyedState
    destroy_type: DestroyType
    destroy_ratio: float  # Fraction of customers removed
    random_seed: int

    def __post_init__(self):
        """Validate state record."""
        if not (0.0 < self.destroy_ratio <= 1.0):
            raise ValueError(f"Invalid destroy ratio: {self.destroy_ratio}")
        # Verify all customers accounted for
        partial_customers = set(self.destroyed_state.partial_solution.get_all_customers())
        removed_customers = set(self.destroyed_state.removed_customers)
        all_customers = partial_customers | removed_customers
        expected_customers = set(range(1, self.instance.n_customers + 1))
        if all_customers != expected_customers:
            raise ValueError(
                f"Customer mismatch: got {all_customers}, expected {expected_customers}"
            )


@dataclass
class PreferenceTaskRecord:
    """A preference-specific repair task."""
    task_id: str  # Unique identifier
    state_id: str  # Reference to StateRecord
    preference_category: PreferenceCategory
    preference_weights: np.ndarray  # Shape (5,), must sum to 1
    teacher_candidates: List[TeacherCandidate]
    selected_teacher: Optional[str] = None
    selected_score: Optional[float] = None

    def __post_init__(self):
        """Validate preference task."""
        # Validate preference weights
        if self.preference_weights.shape != (5,):
            raise ValueError(
                f"Expected 5D preference, got {self.preference_weights.shape}"
            )
        if not np.all(np.isfinite(self.preference_weights)):
            raise ValueError("Weights must be finite")
        if not np.all(self.preference_weights >= 0):
            raise ValueError("Weights must be non-negative")
        if not np.abs(np.sum(self.preference_weights) - 1.0) < 1e-6:
            raise ValueError(
                f"Weights must sum to 1, got {np.sum(self.preference_weights)}"
            )
        if len(self.teacher_candidates) == 0:
            raise ValueError("Must have at least one teacher candidate")
        if self.selected_teacher is not None:
            candidate_names = {candidate.teacher_name for candidate in self.teacher_candidates}
            if self.selected_teacher not in candidate_names:
                raise ValueError("selected_teacher must reference a teacher candidate")
            if self.selected_score is None or not np.isfinite(self.selected_score):
                raise ValueError("selected_score must be finite when selected_teacher is set")
        elif self.selected_score is not None:
            raise ValueError("selected_score requires selected_teacher")


@dataclass
class Normalizer:
    """Normalizer for five objectives."""
    mins: np.ndarray  # Shape (5,)
    ranges: np.ndarray  # Shape (5,), uses 99th percentile - min

    def normalize(self, objectives: np.ndarray) -> np.ndarray:
        """Normalize objectives to [0, 1] range."""
        if objectives.shape[-1] != 5:
            raise ValueError(f"Expected 5 objectives, got {objectives.shape}")
        return np.clip((objectives - self.mins) / (self.ranges + 1e-8), 0.0, 1.0)

    def __post_init__(self):
        """Validate normalizer."""
        if self.mins.shape != (5,):
            raise ValueError(f"Expected shape (5,), got {self.mins.shape}")
        if self.ranges.shape != (5,):
            raise ValueError(f"Expected shape (5,), got {self.ranges.shape}")
        if not np.all(np.isfinite(self.mins)):
            raise ValueError("Mins must be finite")
        if not np.all(np.isfinite(self.ranges)):
            raise ValueError("Ranges must be finite")
        if not np.all(self.ranges > 0):
            raise ValueError("Ranges must be positive")
