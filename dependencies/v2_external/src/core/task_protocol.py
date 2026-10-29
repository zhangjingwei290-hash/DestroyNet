"""Immutable Task Protocol for v0.17."""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import hashlib
import json


@dataclass(frozen=True)
class TaskProtocol:
    """
    Immutable protocol configuration for v0.17.

    This defines the complete data generation and evaluation protocol.
    All fields are frozen to prevent modification after creation.
    """
    # Version
    schema_version: str = "v0.17.0"

    # Instance distribution
    n_train_instances: int = 2000
    n_val_instances: int = 400

    # Instance stratification (target proportions)
    instance_type_distribution: Tuple[float, float, float] = (0.33, 0.34, 0.33)  # R, C, RC
    instance_size_distribution: Tuple[float, float, float] = (0.33, 0.34, 0.33)  # 50, 150, 250

    # Task counts
    n_main_train_tasks: int = 12000  # Main training tasks (unique states)
    n_shared_state_train_tasks: int = 3000  # Shared-state tasks (300 states × 10 prefs)
    n_shared_states: int = 300
    n_preferences_per_shared_state: int = 10

    n_main_val_tasks: int = 2400  # IID validation
    n_shared_state_val_tasks: int = 600  # 60 states × 10 prefs
    n_shared_val_states: int = 60

    n_destroy_challenge_instances: int = 100
    n_destroy_types_challenge: int = 7
    n_fixed_preferences_challenge: int = 12
    n_destroy_challenge_tasks: int = 8400  # 100 × 7 × 12

    n_continuous_test_tasks: int = 3200  # 100 shared states × 32 unseen prefs
    n_unseen_preferences: int = 32

    # Destroy distribution
    destroy_ratios: Tuple[float, float, float] = (0.10, 0.20, 0.30)
    destroy_ratio_distribution: Tuple[float, float, float] = (0.25, 0.50, 0.25)

    # Destroy types (7 types)
    destroy_types: Tuple[str, ...] = (
        "route_repack",
        "worst_edge_related",
        "makespan_segment",
        "waiting_segment",
        "tardiness_segment",
        "generic_random",
        "generic_mixed"
    )

    # 6 types target-specific, 1 generic
    # Each state gets: 5 objective-related + 1 generic (split 50/50 random/mixed)

    # Preference distribution
    preference_categories: Tuple[str, ...] = (
        "extreme",
        "biased_mix",
        "general_mix",
        "near_uniform"
    )
    preference_distribution: Tuple[float, float, float, float] = (0.30, 0.30, 0.25, 0.15)

    # Shared-state preference structure (fixed for v0.17)
    # Each shared state has exactly 10 preferences with fixed structure:
    shared_state_extreme_per_state: int = 5  # One-hot for each of 5 objectives
    shared_state_biased_per_state: int = 1
    shared_state_general_per_state: int = 2
    shared_state_near_uniform_per_state: int = 2

    # Five objectives (order is fixed and immutable)
    objective_names: Tuple[str, ...] = (
        "n_vehicles",
        "total_distance",
        "makespan",
        "total_waiting",
        "total_tardiness"
    )

    # Scalarization parameters
    scalarization_method: str = "augmented_tchebycheff"
    tchebycheff_rho: float = 0.05

    # Teacher parameters
    teacher_selection_tolerance: float = 0.01  # 1% relative difference = tie

    # Curriculum training epochs
    curriculum_phase1_epochs: int = 8  # Anchors only
    curriculum_phase2_epochs: int = 12  # 50% anchor, 50% mixed
    curriculum_phase3_epochs: int = 10  # 30% anchor, 70% mixed
    total_bc_epochs: int = 30

    # PPO sampling
    ppo_total_episodes: int = 20000
    ppo_phase1_episodes: int = 5000  # 50% anchor, 50% mixed
    ppo_phase2_episodes: int = 15000  # 30% anchor, 70% mixed

    def compute_protocol_hash(self) -> str:
        """
        Compute SHA-256 hash of the entire protocol.

        This hash uniquely identifies the protocol configuration.
        Any change to protocol parameters will change this hash.
        """
        # Convert to dict, excluding the hash itself
        protocol_dict = {
            'schema_version': self.schema_version,
            'n_train_instances': self.n_train_instances,
            'n_val_instances': self.n_val_instances,
            'instance_type_distribution': self.instance_type_distribution,
            'instance_size_distribution': self.instance_size_distribution,
            'n_main_train_tasks': self.n_main_train_tasks,
            'n_shared_state_train_tasks': self.n_shared_state_train_tasks,
            'n_shared_states': self.n_shared_states,
            'n_preferences_per_shared_state': self.n_preferences_per_shared_state,
            'n_main_val_tasks': self.n_main_val_tasks,
            'n_shared_state_val_tasks': self.n_shared_state_val_tasks,
            'n_shared_val_states': self.n_shared_val_states,
            'n_destroy_challenge_instances': self.n_destroy_challenge_instances,
            'n_destroy_types_challenge': self.n_destroy_types_challenge,
            'n_fixed_preferences_challenge': self.n_fixed_preferences_challenge,
            'n_destroy_challenge_tasks': self.n_destroy_challenge_tasks,
            'n_continuous_test_tasks': self.n_continuous_test_tasks,
            'n_unseen_preferences': self.n_unseen_preferences,
            'destroy_ratios': self.destroy_ratios,
            'destroy_ratio_distribution': self.destroy_ratio_distribution,
            'destroy_types': self.destroy_types,
            'preference_categories': self.preference_categories,
            'preference_distribution': self.preference_distribution,
            'shared_state_extreme_per_state': self.shared_state_extreme_per_state,
            'shared_state_biased_per_state': self.shared_state_biased_per_state,
            'shared_state_general_per_state': self.shared_state_general_per_state,
            'shared_state_near_uniform_per_state': self.shared_state_near_uniform_per_state,
            'objective_names': self.objective_names,
            'scalarization_method': self.scalarization_method,
            'tchebycheff_rho': self.tchebycheff_rho,
            'teacher_selection_tolerance': self.teacher_selection_tolerance,
            'curriculum_phase1_epochs': self.curriculum_phase1_epochs,
            'curriculum_phase2_epochs': self.curriculum_phase2_epochs,
            'curriculum_phase3_epochs': self.curriculum_phase3_epochs,
            'total_bc_epochs': self.total_bc_epochs,
            'ppo_total_episodes': self.ppo_total_episodes,
            'ppo_phase1_episodes': self.ppo_phase1_episodes,
            'ppo_phase2_episodes': self.ppo_phase2_episodes,
        }

        json_str = json.dumps(protocol_dict, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(json_str.encode('utf-8')).hexdigest()

    def get_main_task_quotas(self, split: str = 'train') -> Dict[str, int]:
        """
        Compute main task preference quotas to achieve overall target distribution.

        The overall preference distribution target (for 15000 training tasks) is:
        - extreme: 30% = 4500
        - biased_mix: 30% = 4500
        - general_mix: 25% = 3750
        - near_uniform: 15% = 2250

        Shared-state contribution (300 states × 10 prefs = 3000 tasks):
        - extreme: 1500 (300 states × 5)
        - biased_mix: 300 (300 states × 1)
        - general_mix: 600 (300 states × 2)
        - near_uniform: 600 (300 states × 2)

        Main tasks (12000) must compensate:
        - extreme: 3000 (4500 - 1500)
        - biased_mix: 4200 (4500 - 300)
        - general_mix: 3150 (3750 - 600)
        - near_uniform: 1650 (2250 - 600)

        Args:
            split: 'train' or 'val'

        Returns:
            Dict mapping category name to main task quota
        """
        if split == 'train':
            n_main_tasks = self.n_main_train_tasks
            n_shared_states = self.n_shared_states
            total_tasks = n_main_tasks + self.n_shared_state_train_tasks
        elif split == 'val':
            n_main_tasks = self.n_main_val_tasks
            n_shared_states = self.n_shared_val_states
            total_tasks = n_main_tasks + self.n_shared_state_val_tasks
        else:
            raise ValueError(f"Unknown split: {split}")

        # Overall targets based on preference_distribution
        target_extreme = int(total_tasks * self.preference_distribution[0])
        target_biased = int(total_tasks * self.preference_distribution[1])
        target_general = int(total_tasks * self.preference_distribution[2])
        target_near_uniform = int(total_tasks * self.preference_distribution[3])

        # Shared-state contributions
        shared_extreme = n_shared_states * self.shared_state_extreme_per_state
        shared_biased = n_shared_states * self.shared_state_biased_per_state
        shared_general = n_shared_states * self.shared_state_general_per_state
        shared_near_uniform = n_shared_states * self.shared_state_near_uniform_per_state

        # Main task compensating quotas
        main_extreme = target_extreme - shared_extreme
        main_biased = target_biased - shared_biased
        main_general = target_general - shared_general
        main_near_uniform = target_near_uniform - shared_near_uniform

        # Validate
        if main_extreme < 0 or main_biased < 0 or main_general < 0 or main_near_uniform < 0:
            raise ValueError(
                f"Negative main task quota for {split}: "
                f"extreme={main_extreme}, biased={main_biased}, "
                f"general={main_general}, near_uniform={main_near_uniform}"
            )

        total_main = main_extreme + main_biased + main_general + main_near_uniform
        if total_main != n_main_tasks:
            raise ValueError(
                f"Main task quotas don't sum to {n_main_tasks} for {split}: "
                f"{main_extreme}+{main_biased}+{main_general}+{main_near_uniform}={total_main}"
            )

        return {
            'extreme': main_extreme,
            'biased_mix': main_biased,
            'general_mix': main_general,
            'near_uniform': main_near_uniform,
        }

    def __post_init__(self):
        """Validate protocol consistency and v0.17 fixed experiment scale."""
        # Check v0.17 fixed experiment scale
        if self.n_train_instances != 2000:
            raise ValueError(f"n_train_instances must be 2000 for v0.17, got {self.n_train_instances}")
        if self.n_val_instances != 400:
            raise ValueError(f"n_val_instances must be 400 for v0.17, got {self.n_val_instances}")

        # Check task counts are positive
        if self.n_main_train_tasks <= 0:
            raise ValueError(f"n_main_train_tasks must be positive, got {self.n_main_train_tasks}")
        if self.n_main_val_tasks <= 0:
            raise ValueError(f"n_main_val_tasks must be positive, got {self.n_main_val_tasks}")
        if self.n_shared_state_train_tasks <= 0:
            raise ValueError(f"n_shared_state_train_tasks must be positive, got {self.n_shared_state_train_tasks}")
        if self.n_shared_state_val_tasks <= 0:
            raise ValueError(f"n_shared_state_val_tasks must be positive, got {self.n_shared_state_val_tasks}")
        if self.n_destroy_challenge_tasks <= 0:
            raise ValueError(f"n_destroy_challenge_tasks must be positive, got {self.n_destroy_challenge_tasks}")
        if self.n_continuous_test_tasks <= 0:
            raise ValueError(f"n_continuous_test_tasks must be positive, got {self.n_continuous_test_tasks}")

        # Check v0.17 total task counts
        total_train = self.n_main_train_tasks + self.n_shared_state_train_tasks
        if total_train != 15000:
            raise ValueError(
                f"Total training tasks must be 15000 for v0.17, got {total_train} "
                f"(main={self.n_main_train_tasks} + shared={self.n_shared_state_train_tasks})"
            )

        total_val = self.n_main_val_tasks + self.n_shared_state_val_tasks
        if total_val != 3000:
            raise ValueError(
                f"Total validation tasks must be 3000 for v0.17, got {total_val} "
                f"(main={self.n_main_val_tasks} + shared={self.n_shared_state_val_tasks})"
            )

        # Check shared state consistency
        if self.n_shared_states <= 0:
            raise ValueError(f"n_shared_states must be positive, got {self.n_shared_states}")
        if self.n_preferences_per_shared_state <= 0:
            raise ValueError(f"n_preferences_per_shared_state must be positive, got {self.n_preferences_per_shared_state}")
        if self.n_shared_state_train_tasks != self.n_shared_states * self.n_preferences_per_shared_state:
            raise ValueError(
                f"Shared state train tasks mismatch: {self.n_shared_state_train_tasks} != "
                f"{self.n_shared_states} × {self.n_preferences_per_shared_state}"
            )

        if self.n_shared_val_states <= 0:
            raise ValueError(f"n_shared_val_states must be positive, got {self.n_shared_val_states}")
        if self.n_shared_state_val_tasks != self.n_shared_val_states * self.n_preferences_per_shared_state:
            raise ValueError(
                f"Shared state val tasks mismatch: {self.n_shared_state_val_tasks} != "
                f"{self.n_shared_val_states} × {self.n_preferences_per_shared_state}"
            )

        # Check shared-state preference structure (v0.17: 5+1+2+2=10)
        if (self.shared_state_extreme_per_state != 5 or
            self.shared_state_biased_per_state != 1 or
            self.shared_state_general_per_state != 2 or
            self.shared_state_near_uniform_per_state != 2):
            raise ValueError(
                f"Shared-state structure must be 5+1+2+2 for v0.17, got "
                f"{self.shared_state_extreme_per_state}+{self.shared_state_biased_per_state}+"
                f"{self.shared_state_general_per_state}+{self.shared_state_near_uniform_per_state}"
            )

        expected_prefs_per_state = (
            self.shared_state_extreme_per_state +
            self.shared_state_biased_per_state +
            self.shared_state_general_per_state +
            self.shared_state_near_uniform_per_state
        )
        if self.n_preferences_per_shared_state != expected_prefs_per_state:
            raise ValueError(
                f"n_preferences_per_shared_state must equal sum of structure components for v0.17: "
                f"{self.n_preferences_per_shared_state} != "
                f"{self.shared_state_extreme_per_state}+{self.shared_state_biased_per_state}+"
                f"{self.shared_state_general_per_state}+{self.shared_state_near_uniform_per_state} = {expected_prefs_per_state}"
            )

        # Verify main task quotas are valid
        try:
            train_quotas = self.get_main_task_quotas('train')
            val_quotas = self.get_main_task_quotas('val')
        except ValueError as e:
            raise ValueError(f"Invalid main task quotas: {e}")


        # Check destroy challenge consistency (v0.17: 100 × 7 × 12 = 8400)
        if self.n_destroy_challenge_instances != 100:
            raise ValueError(f"n_destroy_challenge_instances must be 100 for v0.17, got {self.n_destroy_challenge_instances}")
        if self.n_destroy_types_challenge != 7:
            raise ValueError(f"n_destroy_types_challenge must be 7 for v0.17, got {self.n_destroy_types_challenge}")
        if self.n_fixed_preferences_challenge != 12:
            raise ValueError(f"n_fixed_preferences_challenge must be 12 for v0.17, got {self.n_fixed_preferences_challenge}")

        expected_challenge = (
            self.n_destroy_challenge_instances *
            self.n_destroy_types_challenge *
            self.n_fixed_preferences_challenge
        )
        if self.n_destroy_challenge_tasks != expected_challenge:
            raise ValueError(
                f"Destroy challenge tasks mismatch: {self.n_destroy_challenge_tasks} != "
                f"{self.n_destroy_challenge_instances} × {self.n_destroy_types_challenge} × "
                f"{self.n_fixed_preferences_challenge} = {expected_challenge}"
            )

        # Check continuous test consistency (v0.17: 100 states × 32 unseen = 3200)
        if self.n_unseen_preferences != 32:
            raise ValueError(f"n_unseen_preferences must be 32 for v0.17, got {self.n_unseen_preferences}")
        # Infer n_continuous_test_states from n_continuous_test_tasks
        if self.n_continuous_test_tasks % self.n_unseen_preferences != 0:
            raise ValueError(
                f"n_continuous_test_tasks {self.n_continuous_test_tasks} must be divisible by "
                f"n_unseen_preferences {self.n_unseen_preferences}"
            )
        n_continuous_test_states = self.n_continuous_test_tasks // self.n_unseen_preferences
        if n_continuous_test_states != 100:
            raise ValueError(
                f"n_continuous_test_tasks must be 100 × 32 = 3200 for v0.17, "
                f"got {self.n_continuous_test_tasks} (implies {n_continuous_test_states} states)"
            )

        # Check instance type distribution
        if len(self.instance_type_distribution) != 3:
            raise ValueError(
                f"instance_type_distribution must have 3 elements (R, C, RC), "
                f"got {len(self.instance_type_distribution)}"
            )
        if not all(isinstance(v, (int, float)) and v >= 0 for v in self.instance_type_distribution):
            raise ValueError(
                f"All instance_type_distribution values must be non-negative, "
                f"got {self.instance_type_distribution}"
            )
        if not all(abs(v) < float('inf') for v in self.instance_type_distribution):
            raise ValueError(
                f"All instance_type_distribution values must be finite, "
                f"got {self.instance_type_distribution}"
            )
        if abs(sum(self.instance_type_distribution) - 1.0) > 1e-6:
            raise ValueError(
                f"instance_type_distribution must sum to 1.0, "
                f"got {sum(self.instance_type_distribution)}"
            )

        # Check instance size distribution
        if len(self.instance_size_distribution) != 3:
            raise ValueError(
                f"instance_size_distribution must have 3 elements (50, 150, 250), "
                f"got {len(self.instance_size_distribution)}"
            )
        if not all(isinstance(v, (int, float)) and v >= 0 for v in self.instance_size_distribution):
            raise ValueError(
                f"All instance_size_distribution values must be non-negative, "
                f"got {self.instance_size_distribution}"
            )
        if not all(abs(v) < float('inf') for v in self.instance_size_distribution):
            raise ValueError(
                f"All instance_size_distribution values must be finite, "
                f"got {self.instance_size_distribution}"
            )
        if abs(sum(self.instance_size_distribution) - 1.0) > 1e-6:
            raise ValueError(
                f"instance_size_distribution must sum to 1.0, "
                f"got {sum(self.instance_size_distribution)}"
            )

        # Check destroy ratios and distribution
        if len(self.destroy_ratios) == 0:
            raise ValueError("destroy_ratios must not be empty")
        if not all(isinstance(r, (int, float)) and 0 < r < 1 for r in self.destroy_ratios):
            raise ValueError(f"All destroy_ratios must be in (0, 1), got {self.destroy_ratios}")

        if len(self.destroy_ratio_distribution) != len(self.destroy_ratios):
            raise ValueError(
                f"destroy_ratio_distribution length {len(self.destroy_ratio_distribution)} "
                f"must match destroy_ratios length {len(self.destroy_ratios)}"
            )
        if not all(isinstance(v, (int, float)) and v >= 0 for v in self.destroy_ratio_distribution):
            raise ValueError(
                f"All destroy_ratio_distribution values must be non-negative, "
                f"got {self.destroy_ratio_distribution}"
            )
        if not all(abs(v) < float('inf') for v in self.destroy_ratio_distribution):
            raise ValueError(
                f"All destroy_ratio_distribution values must be finite, "
                f"got {self.destroy_ratio_distribution}"
            )
        if abs(sum(self.destroy_ratio_distribution) - 1.0) > 1e-6:
            raise ValueError(
                f"destroy_ratio_distribution must sum to 1.0, got {sum(self.destroy_ratio_distribution)}"
            )

        # Check destroy types
        if len(self.destroy_types) != self.n_destroy_types_challenge:
            raise ValueError(
                f"Number of destroy_types {len(self.destroy_types)} must match "
                f"n_destroy_types_challenge {self.n_destroy_types_challenge}"
            )

        # Check objective names
        if len(self.objective_names) != 5:
            raise ValueError(f"Must have exactly 5 objectives, got {len(self.objective_names)}")
        expected_objectives = ("n_vehicles", "total_distance", "makespan", "total_waiting", "total_tardiness")
        if self.objective_names != expected_objectives:
            raise ValueError(f"Objective names must be {expected_objectives}, got {self.objective_names}")

        # Check preference categories and distribution
        if len(self.preference_distribution) != len(self.preference_categories):
            raise ValueError(
                f"preference_distribution length {len(self.preference_distribution)} "
                f"must match preference_categories length {len(self.preference_categories)}"
            )
        if not all(isinstance(v, (int, float)) and v >= 0 for v in self.preference_distribution):
            raise ValueError(
                f"All preference_distribution values must be non-negative, "
                f"got {self.preference_distribution}"
            )
        if not all(abs(v) < float('inf') for v in self.preference_distribution):
            raise ValueError(
                f"All preference_distribution values must be finite, "
                f"got {self.preference_distribution}"
            )
        if abs(sum(self.preference_distribution) - 1.0) > 1e-6:
            raise ValueError(f"Preference distribution must sum to 1.0, got {sum(self.preference_distribution)}")

        # Check training epochs consistency
        if self.curriculum_phase1_epochs <= 0:
            raise ValueError(f"curriculum_phase1_epochs must be positive, got {self.curriculum_phase1_epochs}")
        if self.curriculum_phase2_epochs <= 0:
            raise ValueError(f"curriculum_phase2_epochs must be positive, got {self.curriculum_phase2_epochs}")
        if self.curriculum_phase3_epochs <= 0:
            raise ValueError(f"curriculum_phase3_epochs must be positive, got {self.curriculum_phase3_epochs}")
        expected_total_epochs = (
            self.curriculum_phase1_epochs +
            self.curriculum_phase2_epochs +
            self.curriculum_phase3_epochs
        )
        if self.total_bc_epochs != expected_total_epochs:
            raise ValueError(
                f"total_bc_epochs {self.total_bc_epochs} must equal sum of phase epochs "
                f"{expected_total_epochs}"
            )

        # Check PPO episodes consistency
        if self.ppo_phase1_episodes <= 0:
            raise ValueError(f"ppo_phase1_episodes must be positive, got {self.ppo_phase1_episodes}")
        if self.ppo_phase2_episodes <= 0:
            raise ValueError(f"ppo_phase2_episodes must be positive, got {self.ppo_phase2_episodes}")
        expected_total_episodes = self.ppo_phase1_episodes + self.ppo_phase2_episodes
        if self.ppo_total_episodes != expected_total_episodes:
            raise ValueError(
                f"ppo_total_episodes {self.ppo_total_episodes} must equal sum of phase episodes "
                f"{expected_total_episodes}"
            )


# Global immutable protocol instance
DEFAULT_PROTOCOL = TaskProtocol()



# Global immutable protocol instance
DEFAULT_PROTOCOL = TaskProtocol()
