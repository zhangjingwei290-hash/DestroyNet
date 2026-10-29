"""Preference generation and validation."""

import numpy as np
from typing import Tuple, Dict
from ..core.protocol import PreferenceCategory
from ..core.task_protocol import TaskProtocol


def generate_preference_weights(
    category: PreferenceCategory,
    rng: np.random.Generator
) -> Tuple[np.ndarray, PreferenceCategory]:
    """
    Generate preference weights according to category distribution.

    Args:
        category: Desired preference category
        rng: Random number generator

    Returns:
        Tuple of (weights, actual_category)
        actual_category may differ if generated weights don't meet criteria
    """
    if category == PreferenceCategory.EXTREME:
        # One-hot encoding, uniformly distributed across 5 objectives
        idx = rng.integers(0, 5)
        weights = np.zeros(5, dtype=np.float64)
        weights[idx] = 1.0
        return weights, PreferenceCategory.EXTREME

    elif category == PreferenceCategory.BIASED_MIX:
        # Main objective gets U(0.5, 0.8), rest distributed by Dirichlet(1)
        main_idx = rng.integers(0, 5)
        main_weight = rng.uniform(0.5, 0.8)

        # Distribute remaining weight using Dirichlet(1) ~ Uniform on simplex
        remaining_weight = 1.0 - main_weight
        other_weights = rng.dirichlet(np.ones(4)) * remaining_weight

        weights = np.zeros(5, dtype=np.float64)
        weights[main_idx] = main_weight
        other_indices = [i for i in range(5) if i != main_idx]
        weights[other_indices] = other_weights

        # Verify it meets criteria
        max_w = np.max(weights)
        if 0.5 <= max_w <= 0.8:
            return weights, PreferenceCategory.BIASED_MIX
        else:
            # Fallback classification
            return weights, _classify_weights(weights)

    elif category == PreferenceCategory.GENERAL_MIX:
        # Dirichlet(1), accept if 0.30 <= max(w) < 0.50
        max_attempts = 1000
        for _ in range(max_attempts):
            weights = rng.dirichlet(np.ones(5))
            max_w = np.max(weights)
            if 0.30 <= max_w < 0.50:
                return weights, PreferenceCategory.GENERAL_MIX

        # Fallback: use rejection sampling on constructed distribution
        # Generate until we get valid weights
        for _ in range(max_attempts):
            # Generate 5 random values
            raw = rng.uniform(0.15, 0.35, size=5)  # Each in [0.15, 0.35]
            weights = raw / np.sum(raw)
            max_w = np.max(weights)

            # Check if all constraints satisfied
            if 0.30 <= max_w < 0.50:
                return weights, PreferenceCategory.GENERAL_MIX

        # Ultimate fallback: construct deterministically
        # Set all to 0.20, then adjust one to be in valid range
        weights = np.full(5, 0.20)
        weights[0] = 0.35  # Make max = 0.35, which is in [0.30, 0.50)
        # Remaining 4 dimensions: (1.0 - 0.35) / 4 = 0.1625 each
        weights[1:] = (1.0 - 0.35) / 4
        return weights, PreferenceCategory.GENERAL_MIX

    elif category == PreferenceCategory.NEAR_UNIFORM:
        # 5% exactly uniform, 10% Dirichlet(10) with max(w) < 0.30
        if rng.random() < 1/3:  # 5% of 15% = 1/3 of near_uniform
            # Exactly uniform
            weights = np.ones(5, dtype=np.float64) / 5.0
            return weights, PreferenceCategory.NEAR_UNIFORM
        else:
            # Dirichlet(10) with max(w) < 0.30
            max_attempts = 100
            for _ in range(max_attempts):
                weights = rng.dirichlet(np.ones(5) * 10.0)
                if np.max(weights) < 0.30:
                    return weights, PreferenceCategory.NEAR_UNIFORM

            # Fallback: uniform
            weights = np.ones(5, dtype=np.float64) / 5.0
            return weights, PreferenceCategory.NEAR_UNIFORM

    else:
        raise ValueError(f"Unknown preference category: {category}")


def _classify_weights(weights: np.ndarray) -> PreferenceCategory:
    """Classify weights into a category based on their distribution."""
    max_w = np.max(weights)

    # Check for one-hot (extreme)
    if np.abs(max_w - 1.0) < 1e-6:
        return PreferenceCategory.EXTREME

    # Check ranges
    if max_w >= 0.50:
        return PreferenceCategory.BIASED_MIX
    elif max_w >= 0.30:
        return PreferenceCategory.GENERAL_MIX
    else:
        return PreferenceCategory.NEAR_UNIFORM


def validate_preference_weights(weights: np.ndarray) -> Tuple[bool, str]:
    """
    Validate preference weights according to protocol.

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check shape
    if weights.shape != (5,):
        return False, f"Expected shape (5,), got {weights.shape}"

    # Check finite
    if not np.all(np.isfinite(weights)):
        return False, "Weights must be finite"

    # Check non-negative
    if not np.all(weights >= 0):
        return False, f"Weights must be non-negative, got min={np.min(weights)}"

    # Check sum to 1
    weight_sum = np.sum(weights)
    if not np.abs(weight_sum - 1.0) < 1e-6:
        return False, f"Weights must sum to 1.0, got {weight_sum}"

    return True, ""


def generate_dataset_preferences(
    n_samples: int,
    rng: np.random.Generator,
    target_distribution: Dict[PreferenceCategory, float] = None
) -> list:
    """
    Generate preference weights for a dataset according to target distribution.

    Uses exact quota allocation: pre-compute category counts, create list, shuffle.

    Args:
        n_samples: Number of preference vectors to generate
        rng: Random number generator
        target_distribution: Dict mapping PreferenceCategory to fraction
                           Default: {EXTREME: 0.30, BIASED_MIX: 0.30,
                                    GENERAL_MIX: 0.25, NEAR_UNIFORM: 0.15}

    Returns:
        List of (weights, category) tuples
    """
    if target_distribution is None:
        target_distribution = {
            PreferenceCategory.EXTREME: 0.30,
            PreferenceCategory.BIASED_MIX: 0.30,
            PreferenceCategory.GENERAL_MIX: 0.25,
            PreferenceCategory.NEAR_UNIFORM: 0.15,
        }

    # Verify distribution sums to 1
    dist_sum = sum(target_distribution.values())
    if not np.abs(dist_sum - 1.0) < 1e-6:
        raise ValueError(f"Distribution must sum to 1.0, got {dist_sum}")

    # Compute exact quotas
    categories = list(target_distribution.keys())
    quotas = [int(n_samples * target_distribution[cat]) for cat in categories]

    # Distribute remainder to maintain exact total
    remainder = n_samples - sum(quotas)
    for i in range(remainder):
        quotas[i] += 1

    # Create category list with exact counts
    category_list = []
    for cat, quota in zip(categories, quotas):
        category_list.extend([cat] * quota)

    # Shuffle to randomize order
    rng.shuffle(category_list)

    # Generate weights for each category
    preferences = []
    for cat in category_list:
        weights, actual_cat = generate_preference_weights(cat, rng)
        is_valid, error = validate_preference_weights(weights)
        if not is_valid:
            raise ValueError(f"Generated invalid weights: {error}")
        preferences.append((weights, actual_cat))

    return preferences


def generate_main_task_preferences(
    protocol: TaskProtocol,
    split: str,
    rng: np.random.Generator
) -> list:
    """
    Generate preference weights for main tasks using TaskProtocol quotas.

    This function uses protocol.get_main_task_quotas() to obtain the compensating
    quotas that achieve overall target distribution when combined with shared-state
    preferences (5+1+2+2 structure).

    Args:
        protocol: TaskProtocol defining the experiment configuration
        split: 'train' or 'val'
        rng: Random number generator

    Returns:
        List of (weights, category) tuples for main tasks
    """
    # Get main task quotas from protocol
    quotas = protocol.get_main_task_quotas(split)

    # Create category list with exact counts
    category_list = []
    category_list.extend([PreferenceCategory.EXTREME] * quotas['extreme'])
    category_list.extend([PreferenceCategory.BIASED_MIX] * quotas['biased_mix'])
    category_list.extend([PreferenceCategory.GENERAL_MIX] * quotas['general_mix'])
    category_list.extend([PreferenceCategory.NEAR_UNIFORM] * quotas['near_uniform'])

    # Shuffle to randomize order
    rng.shuffle(category_list)

    # Generate weights for each category
    preferences = []
    for cat in category_list:
        weights, actual_cat = generate_preference_weights(cat, rng)
        is_valid, error = validate_preference_weights(weights)
        if not is_valid:
            raise ValueError(f"Generated invalid weights: {error}")
        preferences.append((weights, actual_cat))

    return preferences


def generate_shared_state_preferences(rng: np.random.Generator) -> list:
    """
    Generate the 10 fixed preferences for shared-state tasks.

    Returns:
        List of 10 (weights, category) tuples:
        - 5 EXTREME (one for each objective)
        - 1 BIASED_MIX
        - 2 GENERAL_MIX
        - 2 NEAR_UNIFORM
    """
    preferences = []

    # 5 extreme (one-hot for each objective)
    for i in range(5):
        weights = np.zeros(5, dtype=np.float64)
        weights[i] = 1.0
        preferences.append((weights, PreferenceCategory.EXTREME))

    # 1 biased_mix
    weights, cat = generate_preference_weights(PreferenceCategory.BIASED_MIX, rng)
    preferences.append((weights, cat))

    # 2 general_mix
    for _ in range(2):
        weights, cat = generate_preference_weights(PreferenceCategory.GENERAL_MIX, rng)
        preferences.append((weights, cat))

    # 2 near_uniform
    for _ in range(2):
        weights, cat = generate_preference_weights(PreferenceCategory.NEAR_UNIFORM, rng)
        preferences.append((weights, cat))

    # Validate all
    for weights, cat in preferences:
        is_valid, error = validate_preference_weights(weights)
        if not is_valid:
            raise ValueError(f"Generated invalid shared preference: {error}")

    return preferences


def compute_main_task_quotas(
    n_main_tasks: int,
    n_shared_states: int,
    target_extreme: int,
    target_biased: int,
    target_general: int,
    target_near_uniform: int,
) -> tuple:
    """
    Compute main task quotas to achieve overall target distribution.

    Shared-state contribution (300 states × 10 prefs = 3000 tasks):
    - 1500 extreme (300 states × 5)
    - 300 biased_mix (300 states × 1)
    - 600 general_mix (300 states × 2)
    - 600 near_uniform (300 states × 2)

    To reach overall targets (4500/4500/3750/2250 for 15000 total):
    Main tasks (12000) need:
    - 3000 extreme (4500 - 1500)
    - 4200 biased_mix (4500 - 300)
    - 3150 general_mix (3750 - 600)
    - 1650 near_uniform (2250 - 600)

    Args:
        n_main_tasks: Number of main tasks (12000)
        n_shared_states: Number of shared states (300)
        target_extreme: Overall target for extreme (4500)
        target_biased: Overall target for biased_mix (4500)
        target_general: Overall target for general_mix (3750)
        target_near_uniform: Overall target for near_uniform (2250)

    Returns:
        Tuple of (main_extreme, main_biased, main_general, main_near_uniform)
    """
    # Shared contribution per state: 5 extreme, 1 biased, 2 general, 2 near_uniform
    shared_extreme_per_state = 5
    shared_biased_per_state = 1
    shared_general_per_state = 2
    shared_near_uniform_per_state = 2

    # Total shared contribution
    total_shared_extreme = n_shared_states * shared_extreme_per_state
    total_shared_biased = n_shared_states * shared_biased_per_state
    total_shared_general = n_shared_states * shared_general_per_state
    total_shared_near_uniform = n_shared_states * shared_near_uniform_per_state

    # Main tasks need to compensate
    main_extreme = target_extreme - total_shared_extreme
    main_biased = target_biased - total_shared_biased
    main_general = target_general - total_shared_general
    main_near_uniform = target_near_uniform - total_shared_near_uniform

    # Validate quotas are non-negative
    if main_extreme < 0:
        raise ValueError(
            f"Cannot achieve target_extreme {target_extreme}: "
            f"shared states already contribute {total_shared_extreme}"
        )
    if main_biased < 0:
        raise ValueError(
            f"Cannot achieve target_biased {target_biased}: "
            f"shared states already contribute {total_shared_biased}"
        )
    if main_general < 0:
        raise ValueError(
            f"Cannot achieve target_general {target_general}: "
            f"shared states already contribute {total_shared_general}"
        )
    if main_near_uniform < 0:
        raise ValueError(
            f"Cannot achieve target_near_uniform {target_near_uniform}: "
            f"shared states already contribute {total_shared_near_uniform}"
        )

    # Verify totals
    total_main = main_extreme + main_biased + main_general + main_near_uniform
    if total_main != n_main_tasks:
        raise ValueError(
            f"Main task quotas don't sum to {n_main_tasks}: "
            f"{main_extreme}+{main_biased}+{main_general}+{main_near_uniform}={total_main}"
        )

    return (main_extreme, main_biased, main_general, main_near_uniform)

