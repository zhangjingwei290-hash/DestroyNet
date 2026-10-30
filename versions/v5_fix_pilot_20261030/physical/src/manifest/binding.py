"""Manifest and checkpoint binding with hash verification."""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import numpy as np

from ..core.protocol import (
    StateRecord,
    PreferenceTaskRecord,
    Normalizer,
    VRPTWInstance,
)
from ..core.task_protocol import TaskProtocol
from ..core.replay import replay_trajectory, validate_trajectory_replay
from ..evaluation.vrptw_eval import evaluate_solution, is_solution_feasible


def compute_content_hash(content: Any) -> str:
    """
    Compute SHA-256 hash of content.

    Args:
        content: Content to hash (will be JSON serialized)

    Returns:
        Hex string of hash
    """
    # Convert to JSON string with sorted keys for reproducibility
    if isinstance(content, (np.ndarray, np.generic)):
        content = content.tolist()

    json_str = json.dumps(content, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(json_str.encode('utf-8')).hexdigest()


def compute_normalizer_hash(normalizer: Normalizer) -> str:
    """Compute hash of normalizer parameters."""
    content = {
        'mins': normalizer.mins.tolist(),
        'ranges': normalizer.ranges.tolist(),
    }
    return compute_content_hash(content)


def compute_instance_hash(instance: VRPTWInstance) -> str:
    """Compute hash of VRPTW instance."""
    content = {
        'instance_id': instance.instance_id,
        'n_customers': instance.n_customers,
        'vehicle_capacity': float(instance.vehicle_capacity),
        'max_delay': float(instance.max_delay),
        'depot_time_window': instance.depot_time_window,
        'customer_coords': instance.customer_coords.tolist(),
        'customer_demands': instance.customer_demands.tolist(),
        'customer_time_windows': instance.customer_time_windows.tolist(),
        'customer_service_times': instance.customer_service_times.tolist(),
        'distance_matrix': instance.distance_matrix.tolist(),
        'duration_matrix': instance.duration_matrix.tolist(),
    }
    return compute_content_hash(content)


def compute_state_hash(state: StateRecord) -> str:
    """Compute hash of state record (without preference)."""
    content = {
        'state_id': state.state_id,
        'instance_hash': compute_instance_hash(state.instance),
        'source_solution': [route.customers for route in state.source_solution.routes],
        'source_objectives': state.source_objectives.to_array().tolist(),
        'partial_solution': [route.customers for route in state.destroyed_state.partial_solution.routes],
        'removed_customers': state.destroyed_state.removed_customers,
        'destroy_type': state.destroy_type.value,
        'destroy_ratio': float(state.destroy_ratio),
        'random_seed': state.random_seed,
    }
    return compute_content_hash(content)


def compute_task_hash(task: PreferenceTaskRecord, state_hash: str) -> str:
    """Compute hash of preference task record."""
    # Convert teacher candidates to hashable format
    candidates = []
    for tc in task.teacher_candidates:
        # Convert RepairAction list to dict format
        actions = []
        for action in tc.trajectory.actions:
            actions.append({
                'customer_id': action.customer_id,
                'route_index': action.route_index,
                'insert_position': action.insert_position,
                'opens_new_route': action.opens_new_route,
            })

        candidates.append({
            'teacher_name': tc.teacher_name,
            'actions': actions,
            'final_solution': [route.customers for route in tc.final_solution.routes],
            'objectives': tc.objectives.to_array().tolist(),
            'is_feasible': tc.is_feasible,
            'runtime_seconds': float(tc.runtime_seconds),
        })

    content = {
        'task_id': task.task_id,
        'state_hash': state_hash,
        'preference_category': task.preference_category.value,
        'preference_weights': task.preference_weights.tolist(),
        'teacher_candidates': candidates,
    }
    return compute_content_hash(content)


@dataclass
class DatasetManifest:
    """
    Dataset manifest with hash-based integrity verification.

    All hashes are computed from actual content and verified on load.
    """
    schema_version: str  # e.g., "v0.17.0"
    protocol_hash: str  # Hash of TaskProtocol
    normalizer: Normalizer
    normalizer_hash: str
    instances: Dict[str, VRPTWInstance]  # instance_id -> instance
    instance_hashes: Dict[str, str]  # instance_id -> hash
    states: Dict[str, StateRecord]  # state_id -> state
    state_hashes: Dict[str, str]  # state_id -> hash
    tasks: Dict[str, PreferenceTaskRecord]  # task_id -> task
    task_hashes: Dict[str, str]  # task_id -> hash
    splits: Dict[str, List[str]]  # split_name -> [task_ids]
    model_config_hash: Optional[str] = None  # Hash of model architecture config
    manifest_hash: Optional[str] = None  # Hash of this manifest (excluding itself)

    def verify_integrity(self, expected_protocol: Optional[TaskProtocol] = None) -> List[str]:
        """
        Verify all hashes match actual content.

        Args:
            expected_protocol: Optional TaskProtocol to verify against

        Returns:
            List of error messages (empty if all valid)
        """
        errors = []

        # Verify schema_version is present
        if not self.schema_version:
            errors.append("schema_version is missing")

        # Verify protocol_hash is present
        if not self.protocol_hash:
            errors.append("protocol_hash is missing")

        # Verify against expected protocol if provided
        if expected_protocol is not None:
            expected_schema_version = expected_protocol.schema_version
            expected_protocol_hash = expected_protocol.compute_protocol_hash()

            if self.schema_version != expected_schema_version:
                errors.append(
                    f"schema_version mismatch: manifest has {self.schema_version}, "
                    f"expected {expected_schema_version}"
                )

            if self.protocol_hash != expected_protocol_hash:
                errors.append(
                    f"protocol_hash mismatch: manifest has {self.protocol_hash}, "
                    f"expected {expected_protocol_hash}"
                )

        # Verify manifest_hash is present and matches
        if not self.manifest_hash:
            errors.append("manifest_hash is missing (required for integrity verification)")
        else:
            manifest_content = {
                'schema_version': self.schema_version,
                'protocol_hash': self.protocol_hash,
                'normalizer_hash': self.normalizer_hash,
                'instance_hashes': self.instance_hashes,
                'state_hashes': self.state_hashes,
                'task_hashes': self.task_hashes,
                'splits': self.splits,
                'model_config_hash': self.model_config_hash,
            }
            actual_manifest_hash = compute_content_hash(manifest_content)
            if actual_manifest_hash != self.manifest_hash:
                errors.append(
                    f"Manifest hash mismatch: expected {self.manifest_hash}, "
                    f"got {actual_manifest_hash}"
                )

        # Verify normalizer hash
        actual_norm_hash = compute_normalizer_hash(self.normalizer)
        if actual_norm_hash != self.normalizer_hash:
            errors.append(f"Normalizer hash mismatch: {actual_norm_hash} != {self.normalizer_hash}")

        # Verify instance hashes and key consistency
        if set(self.instance_hashes.keys()) != set(self.instances.keys()):
            missing_in_hash = set(self.instances.keys()) - set(self.instance_hashes.keys())
            missing_in_registry = set(self.instance_hashes.keys()) - set(self.instances.keys())
            if missing_in_hash:
                errors.append(f"Instances missing in hash registry: {missing_in_hash}")
            if missing_in_registry:
                errors.append(f"Instance hashes have no corresponding instance: {missing_in_registry}")

        for instance_id, instance in self.instances.items():
            # Verify instance_id matches dict key
            if instance.instance_id != instance_id:
                errors.append(
                    f"Instance dict key '{instance_id}' does not match "
                    f"instance.instance_id '{instance.instance_id}'"
                )

            actual_hash = compute_instance_hash(instance)
            expected_hash = self.instance_hashes.get(instance_id)
            if expected_hash is None:
                errors.append(f"Instance {instance_id} missing hash in manifest")
            elif actual_hash != expected_hash:
                errors.append(f"Instance {instance_id} hash mismatch: {actual_hash} != {expected_hash}")

        # Verify state hashes and key consistency
        if set(self.state_hashes.keys()) != set(self.states.keys()):
            missing_in_hash = set(self.states.keys()) - set(self.state_hashes.keys())
            missing_in_registry = set(self.state_hashes.keys()) - set(self.states.keys())
            if missing_in_hash:
                errors.append(f"States missing in hash registry: {missing_in_hash}")
            if missing_in_registry:
                errors.append(f"State hashes have no corresponding state: {missing_in_registry}")

        for state_id, state in self.states.items():
            # Verify state_id matches dict key
            if state.state_id != state_id:
                errors.append(
                    f"State dict key '{state_id}' does not match "
                    f"state.state_id '{state.state_id}'"
                )

            actual_hash = compute_state_hash(state)
            expected_hash = self.state_hashes.get(state_id)
            if expected_hash is None:
                errors.append(f"State {state_id} missing hash in manifest")
            elif actual_hash != expected_hash:
                errors.append(f"State {state_id} hash mismatch: {actual_hash} != {expected_hash}")

            # Verify state's embedded instance matches registry instance
            embedded_instance = state.instance
            registry_instance = self.instances.get(embedded_instance.instance_id)
            if registry_instance is None:
                errors.append(
                    f"State {state_id} references instance {embedded_instance.instance_id} "
                    f"not in manifest registry"
                )
            else:
                embedded_hash = compute_instance_hash(embedded_instance)
                registry_hash = compute_instance_hash(registry_instance)
                if embedded_hash != registry_hash:
                    errors.append(
                        f"State {state_id} embedded instance hash {embedded_hash} "
                        f"does not match registry instance hash {registry_hash} "
                        f"for instance_id {embedded_instance.instance_id}"
                    )

            # Verify all customers accounted for
            partial_customers = set(state.destroyed_state.partial_solution.get_all_customers())
            removed_customers = set(state.destroyed_state.removed_customers)
            all_customers = partial_customers | removed_customers
            expected_customers = set(range(1, state.instance.n_customers + 1))

            if all_customers != expected_customers:
                missing = expected_customers - all_customers
                extra = all_customers - expected_customers
                errors.append(
                    f"State {state_id} customer mismatch: "
                    f"missing={missing}, extra={extra}"
                )

            # Verify no duplicates
            if len(partial_customers & removed_customers) > 0:
                errors.append(
                    f"State {state_id} has customers in both partial and removed: "
                    f"{partial_customers & removed_customers}"
                )

        # Verify task hashes and key consistency
        if set(self.task_hashes.keys()) != set(self.tasks.keys()):
            missing_in_hash = set(self.tasks.keys()) - set(self.task_hashes.keys())
            missing_in_registry = set(self.task_hashes.keys()) - set(self.tasks.keys())
            if missing_in_hash:
                errors.append(f"Tasks missing in hash registry: {missing_in_hash}")
            if missing_in_registry:
                errors.append(f"Task hashes have no corresponding task: {missing_in_registry}")

        for task_id, task in self.tasks.items():
            # Verify task_id matches dict key
            if task.task_id != task_id:
                errors.append(
                    f"Task dict key '{task_id}' does not match "
                    f"task.task_id '{task.task_id}'"
                )

            state_hash = self.state_hashes.get(task.state_id)
            if state_hash is None:
                errors.append(f"Task {task_id} references unknown state {task.state_id}")
                continue

            actual_hash = compute_task_hash(task, state_hash)
            expected_hash = self.task_hashes.get(task_id)
            if expected_hash is None:
                errors.append(f"Task {task_id} missing hash in manifest")
            elif actual_hash != expected_hash:
                errors.append(f"Task {task_id} hash mismatch: {actual_hash} != {expected_hash}")

            # Validate trajectory replay for each teacher candidate
            state = self.states.get(task.state_id)
            if state is not None:
                for i, candidate in enumerate(task.teacher_candidates):
                    try:
                        # Replay trajectory
                        replay_valid = validate_trajectory_replay(
                            state.destroyed_state,
                            candidate.trajectory,
                            candidate.final_solution
                        )
                        if not replay_valid:
                            errors.append(
                                f"Task {task_id} candidate {i} ({candidate.teacher_name}): "
                                f"trajectory does not replay to final_solution"
                            )
                    except ValueError as e:
                        errors.append(
                            f"Task {task_id} candidate {i} ({candidate.teacher_name}): "
                            f"trajectory replay error: {e}"
                        )

                    # Recompute objectives and verify
                    try:
                        recomputed_objectives = evaluate_solution(
                            candidate.final_solution,
                            state.instance
                        )
                        if recomputed_objectives != candidate.objectives:
                            errors.append(
                                f"Task {task_id} candidate {i} ({candidate.teacher_name}): "
                                f"objectives mismatch - recorded {candidate.objectives}, "
                                f"recomputed {recomputed_objectives}"
                            )
                    except Exception as e:
                        errors.append(
                            f"Task {task_id} candidate {i} ({candidate.teacher_name}): "
                            f"objective recomputation error: {e}"
                        )

                    # Recompute feasibility and verify
                    try:
                        recomputed_feasible = is_solution_feasible(
                            candidate.final_solution,
                            state.instance
                        )
                        if recomputed_feasible != candidate.is_feasible:
                            errors.append(
                                f"Task {task_id} candidate {i} ({candidate.teacher_name}): "
                                f"feasibility mismatch - recorded {candidate.is_feasible}, "
                                f"recomputed {recomputed_feasible}"
                            )
                    except Exception as e:
                        errors.append(
                            f"Task {task_id} candidate {i} ({candidate.teacher_name}): "
                            f"feasibility recomputation error: {e}"
                        )

        # Verify splits
        all_task_ids = set(self.tasks.keys())
        split_task_ids = set()
        for split_name, task_ids in self.splits.items():
            # Check for duplicates within split
            if len(task_ids) != len(set(task_ids)):
                errors.append(f"Split {split_name} contains duplicate task IDs")

            # Check all tasks exist
            for task_id in task_ids:
                if task_id not in all_task_ids:
                    errors.append(f"Split {split_name} references unknown task {task_id}")

            split_task_ids.update(task_ids)

        # Check for tasks not in any split
        orphaned = all_task_ids - split_task_ids
        if orphaned:
            errors.append(f"Tasks not in any split: {orphaned}")

        # Check for instance-level cross-split contamination
        # Same instance_id must not appear in multiple splits
        instance_to_splits = {}
        for split_name, task_ids in self.splits.items():
            for task_id in task_ids:
                if task_id in self.tasks:
                    state_id = self.tasks[task_id].state_id
                    if state_id in self.states:
                        instance_id = self.states[state_id].instance.instance_id
                        if instance_id not in instance_to_splits:
                            instance_to_splits[instance_id] = []
                        instance_to_splits[instance_id].append(split_name)

        for instance_id, split_names in instance_to_splits.items():
            if len(set(split_names)) > 1:
                errors.append(
                    f"Instance {instance_id} appears in multiple splits: {set(split_names)}"
                )

        return errors

    def save(self, path: Path):
        """
        Save manifest to disk.

        Note: This saves only the metadata. Full dataset should be saved separately
        using appropriate serialization (e.g., pickle, numpy, etc.).
        """
        # Save manifest metadata as JSON
        metadata = {
            'schema_version': self.schema_version,
            'protocol_hash': self.protocol_hash,
            'normalizer_hash': self.normalizer_hash,
            'instance_hashes': self.instance_hashes,
            'state_hashes': self.state_hashes,
            'task_hashes': self.task_hashes,
            'splits': self.splits,
            'model_config_hash': self.model_config_hash,
            'manifest_hash': self.manifest_hash,
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load(cls, path: Path,
             normalizer: Normalizer,
             instances: Dict[str, VRPTWInstance],
             states: Dict[str, StateRecord],
             tasks: Dict[str, PreferenceTaskRecord],
             expected_protocol: Optional[TaskProtocol] = None) -> 'DatasetManifest':
        """
        Load manifest from disk and optionally verify against expected protocol.

        Args:
            path: Path to manifest metadata JSON
            normalizer: Normalizer object
            instances: Instance registry
            states: State registry
            tasks: Task registry
            expected_protocol: Optional TaskProtocol to verify against

        Returns:
            DatasetManifest with loaded metadata

        Raises:
            ValueError: If expected_protocol is provided and verification fails
        """
        with open(path, 'r') as f:
            metadata = json.load(f)

        manifest = cls(
            schema_version=metadata['schema_version'],
            protocol_hash=metadata['protocol_hash'],
            normalizer=normalizer,
            normalizer_hash=metadata['normalizer_hash'],
            instances=instances,
            instance_hashes=metadata['instance_hashes'],
            states=states,
            state_hashes=metadata['state_hashes'],
            tasks=tasks,
            task_hashes=metadata['task_hashes'],
            splits=metadata['splits'],
            model_config_hash=metadata.get('model_config_hash'),
            manifest_hash=metadata['manifest_hash'],
        )

        # Verify against expected protocol if provided
        if expected_protocol is not None:
            errors = manifest.verify_integrity(expected_protocol=expected_protocol)
            if errors:
                raise ValueError(
                    f"Manifest verification failed against expected protocol:\n" +
                    "\n".join(errors)
                )

        return manifest


@dataclass(frozen=True)
class CheckpointMetadata:
    """
    Immutable checkpoint metadata for model checkpoints.

    This binds a model checkpoint to a specific protocol, manifest, and configuration,
    ensuring reproducibility and preventing accidental misuse of checkpoints.
    """
    # Protocol binding
    schema_version: str
    protocol_hash: str

    # Data binding
    manifest_hash: str
    normalizer_hash: str

    # Model binding
    model_config_hash: str

    # Checkpoint identification
    checkpoint_type: str  # 'bc', 'ppo', 'final', etc.
    checkpoint_version: int  # Incremental version number

    # Optional metadata
    epoch: Optional[int] = None
    step: Optional[int] = None
    timestamp: Optional[str] = None

    def verify_compatibility(
        self,
        protocol: TaskProtocol,
        manifest: DatasetManifest,
        model_config_hash: str,
    ) -> List[str]:
        """
        Verify checkpoint is compatible with given protocol, manifest, and model.

        Args:
            protocol: TaskProtocol to verify against
            manifest: DatasetManifest to verify against
            model_config_hash: Model config hash to verify against

        Returns:
            List of error messages (empty if compatible)
        """
        errors = []

        # Verify protocol
        if self.schema_version != protocol.schema_version:
            errors.append(
                f"schema_version mismatch: checkpoint has {self.schema_version}, "
                f"protocol has {protocol.schema_version}"
            )

        expected_protocol_hash = protocol.compute_protocol_hash()
        if self.protocol_hash != expected_protocol_hash:
            errors.append(
                f"protocol_hash mismatch: checkpoint has {self.protocol_hash}, "
                f"expected {expected_protocol_hash}"
            )

        # Verify manifest
        if self.manifest_hash != manifest.manifest_hash:
            errors.append(
                f"manifest_hash mismatch: checkpoint has {self.manifest_hash}, "
                f"manifest has {manifest.manifest_hash}"
            )

        if self.normalizer_hash != manifest.normalizer_hash:
            errors.append(
                f"normalizer_hash mismatch: checkpoint has {self.normalizer_hash}, "
                f"manifest has {manifest.normalizer_hash}"
            )

        # Verify model config
        if self.model_config_hash != model_config_hash:
            errors.append(
                f"model_config_hash mismatch: checkpoint has {self.model_config_hash}, "
                f"provided {model_config_hash}"
            )

        return errors

    def save(self, path: Path):
        """Save checkpoint metadata to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'CheckpointMetadata':
        """Load checkpoint metadata from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


def save_checkpoint_with_metadata(
    checkpoint_path: Path,
    checkpoint_data: Any,
    protocol: TaskProtocol,
    manifest: DatasetManifest,
    model_config_hash: str,
    checkpoint_type: str,
    checkpoint_version: int,
    epoch: Optional[int] = None,
    step: Optional[int] = None,
) -> CheckpointMetadata:
    """
    Save checkpoint with metadata.

    Args:
        checkpoint_path: Path to save checkpoint (e.g., 'checkpoint_v1.pt')
        checkpoint_data: Checkpoint data to save (e.g., model state_dict)
        protocol: TaskProtocol used for training
        manifest: DatasetManifest used for training
        model_config_hash: Hash of model configuration
        checkpoint_type: Type of checkpoint ('bc', 'ppo', 'final')
        checkpoint_version: Version number
        epoch: Optional epoch number
        step: Optional step number

    Returns:
        CheckpointMetadata object
    """
    import torch
    from datetime import datetime

    # Create metadata
    metadata = CheckpointMetadata(
        schema_version=protocol.schema_version,
        protocol_hash=protocol.compute_protocol_hash(),
        manifest_hash=manifest.manifest_hash,
        normalizer_hash=manifest.normalizer_hash,
        model_config_hash=model_config_hash,
        checkpoint_type=checkpoint_type,
        checkpoint_version=checkpoint_version,
        epoch=epoch,
        step=step,
        timestamp=datetime.now().isoformat(),
    )

    # Save checkpoint data
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_data, checkpoint_path)

    # Save metadata
    metadata_path = checkpoint_path.with_suffix('.meta.json')
    metadata.save(metadata_path)

    return metadata


def load_checkpoint_with_verification(
    checkpoint_path: Path,
    protocol: TaskProtocol,
    manifest: DatasetManifest,
    model_config_hash: str,
) -> tuple:
    """
    Load checkpoint and verify compatibility.

    Args:
        checkpoint_path: Path to checkpoint file
        protocol: Expected TaskProtocol
        manifest: Expected DatasetManifest
        model_config_hash: Expected model config hash

    Returns:
        Tuple of (checkpoint_data, metadata)

    Raises:
        ValueError: If checkpoint is incompatible
    """
    import torch

    # Load metadata
    metadata_path = checkpoint_path.with_suffix('.meta.json')
    if not metadata_path.exists():
        raise ValueError(f"Checkpoint metadata not found: {metadata_path}")

    metadata = CheckpointMetadata.load(metadata_path)

    # Verify compatibility
    errors = metadata.verify_compatibility(protocol, manifest, model_config_hash)
    if errors:
        raise ValueError(f"Checkpoint incompatible:\n" + "\n".join(errors))

    # Load checkpoint data
    checkpoint_data = torch.load(checkpoint_path)

    return checkpoint_data, metadata


def create_manifest(
    protocol: TaskProtocol,
    normalizer: Normalizer,
    instances: Dict[str, VRPTWInstance],
    states: Dict[str, StateRecord],
    tasks: Dict[str, PreferenceTaskRecord],
    splits: Dict[str, List[str]],
    model_config_hash: Optional[str] = None,
) -> DatasetManifest:
    """
    Create a dataset manifest with all hashes computed.

    Args:
        protocol: TaskProtocol object defining the experiment configuration
        normalizer: Normalizer for objectives
        instances: Mapping of instance_id to instance
        states: Mapping of state_id to state
        tasks: Mapping of task_id to task
        splits: Mapping of split_name to task_ids
        model_config_hash: Optional hash of model architecture config

    Returns:
        DatasetManifest with all hashes computed

    Raises:
        ValueError: If integrity checks fail or protocol validation fails
    """
    # Extract schema_version and protocol_hash from protocol
    schema_version = protocol.schema_version
    protocol_hash = protocol.compute_protocol_hash()

    # Compute all hashes
    normalizer_hash = compute_normalizer_hash(normalizer)

    instance_hashes = {
        instance_id: compute_instance_hash(instance)
        for instance_id, instance in instances.items()
    }

    state_hashes = {
        state_id: compute_state_hash(state)
        for state_id, state in states.items()
    }

    task_hashes = {}
    for task_id, task in tasks.items():
        state_hash = state_hashes.get(task.state_id)
        if state_hash is None:
            raise ValueError(f"Task {task_id} references unknown state {task.state_id}")
        task_hashes[task_id] = compute_task_hash(task, state_hash)

    manifest = DatasetManifest(
        schema_version=schema_version,
        protocol_hash=protocol_hash,
        normalizer=normalizer,
        normalizer_hash=normalizer_hash,
        instances=instances,
        instance_hashes=instance_hashes,
        states=states,
        state_hashes=state_hashes,
        tasks=tasks,
        task_hashes=task_hashes,
        splits=splits,
        model_config_hash=model_config_hash,
        manifest_hash=None,  # Computed after integrity check
    )

    # Compute manifest hash first (before verify_integrity checks it)
    manifest_content = {
        'schema_version': schema_version,
        'protocol_hash': protocol_hash,
        'normalizer_hash': normalizer_hash,
        'instance_hashes': instance_hashes,
        'state_hashes': state_hashes,
        'task_hashes': task_hashes,
        'splits': splits,
        'model_config_hash': model_config_hash,
    }
    manifest_hash = compute_content_hash(manifest_content)
    manifest.manifest_hash = manifest_hash

    # Verify integrity (includes trajectory replay, objective verification, etc.)
    errors = manifest.verify_integrity()
    if errors:
        raise ValueError(f"Manifest integrity check failed:\n" + "\n".join(errors))

    return manifest
