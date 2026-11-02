"""Core data structures."""

from .protocol import (
    PreferenceCategory,
    DestroyType,
    VRPTWInstance,
    Route,
    Solution,
    FiveObjectives,
    DestroyedState,
    RepairAction,
    ActionTrajectory,
    TeacherCandidate,
    StateRecord,
    PreferenceTaskRecord,
    Normalizer,
)
from .task_protocol import TaskProtocol, DEFAULT_PROTOCOL
from .replay import replay_trajectory, validate_trajectory_replay

__all__ = [
    "PreferenceCategory",
    "DestroyType",
    "VRPTWInstance",
    "Route",
    "Solution",
    "FiveObjectives",
    "DestroyedState",
    "RepairAction",
    "ActionTrajectory",
    "TeacherCandidate",
    "StateRecord",
    "PreferenceTaskRecord",
    "Normalizer",
    "TaskProtocol",
    "DEFAULT_PROTOCOL",
    "replay_trajectory",
    "validate_trajectory_replay",
]
