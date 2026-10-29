"""Small, explicit run configuration. Formal training requires user-set budgets."""
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Config:
    project_root: str = r"D:\WorkSpace\versions\version0.7"
    study_root: str = r"D:\WorkSpace\versions\version0.7\artifacts\hemt_feature_stage_v3_20261013"
    mode: str = "smoke"  # smoke / train / evaluate
    device: str = "cpu"  # GPU use is opt-in, not automatic
    seed: int = 20261015
    hidden_dim: int = 128
    critic_dim: int = 256
    anchors: int = 2
    near_neighbors: int = 10
    learning_rate: float = 1e-5
    gamma: float = .99
    clip_epsilon: float = .2
    ppo_epochs: int = 2
    rollout_steps: int = 600  # derived guard: episode_steps * rollout_episodes
    rollout_episodes: int = 4
    episode_steps: int = 150
    entropy_weight: float = .001
    failure_reward: float = -1.0
    repair_seconds: float = 10.0
    smoke_steps: int = 8
    smoke_seconds: float = 120.0
    train_steps: int | None = None
    train_seconds: float | None = None
    train_instance_ids: tuple[str, ...] = ()  # optional FIT subset, never development
    split_manifest: str = str(Path(__file__).with_name("train_development_v2.json"))
    evaluate_steps: int | None = None  # None: one full episode (150 by default)
    checkpoint: str | None = None
    checkpoint_manifest: str | None = None  # verified sidecar for legacy weights only
    fixed_q: int | None = None  # fixed-q ablation; default learns q
    preference: tuple[float, ...] | None = None  # None: uniform/Dirichlet per episode
    output_root: str = str(Path(__file__).resolve().parents[1] / "runs_mo")

    def to_dict(self):
        return asdict(self)

    def validate(self):
        if self.mode not in {"smoke", "train", "evaluate"}:
            raise ValueError("mode must be smoke, train or evaluate")
        if self.anchors < 1 or self.near_neighbors < 1 or self.hidden_dim < 8:
            raise ValueError("invalid model dimensions")
        if min(self.episode_steps, self.rollout_steps, self.rollout_episodes, self.ppo_epochs, self.smoke_steps) < 1:
            raise ValueError("step counts and PPO epochs must be positive")
        if self.evaluate_steps is not None and not 1 <= self.evaluate_steps <= self.episode_steps:
            raise ValueError("evaluate_steps must be within one training episode; shorter is diagnostic truncation")
        if min(self.repair_seconds, self.smoke_seconds, self.learning_rate) <= 0:
            raise ValueError("time limits and learning rate must be positive")
        if not 0 < self.gamma <= 1 or not 0 < self.clip_epsilon < 1:
            raise ValueError("invalid PPO parameters")
        if self.mode == "train" and (self.train_steps is None or self.train_seconds is None
                                      or self.train_steps < 1 or self.train_seconds <= 0):
            raise ValueError("Set BOTH train_steps and train_seconds before formal training.")
        if self.mode == "train" and self.rollout_steps != self.episode_steps * self.rollout_episodes:
            raise ValueError("rollout_steps must equal episode_steps * rollout_episodes")
        if self.mode == "train" and (self.episode_steps != 150 or self.rollout_episodes != 4):
            raise ValueError("Approved formal protocol is 4 whole 150-action episodes; alternate protocols need separate naming")
        if self.mode == "evaluate" and not self.checkpoint:
            raise ValueError("evaluate requires a checkpoint; random weights are not an evaluation")
