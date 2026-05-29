"""No-video control environment for precise pouring experiments.

This wrapper exposes the Genesis pour as a reset/step loop with low-dimensional
metrics. It intentionally uses the current scripted cup motion; direct action
control is the next milestone after the task and metrics are stable.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from chinatown import robotic_arm_pour_genesis as sim


SCRIPTED_ACTIONS = {None, 0, "scripted", "noop", "hold"}


@dataclass(frozen=True)
class PouringControlEnvConfig:
    target_fraction: float = 0.40
    volume_tolerance_fraction: float = sim.CONTROL_VOLUME_TOLERANCE_FRACTION
    spill_tolerance_fraction: float = sim.CONTROL_SPILL_TOLERANCE_FRACTION
    max_frames: int | None = None
    settled_cache: Path | None = None
    rebake: bool = False
    cuda_device: str | None = "1"
    include_true_viscosity: bool = False
    stop_on_success: bool = False
    spill_reward_weight: float = 2.0
    overshoot_reward_weight: float = 1.0
    success_reward: float = 1.0


class ScriptedPouringControlEnv:
    """Frame-stepped no-video environment around the current scripted pour."""

    def __init__(self, config: PouringControlEnvConfig | None = None) -> None:
        self.config = config or PouringControlEnvConfig()
        self.task = sim.PourControlTask(
            target_fraction=self.config.target_fraction,
            volume_tolerance_fraction=self.config.volume_tolerance_fraction,
            spill_tolerance_fraction=self.config.spill_tolerance_fraction,
        )
        self.max_frames = self.config.max_frames or sim.VIDEO_NUM_FRAMES
        self.demo: sim.RoboticArmPourGenesisDemo | None = None
        self.done = False
        self.last_metrics: sim.PourStepMetrics | None = None
        self.metric_history: list[sim.PourStepMetrics] = []

    def reset(
        self,
        *,
        viscosity: float = sim.WATER_VISCOSITY,
        target_fraction: float | None = None,
        rebake: bool | None = None,
    ) -> np.ndarray:
        if viscosity <= 0.0:
            raise ValueError("viscosity must be positive")
        if self.config.cuda_device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.config.cuda_device
        self.task = sim.PourControlTask(
            target_fraction=(
                self.config.target_fraction if target_fraction is None else target_fraction
            ),
            volume_tolerance_fraction=self.config.volume_tolerance_fraction,
            spill_tolerance_fraction=self.config.spill_tolerance_fraction,
        )

        sim.WATER_VISCOSITY = float(viscosity)
        cache_path = self.config.settled_cache or sim.SETTLED_PARTICLES_CACHE
        should_rebake = self.config.rebake if rebake is None else rebake
        if should_rebake or not cache_path.exists():
            sim.bake_settled_particles(cache_path=cache_path)

        self.demo = sim.RoboticArmPourGenesisDemo(
            num_frames=self.max_frames,
            show_viewer=False,
            enable_camera=False,
        )
        if not self.demo.load_settled_particles(cache_path):
            sim.bake_settled_particles(cache_path=cache_path)
            if not self.demo.load_settled_particles(cache_path):
                raise RuntimeError(f"failed to load settled particle cache {cache_path}")

        self.done = False
        self.last_metrics = self.demo.step_metrics(self.task)
        self.metric_history = [self.last_metrics]
        return self.observation(self.last_metrics)

    def step(self, action: object = None) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        if self.demo is None:
            raise RuntimeError("reset() must be called before step()")
        if self.done:
            raise RuntimeError("episode is done; call reset() before stepping again")
        try:
            is_scripted_action = action in SCRIPTED_ACTIONS
        except TypeError:
            is_scripted_action = False
        if not is_scripted_action:
            raise NotImplementedError(
                "controllable actions start in the next milestone; "
                "use None or 'scripted' for the current no-video environment"
            )

        self.demo.step()
        metrics = self.demo.step_metrics(self.task)
        self.last_metrics = metrics
        self.metric_history.append(metrics)
        self.done = metrics.frame_index >= self.max_frames
        if self.config.stop_on_success and metrics.success:
            self.done = True
        observation = self.observation(metrics)
        reward = self.reward(metrics)
        return observation, reward, self.done, self.info(metrics)

    def observation(self, metrics: sim.PourStepMetrics) -> np.ndarray:
        max_time = max(self.max_frames / sim.FRAME_RATE, sim.VIDEO_DT)
        values = [
            self.task.target_fraction,
            min(metrics.time_seconds / max_time, 1.0),
            metrics.tilt_degrees / 90.0,
            metrics.receiver_fraction,
            metrics.pourer_fraction,
            metrics.spilled_fraction,
        ]
        if self.config.include_true_viscosity:
            values.append(math.log10(metrics.viscosity))
        return np.asarray(values, dtype=np.float32)

    def reward(self, metrics: sim.PourStepMetrics) -> float:
        if metrics.target_error_fraction is None:
            return 0.0
        overshoot = max(metrics.target_error_fraction, 0.0)
        reward = -abs(metrics.target_error_fraction)
        reward -= self.config.spill_reward_weight * metrics.spilled_fraction
        reward -= self.config.overshoot_reward_weight * overshoot
        if metrics.success:
            reward += self.config.success_reward
        return float(reward)

    def info(self, metrics: sim.PourStepMetrics) -> dict[str, Any]:
        return {
            "metrics": metrics,
            "task": self.task,
            "target_fraction": self.task.target_fraction,
            "success": bool(metrics.success),
        }


__all__ = [
    "PouringControlEnvConfig",
    "SCRIPTED_ACTIONS",
    "ScriptedPouringControlEnv",
]
