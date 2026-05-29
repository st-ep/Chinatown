"""Unit tests for precise-pouring control task helpers."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chinatown import robotic_arm_pour_genesis as sim
from chinatown.pouring_control_env import PouringControlEnvConfig, ScriptedPouringControlEnv


class PouringControlEnvTest(unittest.TestCase):
    def test_control_task_validates_target_fraction(self) -> None:
        task = sim.PourControlTask(target_fraction=0.4)
        self.assertEqual(task.target_fraction, 0.4)
        self.assertEqual(sim.CONTROL_TARGET_FRACTIONS, (0.25, 0.40, 0.55, 0.70))
        with self.assertRaises(ValueError):
            sim.PourControlTask(target_fraction=0.0)
        with self.assertRaises(ValueError):
            sim.PourControlTask(target_fraction=1.0)

    def test_observation_and_reward_use_fraction_metrics(self) -> None:
        env = ScriptedPouringControlEnv(
            PouringControlEnvConfig(
                target_fraction=0.4,
                max_frames=100,
                include_true_viscosity=True,
            )
        )
        metrics = sim.PourStepMetrics(
            frame_index=25,
            time_seconds=25 / sim.FRAME_RATE,
            tilt_degrees=45.0,
            viscosity=1.0e-3,
            target_fraction=0.4,
            initial_particle_count=100,
            particles_in_pourer=58,
            particles_in_receiver=40,
            live_particles=100,
            spilled_particles=2,
            pourer_fraction=0.58,
            receiver_fraction=0.40,
            spilled_fraction=0.02,
            target_error_fraction=0.0,
            success=True,
        )

        observation = env.observation(metrics)
        self.assertEqual(observation.shape, (7,))
        self.assertAlmostEqual(float(observation[0]), 0.4, places=6)
        self.assertAlmostEqual(float(observation[2]), 0.5, places=6)
        self.assertAlmostEqual(float(observation[3]), 0.4, places=6)
        self.assertGreater(env.reward(metrics), 0.0)

    def test_step_requires_reset(self) -> None:
        env = ScriptedPouringControlEnv()
        with self.assertRaises(RuntimeError):
            env.step()


if __name__ == "__main__":
    unittest.main()
