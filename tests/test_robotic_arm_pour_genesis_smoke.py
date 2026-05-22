"""Smoke test for the Genesis robot-arm water-pouring variant.

Run from the genesis-sim conda env. The module is imported directly by path to
avoid importing unrelated package side effects.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "chinatown" / "robotic_arm_pour_genesis.py"

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")


def _load_module():
    spec = importlib.util.spec_from_file_location("robotic_arm_pour_genesis", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RoboticArmPourGenesisSmokeTest(unittest.TestCase):
    def _demo_with_settled_water(self, mod):
        demo = mod.RoboticArmPourGenesisDemo(num_frames=0, show_viewer=False, enable_camera=False)
        if not demo.load_settled_particles(mod.SETTLED_PARTICLES_CACHE):
            mod.bake_settled_particles(cache_path=mod.SETTLED_PARTICLES_CACHE)
            self.assertTrue(demo.load_settled_particles(mod.SETTLED_PARTICLES_CACHE))
        return demo

    def test_gripper_stays_on_external_handle(self) -> None:
        mod = _load_module()
        demo = mod.RoboticArmPourGenesisDemo(num_frames=0, show_viewer=False, enable_camera=False)
        inspect_frames = {0, 45, 90, 135, 180, 240, 300, 350}

        for frame in range(max(inspect_frames) + 1):
            time_seconds = frame / mod.FRAME_RATE
            demo._apply_kinematic_pose(time_seconds)
            if frame not in inspect_frames:
                continue
            cup_pos = demo.current_cup_pos
            cup_quat = demo.current_cup_quat

            actual_tcp_pos, _ = demo._actual_standard_robot_grasp_pose()

            tcp_local = mod._inverse_transform_points(cup_pos, cup_quat, actual_tcp_pos[None, :])[0]
            self.assertGreater(np.linalg.norm(tcp_local[:2]), mod.GLASS_OUTER_RADIUS + 0.04)
            np.testing.assert_allclose(tcp_local, mod.PANDA_GRASP_TARGET_LOCAL, atol=2.0e-2)

            hand_link = demo.standard_robot.get_link(name="hand")
            hand_pos = demo.standard_robot.get_links_pos([hand_link.idx_local]).cpu().numpy()[0]
            hand_local = mod._inverse_transform_points(cup_pos, cup_quat, hand_pos[None, :])[0]
            self.assertLess(
                hand_local[0],
                tcp_local[0] - 0.04,
                f"hand palm is on the cup side of the handle at frame {frame}",
            )

            for link_name in ("link6", "link7", "hand", "left_finger", "right_finger"):
                link = demo.standard_robot.get_link(name=link_name)
                link_pos = demo.standard_robot.get_links_pos([link.idx_local]).cpu().numpy()[0]
                local = mod._inverse_transform_points(cup_pos, cup_quat, link_pos[None, :])[0]
                self.assertGreater(
                    np.linalg.norm(local[:2]),
                    mod.GLASS_OUTER_RADIUS + 0.01,
                    f"{link_name} entered the glass footprint at frame {frame}",
                )

            segment_names = ("link4", "link5", "link6", "link7", "hand", "left_finger")
            local_points = []
            for link_name in segment_names:
                link = demo.standard_robot.get_link(name=link_name)
                link_pos = demo.standard_robot.get_links_pos([link.idx_local]).cpu().numpy()[0]
                local_points.append(mod._inverse_transform_points(cup_pos, cup_quat, link_pos[None, :])[0])
            for start_name, end_name, start, end in zip(segment_names, segment_names[1:], local_points, local_points[1:]):
                u = np.linspace(0.0, 1.0, 41)[:, None]
                samples = (1.0 - u) * start + u * end
                within_height = (
                    (samples[:, 2] > -0.5 * mod.GLASS_HEIGHT)
                    & (samples[:, 2] < 0.5 * mod.GLASS_HEIGHT)
                )
                if not within_height.any():
                    continue
                min_radius = np.linalg.norm(samples[within_height, :2], axis=1).min()
                self.assertGreater(
                    min_radius,
                    mod.GLASS_OUTER_RADIUS + 0.04,
                    f"{start_name}->{end_name} crosses the glass footprint at frame {frame}",
                )

    def test_standard_arm_trajectory_is_smooth(self) -> None:
        mod = _load_module()
        qs = []
        for frame in range(mod.VIDEO_NUM_FRAMES):
            qs.append(mod.standard_robot_q_at(frame / mod.FRAME_RATE)[:7].copy())
        qs = np.asarray(qs)
        max_joint_step = np.abs(np.diff(qs, axis=0)).max()
        self.assertLess(max_joint_step, 0.09)

    def test_upper_glass_starts_and_finishes_raised(self) -> None:
        mod = _load_module()
        demo = mod.RoboticArmPourGenesisDemo(num_frames=0, show_viewer=False, enable_camera=False)

        demo._apply_kinematic_pose(0.0)
        start_pos = demo.current_cup_pos.copy()
        np.testing.assert_allclose(start_pos, mod.POURER_CENTER, atol=2.5e-2)
        self.assertGreater(start_pos[2], 0.42)

        demo._apply_kinematic_pose(mod.TILT_SECONDS)
        tilted_pos = demo.current_cup_pos.copy()
        self.assertGreater(demo.current_cup_tilt, 70.0)
        self.assertGreater(tilted_pos[2], 0.35)

        demo._apply_kinematic_pose(mod.VIDEO_NUM_FRAMES / mod.FRAME_RATE)
        final_pos = demo.current_cup_pos.copy()
        np.testing.assert_allclose(final_pos, start_pos, atol=3.0e-2)
        self.assertLess(abs(demo.current_cup_tilt), 2.0)

    def test_glasses_do_not_overlap(self) -> None:
        mod = _load_module()
        demo = mod.RoboticArmPourGenesisDemo(num_frames=0, show_viewer=False, enable_camera=False)

        for frame in range(mod.VIDEO_NUM_FRAMES):
            demo._apply_kinematic_pose(frame / mod.FRAME_RATE)
            overlap_count = mod.glass_overlap_sample_count(
                demo.current_cup_pos,
                demo.current_cup_quat,
            )
            self.assertEqual(
                overlap_count,
                0,
                f"moving glass overlaps the receiving glass at frame {frame}",
            )

    def test_no_floor_leak_during_short_pour(self) -> None:
        mod = _load_module()
        demo = self._demo_with_settled_water(mod)
        inspect_frames = {0, 20, 60, 100, 140}

        for frame in range(max(inspect_frames) + 1):
            if frame in inspect_frames:
                positions = demo._particle_positions()
                live = positions[positions[:, 2] > -1.0]
                cup_pos = demo.pouring_glass.get_pos().cpu().numpy()
                cup_quat = demo.pouring_glass.get_quat().cpu().numpy()
                local = mod._inverse_transform_points(cup_pos, cup_quat, live)
                r_xy = np.linalg.norm(local[:, :2], axis=1)
                below_floor = (
                    (r_xy < mod.GLASS_INNER_RADIUS - mod.WATER_PARTICLE_SIZE)
                    & (local[:, 2] < mod.GLASS_INNER_FLOOR_Z - 0.5 * mod.WATER_PARTICLE_SIZE)
                )
                self.assertEqual(
                    int(below_floor.sum()),
                    0,
                    f"water leaked below moving cup floor at frame {frame}, "
                    f"tilt {mod.tilt_degrees_at(demo.sim_time):.1f} deg",
                )
            if frame < max(inspect_frames):
                demo.step()

    def test_partial_pour_into_second_glass(self) -> None:
        mod = _load_module()
        result = mod.run_simulation(num_frames=mod.VIDEO_NUM_FRAMES)

        self.assertTrue(np.isfinite(result.final_particle_positions).all())
        self.assertGreater(result.initial_particle_count, 13000)

        initial_pos, initial_quat = mod.initial_glass_pose()
        initial_local = mod._inverse_transform_points(
            initial_pos,
            initial_quat,
            result.initial_particle_positions,
        )
        initial_r = np.linalg.norm(initial_local[:, :2], axis=1)
        initial_core = (
            (initial_r < mod.GLASS_INNER_RADIUS - mod.WATER_PARTICLE_SIZE)
            & (initial_local[:, 2] >= mod.GLASS_INNER_FLOOR_Z - 1.0e-3)
            & (initial_local[:, 2] <= mod.GLASS_RIM_Z)
        )
        initial_surface_z = float(np.percentile(initial_local[initial_core, 2], 99.5))
        initial_level_fraction = (
            (initial_surface_z - mod.GLASS_INNER_FLOOR_Z)
            / (mod.GLASS_RIM_Z - mod.GLASS_INNER_FLOOR_Z)
        )
        self.assertGreater(initial_level_fraction, 0.76)
        self.assertLess(initial_level_fraction, 0.84)

        self.assertGreater(result.max_tilt_degrees, 70.0)
        self.assertLess(result.final_tilt_degrees, 2.0)
        self.assertGreater(result.final_particles_in_receiver, 0.45 * result.initial_particle_count)
        self.assertGreater(result.final_particles_in_pourer, 0.40 * result.initial_particle_count)
        self.assertLess(result.max_glass_solid_particles, 25)
        self.assertLess(result.max_pourer_solid_particles, 25)
        self.assertEqual(result.max_pourer_base_particles, 0)

        final_in_pourer_mask = mod._glass_inner_mask(
            result.final_particle_positions,
            result.final_pourer_position,
            result.final_pourer_quat_wxyz,
        )
        final_in_receiver_mask = mod._glass_inner_mask_scaled(
            result.final_particle_positions,
            mod.RECEIVER_CENTER,
            np.array([1.0, 0.0, 0.0, 0.0]),
            scale=mod.RECEIVER_SCALE,
        )
        final_live_mask = result.final_particle_positions[:, 2] > -1.0
        final_outside_count = int((final_live_mask & ~final_in_pourer_mask & ~final_in_receiver_mask).sum())
        self.assertLess(final_outside_count, 0.03 * result.initial_particle_count)

        final_local = mod._inverse_transform_points(
            result.final_pourer_position,
            result.final_pourer_quat_wxyz,
            result.final_particle_positions,
        )
        final_r = np.linalg.norm(final_local[:, :2], axis=1)
        final_in_pourer = (
            (final_r < mod.GLASS_INNER_RADIUS + mod.WATER_PARTICLE_SIZE * 0.75)
            & (final_local[:, 2] <= mod.GLASS_RIM_Z - mod.WATER_BRIM_CLEARANCE)
        )
        self.assertEqual(
            int((final_in_pourer & (final_local[:, 2] < mod.GLASS_INNER_FLOOR_Z - 1.0e-4)).sum()),
            0,
        )
        self.assertEqual(
            int(
                (
                    final_in_pourer
                    & (
                        final_local[:, 2] - 0.5 * mod.WATER_PARTICLE_SIZE
                        < mod.GLASS_INNER_FLOOR_Z - 1.0e-4
                    )
                ).sum()
            ),
            0,
        )


if __name__ == "__main__":
    unittest.main()
