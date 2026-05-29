"""Unit checks for the randomized viscosity-belief dataset generator."""
from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "generate_viscosity_belief_random_actions.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("generate_viscosity_belief_random_actions", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ViscosityBeliefGeneratorTest(unittest.TestCase):
    def test_single_viscosity_uses_minimum(self) -> None:
        mod = _load_module()
        values = mod._viscosity_values(
            SimpleNamespace(
                num_viscosities=1,
                min_viscosity=1.0e-3,
                max_viscosity=3.0e-2,
            )
        )
        self.assertEqual(values, [1.0e-3])

    def test_existing_run_health_rejects_missing_receiver_with_high_spill(self) -> None:
        mod = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = self._fake_run(mod, root)
            self._write_common_files(run, num_frames=2)
            self._write_metrics(
                run.metrics_path,
                rows=[
                    (0, 0.0, 0.0),
                    (1, 0.0, 0.45),
                ],
            )

            healthy, reason = self._existing_run_health_without_ffprobe(mod, run)

        self.assertFalse(healthy)
        self.assertIn("near-zero receiver", reason)

    def test_existing_run_health_accepts_normal_receiver_capture(self) -> None:
        mod = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = self._fake_run(mod, root)
            self._write_common_files(run, num_frames=2)
            self._write_metrics(
                run.metrics_path,
                rows=[
                    (0, 0.0, 0.0),
                    (1, 0.45, 0.01),
                ],
            )

            healthy, reason = self._existing_run_health_without_ffprobe(mod, run)

        self.assertTrue(healthy, reason)

    def test_existing_run_health_rejects_bad_video_frame_count(self) -> None:
        mod = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = self._fake_run(mod, root)
            self._write_common_files(run, num_frames=2)
            self._write_metrics(
                run.metrics_path,
                rows=[
                    (0, 0.0, 0.0),
                    (1, 0.45, 0.01),
                ],
            )
            original = mod._normalize_video_frame_count
            mod._normalize_video_frame_count = lambda *args, **kwargs: (_ for _ in ()).throw(
                ValueError("video frame count 1 != expected 2")
            )
            try:
                healthy, reason = mod._existing_run_health(
                    self._health_args(),
                    run,
                    SimpleNamespace(FRAME_RATE=60),
                )
            finally:
                mod._normalize_video_frame_count = original

        self.assertFalse(healthy)
        self.assertIn("video frame count invalid", reason)

    def _health_args(self):
        return SimpleNamespace(
            min_video_bytes=16,
            bad_run_min_receiver_fraction=0.005,
            bad_run_max_spill_without_receiver_fraction=0.20,
        )

    def _existing_run_health_without_ffprobe(self, mod, run):
        original = mod._normalize_video_frame_count
        mod._normalize_video_frame_count = lambda *args, **kwargs: "video frame count ok"
        try:
            return mod._existing_run_health(
                self._health_args(),
                run,
                SimpleNamespace(FRAME_RATE=60),
            )
        finally:
            mod._normalize_video_frame_count = original

    def _fake_run(self, mod, root: Path):
        action = mod.ActionProgram(
            action_index=0,
            action_id="action_000_probe",
            family="probe",
            keyframes=((0.0, 0.0), (1.0 / 30.0, 0.5)),
            params={},
        )
        return mod.DatasetRun(
            index=0,
            viscosity_index=0,
            action=action,
            run_id="run_0000_action_000_mu_0p001000",
            viscosity=1.0e-3,
            log10_viscosity=-3.0,
            video_path=root / "video.mp4",
            metadata_path=root / "metadata.json",
            action_program_path=root / "action.json",
            action_trace_path=root / "action_trace.csv",
            metrics_path=root / "per_frame_metrics.csv",
            microsteps_per_frame=84,
            source_boundary_correction_interval=1,
        )

    def _write_common_files(self, run, *, num_frames: int) -> None:
        del num_frames
        run.video_path.write_bytes(b"0" * 128)
        run.metadata_path.write_text("{}", encoding="utf-8")
        run.action_trace_path.write_text("frame_index,time_seconds,pose_fraction\n", encoding="utf-8")

    def _write_metrics(self, path: Path, *, rows: list[tuple[int, float, float]]) -> None:
        fieldnames = [
            "frame_index",
            "time_seconds",
            "tilt_degrees",
            "viscosity",
            "target_fraction",
            "initial_particle_count",
            "particles_in_pourer",
            "particles_in_receiver",
            "live_particles",
            "spilled_particles",
            "pourer_fraction",
            "receiver_fraction",
            "spilled_fraction",
            "target_error_fraction",
            "success",
        ]
        with path.open("w", encoding="utf-8", newline="") as metrics_file:
            writer = csv.DictWriter(metrics_file, fieldnames=fieldnames)
            writer.writeheader()
            for frame_index, receiver_fraction, spilled_fraction in rows:
                writer.writerow(
                    {
                        "frame_index": frame_index,
                        "time_seconds": frame_index / 60.0,
                        "tilt_degrees": 0.0,
                        "viscosity": 1.0e-3,
                        "target_fraction": "",
                        "initial_particle_count": 100,
                        "particles_in_pourer": 50,
                        "particles_in_receiver": int(receiver_fraction * 100),
                        "live_particles": 100,
                        "spilled_particles": int(spilled_fraction * 100),
                        "pourer_fraction": 0.5,
                        "receiver_fraction": receiver_fraction,
                        "spilled_fraction": spilled_fraction,
                        "target_error_fraction": "",
                        "success": "",
                    }
                )


if __name__ == "__main__":
    unittest.main()
