"""Genesis SPH scene: a robot arm pours water from one glass into another.

Runs in the ``genesis-sim`` conda env only. Like the other Genesis variant in
this module keeps the Genesis import inside scene construction so package
import remains lightweight.
"""
from __future__ import annotations

import csv
import fcntl
import math
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FRAME_RATE = 60
# Drive the moving cup at the same cadence as the SPH-rigid coupling. Genesis
# accepts external pose commands at scene-step granularity, so we expose the
# former internal substeps as explicit Python microsteps.
MICROSTEPS_PER_FRAME = 84
VIDEO_DT = 1.0 / FRAME_RATE
PHYSICS_DT = VIDEO_DT / MICROSTEPS_PER_FRAME
SIM_SUBSTEPS = 1

GLASS_HEIGHT = 0.24
GLASS_BOTTOM_RADIUS = 0.072
GLASS_TOP_RADIUS = 0.098
GLASS_WALL_THICKNESS = 0.010
GLASS_BASE_THICKNESS = 0.050
GLASS_MESH_SEGMENTS = 48
GLASS_FILLET_SEGMENTS = 8

GLASS_OUTER_RADIUS = 0.5 * (GLASS_BOTTOM_RADIUS + GLASS_TOP_RADIUS) + 0.004
GLASS_INNER_RADIUS = GLASS_OUTER_RADIUS - max(GLASS_WALL_THICKNESS * 2.0, 0.020)
GLASS_INNER_FLOOR_Z = -GLASS_HEIGHT * 0.5 + GLASS_BASE_THICKNESS
GLASS_RIM_Z = GLASS_HEIGHT * 0.5

WATER_PARTICLE_SIZE = 0.006
GLASS_COUP_FRICTION = 0.05
GLASS_COUP_SOFTNESS = 0.0015
GLASS_SDF_CELL_SIZE = 0.0025
GLASS_SDF_MIN_RES = 64
GLASS_SDF_MAX_RES = 256
GLASS_INNER_FILLET_RADIUS = 2.0 * WATER_PARTICLE_SIZE
ENABLE_SOURCE_BOUNDARY_CORRECTION = True
SOURCE_BOUNDARY_CORRECTION_INTERVAL = 1
SOURCE_WALL_CORRECTION_RIM_CLEARANCE = 0.030
SOURCE_WALL_CORRECTION_OUTER_MARGIN = -0.25 * WATER_PARTICLE_SIZE
SOURCE_WALL_CORRECTION_CLEARANCE = 0.10 * WATER_PARTICLE_SIZE
SOURCE_BASE_CORRECTION_CLEARANCE = 0.55 * WATER_PARTICLE_SIZE
WATER_DENSITY = 1000.0
WATER_VISCOSITY = 1.0e-3
LIQUID_SURFACE_TENSION = 0.01
LIQUID_COLOR = (0.25, 0.55, 0.95, 1.0)
LIQUID_VIS_MODE = "particle"
WATER_FILL_FRACTION = 0.80
WATER_BRIM_CLEARANCE = 0.006
WATER_FLOOR_CLEARANCE = WATER_PARTICLE_SIZE
# Genesis's regular particle sampling settles lower than the nominal emission
# cylinder. This factor is calibrated so the settled frame-0 surface is 80% of
# the inner cavity height, not just 80% of the pre-settle emission height.
EMISSION_OVERFILL_FACTOR = 1.405

POURER_CENTER = np.array([0.18, -0.20, 0.48], dtype=np.float64)
RECEIVER_SCALE = 1.0
RECEIVER_CENTER = np.array([0.29, -0.20, GLASS_HEIGHT * RECEIVER_SCALE * 0.5], dtype=np.float64)
PANDA_BASE_POS = np.array([-0.15, 0.0, 0.0], dtype=np.float64)
PANDA_BASE_EULER = np.array([0.0, 0.0, 0.0], dtype=np.float64)
PANDA_Q_UPRIGHT = np.array(
    [
        -1.5916905403137207,
        -1.2717534303665161,
        -0.06664533913135529,
        -2.951836109161377,
        1.5030548572540283,
        1.537463665008545,
        2.2464425563812256,
        0.026,
        0.026,
    ],
    dtype=np.float32,
)
PANDA_Q_FULL_POUR = np.array(
    [
        -1.6516658067703247,
        -0.798383355140686,
        0.6634261012077332,
        -2.0772507190704346,
        0.5259794592857361,
        1.4164435863494873,
        2.8647570610046387,
        0.026,
        0.026,
    ],
    dtype=np.float32,
)
POUR_POSE_FRACTION = 0.80
PANDA_Q_POUR = PANDA_Q_UPRIGHT + (PANDA_Q_FULL_POUR - PANDA_Q_UPRIGHT) * POUR_POSE_FRACTION
PANDA_HOME_Q = PANDA_Q_UPRIGHT.copy()
PANDA_FINGER_OPENING = 0.026
PANDA_TCP_LOCAL_POINT = np.array([0.0, 0.0, 0.092], dtype=np.float32)
HANDLE_LOCAL_POS = np.array([-GLASS_OUTER_RADIUS - 0.060, 0.0, 0.055], dtype=np.float64)
HANDLE_SIZE = (0.120, 0.050, 0.100)
PANDA_GRASP_TARGET_LOCAL = HANDLE_LOCAL_POS.copy()

TILT_SECONDS = 3.00
RETURN_SECONDS = 1.60
MAX_TILT_DEG = 82.6
ACTION_PROGRAM_KEYFRAMES: tuple[tuple[float, float], ...] | None = None
ACTION_PROGRAM_ID: str | None = None

VIDEO_NUM_FRAMES = int(round((TILT_SECONDS + RETURN_SECONDS) * FRAME_RATE))
VIDEO_RESOLUTION = (1280, 720)
VIDEO_FPS = 60
CAMERA_POS = (0.95, -1.35, 0.62)
CAMERA_LOOKAT = (0.08, 0.0, 0.22)
CAMERA_FOV = 48.0
SOLID_CHECK_INTERVAL_FRAMES = 6

SETTLED_PARTICLES_CACHE = (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "_genesis"
    / "robotic_arm_raised_pour_base050_p006_fill080_over1405_clear006_fric005_soft0015_pose080_slow_fillet012_micro084_sdf0025_align0_corrbase_settled_water.npy"
)
GLASS_MESH_PATH = Path(__file__).resolve().parents[2] / "outputs" / "_genesis" / "pouring_glass.obj"
GLASS_MESH_MIN_BYTES = 1024
SETTLE_BAKE_SECONDS = 0.8
STASHED_PARTICLE_POS = np.array([10.0, 10.0, -10.0], dtype=np.float32)

CONTROL_TARGET_FRACTIONS = (0.25, 0.40, 0.55, 0.70)
CONTROL_VOLUME_TOLERANCE_FRACTION = 0.03
CONTROL_SPILL_TOLERANCE_FRACTION = 0.02


@dataclass(frozen=True)
class PourControlTask:
    """Target-volume task expressed as a fraction of initial source particles."""

    target_fraction: float
    volume_tolerance_fraction: float = CONTROL_VOLUME_TOLERANCE_FRACTION
    spill_tolerance_fraction: float = CONTROL_SPILL_TOLERANCE_FRACTION

    def __post_init__(self) -> None:
        if not (0.0 < self.target_fraction < 1.0):
            raise ValueError("target_fraction must be in (0, 1)")
        if self.volume_tolerance_fraction <= 0.0:
            raise ValueError("volume_tolerance_fraction must be positive")
        if self.spill_tolerance_fraction < 0.0:
            raise ValueError("spill_tolerance_fraction must be non-negative")


@dataclass(frozen=True)
class PourStepMetrics:
    """Per-frame metrics for no-video control and RL reward/debugging."""

    frame_index: int
    time_seconds: float
    tilt_degrees: float
    viscosity: float
    target_fraction: float | None
    initial_particle_count: int
    particles_in_pourer: int
    particles_in_receiver: int
    live_particles: int
    spilled_particles: int
    pourer_fraction: float
    receiver_fraction: float
    spilled_fraction: float
    target_error_fraction: float | None
    success: bool | None


def validate_control_target_fraction(target_fraction: float) -> float:
    target = float(target_fraction)
    if not (0.0 < target < 1.0):
        raise ValueError("target_fraction must be in (0, 1)")
    return target


@dataclass(frozen=True)
class SimulationResult:
    initial_particle_positions: np.ndarray
    final_particle_positions: np.ndarray
    final_pourer_position: np.ndarray
    final_pourer_quat_wxyz: np.ndarray
    final_tilt_degrees: float
    max_tilt_degrees: float
    initial_particle_count: int
    final_particles_in_pourer: int
    final_particles_in_receiver: int
    final_live_particles: int
    max_glass_solid_particles: int
    max_pourer_solid_particles: int
    max_receiver_solid_particles: int
    max_pourer_base_particles: int

    @property
    def receiver_fraction(self) -> float:
        return self.final_particles_in_receiver / max(1, self.initial_particle_count)

    @property
    def pourer_fraction(self) -> float:
        return self.final_particles_in_pourer / max(1, self.initial_particle_count)


def _smoothstep(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def configure_pour_action_program(
    keyframes: list[tuple[float, float]] | tuple[tuple[float, float], ...],
    *,
    action_id: str | None = None,
) -> None:
    """Configure a piecewise cup-pose-fraction trajectory.

    Keyframes are ``(time_seconds, full_pose_fraction)`` pairs. Fractions are
    absolute interpolation values from the upright Panda pose toward
    ``PANDA_Q_FULL_POUR``. Segment interpolation uses smoothstep.
    """
    global ACTION_PROGRAM_ID, ACTION_PROGRAM_KEYFRAMES, PANDA_Q_POUR, POUR_POSE_FRACTION
    global RETURN_SECONDS, TILT_SECONDS, VIDEO_NUM_FRAMES

    if len(keyframes) < 2:
        raise ValueError("an action program needs at least two keyframes")
    normalized = tuple((float(time), float(fraction)) for time, fraction in keyframes)
    if normalized[0][0] != 0.0:
        raise ValueError("the first action keyframe must start at time 0.0")
    previous_time = -math.inf
    for time_seconds, fraction in normalized:
        if time_seconds < 0.0:
            raise ValueError("action keyframe times must be non-negative")
        if time_seconds <= previous_time:
            raise ValueError("action keyframe times must be strictly increasing")
        if not (0.0 <= fraction <= 1.0):
            raise ValueError("action keyframe fractions must be in [0, 1]")
        previous_time = time_seconds

    ACTION_PROGRAM_ID = action_id
    ACTION_PROGRAM_KEYFRAMES = normalized
    POUR_POSE_FRACTION = 1.0
    PANDA_Q_POUR = PANDA_Q_FULL_POUR.copy()
    TILT_SECONDS = normalized[-1][0]
    RETURN_SECONDS = 0.0
    VIDEO_NUM_FRAMES = max(1, int(round(normalized[-1][0] * FRAME_RATE)))


def clear_pour_action_program() -> None:
    global ACTION_PROGRAM_ID, ACTION_PROGRAM_KEYFRAMES

    ACTION_PROGRAM_ID = None
    ACTION_PROGRAM_KEYFRAMES = None


def _piecewise_action_fraction_at(time_seconds: float) -> float:
    if ACTION_PROGRAM_KEYFRAMES is None:
        raise RuntimeError("no action program is configured")
    t = float(time_seconds)
    if t <= ACTION_PROGRAM_KEYFRAMES[0][0]:
        return ACTION_PROGRAM_KEYFRAMES[0][1]
    for (t0, f0), (t1, f1) in zip(ACTION_PROGRAM_KEYFRAMES[:-1], ACTION_PROGRAM_KEYFRAMES[1:]):
        if t <= t1:
            alpha = _smoothstep((t - t0) / (t1 - t0))
            return float(f0 + (f1 - f0) * alpha)
    return ACTION_PROGRAM_KEYFRAMES[-1][1]


def action_command_at(time_seconds: float) -> dict[str, float | str | None]:
    motion_fraction = pour_motion_fraction_at(time_seconds)
    if ACTION_PROGRAM_KEYFRAMES is None:
        full_pose_fraction = motion_fraction * POUR_POSE_FRACTION
    else:
        full_pose_fraction = motion_fraction
    return {
        "action_program_id": ACTION_PROGRAM_ID,
        "time_seconds": float(time_seconds),
        "motion_fraction": float(motion_fraction),
        "full_pose_fraction": float(full_pose_fraction),
    }


def _quat_to_matrix_wxyz(q: np.ndarray) -> np.ndarray:
    w, x, y, z = [float(v) for v in q]
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _quat_wxyz_from_matrix(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                0.25 * s,
                (m[2, 1] - m[1, 2]) / s,
                (m[0, 2] - m[2, 0]) / s,
                (m[1, 0] - m[0, 1]) / s,
            ],
            dtype=np.float64,
        )
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(max(1.0 + m[0, 0] - m[1, 1] - m[2, 2], 1.0e-12)) * 2.0
        quat = np.array(
            [
                (m[2, 1] - m[1, 2]) / s,
                0.25 * s,
                (m[0, 1] + m[1, 0]) / s,
                (m[0, 2] + m[2, 0]) / s,
            ],
            dtype=np.float64,
        )
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(max(1.0 + m[1, 1] - m[0, 0] - m[2, 2], 1.0e-12)) * 2.0
        quat = np.array(
            [
                (m[0, 2] - m[2, 0]) / s,
                (m[0, 1] + m[1, 0]) / s,
                0.25 * s,
                (m[1, 2] + m[2, 1]) / s,
            ],
            dtype=np.float64,
        )
    else:
        s = math.sqrt(max(1.0 + m[2, 2] - m[0, 0] - m[1, 1], 1.0e-12)) * 2.0
        quat = np.array(
            [
                (m[1, 0] - m[0, 1]) / s,
                (m[0, 2] + m[2, 0]) / s,
                (m[1, 2] + m[2, 1]) / s,
                0.25 * s,
            ],
            dtype=np.float64,
        )
    return quat / np.linalg.norm(quat)


def _quat_multiply_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = [float(v) for v in a]
    bw, bx, by, bz = [float(v) for v in b]
    quat = np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )
    return quat / np.linalg.norm(quat)


def _quat_inverse_wxyz(q: np.ndarray) -> np.ndarray:
    quat = np.asarray(q, dtype=np.float64)
    return np.array([quat[0], -quat[1], -quat[2], -quat[3]], dtype=np.float64) / np.dot(quat, quat)


def _quat_slerp_wxyz(q0: np.ndarray, q1: np.ndarray, fraction: float) -> np.ndarray:
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    fraction = float(np.clip(fraction, 0.0, 1.0))
    if dot > 0.9995:
        quat = q0 + fraction * (q1 - q0)
        return quat / np.linalg.norm(quat)
    theta_0 = math.acos(np.clip(dot, -1.0, 1.0))
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * fraction
    s0 = math.cos(theta) - dot * math.sin(theta) / sin_theta_0
    s1 = math.sin(theta) / sin_theta_0
    quat = s0 * q0 + s1 * q1
    return quat / np.linalg.norm(quat)


def _angular_velocity_between_wxyz(q0: np.ndarray, q1: np.ndarray, dt: float) -> np.ndarray:
    delta = _quat_multiply_wxyz(np.asarray(q1, dtype=np.float64), _quat_inverse_wxyz(np.asarray(q0, dtype=np.float64)))
    if delta[0] < 0.0:
        delta = -delta
    vector_norm = float(np.linalg.norm(delta[1:]))
    if vector_norm < 1.0e-9:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * math.atan2(vector_norm, float(delta[0]))
    return delta[1:] / vector_norm * (angle / dt)


def _transform_local(pos: np.ndarray, quat_wxyz: np.ndarray, local: np.ndarray) -> np.ndarray:
    return np.asarray(pos, dtype=np.float64) + _quat_to_matrix_wxyz(quat_wxyz) @ np.asarray(local, dtype=np.float64)


def _inverse_transform_points(pos: np.ndarray, quat_wxyz: np.ndarray, points: np.ndarray) -> np.ndarray:
    rotation = _quat_to_matrix_wxyz(quat_wxyz)
    return (np.asarray(points, dtype=np.float64) - np.asarray(pos, dtype=np.float64)) @ rotation


def tilt_degrees_at(time_seconds: float) -> float:
    return MAX_TILT_DEG * pour_motion_fraction_at(time_seconds)


def pour_motion_fraction_at(time_seconds: float) -> float:
    if ACTION_PROGRAM_KEYFRAMES is not None:
        return _piecewise_action_fraction_at(time_seconds)
    t = float(time_seconds)
    if t < 0.0:
        return 0.0
    if t < TILT_SECONDS:
        return _smoothstep(t / TILT_SECONDS)
    t -= TILT_SECONDS
    if t < RETURN_SECONDS:
        return 1.0 - _smoothstep(t / RETURN_SECONDS)
    return 0.0


def standard_robot_q_at(time_seconds: float) -> np.ndarray:
    fraction = pour_motion_fraction_at(time_seconds)
    q = PANDA_Q_UPRIGHT + (PANDA_Q_POUR - PANDA_Q_UPRIGHT) * fraction
    q = q.astype(np.float32, copy=True)
    q[7:] = PANDA_FINGER_OPENING
    return q


def initial_glass_pose() -> tuple[np.ndarray, np.ndarray]:
    return POURER_CENTER.copy(), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def _panda_asset_path(gs) -> Path:
    return Path(gs.__file__).resolve().parent / "assets" / "xml" / "franka_emika_panda" / "panda.xml"


def _cup_pose_from_grasp_tcp(tcp_pos: np.ndarray, hand_quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hand_rotation = _quat_to_matrix_wxyz(hand_quat)
    cup_to_hand = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    cup_rotation = hand_rotation @ cup_to_hand.T
    cup_quat = _quat_wxyz_from_matrix(cup_rotation)
    cup_pos = np.asarray(tcp_pos, dtype=np.float64) - cup_rotation @ PANDA_GRASP_TARGET_LOCAL
    return cup_pos, cup_quat


def _glass_inner_mask(points: np.ndarray, center: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    return _glass_inner_mask_scaled(points, center, quat_wxyz, scale=1.0)


def _glass_inner_radius_at_z(local_z: np.ndarray | float) -> np.ndarray:
    local_z = np.asarray(local_z, dtype=np.float64)
    fillet_r = min(max(GLASS_INNER_FILLET_RADIUS, 0.0), GLASS_INNER_RADIUS - WATER_PARTICLE_SIZE)
    radius = np.full_like(local_z, GLASS_INNER_RADIUS, dtype=np.float64)
    if fillet_r > 0.0:
        corner = (local_z >= GLASS_INNER_FLOOR_Z) & (local_z < GLASS_INNER_FLOOR_Z + fillet_r)
        dz = local_z[corner] - (GLASS_INNER_FLOOR_Z + fillet_r)
        radius[corner] = GLASS_INNER_RADIUS - fillet_r + np.sqrt(np.maximum(fillet_r * fillet_r - dz * dz, 0.0))
        radius = np.where(local_z < GLASS_INNER_FLOOR_Z, GLASS_INNER_RADIUS - fillet_r, radius)
    return radius


def _glass_inner_mask_scaled(
    points: np.ndarray,
    center: np.ndarray,
    quat_wxyz: np.ndarray,
    *,
    scale: float,
) -> np.ndarray:
    local = _inverse_transform_points(center, quat_wxyz, points) / scale
    r_xy = np.linalg.norm(local[:, :2], axis=1)
    inner_radius = _glass_inner_radius_at_z(local[:, 2])
    return (
        (r_xy < inner_radius + WATER_PARTICLE_SIZE * 0.75)
        & (local[:, 2] >= GLASS_INNER_FLOOR_Z - 0.001)
        & (local[:, 2] <= GLASS_RIM_Z - WATER_BRIM_CLEARANCE)
    )


def _glass_solid_mask(
    points: np.ndarray,
    center: np.ndarray,
    quat_wxyz: np.ndarray,
    *,
    tolerance: float | None = None,
    scale: float = 1.0,
) -> np.ndarray:
    if tolerance is None:
        tolerance = WATER_PARTICLE_SIZE * 1.25

    local = _inverse_transform_points(center, quat_wxyz, points) / scale
    r_xy = np.linalg.norm(local[:, :2], axis=1)
    base_z = -GLASS_HEIGHT * 0.5
    floor_z = GLASS_INNER_FLOOR_Z
    rim_z = GLASS_RIM_Z
    inner_radius = _glass_inner_radius_at_z(local[:, 2])

    side_wall = (
        (r_xy > inner_radius + tolerance)
        & (r_xy < GLASS_OUTER_RADIUS - tolerance)
        & (local[:, 2] > floor_z - tolerance)
        & (local[:, 2] < rim_z - tolerance)
    )
    base = (
        (r_xy < GLASS_OUTER_RADIUS - tolerance)
        & (local[:, 2] > base_z + tolerance)
        & (local[:, 2] < floor_z - tolerance)
    )
    return side_wall | base


def _glass_base_solid_mask(
    points: np.ndarray,
    center: np.ndarray,
    quat_wxyz: np.ndarray,
    *,
    tolerance: float | None = None,
    scale: float = 1.0,
) -> np.ndarray:
    if tolerance is None:
        tolerance = WATER_PARTICLE_SIZE

    local = _inverse_transform_points(center, quat_wxyz, points) / scale
    r_xy = np.linalg.norm(local[:, :2], axis=1)
    base_z = -GLASS_HEIGHT * 0.5
    inner_radius = _glass_inner_radius_at_z(local[:, 2])
    return (
        (r_xy < inner_radius - tolerance)
        & (local[:, 2] > base_z + tolerance)
        & (local[:, 2] < GLASS_INNER_FLOOR_Z - tolerance)
    )


def _glass_overlap_sample_points() -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, GLASS_MESH_SEGMENTS, endpoint=False)
    zs = np.linspace(-0.5 * GLASS_HEIGHT, 0.5 * GLASS_HEIGHT, 15)
    side = np.array(
        [
            [GLASS_OUTER_RADIUS * math.cos(angle), GLASS_OUTER_RADIUS * math.sin(angle), z]
            for z in zs
            for angle in angles
        ],
        dtype=np.float64,
    )
    rims = np.array(
        [
            [radius * math.cos(angle), radius * math.sin(angle), z]
            for z in (-0.5 * GLASS_HEIGHT, 0.5 * GLASS_HEIGHT)
            for radius in (GLASS_INNER_RADIUS, GLASS_OUTER_RADIUS)
            for angle in angles
        ],
        dtype=np.float64,
    )
    return np.concatenate([side, rims], axis=0)


def _glass_outer_volume_mask_scaled(
    points: np.ndarray,
    center: np.ndarray,
    quat_wxyz: np.ndarray,
    *,
    scale: float,
    tolerance: float = 0.006,
) -> np.ndarray:
    local = _inverse_transform_points(center, quat_wxyz, points) / scale
    r_xy = np.linalg.norm(local[:, :2], axis=1)
    return (
        (r_xy < GLASS_OUTER_RADIUS + tolerance)
        & (local[:, 2] > -0.5 * GLASS_HEIGHT - tolerance)
        & (local[:, 2] < 0.5 * GLASS_HEIGHT + tolerance)
    )


def glass_overlap_sample_count(
    cup_pos: np.ndarray,
    cup_quat: np.ndarray,
    receiver_pos: np.ndarray = RECEIVER_CENTER,
    receiver_quat: np.ndarray | None = None,
    *,
    receiver_scale: float = RECEIVER_SCALE,
) -> int:
    if receiver_quat is None:
        receiver_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    local_points = _glass_overlap_sample_points()
    cup_points = cup_pos + local_points @ _quat_to_matrix_wxyz(cup_quat).T
    receiver_points = receiver_pos + receiver_scale * (local_points @ _quat_to_matrix_wxyz(receiver_quat).T)
    cup_in_receiver = _glass_outer_volume_mask_scaled(
        cup_points,
        receiver_pos,
        receiver_quat,
        scale=receiver_scale,
    )
    receiver_in_cup = _glass_outer_volume_mask_scaled(
        receiver_points,
        cup_pos,
        cup_quat,
        scale=1.0,
    )
    return int(cup_in_receiver.sum() + receiver_in_cup.sum())


def _write_glass_mesh(
    path: Path,
    *,
    inner_radius: float,
    inner_floor_z: float,
    fillet_radius: float | None = None,
) -> Path:
    """Write one watertight open-top glass mesh."""
    import trimesh

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")

    n = GLASS_MESH_SEGMENTS
    half_h = GLASS_HEIGHT * 0.5
    base_z = -half_h
    rim_z = half_h
    inner_base_z = inner_floor_z
    outer_r = GLASS_OUTER_RADIUS
    inner_r = inner_radius
    if fillet_radius is None:
        fillet_radius = GLASS_INNER_FILLET_RADIUS
    fillet_r = min(max(float(fillet_radius), 0.0), inner_r - WATER_PARTICLE_SIZE)

    def ring(radius: float, z: float) -> np.ndarray:
        angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        return np.stack([radius * np.cos(angles), radius * np.sin(angles), np.full(n, z)], axis=1)

    outer_bot = ring(outer_r, base_z)
    outer_top = ring(outer_r, rim_z)
    if fillet_r > 0.0:
        theta = np.linspace(0.0, 0.5 * np.pi, GLASS_FILLET_SEGMENTS + 1)
        fillet_rings = [
            ring(inner_r - fillet_r + fillet_r * math.sin(t), inner_base_z + fillet_r * (1.0 - math.cos(t)))
            for t in theta
        ]
        inner_wall_bottom = fillet_rings[-1]
        inner_floor = fillet_rings[0]
    else:
        fillet_rings = []
        inner_wall_bottom = ring(inner_r, inner_base_z)
        inner_floor = inner_wall_bottom
    inner_top = ring(inner_r, rim_z)
    outer_base = ring(outer_r, base_z)

    rings = [outer_bot, outer_top, inner_wall_bottom, inner_top, outer_base, inner_floor]
    off_ob, off_ot, off_ib, off_it, off_base, off_floor = 0, n, 2 * n, 3 * n, 4 * n, 5 * n
    off_fillet = 6 * n
    if fillet_r > 0.0:
        rings.extend(fillet_rings[1:-1])
    verts = np.concatenate(rings, axis=0)
    base_center_idx = verts.shape[0]
    verts = np.concatenate([verts, np.array([[0.0, 0.0, base_z]])], axis=0)
    floor_center_idx = verts.shape[0]
    verts = np.concatenate([verts, np.array([[0.0, 0.0, inner_base_z]])], axis=0)

    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.append([off_ob + i, off_ob + j, off_ot + j])
        faces.append([off_ob + i, off_ot + j, off_ot + i])
        faces.append([off_ib + i, off_it + j, off_ib + j])
        faces.append([off_ib + i, off_it + i, off_it + j])
        faces.append([off_ot + i, off_it + j, off_it + i])
        faces.append([off_ot + i, off_ot + j, off_it + j])
        faces.append([base_center_idx, off_base + j, off_base + i])
        faces.append([floor_center_idx, off_floor + i, off_floor + j])
        if fillet_r > 0.0:
            fillet_offsets = [off_floor] + [
                off_fillet + k * n for k in range(GLASS_FILLET_SEGMENTS - 1)
            ] + [off_ib]
            for a, b in zip(fillet_offsets[:-1], fillet_offsets[1:]):
                faces.append([a + i, b + j, a + j])
                faces.append([a + i, b + i, b + j])

    mesh = trimesh.Trimesh(vertices=verts, faces=np.asarray(faces), process=True)
    mesh.fix_normals()
    try:
        mesh.export(tmp_path, file_type="obj")
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return path


@contextmanager
def _file_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _glass_mesh_exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size >= GLASS_MESH_MIN_BYTES


def build_glass_mesh(path: Path | None = None, *, force: bool = False) -> Path:
    """Ensure the watertight cup mesh used for rendering and Genesis SDF coupling exists."""
    if path is None:
        path = GLASS_MESH_PATH
    path = Path(path)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with _file_lock(lock_path):
        if not force and _glass_mesh_exists(path):
            return path
        return _write_glass_mesh(
            path,
            inner_radius=GLASS_INNER_RADIUS,
            inner_floor_z=GLASS_INNER_FLOOR_Z,
        )


class RoboticArmPourGenesisDemo:
    def __init__(
        self,
        *,
        num_frames: int,
        show_viewer: bool = False,
        enable_camera: bool = False,
    ):
        import genesis as gs

        try:
            gs.init(backend=gs.gpu, logging_level="warning")
        except gs.GenesisException as exc:
            if "already initialized" not in str(exc).lower():
                raise

        self.gs = gs
        self.num_frames = num_frames
        self.frame_dt = VIDEO_DT
        self.physics_dt = PHYSICS_DT
        self.sim_time = 0.0
        self.frame_index = 0
        self.max_tilt_degrees = 0.0
        self.max_glass_solid_particles = 0
        self.max_pourer_solid_particles = 0
        self.max_receiver_solid_particles = 0
        self.max_pourer_base_particles = 0
        self.standard_robot = None
        self.standard_robot_hand = None
        cup_pos, cup_quat = initial_glass_pose()
        self.current_cup_pos = cup_pos
        self.current_cup_quat = cup_quat
        self.current_cup_linear_velocity = np.zeros(3, dtype=np.float64)
        self.current_cup_angular_velocity = np.zeros(3, dtype=np.float64)
        self.current_cup_tilt = 0.0

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.physics_dt,
                substeps=SIM_SUBSTEPS,
                gravity=(0.0, 0.0, -9.81),
            ),
            sph_options=gs.options.SPHOptions(
                particle_size=WATER_PARTICLE_SIZE,
                pressure_solver="DFSPH",
                hash_grid_cell_size=2.0 * WATER_PARTICLE_SIZE,
                lower_bound=(-0.85, -0.85, -0.2),
                upper_bound=(1.05, 0.85, 1.15),
            ),
            rigid_options=gs.options.RigidOptions(
                dt=self.physics_dt,
                gravity=(0.0, 0.0, -9.81),
            ),
            show_viewer=show_viewer,
        )

        self._add_floor()
        self._add_robot_visuals()
        self._add_glasses()
        self._add_water()

        self.camera = None
        if enable_camera:
            self.camera = self.scene.add_camera(
                res=VIDEO_RESOLUTION,
                pos=CAMERA_POS,
                lookat=CAMERA_LOOKAT,
                fov=CAMERA_FOV,
                GUI=False,
            )

        self.scene.build()
        self._finish_robot_visuals()
        self._apply_kinematic_pose(0.0, self.physics_dt)

        self.initial_cup_pos = self.current_cup_pos.copy()
        self.initial_cup_quat = self.current_cup_quat.copy()
        self.initial_particles = self._particle_positions()

    def _glass_material(
        self,
        *,
        rho: float = 650.0,
        gravity_compensation: float = 0.0,
        coup_softness: float = GLASS_COUP_SOFTNESS,
    ):
        gs = self.gs
        return gs.materials.Rigid(
            rho=rho,
            coup_softness=coup_softness,
            coup_friction=GLASS_COUP_FRICTION,
            coup_restitution=0.0,
            sdf_cell_size=GLASS_SDF_CELL_SIZE,
            sdf_min_res=GLASS_SDF_MIN_RES,
            sdf_max_res=GLASS_SDF_MAX_RES,
            gravity_compensation=gravity_compensation,
        )

    def _add_floor(self) -> None:
        gs = self.gs
        self.floor = self.scene.add_entity(
            material=gs.materials.Rigid(),
            morph=gs.morphs.Box(
                pos=(0.08, 0.0, -0.025),
                size=(1.45, 1.15, 0.05),
                fixed=True,
            ),
            surface=gs.surfaces.Default(color=(0.55, 0.58, 0.56), roughness=0.8),
        )

    def _add_glasses(self) -> None:
        gs = self.gs
        mesh_path = build_glass_mesh()
        glass_surface = gs.surfaces.Default(
            color=(0.76, 0.92, 1.0),
            opacity=0.30,
            roughness=0.08,
        )

        pos, quat = initial_glass_pose()
        self.pouring_glass = self.scene.add_entity(
            material=self._glass_material(rho=50000.0, gravity_compensation=1.0),
            morph=gs.morphs.Mesh(
                file=str(mesh_path),
                pos=tuple(pos),
                quat=tuple(quat),
                fixed=False,
                decimate=False,
                convexify=False,
                align=False,
            ),
            surface=glass_surface,
        )
        handle_pos = _transform_local(pos, quat, HANDLE_LOCAL_POS)
        self.pouring_glass_handle = self.scene.add_entity(
            material=gs.materials.Rigid(needs_coup=False),
            morph=gs.morphs.Box(
                pos=tuple(handle_pos),
                quat=tuple(quat),
                size=HANDLE_SIZE,
                fixed=False,
                collision=False,
            ),
            surface=gs.surfaces.Default(color=(0.04, 0.05, 0.06), roughness=0.45),
        )
        self.receiving_glass = self.scene.add_entity(
            material=self._glass_material(),
            morph=gs.morphs.Mesh(
                file=str(mesh_path),
                pos=tuple(RECEIVER_CENTER),
                scale=RECEIVER_SCALE,
                fixed=True,
                batch_fixed_verts=True,
                decimate=False,
                convexify=False,
                align=False,
            ),
            surface=glass_surface,
        )

    def _add_robot_visuals(self) -> None:
        gs = self.gs
        self.standard_robot = self.scene.add_entity(
            material=gs.materials.Rigid(needs_coup=False),
            morph=gs.morphs.MJCF(
                file=str(_panda_asset_path(gs)),
                pos=tuple(PANDA_BASE_POS),
                euler=tuple(PANDA_BASE_EULER),
                collision=False,
            ),
        )

    def _finish_robot_visuals(self) -> None:
        self.standard_robot_hand = self.standard_robot.get_link(name="hand")
        self.standard_robot.set_dofs_position(PANDA_HOME_Q.copy(), zero_velocity=True)

    def _add_water(self) -> None:
        gs = self.gs
        fill_bottom_local_z = GLASS_INNER_FLOOR_Z + WATER_FLOOR_CLEARANCE
        fill_height = (
            WATER_FILL_FRACTION
            * (GLASS_HEIGHT - GLASS_BASE_THICKNESS - WATER_BRIM_CLEARANCE - WATER_FLOOR_CLEARANCE)
        )
        target_volume = math.pi * (GLASS_INNER_RADIUS - WATER_PARTICLE_SIZE) ** 2 * fill_height
        rest_volume_per_particle = 0.8 * WATER_PARTICLE_SIZE ** 3
        target_n_particles = target_volume / rest_volume_per_particle
        emission_volume = target_n_particles * EMISSION_OVERFILL_FACTOR * WATER_PARTICLE_SIZE ** 3
        cylinder_radius = max(GLASS_INNER_RADIUS - WATER_PARTICLE_SIZE, WATER_PARTICLE_SIZE)
        emission_height = emission_volume / (math.pi * cylinder_radius ** 2)
        cylinder_center_local_z = fill_bottom_local_z + emission_height * 0.5
        cup_pos, _ = initial_glass_pose()
        water_center = cup_pos + np.array([0.0, 0.0, cylinder_center_local_z], dtype=np.float64)

        self.water = self.scene.add_entity(
            material=gs.materials.SPH.Liquid(
                rho=WATER_DENSITY,
                mu=WATER_VISCOSITY,
                gamma=LIQUID_SURFACE_TENSION,
            ),
            morph=gs.morphs.Cylinder(
                pos=tuple(water_center),
                radius=cylinder_radius,
                height=emission_height,
            ),
            surface=gs.surfaces.Default(color=LIQUID_COLOR, vis_mode=LIQUID_VIS_MODE),
        )

    def _set_pouring_glass_pose(
        self,
        pos: np.ndarray,
        quat: np.ndarray,
        linear_velocity: np.ndarray,
        angular_velocity: np.ndarray,
    ) -> None:
        pos = np.asarray(pos, dtype=np.float64)
        quat = np.asarray(quat, dtype=np.float64)
        linear_velocity = np.asarray(linear_velocity, dtype=np.float64)
        angular_velocity = np.asarray(angular_velocity, dtype=np.float64)
        qpos = np.concatenate([pos, quat]).astype(np.float32)
        qvel = np.concatenate([linear_velocity, angular_velocity]).astype(np.float32)
        self.pouring_glass.set_qpos(qpos, zero_velocity=False, skip_forward=True)
        self.pouring_glass.set_dofs_velocity(
            qvel,
            skip_forward=False,
        )
        self.current_cup_pos = pos.copy()
        self.current_cup_quat = quat.copy()
        self.current_cup_linear_velocity = linear_velocity.copy()
        self.current_cup_angular_velocity = angular_velocity.copy()

    def _set_handle_pose(self, cup_pos: np.ndarray, cup_quat: np.ndarray) -> None:
        handle_pos = _transform_local(cup_pos, cup_quat, HANDLE_LOCAL_POS)
        self.pouring_glass_handle.set_pos(handle_pos.astype(np.float32), zero_velocity=True, relative=False, skip_forward=True)
        self.pouring_glass_handle.set_quat(cup_quat.astype(np.float32), zero_velocity=True, relative=False)

    def _actual_standard_robot_grasp_pose(self) -> tuple[np.ndarray, np.ndarray]:
        hand_pos = self.standard_robot.get_links_pos([self.standard_robot_hand.idx_local]).cpu().numpy()[0]
        hand_quat = self.standard_robot.get_links_quat([self.standard_robot_hand.idx_local]).cpu().numpy()[0]
        tcp_pos = _transform_local(hand_pos, hand_quat, PANDA_TCP_LOCAL_POINT)
        return tcp_pos, hand_quat

    def _standard_robot_grasp_pose_at(
        self,
        time_seconds: float,
        *,
        velocity_dt: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        q_current = standard_robot_q_at(time_seconds)
        self.standard_robot.set_dofs_position(q_current, zero_velocity=False)
        if velocity_dt is not None:
            q_next = standard_robot_q_at(time_seconds + velocity_dt)
            self.standard_robot.set_dofs_velocity((q_next - q_current) / velocity_dt)
        return self._actual_standard_robot_grasp_pose()

    def _cup_pose_at(
        self,
        time_seconds: float,
        *,
        velocity_dt: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        tcp_pos, hand_quat = self._standard_robot_grasp_pose_at(time_seconds, velocity_dt=velocity_dt)
        return _cup_pose_from_grasp_tcp(tcp_pos, hand_quat)

    def _apply_kinematic_pose(self, time_seconds: float, dt: float | None = None) -> None:
        if dt is None:
            dt = self.frame_dt
        tcp_pos, hand_quat = self._standard_robot_grasp_pose_at(time_seconds, velocity_dt=dt)
        next_tcp_pos, next_hand_quat = self._standard_robot_grasp_pose_at(time_seconds + dt)
        self._standard_robot_grasp_pose_at(time_seconds, velocity_dt=dt)
        cup_pos, cup_quat = _cup_pose_from_grasp_tcp(tcp_pos, hand_quat)
        next_cup_pos, next_cup_quat = _cup_pose_from_grasp_tcp(next_tcp_pos, next_hand_quat)
        # The glass pose is the fixed handle grasp transform from the actual
        # Panda hand FK. The default trajectory starts at the raised upright
        # pour pose, then moves only through robot joint commands.
        # Genesis's attach() path made SPH leak through the held mesh, so this
        # keeps the working rigid-SPH coupling while preserving the robot as
        # the source of motion.
        self._set_pouring_glass_pose(
            cup_pos,
            cup_quat,
            (next_cup_pos - cup_pos) / dt,
            _angular_velocity_between_wxyz(cup_quat, next_cup_quat, dt),
        )
        self._set_handle_pose(cup_pos, cup_quat)
        rotation = _quat_to_matrix_wxyz(cup_quat)
        cup_axis_z = float(np.clip(rotation[2, 2], -1.0, 1.0))
        actual_tilt = math.degrees(math.acos(cup_axis_z))
        self.max_tilt_degrees = max(self.max_tilt_degrees, abs(actual_tilt))
        self.current_cup_tilt = actual_tilt

    def _interpolated_cup_pose(
        self,
        start_pos: np.ndarray,
        start_quat: np.ndarray,
        end_pos: np.ndarray,
        end_quat: np.ndarray,
        fraction: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        fraction = float(np.clip(fraction, 0.0, 1.0))
        pos = start_pos + (end_pos - start_pos) * fraction
        quat = _quat_slerp_wxyz(start_quat, end_quat, fraction)
        return pos, quat

    def _particle_positions(self) -> np.ndarray:
        return self.water.get_particles_pos().cpu().numpy().copy()

    def _particle_velocities(self) -> np.ndarray:
        return self.water.get_particles_vel().cpu().numpy().copy()

    def _correct_source_wall_particles(self) -> int:
        """Project missed source-cup wall contacts back into the rounded cavity."""
        positions = self._particle_positions()
        live_indices = np.flatnonzero(positions[:, 2] > -1.0)
        if live_indices.size == 0:
            return 0

        velocities = self._particle_velocities()
        rotation = _quat_to_matrix_wxyz(self.current_cup_quat)
        local = (positions[live_indices] - self.current_cup_pos) @ rotation
        local_z = local[:, 2]
        radial = np.linalg.norm(local[:, :2], axis=1)
        inner_radius = _glass_inner_radius_at_z(local_z)

        side_wall = (
            (local_z >= GLASS_INNER_FLOOR_Z - 0.25 * WATER_PARTICLE_SIZE)
            & (local_z < GLASS_RIM_Z - SOURCE_WALL_CORRECTION_RIM_CLEARANCE)
            & (radial > inner_radius)
            & (radial < GLASS_OUTER_RADIUS + SOURCE_WALL_CORRECTION_OUTER_MARGIN)
        )
        base = (
            (local_z < GLASS_INNER_FLOOR_Z + SOURCE_BASE_CORRECTION_CLEARANCE)
            & (local_z > -0.5 * GLASS_HEIGHT - 0.5 * WATER_PARTICLE_SIZE)
            & (radial < inner_radius)
        )
        changed = side_wall | base
        if not changed.any():
            return 0

        local_normals = np.zeros_like(local)
        if side_wall.any():
            side_indices = np.flatnonzero(side_wall)
            local_normals[side_indices, :2] = local[side_indices, :2] / np.maximum(radial[side_indices, None], 1.0e-9)
            target_radius = inner_radius[side_indices] - SOURCE_WALL_CORRECTION_CLEARANCE
            local[side_indices, :2] = local_normals[side_indices, :2] * target_radius[:, None]

        if base.any():
            base_indices = np.flatnonzero(base)
            local_normals[base_indices, 2] = -1.0
            target_z = GLASS_INNER_FLOOR_Z + SOURCE_BASE_CORRECTION_CLEARANCE
            local[base_indices, 2] = target_z

        changed_indices = live_indices[changed]
        positions[changed_indices] = self.current_cup_pos + local[changed] @ rotation.T
        normals_world = local_normals[changed] @ rotation.T
        rel_points = positions[changed_indices] - self.current_cup_pos
        wall_velocity = self.current_cup_linear_velocity + np.cross(self.current_cup_angular_velocity, rel_points)
        relative_velocity = velocities[changed_indices] - wall_velocity
        outward_speed = np.sum(relative_velocity * normals_world, axis=1)
        relative_velocity -= normals_world * np.maximum(outward_speed, 0.0)[:, None]
        velocities[changed_indices] = wall_velocity + relative_velocity
        self.water.set_particles_pos(positions.astype(np.float32))
        self.water.set_particles_vel(velocities.astype(np.float32))
        return int(changed.sum())

    def _count_glass_solid_particles_by_glass(self, positions: np.ndarray, time_seconds: float) -> tuple[int, int]:
        del time_seconds
        cup_pos = self.current_cup_pos
        cup_quat = self.current_cup_quat
        receiver_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        live = positions[positions[:, 2] > -1.0]
        if live.size:
            pourer_count = int(_glass_solid_mask(live, cup_pos, cup_quat).sum())
            receiver_count = int(_glass_solid_mask(live, RECEIVER_CENTER, receiver_quat, scale=RECEIVER_SCALE).sum())
            return pourer_count, receiver_count
        return 0, 0

    def _count_pourer_base_particles(self, positions: np.ndarray) -> int:
        live = positions[positions[:, 2] > -1.0]
        if not live.size:
            return 0
        return int(_glass_base_solid_mask(live, self.current_cup_pos, self.current_cup_quat).sum())

    def step(self) -> None:
        frame_start_time = self.sim_time
        start_pos, start_quat = self._cup_pose_at(frame_start_time)
        end_pos, end_quat = self._cup_pose_at(frame_start_time + self.frame_dt)
        for microstep in range(MICROSTEPS_PER_FRAME):
            alpha = microstep / MICROSTEPS_PER_FRAME
            next_alpha = (microstep + 1) / MICROSTEPS_PER_FRAME
            cup_pos, cup_quat = self._interpolated_cup_pose(start_pos, start_quat, end_pos, end_quat, alpha)
            next_cup_pos, next_cup_quat = self._interpolated_cup_pose(
                start_pos,
                start_quat,
                end_pos,
                end_quat,
                next_alpha,
            )
            self._set_pouring_glass_pose(
                cup_pos,
                cup_quat,
                (next_cup_pos - cup_pos) / self.physics_dt,
                _angular_velocity_between_wxyz(cup_quat, next_cup_quat, self.physics_dt),
            )
            self.scene.step()
            should_correct = (
                ENABLE_SOURCE_BOUNDARY_CORRECTION
                and (
                    (microstep + 1) % SOURCE_BOUNDARY_CORRECTION_INTERVAL == 0
                    or microstep == MICROSTEPS_PER_FRAME - 1
                )
            )
            if should_correct:
                self._correct_source_wall_particles()
            self.sim_time += self.physics_dt

        self._apply_kinematic_pose(self.sim_time, self.physics_dt)
        self.frame_index += 1
        positions = self._particle_positions()
        self.max_pourer_base_particles = max(
            self.max_pourer_base_particles,
            self._count_pourer_base_particles(positions),
        )
        if self.frame_index % SOLID_CHECK_INTERVAL_FRAMES == 0:
            pourer_count, receiver_count = self._count_glass_solid_particles_by_glass(
                positions,
                self.sim_time,
            )
            self.max_pourer_solid_particles = max(self.max_pourer_solid_particles, pourer_count)
            self.max_receiver_solid_particles = max(self.max_receiver_solid_particles, receiver_count)
            self.max_glass_solid_particles = max(
                self.max_glass_solid_particles,
                pourer_count + receiver_count,
            )

    def pre_settle(self, seconds: float = SETTLE_BAKE_SECONDS) -> None:
        for step_index in range(int(round(seconds / self.physics_dt))):
            self._apply_kinematic_pose(0.0, self.physics_dt)
            self.scene.step()
            should_correct = (
                ENABLE_SOURCE_BOUNDARY_CORRECTION
                and (
                    (step_index + 1) % SOURCE_BOUNDARY_CORRECTION_INTERVAL == 0
                    or step_index == int(round(seconds / self.physics_dt)) - 1
                )
            )
            if should_correct:
                self._correct_source_wall_particles()
        self._apply_kinematic_pose(0.0, self.physics_dt)
        if ENABLE_SOURCE_BOUNDARY_CORRECTION:
            self._correct_source_wall_particles()

    def trim_overflow_particles(self) -> None:
        positions = self._particle_positions()
        in_pourer = _glass_inner_mask(positions, self.current_cup_pos, self.current_cup_quat)
        if int((~in_pourer).sum()) > 0:
            positions[~in_pourer] = STASHED_PARTICLE_POS.astype(positions.dtype)
            self.water.set_particles_pos(positions.astype(np.float32))
            self.water.set_particles_vel(np.zeros_like(positions, dtype=np.float32))

    def load_settled_particles(self, path: Path) -> bool:
        if not path.exists():
            return False
        settled = np.load(path).astype(np.float32)
        if settled.shape[0] != self.water.n_particles:
            return False
        in_pourer = _glass_inner_mask(settled, self.current_cup_pos, self.current_cup_quat)
        if int(in_pourer.sum()) < 0.5 * settled.shape[0]:
            return False
        self.water.set_particles_pos(settled)
        self.water.set_particles_vel(np.zeros_like(settled))
        if ENABLE_SOURCE_BOUNDARY_CORRECTION:
            self._correct_source_wall_particles()
        settled = self._particle_positions()
        self.initial_particles = settled.copy()
        return True

    def initial_pourer_particle_count(self) -> int:
        initial_in_pourer, _, _ = self.particle_counts(
            self.initial_particles,
            cup_pos=self.initial_cup_pos,
            cup_quat=self.initial_cup_quat,
        )
        return initial_in_pourer

    def particle_region_counts(
        self,
        particles: np.ndarray,
        *,
        cup_pos: np.ndarray | None = None,
        cup_quat: np.ndarray | None = None,
    ) -> tuple[int, int, int, int]:
        if cup_pos is None:
            cup_pos = self.current_cup_pos
        if cup_quat is None:
            cup_quat = self.current_cup_quat
        live = particles[:, 2] > -1.0
        in_pourer = _glass_inner_mask(particles, cup_pos, cup_quat)
        in_receiver = _glass_inner_mask_scaled(
            particles,
            RECEIVER_CENTER,
            np.array([1.0, 0.0, 0.0, 0.0]),
            scale=RECEIVER_SCALE,
        )
        spilled = live & ~in_pourer & ~in_receiver
        return int(in_pourer.sum()), int(in_receiver.sum()), int(spilled.sum()), int(live.sum())

    def particle_counts(
        self,
        particles: np.ndarray,
        *,
        cup_pos: np.ndarray | None = None,
        cup_quat: np.ndarray | None = None,
    ) -> tuple[int, int, int]:
        if cup_pos is None:
            cup_pos = self.current_cup_pos
        if cup_quat is None:
            cup_quat = self.current_cup_quat
        live = particles[:, 2] > -1.0
        in_pourer = _glass_inner_mask(particles, cup_pos, cup_quat)
        in_receiver = _glass_inner_mask_scaled(
            particles,
            RECEIVER_CENTER,
            np.array([1.0, 0.0, 0.0, 0.0]),
            scale=RECEIVER_SCALE,
        )
        return int(in_pourer.sum()), int(in_receiver.sum()), int(live.sum())

    def step_metrics(self, task: PourControlTask | None = None) -> PourStepMetrics:
        positions = self._particle_positions()
        in_pourer, in_receiver, spilled, live = self.particle_region_counts(positions)
        initial_count = max(1, self.initial_pourer_particle_count())
        receiver_fraction = in_receiver / initial_count
        spilled_fraction = spilled / initial_count
        target_fraction = None if task is None else task.target_fraction
        target_error = None
        success = None
        if task is not None:
            target_error = receiver_fraction - task.target_fraction
            success = (
                abs(target_error) <= task.volume_tolerance_fraction
                and spilled_fraction <= task.spill_tolerance_fraction
            )
        return PourStepMetrics(
            frame_index=int(self.frame_index),
            time_seconds=float(self.sim_time),
            tilt_degrees=float(self.current_cup_tilt),
            viscosity=float(WATER_VISCOSITY),
            target_fraction=target_fraction,
            initial_particle_count=initial_count,
            particles_in_pourer=in_pourer,
            particles_in_receiver=in_receiver,
            live_particles=live,
            spilled_particles=spilled,
            pourer_fraction=in_pourer / initial_count,
            receiver_fraction=receiver_fraction,
            spilled_fraction=spilled_fraction,
            target_error_fraction=target_error,
            success=success,
        )

    def run(self) -> SimulationResult:
        for _ in range(self.num_frames):
            self.step()
        return self.result()

    def result(self) -> SimulationResult:
        final_particles = self._particle_positions()
        in_pourer, in_receiver, live = self.particle_counts(final_particles)
        initial_in_pourer, _, _ = self.particle_counts(
            self.initial_particles,
            cup_pos=self.initial_cup_pos,
            cup_quat=self.initial_cup_quat,
        )
        final_pos = self.current_cup_pos.copy()
        final_quat = self.current_cup_quat.copy()
        final_tilt = self.current_cup_tilt
        final_pourer_solid_count, final_receiver_solid_count = self._count_glass_solid_particles_by_glass(
            final_particles,
            self.sim_time,
        )
        final_solid_count = final_pourer_solid_count + final_receiver_solid_count
        final_pourer_base_count = self._count_pourer_base_particles(final_particles)
        return SimulationResult(
            initial_particle_positions=self.initial_particles.copy(),
            final_particle_positions=final_particles,
            final_pourer_position=final_pos,
            final_pourer_quat_wxyz=final_quat,
            final_tilt_degrees=float(final_tilt),
            max_tilt_degrees=float(self.max_tilt_degrees),
            initial_particle_count=initial_in_pourer,
            final_particles_in_pourer=in_pourer,
            final_particles_in_receiver=in_receiver,
            final_live_particles=live,
            max_glass_solid_particles=int(max(self.max_glass_solid_particles, final_solid_count)),
            max_pourer_solid_particles=int(max(self.max_pourer_solid_particles, final_pourer_solid_count)),
            max_receiver_solid_particles=int(max(self.max_receiver_solid_particles, final_receiver_solid_count)),
            max_pourer_base_particles=int(max(self.max_pourer_base_particles, final_pourer_base_count)),
        )


def bake_settled_particles(
    *,
    cache_path: Path = SETTLED_PARTICLES_CACHE,
    settle_seconds: float = SETTLE_BAKE_SECONDS,
) -> Path:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    demo = RoboticArmPourGenesisDemo(num_frames=0, show_viewer=False, enable_camera=False)
    demo.pre_settle(settle_seconds)
    demo.trim_overflow_particles()
    demo.pre_settle(0.2)
    demo.trim_overflow_particles()
    settled = demo._particle_positions().astype(np.float32)
    np.save(cache_path, settled)
    return cache_path


def run_simulation(
    *,
    num_frames: int = VIDEO_NUM_FRAMES,
    show_viewer: bool = False,
    settled_cache: Path = SETTLED_PARTICLES_CACHE,
    rebake: bool = False,
) -> SimulationResult:
    if rebake or not settled_cache.exists():
        bake_settled_particles(cache_path=settled_cache)

    demo = RoboticArmPourGenesisDemo(num_frames=num_frames, show_viewer=show_viewer, enable_camera=False)
    if not demo.load_settled_particles(settled_cache):
        bake_settled_particles(cache_path=settled_cache)
        demo.load_settled_particles(settled_cache)
    return demo.run()


def _write_csv_rows(path: str | Path, rows: list[dict]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_video(
    *,
    output_path: str = "outputs/robotic_arm_pour_genesis.mp4",
    num_frames: int = VIDEO_NUM_FRAMES,
    settled_cache: Path = SETTLED_PARTICLES_CACHE,
    rebake: bool = False,
    control_task: PourControlTask | None = None,
    metrics_path: str | Path | None = None,
    action_trace_path: str | Path | None = None,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if rebake or not settled_cache.exists():
        bake_settled_particles(cache_path=settled_cache)

    demo = RoboticArmPourGenesisDemo(num_frames=num_frames, show_viewer=False, enable_camera=True)
    assert demo.camera is not None
    if not demo.load_settled_particles(settled_cache):
        bake_settled_particles(cache_path=settled_cache)
        demo.load_settled_particles(settled_cache)

    metrics_rows: list[dict] = []
    action_trace_rows: list[dict] = []

    def record_trace() -> None:
        if metrics_path is not None:
            metrics_rows.append(demo.step_metrics(control_task).__dict__)
        if action_trace_path is not None:
            action_trace_rows.append(
                {
                    "frame_index": int(demo.frame_index),
                    "tilt_degrees": float(demo.current_cup_tilt),
                    **action_command_at(demo.sim_time),
                }
            )

    demo.scene.visualizer.update(force=True)
    demo.camera.render(force_render=True)
    demo.camera.start_recording()
    demo.camera.render(force_render=True)
    record_trace()

    for _ in range(num_frames - 1):
        demo.step()
        demo.camera.render()
        record_trace()

    demo.camera.stop_recording(save_to_filename=str(output), fps=VIDEO_FPS)
    if metrics_path is not None:
        _write_csv_rows(metrics_path, metrics_rows)
    if action_trace_path is not None:
        _write_csv_rows(action_trace_path, action_trace_rows)
    return output
