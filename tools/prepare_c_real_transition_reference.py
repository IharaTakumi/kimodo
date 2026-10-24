#!/usr/bin/env python3
"""Wrap the repaired four-second C motion with smooth SONIC entry/exit phases.

Frame zero is the exact SONIC default standing pose.  This makes policy
engagement (the `]` key) a true hold instead of exposing C frame zero before
the operator presses `T`.  Playback then contains a minimum-jerk 1.5 s entry,
the unchanged 200-frame C motion, and a minimum-jerk 1.5 s return.
The source candidate is never modified.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/C_proud_wave_dynamic_repaired_v2/C_proud_wave_dynamic_repaired_4s_50hz.npz"
OUT = ROOT / "outputs/C_proud_wave_real_transition_v1"
REF = OUT / "sonic_reference/C_proud_wave_entry_C_exit"
FPS = 50.0
DT = 1.0 / FPS
BLEND_S = 1.5
BLEND_INTERVALS = int(round(BLEND_S * FPS))

# Hardware/MuJoCo order, copied from the reviewed official policy parameters.
DEFAULT_Q = np.array([
    -.312, 0, 0, .669, -.363, 0,
    -.312, 0, 0, .669, -.363, 0,
    0, 0, 0,
    .2, .2, 0, .6, 0, 0, 0,
    .2, -.2, 0, .6, 0, 0, 0,
], dtype=np.float64)

MUJOCO_TO_ISAACLAB = np.array(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
     16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]
)


def write_csv(path: Path, values: np.ndarray, header: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(values)


def quintic_boundary(q0: np.ndarray, v0: np.ndarray, a0: np.ndarray,
                     q1: np.ndarray, v1: np.ndarray, a1: np.ndarray,
                     duration: float, intervals: int) -> tuple[np.ndarray, np.ndarray]:
    """Quintic trajectory including both endpoints and exact endpoint rates."""
    a = np.empty((6, q0.size), dtype=np.float64)
    a[0], a[1], a[2] = q0, duration * v0, .5 * duration**2 * a0
    rhs = np.stack([
        q1 - (a[0] + a[1] + a[2]),
        duration * v1 - (a[1] + 2 * a[2]),
        duration**2 * a1 - 2 * a[2],
    ])
    matrix = np.array([[1., 1., 1.], [3., 4., 5.], [6., 12., 20.]])
    a[3:] = np.linalg.solve(matrix, rhs)
    s = np.linspace(0., 1., intervals + 1)
    powers = np.stack([s**i for i in range(6)], axis=1)
    dpowers = np.stack([
        np.zeros_like(s), np.ones_like(s), 2*s, 3*s**2, 4*s**3, 5*s**4
    ], axis=1) / duration
    return powers @ a, dpowers @ a


def smooth_quaternions(q0: np.ndarray, q1: np.ndarray, intervals: int) -> np.ndarray:
    """WXYZ quaternion interpolation with quintic time shaping."""
    xyzw = np.stack([q0[[1, 2, 3, 0]], q1[[1, 2, 3, 0]]])
    slerp = Slerp([0., 1.], Rotation.from_quat(xyzw))
    u = np.linspace(0., 1., intervals + 1)
    shaped = 10*u**3 - 15*u**4 + 6*u**5
    result = slerp(shaped).as_quat()
    return result[:, [3, 0, 1, 2]]


def main() -> int:
    with np.load(SOURCE) as archive:
        source_qpos = archive["qpos_mujoco"].astype(np.float64)
        source_q = archive["joint_pos_mujoco"].astype(np.float64)
        source_v = archive["joint_vel_mujoco"].astype(np.float64)
    if source_qpos.shape != (200, 36):
        raise ValueError(f"Unexpected source shape: {source_qpos.shape}")

    zero = np.zeros(29)
    entry_q, entry_v = quintic_boundary(
        DEFAULT_Q, zero, zero, source_q[0], source_v[0], zero,
        BLEND_S, BLEND_INTERVALS)
    exit_q, exit_v = quintic_boundary(
        source_q[-1], source_v[-1], zero, DEFAULT_Q, zero, zero,
        BLEND_S, BLEND_INTERVALS)
    entry_q[0], entry_v[0] = DEFAULT_Q, zero
    entry_q[-1], entry_v[-1] = source_q[0], source_v[0]
    exit_q[0], exit_v[0] = source_q[-1], source_v[-1]
    exit_q[-1], exit_v[-1] = DEFAULT_Q, zero

    root_p = source_qpos[:, :3]
    root_v = np.gradient(root_p, DT, axis=0, edge_order=2)
    entry_p, _ = quintic_boundary(
        root_p[0], np.zeros(3), np.zeros(3), root_p[0], np.zeros(3), np.zeros(3),
        BLEND_S, BLEND_INTERVALS)
    exit_p, _ = quintic_boundary(
        root_p[-1], np.zeros(3), np.zeros(3), root_p[0], np.zeros(3), np.zeros(3),
        BLEND_S, BLEND_INTERVALS)
    entry_quat = np.repeat(source_qpos[0, 3:7][None, :], BLEND_INTERVALS + 1, axis=0)
    exit_quat = smooth_quaternions(source_qpos[-1, 3:7], source_qpos[0, 3:7], BLEND_INTERVALS)

    # Keep every one of the original 200 C samples exactly.  Endpoint samples
    # are intentionally duplicated at the phase joins for a one-tick settle.
    q = np.vstack([entry_q, source_q, exit_q])
    v = np.vstack([entry_v, source_v, exit_v])
    pos = np.vstack([entry_p, root_p, exit_p])
    quat = np.vstack([entry_quat, source_qpos[:, 3:7], exit_quat])
    qpos = np.hstack([pos, quat, q])

    source_start = len(entry_q)
    source_stop = source_start + len(source_q)
    assert np.array_equal(q[source_start:source_stop], source_q)
    assert np.array_equal(pos[source_start:source_stop], root_p)
    assert np.array_equal(quat[source_start:source_stop], source_qpos[:, 3:7])
    assert np.array_equal(q[0], DEFAULT_Q) and np.array_equal(q[-1], DEFAULT_Q)

    # Quantify (do not clip) the small Hermite overshoot beyond the endpoints'
    # envelope.  Hard model-limit checking is performed in the MuJoCo pass.
    envelope_low = np.minimum(DEFAULT_Q, source_q.min(axis=0)) - 1e-9
    envelope_high = np.maximum(DEFAULT_Q, source_q.max(axis=0)) + 1e-9
    outside = np.argwhere((q < envelope_low) | (q > envelope_high))
    envelope_overshoot = np.maximum(envelope_low - q, q - envelope_high)

    REF.mkdir(parents=True, exist_ok=True)
    write_csv(REF / "joint_pos.csv", q[:, MUJOCO_TO_ISAACLAB],
              [f"joint_{i}" for i in range(29)])
    write_csv(REF / "joint_vel.csv", v[:, MUJOCO_TO_ISAACLAB],
              [f"joint_vel_{i}" for i in range(29)])
    write_csv(REF / "body_pos.csv", pos, [f"body_0_{x}" for x in "xyz"])
    write_csv(REF / "body_quat.csv", quat, [f"body_0_{x}" for x in "wxyz"])
    (REF / "metadata.txt").write_text(
        "Metadata for: C_proud_wave_entry_C_exit\n"
        "================================================\n\n"
        f"Body part indexes:\n[0]\n\nTotal timesteps: {len(q)}\n")
    (REF / "info.txt").write_text(
        "Real-trial transition wrapper; 50 Hz. Frame 0 is SONIC default hold.\n"
        "T plays a 1.5 s minimum-jerk entry, the exact repaired C 200-frame/4 s body, "
        "then a 1.5 s minimum-jerk return.\n")
    np.savez(OUT / "C_proud_wave_entry_C_exit_50hz.npz",
             time_s=np.arange(len(q)) / FPS, qpos_mujoco=qpos,
             joint_pos_mujoco=q, joint_vel_mujoco=v,
             root_pos=pos, root_quat_wxyz=quat,
             c_start_frame=source_start, c_stop_frame_exclusive=source_stop)

    transition_acc = np.gradient(v, DT, axis=0, edge_order=2)
    report = {
        "schema": "kimodo-g1.real-transition.v1",
        "source": str(SOURCE),
        "reference": str(REF),
        "fps": FPS,
        "entry_duration_s": BLEND_S,
        "c_start_frame": source_start,
        "c_stop_frame_exclusive": source_stop,
        "c_duration_s": 4.0,
        "c_frames_bit_identical": True,
        "exit_duration_s": BLEND_S,
        "total_frames": int(len(q)),
        "total_nominal_duration_s": len(q) / FPS,
        "hold_frame_is_sonic_default": True,
        "max_reference_speed_entry_rad_s": float(np.max(np.abs(entry_v))),
        "max_reference_accel_entry_rad_s2": float(np.max(np.abs(transition_acc[:len(entry_q)]))),
        "max_reference_speed_exit_rad_s": float(np.max(np.abs(exit_v))),
        "max_reference_accel_exit_rad_s2": float(np.max(np.abs(transition_acc[-len(exit_q):]))),
        "position_envelope_overshoot_count": int(len(outside)),
        "position_envelope_max_overshoot_rad": float(max(0., envelope_overshoot.max())),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "transition_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
