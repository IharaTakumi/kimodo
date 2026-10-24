#!/usr/bin/env python3
"""Make a narrowly repaired 4 s SONIC reference for Kimodo motion C.

Only three joints may change: one shoulder per hand/hip interference and the
left wrist roll around its acceleration spikes.  Root motion, duration, and all
other joints remain bit-identical to the original 50 Hz conversion.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/C_proud_wave_dynamic/C_proud_wave_original_4s_50hz.npz"
OUT = ROOT / "outputs/C_proud_wave_dynamic_repaired_v2"
REF = OUT / "sonic_reference/C_proud_wave_dynamic_repaired_4s"
MODEL_XML = Path("/home/ubuntu/dev/sonic-g1-eval/gear_sonic_deploy/g1/scene_29dof.xml")
FPS = 50.0
CLEARANCE_M = 0.005

sys.path.insert(0, "/home/ubuntu/dev/motiondecode-test/scripts")
from common import JOINTS, NAMES  # noqa: E402
from scene import clearance_pairs, pair_clearances  # noqa: E402

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


def set_pose(model: mujoco.MjModel, data: mujoco.MjData, row: np.ndarray) -> None:
    data.qpos[:] = model.qpos0
    data.qpos[:7] = row[:7]
    data.qpos[3:7] /= np.linalg.norm(data.qpos[3:7])
    for i, name in enumerate(NAMES):
        data.qpos[model.joint(name).qposadr[0]] = row[7 + i]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def side_hand_hip_clearance(model, data, pairs, row, side: str) -> float:
    set_pose(model, data, row)
    values = pair_clearances(model, data, pairs)
    selected = [value for key, value in values.items()
                if f"{side}_rubber_hand" in key and f"{side}_hip_" in key]
    if not selected:
        raise RuntimeError(f"No configured {side} hand/hip clearance pair")
    return float(min(selected))


def repair_hand_hip(model, original: np.ndarray) -> tuple[np.ndarray, dict]:
    repaired = original.copy()
    data = mujoco.MjData(model)
    pairs = clearance_pairs(model)
    specs = {
        "right": (NAMES.index("right_shoulder_pitch_joint"), -1.0),
        "left": (NAMES.index("left_shoulder_roll_joint"), 1.0),
    }
    report = {}
    for side, (joint_index, direction) in specs.items():
        offsets = np.zeros(len(repaired))
        before = []
        for frame in range(len(repaired)):
            base = repaired[frame].copy()
            distance = side_hand_hip_clearance(model, data, pairs, base, side)
            before.append(distance)
            if distance >= CLEARANCE_M:
                continue
            low, high = 0.0, 0.25
            trial = base.copy()
            trial[7 + joint_index] += direction * high
            if side_hand_hip_clearance(model, data, pairs, trial, side) < CLEARANCE_M:
                raise RuntimeError(f"{side} hand/hip repair exceeds 0.25 rad at frame {frame}")
            for _ in range(24):
                mid = (low + high) / 2
                trial = base.copy()
                trial[7 + joint_index] += direction * mid
                if side_hand_hip_clearance(model, data, pairs, trial, side) >= CLEARANCE_M:
                    high = mid
                else:
                    low = mid
            offsets[frame] = direction * high
            repaired[frame, 7 + joint_index] += offsets[frame]
        after = [side_hand_hip_clearance(model, data, pairs, row, side)
                 for row in repaired]
        changed = np.flatnonzero(np.abs(offsets) > 1e-9)
        report[side] = {
            "joint": NAMES[joint_index],
            "minimum_before_m": float(min(before)),
            "minimum_after_m": float(min(after)),
            "changed_frame_count": int(len(changed)),
            "first_changed_frame": int(changed[0]) if len(changed) else None,
            "last_changed_frame": int(changed[-1]) if len(changed) else None,
            "maximum_abs_offset_rad": float(np.max(np.abs(offsets))),
        }
    return repaired, report


def smooth_wrist(repaired: np.ndarray) -> dict:
    index = NAMES.index("left_wrist_roll_joint")
    original = repaired[:, 7 + index].copy()
    smoothed = gaussian_filter1d(original, sigma=1.75, mode="reflect")
    # Acceleration peaks span roughly frames 50--132.  A cosine feather avoids
    # creating new jerk at either edge of the corrected interval.
    weight = np.zeros(len(original))
    weight[35:151] = 1.0
    ramp = 12
    weight[23:35] = .5 - .5 * np.cos(np.pi * np.arange(1, ramp + 1) / (ramp + 1))
    weight[151:163] = .5 + .5 * np.cos(np.pi * np.arange(1, ramp + 1) / (ramp + 1))
    corrected = original + weight * (smoothed - original)
    repaired[:, 7 + index] = corrected

    def dynamics(x):
        velocity = np.gradient(x, 1 / FPS, edge_order=2)
        acceleration = np.gradient(velocity, 1 / FPS, edge_order=2)
        return velocity, acceleration

    v0, a0 = dynamics(original)
    v1, a1 = dynamics(corrected)
    return {
        "joint": NAMES[index],
        "method": "Gaussian sigma=1.75 frames with cosine-feathered local window 23:162",
        "changed_frame_count": int(np.count_nonzero(np.abs(corrected - original) > 1e-9)),
        "maximum_abs_delta_rad": float(np.max(np.abs(corrected - original))),
        "maximum_speed_before_rad_s": float(np.max(np.abs(v0))),
        "maximum_speed_after_rad_s": float(np.max(np.abs(v1))),
        "maximum_acceleration_before_rad_s2": float(np.max(np.abs(a0))),
        "maximum_acceleration_after_rad_s2": float(np.max(np.abs(a1))),
        "range_retention_ratio": float(np.ptp(corrected) / np.ptp(original)),
    }


def add_sim_tracking_margins(repaired: np.ndarray) -> dict:
    """Add tiny smooth offsets only around residual Sim2Sim contact frames."""
    result = {}
    specs = {
        "right": (NAMES.index("right_shoulder_pitch_joint"), 126, -1.0, 7.0),
        "left": (NAMES.index("left_shoulder_roll_joint"), 194, 1.0, 7.0),
    }
    frames = np.arange(len(repaired))
    for side, (joint, center, direction, sigma) in specs.items():
        bump = 0.025 * np.exp(-0.5 * ((frames - center) / sigma) ** 2)
        bump[bump < 1e-4] = 0.0
        repaired[:, 7 + joint] += direction * bump
        changed = np.flatnonzero(bump > 0)
        result[side] = {
            "joint": NAMES[joint], "center_frame": center,
            "center_time_s": center / FPS,
            "maximum_additional_offset_rad": float(np.max(bump)),
            "first_changed_frame": int(changed[0]),
            "last_changed_frame": int(changed[-1]),
            "reason": "residual articulated-hand contact in first Sim2Sim validation",
        }
    return result


def render_comparison(model, original, repaired, path):
    renderers = [mujoco.Renderer(model, height=480, width=480) for _ in range(2)]
    datas = [mujoco.MjData(model), mujoco.MjData(model)]
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0, 0, .85]
    camera.distance, camera.azimuth, camera.elevation = 2.6, 135, -10
    with imageio.get_writer(path, fps=50, codec="libx264", quality=8,
                            macro_block_size=None) as writer:
        for frame in range(len(original)):
            images = []
            for i, row in enumerate((original[frame], repaired[frame])):
                set_pose(model, datas[i], row)
                renderers[i].update_scene(datas[i], camera=camera)
                image = Image.fromarray(renderers[i].render().copy())
                label = ("ORIGINAL" if i == 0 else "LOCAL REPAIR") + f"  {frame/FPS:04.2f}s"
                ImageDraw.Draw(image).text((12, 12), label, fill="white",
                                           stroke_width=2, stroke_fill="black")
                images.append(np.asarray(image))
            writer.append_data(np.hstack(images))
    for renderer in renderers:
        renderer.close()


def main() -> int:
    with np.load(SOURCE) as archive:
        original = archive["qpos_mujoco"].copy()
    if original.shape != (200, 36):
        raise ValueError(f"Unexpected original shape {original.shape}")
    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    repaired, clearance = repair_hand_hip(model, original)
    tracking_margins = add_sim_tracking_margins(repaired)
    wrist = smooth_wrist(repaired)

    q0, q1 = original[:, 7:], repaired[:, 7:]
    changed_joints = [NAMES[i] for i in np.flatnonzero(np.any(q0 != q1, axis=0))]
    allowed = {"left_shoulder_roll_joint", "right_shoulder_pitch_joint",
               "left_wrist_roll_joint"}
    assert set(changed_joints) <= allowed
    assert np.array_equal(repaired[:, :7], original[:, :7])
    lower = np.array([joint["lower"] for joint in JOINTS])
    upper = np.array([joint["upper"] for joint in JOINTS])
    violations = np.argwhere((q1 < lower) | (q1 > upper))
    if len(violations):
        raise RuntimeError(f"Repair causes {len(violations)} joint-limit violations")

    velocity = np.gradient(q1, 1 / FPS, axis=0, edge_order=2)
    q_isaac = q1[:, MUJOCO_TO_ISAACLAB]
    v_isaac = velocity[:, MUJOCO_TO_ISAACLAB]
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "C_proud_wave_dynamic_repaired_4s_50hz.npz",
             time_s=np.arange(200) / FPS, qpos_mujoco=repaired,
             joint_pos_mujoco=q1, joint_vel_mujoco=velocity,
             root_pos=repaired[:, :3], root_quat_wxyz=repaired[:, 3:7])
    write_csv(REF / "joint_pos.csv", q_isaac, [f"joint_{i}" for i in range(29)])
    write_csv(REF / "joint_vel.csv", v_isaac, [f"joint_vel_{i}" for i in range(29)])
    write_csv(REF / "body_pos.csv", repaired[:, :3], [f"body_0_{x}" for x in "xyz"])
    write_csv(REF / "body_quat.csv", repaired[:, 3:7], [f"body_0_{x}" for x in "wxyz"])
    (REF / "metadata.txt").write_text(
        "Metadata for: C_proud_wave_dynamic_repaired_4s\n"
        "================================================\n\n"
        "Body part indexes:\n[0]\n\nTotal timesteps: 200\n")
    (REF / "info.txt").write_text(
        "Local repair of original Kimodo C; 4.0 s, 200 frames at 50 Hz.\n"
        "Root/timing unchanged; no global scaling, clipping, freezing, or retiming.\n")

    ranges0, ranges1 = np.ptp(q0, axis=0), np.ptp(q1, axis=0)
    report = {
        "schema": "kimodo-g1.dynamic-local-repair.v1",
        "source_preserved": str(SOURCE),
        "frames": 200, "fps": FPS, "duration_s": 4.0,
        "root_bit_identical": True, "global_amplitude_scaling": False,
        "joint_freezing": False, "time_stretching": False, "joint_clipping": False,
        "changed_joints": changed_joints,
        "unchanged_joint_count": 29 - len(changed_joints),
        "clearance_repairs": clearance,
        "sim_tracking_margin_repairs": tracking_margins,
        "wrist_repair": wrist,
        "whole_body_joint_range_sum_retention_ratio": float(ranges1.sum() / ranges0.sum()),
        "left_arm_range_sum_retention_ratio": float(ranges1[15:22].sum() / ranges0[15:22].sum()),
        "hard_joint_limit_violation_count": 0,
    }
    (OUT / "local_repair_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    render_comparison(model, original, repaired, OUT / "C_original_vs_local_repair.mp4")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
