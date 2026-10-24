#!/usr/bin/env python3
"""Preserve the original 4 s Kimodo C motion as a SONIC G1 reference.

This is an offline-only converter/analyzer/renderer.  It performs temporal
resampling from 120 frames to 200 frames without amplitude scaling, joint
freezing, joint clipping, or time stretching.  No robot SDK is imported.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/C_proud_wave_dynamic"
SONIC_REF = OUT / "sonic_reference/C_proud_wave_original_4s"
SOURCE_NPZ = ROOT / "outputs/C_proud_wave.npz"
SOURCE_CSV = ROOT / "outputs/C_proud_wave_g1.csv"
OLD_SAFE = ROOT / "outputs/C_proud_wave_safe_candidate/C_proud_wave_safe_50hz.npz"
MOTIONDECODE = Path("/home/ubuntu/dev/motiondecode-test")
SONIC = Path("/home/ubuntu/dev/sonic-g1-eval")
MODEL_XML = SONIC / "gear_sonic_deploy/g1/scene_29dof.xml"
SOURCE_FRAMES = 120
SOURCE_FPS = 30.0
OUTPUT_FRAMES = 200
OUTPUT_FPS = 50.0

ISAACLAB_TO_MUJOCO = np.array(
    [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8,
     11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28]
)
MUJOCO_TO_ISAACLAB = np.array(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
     16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]
)

sys.path.insert(0, str(MOTIONDECODE / "scripts"))
from common import JOINTS, NAMES  # noqa: E402
from scene import clearance_pairs, pair_clearances  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, a: np.ndarray, header: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(a)


def set_pose(model: mujoco.MjModel, data: mujoco.MjData, row: np.ndarray) -> None:
    data.qpos[:] = model.qpos0
    data.qpos[:3] = row[:3]
    data.qpos[3:7] = row[3:7] / np.linalg.norm(row[3:7])
    for i, name in enumerate(NAMES):
        data.qpos[model.joint(name).qposadr[0]] = row[7 + i]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def resample_original(a: np.ndarray) -> np.ndarray:
    src = np.linspace(0.0, 1.0, SOURCE_FRAMES)
    dst = np.linspace(0.0, 1.0, OUTPUT_FRAMES)
    root = np.stack([np.interp(dst, src, a[:, i]) for i in range(3)], axis=1)
    q_wxyz = a[:, 3:7] / np.linalg.norm(a[:, 3:7], axis=1, keepdims=True)
    x = dst * (SOURCE_FRAMES - 1)
    lo = np.floor(x).astype(int)
    hi = np.minimum(lo + 1, SOURCE_FRAMES - 1)
    u = x - lo
    qa, qb = q_wxyz[lo], q_wxyz[hi].copy()
    dots = np.sum(qa * qb, axis=1)
    qb[dots < 0] *= -1
    dots = np.abs(dots).clip(-1, 1)
    theta = np.arccos(dots)
    sin_theta = np.sin(theta)
    near = sin_theta < 1e-8
    wa = np.empty_like(theta)
    wb = np.empty_like(theta)
    wa[near], wb[near] = 1 - u[near], u[near]
    wa[~near] = np.sin((1 - u[~near]) * theta[~near]) / sin_theta[~near]
    wb[~near] = np.sin(u[~near] * theta[~near]) / sin_theta[~near]
    quat_wxyz = wa[:, None] * qa + wb[:, None] * qb
    quat_wxyz /= np.linalg.norm(quat_wxyz, axis=1, keepdims=True)
    joints = np.stack([np.interp(dst, src, a[:, 7 + i]) for i in range(29)], axis=1)
    return np.c_[root, quat_wxyz, joints]


def analyze(model: mujoco.MjModel, a: np.ndarray, source_npz: dict) -> dict:
    data = mujoco.MjData(model)
    q = a[:, 7:]
    dt = 1.0 / OUTPUT_FPS
    qd = np.gradient(q, dt, axis=0, edge_order=2)
    qdd = np.gradient(qd, dt, axis=0, edge_order=2)
    lower = np.array([j["lower"] for j in JOINTS])
    upper = np.array([j["upper"] for j in JOINTS])
    lower_margin, upper_margin = lower + .03, upper - .03
    hard = np.argwhere((q < lower) | (q > upper))
    margin = np.argwhere((q < lower_margin) | (q > upper_margin))

    pairs = clearance_pairs(model)
    pair_minima: dict[str, dict] = {}
    com, floor_pen, floor_sides = [], [], []
    floor_id = model.geom("floor").id
    for frame, row in enumerate(a):
        set_pose(model, data, row)
        com.append(data.subtree_com[0].copy())
        for key, value in pair_clearances(model, data, pairs).items():
            prev = pair_minima.get(key)
            if prev is None or value < prev["distance_m"]:
                pair_minima[key] = {"distance_m": value, "frame": frame,
                                    "time_s": frame / OUTPUT_FPS}
        pen, sides = 0.0, set()
        for k in range(data.ncon):
            c = data.contact[k]
            if floor_id not in (int(c.geom1), int(c.geom2)):
                continue
            pen = max(pen, max(0.0, -float(c.dist)))
            other = int(c.geom2) if int(c.geom1) == floor_id else int(c.geom1)
            body = model.body(int(model.geom_bodyid[other])).name
            if body.startswith("left_"):
                sides.add("left")
            elif body.startswith("right_"):
                sides.add("right")
        floor_pen.append(pen)
        floor_sides.append(sorted(sides))
    com = np.asarray(com)

    group_defs = {
        "left_leg": np.arange(0, 6), "right_leg": np.arange(6, 12),
        "waist": np.arange(12, 15), "left_arm": np.arange(15, 22),
        "right_arm": np.arange(22, 29),
    }
    groups = {}
    for name, idx in group_defs.items():
        ranges = np.ptp(q[:, idx], axis=0)
        groups[name] = {
            "joint_range_sum_rad": float(ranges.sum()),
            "maximum_single_joint_range_rad": float(ranges.max()),
            "maximum_abs_velocity_rad_s": float(np.max(np.abs(qd[:, idx]))),
            "maximum_abs_acceleration_rad_s2": float(np.max(np.abs(qdd[:, idx]))),
            "per_joint_range_rad": {NAMES[i]: float(np.ptp(q[:, i])) for i in idx},
        }
    worst_v = np.unravel_index(np.argmax(np.abs(qd)), qd.shape)
    worst_a = np.unravel_index(np.argmax(np.abs(qdd)), qdd.shape)
    deficits = [dict(pair=k.split(" | "), **v) for k, v in pair_minima.items()
                if v["distance_m"] < .015]
    collisions = [x for x in deficits if x["distance_m"] < 0.0]
    return {
        "source": {"npz": str(SOURCE_NPZ), "npz_sha256": digest(SOURCE_NPZ),
                   "csv": str(SOURCE_CSV), "csv_sha256": digest(SOURCE_CSV),
                   "frames": SOURCE_FRAMES, "fps": SOURCE_FPS,
                   "nominal_duration_s": 4.0},
        "conversion": {"method": "linear temporal interpolation + quaternion SLERP",
                       "amplitude_scaling": False, "joint_freezing": False,
                       "joint_clipping": False, "time_stretching": False,
                       "frames": OUTPUT_FRAMES, "fps": OUTPUT_FPS,
                       "nominal_duration_s": 4.0},
        "motion": {
            "groups": groups,
            "maximum_abs_velocity_rad_s": float(abs(qd[worst_v])),
            "maximum_velocity_joint": NAMES[worst_v[1]],
            "maximum_velocity_frame": int(worst_v[0]),
            "maximum_abs_acceleration_rad_s2": float(abs(qdd[worst_a])),
            "maximum_acceleration_joint": NAMES[worst_a[1]],
            "maximum_acceleration_frame": int(worst_a[0]),
            "root_axis_range_m": np.ptp(a[:, :3], axis=0).tolist(),
            "root_path_length_m": float(np.linalg.norm(np.diff(a[:, :3], axis=0), axis=1).sum()),
            "whole_body_com_axis_range_m": np.ptp(com, axis=0).tolist(),
            "whole_body_com_path_length_m": float(np.linalg.norm(np.diff(com, axis=0), axis=1).sum()),
            "kimodo_foot_contacts_all_true": bool(np.all(source_npz["foot_contacts"])),
        },
        "feasibility": {
            "hard_joint_limit_violation_count": int(len(hard)),
            "hard_joint_limit_violations": [
                {"frame": int(f), "time_s": float(f / OUTPUT_FPS), "joint": NAMES[j],
                 "value_rad": float(q[f, j]), "lower_rad": float(lower[j]), "upper_rad": float(upper[j])}
                for f, j in hard[:100]
            ],
            "joint_30mrad_margin_violation_count": int(len(margin)),
            "minimum_configured_self_clearance_m": float(min(x["distance_m"] for x in pair_minima.values())),
            "clearance_threshold_m": .015,
            "clearance_deficits": sorted(deficits, key=lambda x: x["distance_m"]),
            "penetrating_pairs": sorted(collisions, key=lambda x: x["distance_m"]),
            "maximum_floor_penetration_m": float(max(floor_pen)),
            "both_feet_contact_frame_fraction": float(np.mean([x == ["left", "right"] for x in floor_sides])),
        },
    }


def camera() -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.0, 0.0, .85]
    cam.distance, cam.azimuth, cam.elevation = 2.6, 135, -10
    return cam


def render_motion(model: mujoco.MjModel, a: np.ndarray, path: Path,
                  label: str, fps: int = 50) -> None:
    renderer = mujoco.Renderer(model, height=480, width=480)
    data, cam = mujoco.MjData(model), camera()
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=8,
                            macro_block_size=None) as writer:
        for i, row in enumerate(a):
            set_pose(model, data, row)
            renderer.update_scene(data, camera=cam)
            image = Image.fromarray(renderer.render().copy())
            d = ImageDraw.Draw(image)
            d.text((14, 14), f"{label}  {i / fps:04.2f} s", fill="white",
                   stroke_width=2, stroke_fill="black")
            writer.append_data(np.asarray(image))
    renderer.close()


def render_comparison(model: mujoco.MjModel, original: np.ndarray,
                      rejected: np.ndarray, path: Path) -> None:
    renderers = [mujoco.Renderer(model, height=480, width=480) for _ in range(2)]
    datas = [mujoco.MjData(model), mujoco.MjData(model)]
    cam = camera()
    with imageio.get_writer(path, fps=50, codec="libx264", quality=8,
                            macro_block_size=None) as writer:
        for i in range(OUTPUT_FRAMES):
            # Compare equal semantic progress.  Labels retain the actual clip
            # durations so this does not disguise the rejected 5x slowdown.
            j = int(round(i / (OUTPUT_FRAMES - 1) * (len(rejected) - 1)))
            frames = []
            for k, row in enumerate((original[i], rejected[j])):
                set_pose(model, datas[k], row)
                renderers[k].update_scene(datas[k], camera=cam)
                frames.append(Image.fromarray(renderers[k].render().copy()))
            labels = (f"ORIGINAL full body 4 s  {i/50:04.2f}s",
                      f"REJECTED left arm 20 s  {j/50:04.2f}s")
            for frame, label in zip(frames, labels):
                ImageDraw.Draw(frame).text((14, 14), label, fill="white",
                                           stroke_width=2, stroke_fill="black")
            writer.append_data(np.hstack([np.asarray(x) for x in frames]))
    for renderer in renderers:
        renderer.close()


def main() -> int:
    raw = np.loadtxt(SOURCE_CSV, delimiter=",")
    if raw.shape != (SOURCE_FRAMES, 36) or not np.isfinite(raw).all():
        raise ValueError(f"Unexpected source CSV {raw.shape}")
    with np.load(SOURCE_NPZ, allow_pickle=False) as z:
        source_npz = {k: z[k].copy() for k in z.files}
    a = resample_original(raw)
    q = a[:, 7:]
    qd = np.gradient(q, 1 / OUTPUT_FPS, axis=0, edge_order=2)
    q_isaac, qd_isaac = q[:, MUJOCO_TO_ISAACLAB], qd[:, MUJOCO_TO_ISAACLAB]
    assert np.allclose(q_isaac[:, ISAACLAB_TO_MUJOCO], q)

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "C_proud_wave_original_4s_50hz.npz",
             time_s=np.arange(OUTPUT_FRAMES) / OUTPUT_FPS,
             qpos_mujoco=a, joint_pos_mujoco=q, joint_vel_mujoco=qd,
             root_pos=a[:, :3], root_quat_wxyz=a[:, 3:7])
    write_csv(OUT / "C_proud_wave_original_4s_g1_qpos_50hz.csv", a,
              [f"qpos_{i}" for i in range(36)])
    write_csv(SONIC_REF / "joint_pos.csv", q_isaac, [f"joint_{i}" for i in range(29)])
    write_csv(SONIC_REF / "joint_vel.csv", qd_isaac, [f"joint_vel_{i}" for i in range(29)])
    write_csv(SONIC_REF / "body_pos.csv", a[:, :3], [f"body_0_{x}" for x in "xyz"])
    write_csv(SONIC_REF / "body_quat.csv", a[:, 3:7], [f"body_0_{x}" for x in "wxyz"])
    (SONIC_REF / "metadata.txt").write_text(
        "Metadata for: C_proud_wave_original_4s\n"
        "========================================\n\n"
        "Body part indexes:\n[0]\n\nTotal timesteps: 200\n"
    )
    (SONIC_REF / "info.txt").write_text(
        "Original Kimodo C; 4.0 s nominal; 200 frames at 50 Hz.\n"
        "No amplitude scaling, joint freezing/clipping, or time stretching.\n"
    )

    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    report = analyze(model, a, source_npz)
    (OUT / "original_motion_analysis.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    render_motion(model, a, OUT / "C_proud_wave_original_4s.mp4",
                  "Kimodo C original full body", 50)
    if OLD_SAFE.exists():
        with np.load(OLD_SAFE) as z:
            rejected = np.c_[z["root_pos"], z["root_quat_wxyz"], z["joint_pos_mujoco"]]
        render_comparison(model, a, rejected, OUT / "C_original_vs_rejected_safe.mp4")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
