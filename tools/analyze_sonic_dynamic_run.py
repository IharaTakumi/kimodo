#!/usr/bin/env python3
"""Analyze and render the completed SONIC Sim2Sim run for original Kimodo C."""

from __future__ import annotations

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
OUT = Path(os.environ.get("SONIC_ANALYSIS_OUT", ROOT / "outputs/C_proud_wave_dynamic"))
REF = Path(os.environ.get("SONIC_ANALYSIS_REF", OUT / "C_proud_wave_original_4s_50hz.npz"))
RAW = Path(os.environ.get("SONIC_ANALYSIS_RAW", OUT / "sonic_sim_raw.npz"))
LOGS = Path(os.environ.get("SONIC_ANALYSIS_LOGS", OUT / "sonic_logs_run2"))
IS_LOCAL_REPAIR = os.environ.get("SONIC_ANALYSIS_LOCAL_REPAIR") == "1"
SONIC = Path("/home/ubuntu/dev/sonic-g1-eval")
MODEL_XML = SONIC / "gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml"
FPS = 50.0

sys.path.insert(0, "/home/ubuntu/dev/motiondecode-test/scripts")
from common import NAMES  # noqa: E402


def csv_values(name: str) -> np.ndarray:
    return np.genfromtxt(LOGS / name, delimiter=",", skip_header=1)


def tilt_rad(quat: np.ndarray) -> float:
    mat = np.empty(9)
    mujoco.mju_quat2Mat(mat, quat)
    return float(math.acos(np.clip(mat.reshape(3, 3)[2, 2], -1.0, 1.0)))


def align_sim(model: mujoco.MjModel, raw_qpos: np.ndarray,
              measured_log: np.ndarray) -> tuple[int, float, float]:
    sim_q = np.stack(
        [raw_qpos[:, model.joint(n).qposadr[0]] for n in NAMES], axis=1
    )
    length = len(measured_log)
    scores = [np.mean((sim_q[o:o + length] - measured_log) ** 2)
              for o in range(len(sim_q) - length + 1)]
    offset = int(np.argmin(scores))
    maximum = float(np.max(np.abs(sim_q[offset:offset + length] - measured_log)))
    return offset, float(scores[offset]), maximum


def tracking_stats(actual: np.ndarray, target: np.ndarray, idx: np.ndarray) -> dict:
    error = actual[:, idx] - target[:, idx]
    trange = np.ptp(target[:, idx], axis=0)
    arange = np.ptp(actual[:, idx], axis=0)
    return {
        "rms_error_rad": float(np.sqrt(np.mean(error * error))),
        "maximum_error_rad": float(np.max(np.abs(error))),
        "target_joint_range_sum_rad": float(trange.sum()),
        "actual_joint_range_sum_rad": float(arange.sum()),
        "range_sum_retention_ratio": float(arange.sum() / trange.sum()),
        "target_max_single_joint_range_rad": float(trange.max()),
        "actual_max_single_joint_range_rad": float(arange.max()),
    }


def evaluate_physics(model: mujoco.MjModel, qpos: np.ndarray) -> dict:
    data = mujoco.MjData(model)
    floor = model.geom("floor").id
    falls, root_z, tilts, coms, floor_sides = [], [], [], [], []
    self_contacts: dict[str, dict] = {}
    max_floor_pen = 0.0
    for frame, row in enumerate(qpos):
        data.qpos[:] = row
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        z = float(row[2])
        tilt = tilt_rad(row[3:7])
        root_z.append(z)
        tilts.append(tilt)
        coms.append(data.subtree_com[0].copy())
        if z < .55 or tilt > .7:
            falls.append(frame)
        sides = set()
        for k in range(data.ncon):
            c = data.contact[k]
            g1, g2 = int(c.geom1), int(c.geom2)
            if floor in (g1, g2):
                max_floor_pen = max(max_floor_pen, max(0.0, -float(c.dist)))
                other = g2 if g1 == floor else g1
                body = model.body(int(model.geom_bodyid[other])).name
                if body.startswith("left_"):
                    sides.add("left")
                if body.startswith("right_"):
                    sides.add("right")
            else:
                b1 = model.body(int(model.geom_bodyid[g1])).name
                b2 = model.body(int(model.geom_bodyid[g2])).name
                if b1 == b2:
                    continue
                key = " | ".join(sorted((b1, b2)))
                old = self_contacts.get(key)
                if old is None or float(c.dist) < old["minimum_distance_m"]:
                    self_contacts[key] = {
                        "bodies": sorted((b1, b2)),
                        "minimum_distance_m": float(c.dist),
                        "frame": frame,
                        "time_s": frame / FPS,
                    }
        floor_sides.append(sorted(sides))
    coms = np.asarray(coms)
    return {
        "fall_detected": bool(falls),
        "fall_frames": falls,
        "minimum_root_height_m": float(min(root_z)),
        "maximum_root_height_m": float(max(root_z)),
        "maximum_root_tilt_rad": float(max(tilts)),
        "maximum_floor_penetration_m": max_floor_pen,
        "both_feet_contact_fraction": float(np.mean([x == ["left", "right"] for x in floor_sides])),
        "whole_body_com_axis_range_m": np.ptp(coms, axis=0).tolist(),
        "whole_body_com_path_length_m": float(np.linalg.norm(np.diff(coms, axis=0), axis=1).sum()),
        "self_contact_pairs": sorted(self_contacts.values(), key=lambda x: x["minimum_distance_m"]),
    }


def target_qpos(model: mujoco.MjModel, root_and_joints: np.ndarray) -> np.ndarray:
    result = np.tile(model.qpos0, (len(root_and_joints), 1))
    result[:, :7] = root_and_joints[:, :7]
    for i, name in enumerate(NAMES):
        result[:, model.joint(name).qposadr[0]] = root_and_joints[:, 7 + i]
    return result


def renderer_camera() -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.0, 0.0, .85]
    camera.distance, camera.azimuth, camera.elevation = 2.6, 135, -10
    return camera


def render_video(model: mujoco.MjModel, rows: list[np.ndarray], labels: list[str],
                 path: Path) -> None:
    renderers = [mujoco.Renderer(model, height=480, width=480) for _ in rows]
    datas = [mujoco.MjData(model) for _ in rows]
    camera = renderer_camera()
    with imageio.get_writer(path, fps=50, codec="libx264", quality=8,
                            macro_block_size=None) as writer:
        for frame in range(len(rows[0])):
            images = []
            for i, trajectory in enumerate(rows):
                datas[i].qpos[:] = trajectory[frame]
                datas[i].qvel[:] = 0.0
                mujoco.mj_forward(model, datas[i])
                renderers[i].update_scene(datas[i], camera=camera)
                image = Image.fromarray(renderers[i].render().copy())
                ImageDraw.Draw(image).text(
                    (12, 12), f"{labels[i]}  {frame/FPS:04.2f}s", fill="white",
                    stroke_width=2, stroke_fill="black")
                images.append(np.asarray(image))
            writer.append_data(np.hstack(images))
    for renderer in renderers:
        renderer.close()


def main() -> int:
    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    with np.load(REF) as z:
        target_joint = z["joint_pos_mujoco"]
        target_full = np.c_[z["root_pos"], z["root_quat_wxyz"], target_joint]
    with np.load(RAW) as z:
        raw_time, raw_qpos = z["time_s"], z["qpos"]
    q_log = csv_values("q.csv")[:, 5:]
    playing = csv_values("motion_playing.csv")[:, -1] > .5
    play_log_indices = np.where(playing)[0]
    if len(play_log_indices) != 200 or not np.all(np.diff(play_log_indices) == 1):
        raise ValueError(f"Expected one complete 200-frame playback, got {len(play_log_indices)}")
    offset, align_mse, align_max = align_sim(model, raw_qpos, q_log)
    raw_indices = offset + play_log_indices
    measured_qpos = raw_qpos[raw_indices]
    actual_joint = q_log[playing]

    groups = {
        "left_leg": np.arange(0, 6), "right_leg": np.arange(6, 12),
        "waist": np.arange(12, 15), "left_arm": np.arange(15, 22),
        "right_arm": np.arange(22, 29), "all": np.arange(29),
    }
    tracking = {name: tracking_stats(actual_joint, target_joint, idx)
                for name, idx in groups.items()}
    err = actual_joint - target_joint
    worst = np.unravel_index(np.argmax(np.abs(err)), err.shape)
    dq_actual = np.gradient(actual_joint, 1 / FPS, axis=0, edge_order=2)
    torque = csv_values("motor_torque.csv")[playing, 5:]
    physics = evaluate_physics(model, measured_qpos)
    target_model_qpos = target_qpos(model, target_full)

    render_video(model, [measured_qpos], ["SONIC Sim2Sim"], OUT / "C_sonic_sim2sim.mp4")
    render_video(model, [target_model_qpos, measured_qpos],
                 ["Kimodo target", "SONIC result"], OUT / "C_target_vs_sonic.mp4")
    np.savez(OUT / "C_sonic_playback_50hz.npz", time_s=np.arange(200) / FPS,
             qpos=measured_qpos, joint_pos_mujoco=actual_joint,
             joint_vel_mujoco=dq_actual, target_joint_pos_mujoco=target_joint,
             raw_sample_indices=raw_indices)

    report = {
        "schema": "kimodo-g1.sonic-dynamic-validation.v1",
        "offline_simulation_only": True,
        "robot_commands_sent": False,
        "reference": {"frames": 200, "fps": FPS, "duration_s": 4.0,
                      "amplitude_scaled": False, "joints_frozen": False,
                      "time_stretched": False},
        "sonic": {
            "repository_commit": "45c03facbf848e0c743091b7d9c17952096ff908",
            "policy": "official nvidia/GEAR-SONIC default encoder+decoder",
            "tensor_rt_version": "10.13.3.9", "policy_precision": "FP32",
            "motion_completed": True, "frames_played": 200,
            "control_frequency_hz": 50,
            "observed_policy_inference_ms_range": [0.131, 0.136],
        },
        "alignment": {"sim_raw_offset": offset,
                      "sim_motion_start_time_s": float(raw_time[raw_indices[0]]),
                      "sim_motion_end_time_s": float(raw_time[raw_indices[-1]]),
                      "raw_vs_cpp_q_mse": align_mse,
                      "raw_vs_cpp_q_max_abs_rad": align_max},
        "tracking": tracking,
        "worst_tracking_sample": {"frame": int(worst[0]), "time_s": worst[0] / FPS,
                                  "joint": NAMES[worst[1]],
                                  "error_rad": float(err[worst]),
                                  "target_rad": float(target_joint[worst]),
                                  "actual_rad": float(actual_joint[worst])},
        "dynamics": {"maximum_actual_joint_speed_rad_s": float(np.max(np.abs(dq_actual))),
                     "maximum_logged_motor_torque_nm": float(np.max(np.abs(torque)))},
        "physics": physics,
        "verdict": {
            "full_4s_motion_completed": True,
            "stable_without_elastic_support": not physics["fall_detected"],
            "tempo_preserved": True,
            "major_left_arm_motion_retained": tracking["left_arm"]["range_sum_retention_ratio"] >= .7,
            "tracking_quality": "PASS_WITH_VISIBLE_TRACKING_ERROR",
            "real_robot_ready": False,
            "remaining_blockers": (([] if IS_LOCAL_REPAIR else [
                "Original kinematic target contains hand-to-hip penetrations; locally repair those frames before hardware consideration.",
            ]) + [
                "Maximum simulated tracking error exceeds the existing 0.1 rad MotionDecode criterion.",
                "Hardware controller exclusion, fresh posture checks, support rig, and emergency stop remain mandatory.",
            ]),
        },
    }
    (OUT / "sonic_dynamic_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
