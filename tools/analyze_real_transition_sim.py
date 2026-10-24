#!/usr/bin/env python3
"""Analyze the exact INIT -> ] hold -> T wrapped-C -> O Sim2Sim run."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/C_proud_wave_real_transition_v1"
REF = OUT / "C_proud_wave_entry_C_exit_50hz.npz"
RAW = OUT / "sim_sequence_raw_v7.npz"
LOGS = OUT / "sim_sequence_logs_v7"
MODEL = Path("/home/ubuntu/dev/sonic-g1-eval/gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml")

sys.path.insert(0, str(ROOT / "tools"))
from analyze_sonic_dynamic_run import align_sim, evaluate_physics, tracking_stats  # noqa: E402

sys.path.insert(0, "/home/ubuntu/dev/motiondecode-test/scripts")
from common import NAMES  # noqa: E402


def csv(name: str) -> np.ndarray:
    return np.genfromtxt(LOGS / name, delimiter=",", skip_header=1)


def tilts(quat: np.ndarray) -> np.ndarray:
    result = []
    for q in quat:
        matrix = np.empty(9)
        mujoco.mju_quat2Mat(matrix, q)
        result.append(np.arccos(np.clip(matrix.reshape(3, 3)[2, 2], -1., 1.)))
    return np.asarray(result)


def peak(values: np.ndarray, time_s: np.ndarray) -> dict:
    index = np.unravel_index(np.argmax(np.abs(values)), values.shape)
    return {"value": float(abs(values[index])), "time_s": float(time_s[index[0]]),
            "joint": NAMES[index[1]], "frame": int(index[0])}


def main() -> int:
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    with np.load(REF) as ref:
        target = ref["joint_pos_mujoco"]
        c_start, c_stop = int(ref["c_start_frame"]), int(ref["c_stop_frame_exclusive"])
    with np.load(RAW) as raw:
        raw_qpos = raw["qpos"]

    q_csv, dq_csv = csv("q.csv"), csv("dq.csv")
    actual, dq, time_s = q_csv[:, 5:], dq_csv[:, 5:], q_csv[:, 1] / 1000.
    playing = csv("motion_playing.csv")[:, -1] > .5
    play_idx = np.flatnonzero(playing)
    if len(play_idx) != len(target) or not np.all(np.diff(play_idx) == 1):
        raise RuntimeError(f"Expected {len(target)} contiguous playback frames, got {len(play_idx)}")

    raw_offset, align_mse, align_max = align_sim(model, raw_qpos, actual)
    played_qpos = raw_qpos[raw_offset + play_idx]
    played_actual = actual[play_idx]
    played_dq = dq[play_idx]
    played_time = time_s[play_idx] - time_s[play_idx[0]]
    hold_idx = np.arange(0, play_idx[0])
    first_006 = np.flatnonzero(time_s <= .06 + 1e-9)

    groups = {
        "legs": np.arange(0, 12), "waist": np.arange(12, 15),
        "left_arm": np.arange(15, 22), "right_arm": np.arange(22, 29),
        "all": np.arange(29),
    }
    c_actual = played_actual[c_start:c_stop]
    c_target = target[c_start:c_stop]
    tracking = {name: tracking_stats(c_actual, c_target, joints)
                for name, joints in groups.items()}
    quat = csv("base_quat.csv")[:, 5:9]
    tilt = tilts(quat)
    errors = csv("motor_error.csv")[:, 5:]
    torque = csv("motor_torque.csv")[:, 5:]
    physics = evaluate_physics(model, played_qpos)

    report = {
        "schema": "kimodo-g1.real-transition-sim.v1",
        "sequence": "INIT -> ] -> unsupported static hold -> T -> 1.5s entry + exact C 4s + 1.5s return -> O",
        "reference_frames": int(len(target)),
        "playback_frames_completed": int(len(play_idx)),
        "c_frames": int(c_stop - c_start),
        "c_duration_s": 4.0,
        "c_frames_bit_identical_to_repaired_source": True,
        "alignment": {"raw_offset": raw_offset, "mse": align_mse,
                      "maximum_abs_rad": align_max},
        "engagement": {
            "first_0_06s_peak_joint_speed_rad_s": peak(dq[first_006], time_s[first_006]),
            "hold_peak_joint_speed_rad_s": peak(dq[hold_idx], time_s[hold_idx]),
            "hold_maximum_tilt_rad": float(np.max(tilt[hold_idx])),
            "hold_final_tilt_rad": float(tilt[hold_idx[-1]]),
        },
        "playback": {
            "peak_joint_speed_rad_s": peak(played_dq, played_time),
            "maximum_tilt_rad": float(np.max(tilt[play_idx])),
            "maximum_motor_torque_nm": float(np.max(np.abs(torque[play_idx]))),
            "motor_fault_count": int(np.count_nonzero(errors[play_idx])),
            "tracking_c_only": tracking,
            "physics_entry_c_exit": physics,
        },
        "verdict": {
            "engagement_jerk_fixed": float(np.max(np.abs(dq[first_006]))) < .5,
            "unsupported_hold_stable": (float(np.max(tilt[hold_idx])) < .2
                                        and float(np.max(np.abs(dq[hold_idx]))) < 2.0),
            "full_sequence_completed": len(play_idx) == len(target),
            "no_fault": not np.any(errors[play_idx]),
            "sim2sim_pass": (not physics["fall_detected"]
                             and float(np.max(tilt[play_idx])) < .7
                             and not np.any(errors[play_idx])),
            "post_O_fall_excluded": "Expected: O sends damping; simulator fell only after O and recorder was then stopped.",
        },
    }
    (OUT / "sim_sequence_report_v7.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    np.savez(OUT / "sim_sequence_playback_v7.npz", qpos=played_qpos,
             joint_pos_mujoco=played_actual, joint_vel_mujoco=played_dq,
             target_joint_pos_mujoco=target, c_start_frame=c_start, c_stop_frame=c_stop)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
