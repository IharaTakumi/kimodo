#!/usr/bin/env python3
"""Convert a folder of Kimodo G1 NPZ reactions to MotionDecode CSVs and summarize them.

Offline only: never imports Unitree/DDS code.
Usage: convert_reaction_batch.py <npz_dir> <out_dir>
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kimodo.exports.motion_convert_lib import convert_motion_files  # noqa: E402

HEADER_SOURCE = ROOT / "outputs/A_happy_celebration_motiondecode.csv"
FPS = 30.0
UPPER = slice(19, 36)  # waist(3) + arms(14), same window as MotionDecode reaction_profiles


def main() -> None:
    npz_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    g1_dir, md_dir = out_dir / "g1_csv", out_dir / "motiondecode_csv"
    g1_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)
    header = HEADER_SOURCE.read_text().splitlines()[0]
    names = header.split(",")
    rows = []
    for i, npz in enumerate(sorted(npz_dir.glob("*.npz")), 1):
        rid = f"r{i:02d}"
        g1_csv = g1_dir / f"{rid}.csv"
        convert_motion_files(str(npz), str(g1_csv), to_fmt="g1-csv")
        md_csv = md_dir / f"{rid}.csv"
        md_csv.write_text(header + "\n" + g1_csv.read_text())
        a = np.loadtxt(g1_csv, delimiter=",")
        up = a[:, UPPER]
        vel = np.abs(np.diff(up, axis=0)) * FPS
        worst = int(np.argmax(vel.max(axis=0)))
        root_xy = float(np.linalg.norm(a[-1, :2] - a[0, :2]))
        rows.append({
            "id": rid,
            "name": npz.stem,
            "frames": len(a),
            "duration_s": round(len(a) / FPS, 2),
            "upper_max_speed_rad_s": round(float(vel.max()), 2),
            "fastest_joint": names[19 + worst].replace("dof_", "").replace("_joint(rad)", ""),
            "upper_max_range_rad": round(float((up.max(0) - up.min(0)).max()), 2),
            "root_drift_m": round(root_xy, 2),
            "root_z_min_m": round(float(a[:, 2].min()), 2),
        })
    with (out_dir / "index.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out_dir / "index.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    for r in rows:
        print(r["id"], r["name"], r["duration_s"], r["upper_max_speed_rad_s"], r["fastest_joint"],
              r["upper_max_range_rad"], r["root_drift_m"], r["root_z_min_m"], sep="\t")


if __name__ == "__main__":
    main()
