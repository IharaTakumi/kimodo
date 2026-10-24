#!/usr/bin/env python3
"""Run the official SONIC MuJoCo bridge headlessly and record simulator state.

The official quick start uses viewer key 9 to release its elastic support.  In
headless mode we disable that support in-memory, without editing the official
configuration.  Communication remains on loopback (simulation only).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from gear_sonic.scripts.run_sim_loop import SimWrapper
from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
from gear_sonic.utils.mujoco_sim.simulator_factory import SimulatorFactory
from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--duration", type=float, default=120.0)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--elastic-release-s", type=float, default=0.0,
                   help="Keep the simulator support enabled until this wall time")
    p.add_argument("--elastic-release-file", type=Path,
                   help="Release support when this marker file appears")
    p.add_argument("--elastic-length", type=float, default=-0.2,
                   help="Vertical support offset; -0.2 keeps G1 near standing height")
    args = p.parse_args()

    config = SimLoopConfig(
        interface="sim", enable_onscreen=False, enable_offscreen=False,
        enable_real_device=False, data_collection=False, with_hands=False,
    )
    wbc = config.load_wbc_yaml()
    wbc["ENV_NAME"] = config.env_name
    wbc["ENABLE_ELASTIC_BAND"] = args.elastic_release_s > 0 or args.elastic_release_file is not None
    wbc["PRINT_SCENE_INFORMATION"] = False

    robot = instantiate_g1_robot_model()
    wrapper = SimWrapper(robot_model=robot, env_name=config.env_name, config=wbc,
                         onscreen=False, offscreen=False, enable_image_publish=False)
    # The upstream headless path only creates this attribute when the band is
    # enabled, while sim_step accesses it unconditionally.
    if not hasattr(wrapper.sim.sim_env, "elastic_band"):
        wrapper.sim.sim_env.elastic_band = None
    SimulatorFactory.start_simulator(wrapper.sim, as_thread=True)
    env = wrapper.sim.sim_env
    if env.elastic_band is not None:
        env.elastic_band.length = args.elastic_length
    rows, velocities, controls, contacts, wall_times = [], [], [], [], []
    start = time.monotonic()
    next_sample = start
    released = args.elastic_release_s <= 0 and args.elastic_release_file is None
    try:
        while time.monotonic() - start < args.duration:
            now = time.monotonic()
            timed_release = args.elastic_release_s > 0 and now - start >= args.elastic_release_s
            file_release = (args.elastic_release_file is not None
                            and args.elastic_release_file.exists())
            if (not released and (timed_release or file_release)
                    and env.elastic_band is not None):
                env.elastic_band.enable = False
                released = True
                print(f"elastic support released at {now - start:.3f} s")
            if now >= next_sample:
                rows.append(env.mj_data.qpos.copy())
                velocities.append(env.mj_data.qvel.copy())
                controls.append(env.mj_data.ctrl.copy())
                contacts.append(env.mj_data.ncon)
                wall_times.append(now - start)
                next_sample += 0.02
            time.sleep(0.001)
    except KeyboardInterrupt:
        print("recording interrupted; saving captured simulator samples")
    finally:
        wrapper.sim.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, time_s=np.asarray(wall_times), qpos=np.asarray(rows),
             qvel=np.asarray(velocities), ctrl=np.asarray(controls),
             contact_count=np.asarray(contacts), elastic_band_enabled=False,
             elastic_release_s=args.elastic_release_s, loopback_only=True)
    print(f"saved {len(rows)} simulator samples to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
