"""Trace the physical cup pickup without changing the episode implementation."""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from episode import BimanualEnv, CLOSE_Q, object_pos, pick_cup  # noqa: E402


def snapshot(env: BimanualEnv, label: str) -> None:
    cup = object_pos(env.data, env.idx, "cup").copy()
    grasp, rotation = env.actual_grasp("right")
    local = rotation.T @ (cup - grasp)
    q = env.data.qpos[env.idx.arm_qpos_ids["right"]].copy()
    contacts = []
    target = env.idx.obj_body["cup"]
    for i in range(env.data.ncon):
        con = env.data.contact[i]
        bodies = [env.model.geom_bodyid[con.geom1], env.model.geom_bodyid[con.geom2]]
        if target not in bodies:
            continue
        contacts.append(tuple(env.model.body(body).name for body in bodies))
    print(
        f"{label}: cup={np.round(cup, 4)} grasp={np.round(grasp, 4)} "
        f"cup_local={np.round(local, 4)} q={np.round(q, 3)} "
        f"held={env.holding('right', 'cup')} contacts={contacts}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--dz", type=float)
    parser.add_argument("--axis-min", type=float)
    parser.add_argument("--close", type=float)
    args = parser.parse_args()
    env = BimanualEnv(record_images=False, seed=args.seed)
    if args.close is not None:
        CLOSE_Q["cup"] = args.close
    env.reset()
    env.randomize_objects()
    if args.axis_min is not None:
        for table in env.table.values():
            original_grasp_poses = table.grasp_poses

            def axis_aligned(*call_args, _table=table,
                             _original=original_grasp_poses, **call_kwargs):
                idx = _original(*call_args, **call_kwargs)
                site_y = np.cross(_table.jaw[idx], _table.fingers[idx])
                return idx[np.abs(site_y[:, 2]) >= args.axis_min]

            table.grasp_poses = axis_aligned
    snapshot(env, "initial")

    original_joint = env.run_joint_phase
    original_grip = env.grip_phase
    joint_count = 0
    grip_count = 0

    def traced_joint(*args, **kwargs):
        nonlocal joint_count
        joint_count += 1
        result = original_joint(*args, **kwargs)
        snapshot(env, f"after joint phase {joint_count}")
        return result

    def traced_grip(*args, **kwargs):
        nonlocal grip_count
        grip_count += 1
        result = original_grip(*args, **kwargs)
        snapshot(env, f"after grip phase {grip_count}")
        return result

    env.run_joint_phase = traced_joint
    env.grip_phase = traced_grip
    pick_cup(env, "right", dz=args.dz)
    snapshot(env, "final")
    print("faults:", env.rec.faults)
    env.close()


if __name__ == "__main__":
    main()
