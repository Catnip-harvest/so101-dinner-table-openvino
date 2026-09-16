"""Print object/contact state after every physical cup-relay primitive."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bimanual_scene import CUP_HALF_H, PLATE_TOP_Z  # noqa: E402
from episode import (  # noqa: E402
    BimanualEnv,
    go_home,
    object_pos,
    pick_cup,
    place_cup,
    success_cup_handoff,
)


def snapshot(env: BimanualEnv, label: str) -> None:
    cup = object_pos(env.data, env.idx, "cup").copy()
    fields = []
    for arm in ("right", "left"):
        grasp, rotation = env.actual_grasp(arm)
        local = rotation.T @ (cup - grasp)
        fields.append(
            f"{arm}:grasp={np.round(grasp, 3)} local={np.round(local, 3)} "
            f"held={env.holding(arm, 'cup')} qg="
            f"{env.data.qpos[env.idx.arm_qpos_ids[arm][5]]:.3f}"
        )
    print(f"{label}: cup={np.round(cup, 3)} | " + " | ".join(fields))
    if env.rec.faults:
        print("  faults:", env.rec.faults)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--rep", type=int, default=0)
    args = parser.parse_args()
    env = BimanualEnv(record_images=False, seed=args.seed)
    for _ in range(args.rep + 1):
        env.reset()
        env.randomize_objects()
    plate0 = object_pos(env.data, env.idx, "plate").copy()
    relay = env.relay
    state = {"stage": "setup", "right": False, "left": False}
    original_joint_step = env.joint_step

    def traced_joint_step(*call_args, **call_kwargs):
        original_joint_step(*call_args, **call_kwargs)
        if state["stage"] == "right place" and state["right"]:
            target = np.array([relay["xy"][0], relay["xy"][1], CUP_HALF_H + 0.001])
            err = np.linalg.norm(object_pos(env.data, env.idx, "cup") - target)
            if err < 0.025:
                print(f"near relay while held: err={err*1000:.1f}mm")
        for arm in ("right", "left"):
            now = env.holding(arm, "cup")
            if now != state[arm]:
                print(f"contact transition during {state['stage']}: {arm} "
                      f"{state[arm]} -> {now}")
                snapshot(env, "  transition")
                state[arm] = now

    env.joint_step = traced_joint_step
    print("layout:", "cup_xy", env.cup_xy, "plate_xy", env.plate_xy,
          "relay", relay)
    snapshot(env, "initial")
    state["stage"] = "right pick"
    pick_cup(env, "right")
    snapshot(env, "after right pick")
    state["stage"] = "right place"
    place_cup(env, "right", relay["xy"], CUP_HALF_H + 0.001,
              dz=relay["dz_put"])
    snapshot(env, "after right place")
    state["stage"] = "right home"
    go_home(env, "right")
    snapshot(env, "after right home")
    state["stage"] = "left pick"
    pick_cup(env, "left", dz=relay["dz_take"])
    snapshot(env, "after left pick")
    state["stage"] = "left place"
    place_cup(env, "left", plate0[:2], PLATE_TOP_Z + CUP_HALF_H)
    snapshot(env, "after left place")
    go_home(env, "left")
    env.hold(0.6)
    snapshot(env, "final")
    print("predicate:", success_cup_handoff(env, {"plate_xy": plate0[:2]}))
    env.close()


if __name__ == "__main__":
    main()
