"""Prune the reachability table to configurations the robot can actually hold.

Why this exists
---------------
`reach_table.py` samples joint space and records where forward kinematics puts
the gripper. Forward kinematics knows nothing about gravity, the actuators'
torque limits, the table, or the other arm. So a large share of the sampled
configurations are geometrically fine and physically impossible: the planner
picked one, the joint command went out, and `shoulder_lift` settled 2043 mrad
(117 degrees) away from its command because the arm simply folded onto the
table under its own weight.

This pass sets each configuration in the full bimanual scene -- with the other
arm parked and the objects cleared away -- and keeps it only if

  * no arm geom is touching the table, the other arm, or itself, and
  * the gravity torque at that posture is inside the actuator force range,
    with margin, so a position servo can hold it, and
  * every fingertip is above the table.

The surviving set is what the planner is allowed to use. Run this once after
building the table; the result is cached alongside it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from mj_backend import select_backend  # noqa: E402

select_backend()

import mujoco  # noqa: E402

from bimanual_scene import (  # noqa: E402
    ARMS,
    HOME_QPOS,
    OBJECTS,
    build_model,
    set_object_pose,
)
from reach_table import CACHE, load  # noqa: E402  (same package)

# Keep a margin under the rated torque: a servo that is exactly at its limit
# holds the pose only in the absence of any disturbance, and these arms get
# bumped by the object they are carrying.
TORQUE_MARGIN = 0.85

# Fingertips reach this far past the grasp centre along the approach axis.
FINGER_REACH = 0.0225


def validate(arm: str, verbose: bool = True) -> np.ndarray:
    d = load(arm)
    model, idx = build_model()
    data = mujoco.MjData(model)

    # park the other arm, and move the free objects off the table entirely so
    # they cannot be what a contact is found against
    other = [a for a in ARMS if a != arm][0]
    for k in range(6):
        data.qpos[idx.arm_qpos_ids[other][k]] = HOME_QPOS[k]
        data.ctrl[idx.arm_act_ids[other][k]] = HOME_QPOS[k]
    for o in OBJECTS:
        set_object_pose(model, data, idx, o, [0.0, 0.0, -5.0])

    arm_bodies = {
        b
        for b in range(model.nbody)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith(
            f"{arm}_"
        )
    }
    force_hi = np.array(
        [model.actuator_forcerange[idx.arm_act_ids[arm][k]][1] for k in range(6)]
    )
    site_id = idx.ee_site[arm]

    q = d["q"]
    n = len(q)
    ok = np.zeros(n, dtype=bool)
    for i in range(n):
        for k in range(5):
            data.qpos[idx.arm_qpos_ids[arm][k]] = q[i, k]
        data.qpos[idx.arm_qpos_ids[arm][5]] = 1.20
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

        # 1. gravity torque must be holdable by the position servos
        bias = np.array(
            [abs(data.qfrc_bias[model.jnt_dofadr[j]]) for j in idx.arm_joint_ids[arm]]
        )
        if np.any(bias > TORQUE_MARGIN * force_hi):
            continue

        # 2. nothing of this arm may be in contact with anything
        touching = False
        for c in range(data.ncon):
            con = data.contact[c]
            b1 = model.geom_bodyid[con.geom1]
            b2 = model.geom_bodyid[con.geom2]
            if b1 in arm_bodies or b2 in arm_bodies:
                touching = True
                break
        if touching:
            continue

        # 3. both ends of the fingers must clear the table. The site sits at
        #    the wrist end in the TRS file and at the fingertip in the
        #    Menagerie file; either way the other end is 100 mm along site -x.
        sp = data.site_xpos[site_id]
        sR = data.site_xmat[site_id].reshape(3, 3)
        far = sp + sR @ np.array([-0.100, 0.0, 0.0])
        if min(sp[2], far[2]) < 0.004:
            continue

        ok[i] = True
        if verbose and ok.sum() % 2000 == 0:
            print(f"  {arm}: {ok.sum()} held of {i+1} checked", flush=True)

    path = Path(str(CACHE).format(arm=arm))
    out = {k: d[k][ok] for k in ("q", "centre", "fingers", "jaw")}
    np.savez_compressed(path, **out)
    if verbose:
        print(
            f"{arm}: kept {ok.sum()} of {n} configurations "
            f"({100*ok.mean():.1f}%) -- written to {path.name}"
        )
    return ok


if __name__ == "__main__":
    for a in ("right", "left"):
        validate(a)
