"""Mark which poses can grip an object that is STANDING on a surface.

Why this exists
---------------
`validate_reach.py` checks that a configuration is holdable with the table
empty. That is not enough. A grasp pose is a place for the object's centre,
and reaching it requires the whole gripper -- wrist, body, both fingers -- to
get there without passing through the object on the way.

For a 90 mm cup standing on the table that rules out most approach angles. A
grasp planned at 25 degrees below horizontal put the wrist 21 mm and the
gripper body 37 mm INSIDE the cup: the only route to the grasp point ran
through the cup's own upper body. The arm then drove into it, three actuators
saturated, and the gripper stopped 94 mm above where it was sent.

So this pass puts a cup where the pose says the cup would be, standing on the
surface, and keeps the pose only if nothing on the arm penetrates it. What
survives is the set of grasps that can actually be reached from outside.

Two masks are produced per arm, because the surfaces differ in height:
  table -- cup standing on the table top
  plate -- cup standing on the plate, which is PLATE_TOP_Z higher
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
    CUP_HALF_H,
    HOME_QPOS,
    OBJECTS,
    PLATE_TOP_Z,
    build_model,
    set_object_pose,
)
from reach_table import CACHE, load  # noqa: E402

# A contact shallower than this is hull noise between touching convex hulls,
# not the gripper being inside the cup.
MAX_PENETRATION = 0.0015

SURFACES = {"table": 0.0, "plate": PLATE_TOP_Z}

# Only poses in this band above the surface could grip a standing cup at all.
BAND = (0.045, 0.125)


def mask_path(arm: str, surface: str) -> Path:
    return Path(str(CACHE).format(arm=arm)).with_name(f"standing_{surface}_{arm}.npy")


def validate(arm: str, surface: str, verbose: bool = True) -> np.ndarray:
    d = load(arm)
    q, centre = d["q"], d["centre"]
    surf_z = SURFACES[surface]
    lo, hi = surf_z + BAND[0], surf_z + BAND[1]
    cand = np.flatnonzero((centre[:, 2] >= lo) & (centre[:, 2] < hi))

    model, idx = build_model()
    data = mujoco.MjData(model)
    other = [a for a in ARMS if a != arm][0]
    for k in range(6):
        data.qpos[idx.arm_qpos_ids[other][k]] = HOME_QPOS[k]
        data.ctrl[idx.arm_act_ids[other][k]] = HOME_QPOS[k]
    for o in OBJECTS:
        if o != "cup":
            set_object_pose(model, data, idx, o, [0.0, 0.0, -5.0])

    arm_bodies = {
        b
        for b in range(model.nbody)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith(
            f"{arm}_"
        )
    }
    cup_body = idx.obj_body["cup"]

    ok = np.zeros(len(q), dtype=bool)
    for n, i in enumerate(cand):
        for k in range(5):
            data.qpos[idx.arm_qpos_ids[arm][k]] = q[i, k]
        data.qpos[idx.arm_qpos_ids[arm][5]] = 1.20
        # the cup stands on the surface, directly under the grasp centre
        set_object_pose(
            model, data, idx, "cup",
            [centre[i, 0], centre[i, 1], surf_z + CUP_HALF_H + 0.001],
        )
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

        worst = 0.0
        for c in range(data.ncon):
            con = data.contact[c]
            b1 = model.geom_bodyid[con.geom1]
            b2 = model.geom_bodyid[con.geom2]
            pair = {b1, b2}
            if cup_body in pair and (pair & arm_bodies):
                worst = min(worst, con.dist)
        if worst > -MAX_PENETRATION:
            ok[i] = True
        if verbose and n and n % 20000 == 0:
            print(f"  {arm}/{surface}: {ok.sum()} clear of {n} checked", flush=True)

    np.save(mask_path(arm, surface), ok)
    if verbose:
        print(
            f"{arm}/{surface}: {ok.sum()} of {len(cand)} poses in the band can reach "
            f"a standing cup without going through it "
            f"({100*ok.sum()/max(1,len(cand)):.1f}%)"
        )
    return ok


if __name__ == "__main__":
    for a in ("right", "left"):
        for s in ("table", "plate"):
            validate(a, s)
