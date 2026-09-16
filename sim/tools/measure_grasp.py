"""Find where the cup must sit in the gripper's site frame to be physically held.

This replaces the guessed grasp constants for a new arm model with measured ones.
For each candidate offset (x, z) of the cup's centre in the `gripperframe` site
frame (y = 0, cup axis along site y so the jaws close across its diameter):

  1. arm held in the air at a clear posture, gripper open, gravity OFF
  2. cup teleported to the offset, jaw commanded to `q_close`, 0.8 s of physics
  3. gravity ON for 0.6 s
  4. HELD if the cup is still touching BOTH jaw bodies and has moved less than
     HOLD_TOL relative to the site

The output is the held set's centroid (the grasp centre), its extent (the
holding window), the jaw angle the servo actually reached (the contact angle)
and a text map, for each close command tried. Everything downstream --
`ik_control.GRASP_CENTRE_SITE`, `GRASP_WINDOW`, `CLOSE_Q` -- is read off this.

Run from `sim/`:  python tools/measure_grasp.py [--fine]
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
    ARM_MODEL,
    ARMS,
    HOME_QPOS,
    OBJECTS,
    build_model,
    set_object_pose,
)

ARM = "right"
POSE = np.array([0.0, -0.9, 0.9, 0.9, 0.0])   # gripper clear of everything
OPEN = 1.20
HOLD_TOL = 0.004
CLOSE_STEPS = 400   # 0.8 s at 200 Hz
GRAV_STEPS = 300    # 0.6 s


def cup_quat_axis_along_site_y(R: np.ndarray) -> np.ndarray:
    """Quaternion putting the cup's own z axis along the site's y axis."""
    Rc = np.column_stack([R[:, 2], R[:, 0], R[:, 1]])
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, Rc.reshape(-1))
    return q


def main() -> int:
    fine = "--fine" in sys.argv
    step = 0.005 if fine else 0.010
    xs = np.arange(-0.100, 0.0101, step)
    zs = np.arange(-0.020, 0.0601, step)
    closes = (0.0, 0.2, 0.4) if fine else (0.0, 0.3)

    model, idx = build_model()
    data = mujoco.MjData(model)
    other = [a for a in ARMS if a != ARM][0]
    site = idx.ee_site[ARM]
    cup = idx.obj_body["cup"]
    fixed_body = idx.gripper_body[ARM]
    moving_body = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, f"{ARM}_moving_jaw_so101_v1"
    )
    gravity = model.opt.gravity.copy()

    def touching():
        f = m = False
        for c in range(data.ncon):
            con = data.contact[c]
            b = {model.geom_bodyid[con.geom1], model.geom_bodyid[con.geom2]}
            if cup in b:
                f |= fixed_body in b
                m |= moving_body in b
        return f, m

    def reset(q_grip):
        mujoco.mj_resetData(model, data)
        for k in range(6):
            data.qpos[idx.arm_qpos_ids[other][k]] = HOME_QPOS[k]
            data.ctrl[idx.arm_act_ids[other][k]] = HOME_QPOS[k]
        for k in range(5):
            data.qpos[idx.arm_qpos_ids[ARM][k]] = POSE[k]
            data.ctrl[idx.arm_act_ids[ARM][k]] = POSE[k]
        data.qpos[idx.arm_qpos_ids[ARM][5]] = q_grip
        data.ctrl[idx.arm_act_ids[ARM][5]] = q_grip
        for o in OBJECTS:
            set_object_pose(model, data, idx, o, [0.0, 0.0, -5.0])
        mujoco.mj_forward(model, data)

    print(f"arm model: {ARM_MODEL}   grid step {step*1000:.0f} mm   "
          f"x {xs[0]:+.3f}..{xs[-1]:+.3f}  z {zs[0]:+.3f}..{zs[-1]:+.3f}")

    for q_close in closes:
        held = []
        reached = []
        grid = {}
        for x in xs:
            for z in zs:
                reset(OPEN)
                p = data.site_xpos[site].copy()
                R = data.site_xmat[site].reshape(3, 3).copy()
                pos = p + R @ np.array([x, 0.0, z])
                quat = cup_quat_axis_along_site_y(R)
                adr = idx.obj_qpos_adr["cup"]
                data.qpos[adr:adr + 3] = pos
                data.qpos[adr + 3:adr + 7] = quat
                data.qvel[:] = 0.0
                model.opt.gravity[:] = 0.0
                mujoco.mj_forward(model, data)
                # the cup must start clear of the open gripper
                f0, m0 = touching()
                if f0 or m0:
                    grid[(x, z)] = "x"      # starts inside the gripper
                    model.opt.gravity[:] = gravity
                    continue
                data.ctrl[idx.arm_act_ids[ARM][5]] = q_close
                for _ in range(CLOSE_STEPS):
                    mujoco.mj_step(model, data)
                f1, m1 = touching()
                if not (f1 and m1):
                    grid[(x, z)] = "s" if f1 else ("m" if m1 else ".")
                    model.opt.gravity[:] = gravity
                    continue
                rel0 = R.T @ (data.xpos[cup] - data.site_xpos[site])
                model.opt.gravity[:] = gravity
                for _ in range(GRAV_STEPS):
                    mujoco.mj_step(model, data)
                R2 = data.site_xmat[site].reshape(3, 3)
                rel1 = R2.T @ (data.xpos[cup] - data.site_xpos[site])
                f2, m2 = touching()
                moved = float(np.linalg.norm(rel1 - rel0))
                if f2 and m2 and moved < HOLD_TOL:
                    grid[(x, z)] = "H"
                    held.append(rel1.copy())
                    reached.append(float(data.qpos[idx.arm_qpos_ids[ARM][5]]))
                else:
                    grid[(x, z)] = "d"      # pinched, then dropped or slid

        print(f"\nclose command q = {q_close:+.2f}   ('H' held, 'd' pinched then "
              f"lost, 's'/'m' one jaw only, '.' nothing, 'x' started inside)")
        for z in zs[::-1]:
            row = " ".join(grid[(x, z)] for x in xs)
            print(f"  z={z:+.3f}  {row}")
        print("           x: " + "  ".join(f"{x:+.2f}" for x in xs[::4]))
        if held:
            A = np.array(held)
            print(f"  held {len(A)} of {len(xs)*len(zs)}: centre (settled, site frame) "
                  f"{np.round(A.mean(axis=0), 4)}")
            print(f"  extent x [{A[:,0].min():+.4f},{A[:,0].max():+.4f}]  "
                  f"y [{A[:,1].min():+.4f},{A[:,1].max():+.4f}]  "
                  f"z [{A[:,2].min():+.4f},{A[:,2].max():+.4f}]")
            print(f"  jaw angle reached while holding: "
                  f"{np.mean(reached):.3f} +/- {np.std(reached):.3f} rad")
        else:
            print("  held: NONE")

    # sanity: where is the park pose now, in world coordinates
    for arm in ARMS:
        reset(OPEN)
        for k in range(6):
            data.qpos[idx.arm_qpos_ids[arm][k]] = HOME_QPOS[k]
        mujoco.mj_forward(model, data)
        print(f"\nHOME site position, {arm}: {np.round(data.site_xpos[idx.ee_site[arm]], 3)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
