"""Map the gripper's free space and its true pinch point with a small probe.

Why a probe and not the cup
---------------------------
The first attempt at this measured where a 48 mm cup "settles when the jaws
close", and reported a holding window. That conflates two different things: an
object can be reported as touching both fingers while also being 37 mm inside
the convex hull of one of them. A wide object cannot distinguish "sitting in
the gap" from "jammed through the metal".

So this walks a 4 mm sphere through the gripper's own frame and asks the
simulator, purely geometrically, two questions:

  free    -- with the jaws OPEN, does the probe touch anything?
  pinched -- with the jaws CLOSED, is the probe touched by BOTH the static
             jaw and the moving jaw?

The set of free points is the actual opening. The set of pinched points is
where a grasp can actually happen. The probe is static -- moved by rewriting
its body position and calling mj_forward -- so this is a pure collision query
with no dynamics, no ejection, and nothing to settle.

The answer decides how to decompose the static jaw's collision geometry, which
is the one change standing between this project and working demonstrations.
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

from bimanual_scene import ARM_XML, replace_gripper_collision  # noqa: E402

PROBE_R = 0.004

# A posture that holds the gripper clear of everything, so the only thing the
# probe can touch is the gripper itself.
POSE = np.array([0.0, -0.9, 0.9, 0.9, 0.0])

JAW_OPEN = 1.20
JAW_CLOSED = 0.10


def build():
    """Single arm plus a static probe sphere, and no floor to confuse matters."""
    spec = mujoco.MjSpec.from_file(str(ARM_XML))
    if "--raw" not in sys.argv:
        replace_gripper_collision(spec, "gripper", "moving_jaw_so101_v1")
    body = spec.worldbody.add_body(name="probe", pos=[0, 0, 0])
    body.add_geom(
        name="probe_g",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[PROBE_R, 0, 0],
        contype=1,
        conaffinity=1,
    )
    model = spec.compile()
    return model, mujoco.MjData(model)


def gripper_geoms(model):
    """Collision geoms of the gripper, split into static side and moving jaw."""
    static, moving = [], []
    for gid in range(model.ngeom):
        if not model.geom_contype[gid]:
            continue
        b = model.geom_bodyid[gid]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if "moving_jaw" in name:
            moving.append(gid)
        elif "gripper" in name:
            static.append(gid)
    return static, moving


def probe_at(model, data, probe_bid, probe_gid, site_id, local, jaw):
    """Place the probe at `local` in the site frame; return what it touches."""
    for k in range(5):
        data.qpos[k] = POSE[k]
    data.qpos[5] = jaw
    mujoco.mj_forward(model, data)
    sp = data.site_xpos[site_id].copy()
    sR = data.site_xmat[site_id].reshape(3, 3).copy()
    model.body_pos[probe_bid] = sp + sR @ np.asarray(local, float)
    mujoco.mj_forward(model, data)
    hits = {}
    for c in range(data.ncon):
        con = data.contact[c]
        if probe_gid in (con.geom1, con.geom2):
            other = con.geom2 if con.geom1 == probe_gid else con.geom1
            hits[other] = min(hits.get(other, 0.0), con.dist)
    return hits


def main() -> int:
    model, data = build()
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    probe_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "probe")
    probe_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "probe_g")
    static, moving = gripper_geoms(model)
    names = {
        g: (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) or "?")
        for g in static + moving
    }
    print(f"static side geoms {static} -> {[names[g] for g in static]}")
    print(f"moving jaw geoms  {moving} -> {[names[g] for g in moving]}")

    xs = np.arange(-0.115, 0.0051, 0.005)
    zs = np.arange(-0.035, 0.0551, 0.005)

    for jaw, tag in ((JAW_OPEN, "JAWS OPEN"), (JAW_CLOSED, "JAWS CLOSED")):
        print(f"\n{tag} (gripper q = {jaw:+.2f}), slice y = 0, site frame")
        print("  '.'=free   's'=static side only   'm'=moving jaw only   'B'=BOTH")
        print("        " + "".join(f"{x:+.3f}"[:6].rjust(2)[:2] for x in xs))
        for z in zs[::-1]:
            row = ""
            for x in xs:
                hits = probe_at(model, data, probe_bid, probe_gid, site_id,
                                [x, 0.0, z], jaw)
                hs = any(g in static for g in hits)
                hm = any(g in moving for g in hits)
                row += " B" if (hs and hm) else (" s" if hs else (" m" if hm else " ."))
            print(f"  z={z:+.3f}" + row)
        print("   x:   " + "   ".join(f"{x:+.3f}" for x in xs[::4]))

    # Where can a grasp actually happen? Free when open, pinched when closed.
    print("\ncandidate grasp points: free with the jaws open, touched by BOTH "
          "faces when closed")
    found = []
    for x in xs:
        for z in zs:
            for y in (-0.008, 0.0, 0.008):
                loc = [x, y, z]
                if probe_at(model, data, probe_bid, probe_gid, site_id, loc, JAW_OPEN):
                    continue
                hits = probe_at(model, data, probe_bid, probe_gid, site_id, loc, JAW_CLOSED)
                if any(g in static for g in hits) and any(g in moving for g in hits):
                    found.append(loc)
    if not found:
        print("  NONE -- no point is both clear when open and pinched when closed.")
        print("  That means the shipped collision geometry cannot grasp anything:")
        print("  the static side's convex hull fills its own jaw opening.")
    else:
        A = np.array(found)
        print(f"  {len(A)} points")
        print(f"  x [{A[:,0].min():+.4f},{A[:,0].max():+.4f}]  "
              f"y [{A[:,1].min():+.4f},{A[:,1].max():+.4f}]  "
              f"z [{A[:,2].min():+.4f},{A[:,2].max():+.4f}]")
        print(f"  centroid {np.round(A.mean(axis=0), 4)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
