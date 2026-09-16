"""Build and cache a reachability table for one SO-101 arm.

Why this exists
---------------
The arm has five joints driving the gripper (the sixth is the jaws), so a
fully specified six-DOF grasp pose is generically UNREACHABLE. Differential IK
does not tell you that clearly: mink happily returns a configuration 70-250 mm
short while reporting a converged solve, and multi-start does not rescue it
because the target is not in the reachable set at all.

So instead of asking "can you hit this pose I invented", we sample the joint
space once, record where each configuration actually puts the grasp centre and
which way the fingers point, and afterwards plan only inside that set. A
desired grasp point is matched to the nearest sampled configuration, which is
then used as the IK seed -- so the solver starts inside the right basin and
only has to polish.

Measured consequence worth knowing before designing any task: with the fingers
within 26 degrees of straight down, this arm cannot put its grasp centre below
z = 0.071 m. A cup standing on the table has its centre at z = 0.045, which is
why a top-down grasp has to take the cup near its rim, not its middle.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from ik_control import FINGER_SIGN, GRASP_CENTRE_SITE, ArmIK  # noqa: E402

CACHE = HERE.parent / "assets" / "reach_{arm}.npz"

JOINT_LO = np.array([-1.9198, -1.7453, -1.69, -1.6581, -2.7438])
JOINT_HI = np.array([1.9198, 1.7453, 1.69, 1.6581, 2.8412])


def build(arm: str, n: int = 240_000, seed: int = 0) -> dict:
    """Sample joint space and record grasp centre + finger direction."""
    ik = ArmIK(arm)
    rng = np.random.default_rng(seed)
    Q = rng.uniform(JOINT_LO, JOINT_HI, size=(n, 5))
    centre = np.empty((n, 3))
    fingers = np.empty((n, 3))
    jaw = np.empty((n, 3))
    for i, q5 in enumerate(Q):
        p, R = ik.fk(np.concatenate([q5, [1.2]]))
        centre[i] = p + R @ GRASP_CENTRE_SITE
        fingers[i] = FINGER_SIGN * R[:, 0]
        jaw[i] = R[:, 2]
    keep = centre[:, 2] > 0.004  # above the table
    return {
        "q": Q[keep],
        "centre": centre[keep],
        "fingers": fingers[keep],
        "jaw": jaw[keep],
    }


def load(arm: str, n: int = 240_000) -> dict:
    path = Path(str(CACHE).format(arm=arm))
    if path.exists():
        z = np.load(path)
        return {k: z[k] for k in ("q", "centre", "fingers", "jaw")}
    d = build(arm, n=n)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **d)
    return d


class ReachTable:
    """Nearest-configuration lookup for one arm."""

    def __init__(self, arm: str) -> None:
        d = load(arm)
        self.q = d["q"]
        self.centre = d["centre"]
        self.fingers = d["fingers"]
        self.jaw = d["jaw"]
        # Poses that can reach a cup STANDING on a surface without the gripper
        # passing through it -- see tools/validate_standing.py. Only about one
        # pose in forty qualifies, which is why planning without this filter
        # kept choosing grasps that drove the wrist 20-37 mm into the cup.
        self.standing = {}
        base = Path(str(CACHE).format(arm=arm))
        for surface in ("table", "plate"):
            f = base.with_name(f"standing_{surface}_{arm}.npy")
            if f.exists():
                self.standing[surface] = np.load(f)

    def nearest(self, point, finger_dir, jaw_dir=None,
                finger_cone: float = 0.35, jaw_cone: float = 0.60):
        """Configuration closest to `point` whose axes are within tolerance.

        The axis requirements are FILTERS, not penalties. An earlier version
        added them to the squared position distance, which quietly made them
        dominate -- an alignment term worth up to 0.24 against a squared
        distance of 0.0025 at five centimetres -- so it returned beautifully
        aligned configurations 120 mm from the object. Cone first, then nearest.

        `finger_cone` and `jaw_cone` are half-angles in radians. The jaw axis is
        compared without sign, because a two-finger gripper closing along +z
        and along -z are the same grasp.
        """
        f = np.asarray(finger_dir, float)
        f = f / np.linalg.norm(f)
        ok = self.fingers @ f >= np.cos(finger_cone)
        if jaw_dir is not None:
            j = np.asarray(jaw_dir, float)
            j = j / np.linalg.norm(j)
            ok = ok & (np.abs(self.jaw @ j) >= np.cos(jaw_cone))
        if not ok.any():  # nothing in the cone: fall back to alignment only
            i = int(np.argmax(self.fingers @ f))
            return self.q[i].copy(), self.centre[i].copy(), float(
                np.linalg.norm(self.centre[i] - np.asarray(point, float))
            )
        idx = np.flatnonzero(ok)
        d = self.centre[idx] - np.asarray(point, float)
        k = int(np.argmin(np.einsum("ij,ij->i", d, d)))
        i = idx[k]
        return self.q[i].copy(), self.centre[i].copy(), float(np.linalg.norm(d[k]))

    def grasp_poses(self, z_lo: float, z_hi: float, fingers_down: float = 0.25,
                     jaw_horizontal: float = 0.35, x_lim=None, y_lim=None,
                     standing: str | None = None):
        """Indices of configurations that could grip a standing cylinder.

        `fingers_down` is the minimum downward component of the finger
        direction and `jaw_horizontal` the maximum vertical component of the
        jaw axis -- together they say "coming down onto it at some angle, jaws
        closing around the body rather than squashing it end to end". Both are
        loose on purpose: this arm has five joints for a six-DOF pose, so
        pinning the approach exactly leaves almost nothing reachable, while
        asking only for what the grasp actually requires leaves hundreds of
        poses per object.
        """
        m = (
            (self.fingers[:, 2] <= -fingers_down)
            & (np.abs(self.jaw[:, 2]) <= jaw_horizontal)
            & (self.centre[:, 2] >= z_lo)
            & (self.centre[:, 2] < z_hi)
        )
        if x_lim is not None:
            m &= (self.centre[:, 0] >= x_lim[0]) & (self.centre[:, 0] <= x_lim[1])
        if y_lim is not None:
            m &= (self.centre[:, 1] >= y_lim[0]) & (self.centre[:, 1] <= y_lim[1])
        if standing is not None and standing in self.standing:
            m &= self.standing[standing]
        return np.flatnonzero(m)

    def nearest_in(self, point, idx):
        """Nearest configuration to `point` among `idx`, as (q, centre, dist).

        Pairs with `grasp_poses`: the same predicate that decides which poses
        are acceptable to SAMPLE also decides which are acceptable to PLAN
        toward. When those two disagree -- a loose filter for sampling and a
        20 degree cone for planning -- the layout looks performable and then
        every waypoint comes out 40-120 mm away.
        """
        d = self.centre[idx] - np.asarray(point, float)
        k = int(np.argmin(np.einsum("ij,ij->i", d, d)))
        i = int(idx[k])
        return self.q[i].copy(), self.centre[i].copy(), float(np.linalg.norm(d[k]))

    def nearest_in_near_config(self, point, idx, q_ref, w_q: float = 0.04):
        """Nearest pose to `point` that is also a short joint move from `q_ref`.

        Two configurations can put the gripper in almost the same place with
        the elbow on opposite sides. Interpolating between them swings the arm
        through everything in between -- including the object it is about to
        pick up -- so a waypoint is chosen for proximity in joint space as
        well as in task space.
        """
        d = self.centre[idx] - np.asarray(point, float)
        pos = np.sqrt(np.einsum("ij,ij->i", d, d))
        dq = self.q[idx] - np.asarray(q_ref, float)[:5]
        cost = pos + w_q * np.sqrt(np.einsum("ij,ij->i", dq, dq))
        k = int(np.argmin(cost))
        i = int(idx[k])
        return self.q[i].copy(), self.centre[i].copy(), float(pos[k])

    def nearest_in_k(self, point, idx, k: int = 64, q_ref=None, tol=None):
        """Candidates near `point`, ordered for the caller to filter.

        Position is a HARD constraint and joint proximity only a preference.
        Blending them into one cost was a mistake: at 0.04 per radian a two
        radian arm swing outweighed eighty millimetres of position error, so
        the planner returned beautifully short moves to the wrong place and
        every waypoint came out 17-26 mm wide of its target.

        With `tol` given, candidates outside it are dropped and the rest are
        ordered by how small a joint move they need from `q_ref`. Otherwise
        they come back nearest-first.
        """
        d = self.centre[idx] - np.asarray(point, float)
        dist = np.sqrt(np.einsum("ij,ij->i", d, d))
        if tol is not None:
            inside = np.flatnonzero(dist <= tol)
            if len(inside) == 0:
                inside = np.argsort(dist)[:k]
            if q_ref is not None:
                dq = self.q[idx[inside]] - np.asarray(q_ref, float)[:5]
                order = inside[np.argsort(np.einsum("ij,ij->i", dq, dq))][:k]
            else:
                order = inside[np.argsort(dist[inside])][:k]
        else:
            order = np.argsort(dist)[:k]
        return [
            (self.q[int(idx[o])].copy(), self.centre[int(idx[o])].copy(),
             float(dist[o]))
            for o in order
        ]

    def sample_pose(self, rng, idx):
        """One configuration from `idx`, as (q, grasp centre)."""
        i = int(idx[rng.integers(len(idx))])
        return self.q[i].copy(), self.centre[i].copy()

    def best_point_near(self, point, finger_dir, max_move: float = 0.05):
        """The closest point to `point` this arm can actually reach."""
        q, c, dist = self.nearest(point, finger_dir)
        return (c if dist <= max_move else None), dist


if __name__ == "__main__":
    for arm in ("left", "right"):
        d = load(arm)
        down = d["fingers"][:, 2] < -0.90
        band = d["centre"][down]
        print(f"{arm}: {len(d['q'])} configs above the table, "
              f"{down.sum()} with fingers within 26 deg of down")
        if down.sum():
            print(f"   fingers-down grasp centre z range "
                  f"[{band[:,2].min():.3f}, {band[:,2].max():.3f}]")
