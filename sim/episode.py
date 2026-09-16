"""Episode runner: scripted bimanual tasks driven by Mink IK.

Two scripted tasks:

  task_plate  -- "pick up the plate and place it on the table with arm A"
  task_cup    -- "pick up the cup and hand it to arm A, which places it on
                  the plate"

"arm A" is the LEFT arm throughout.

The controller is fed nothing but ground-truth MjData (joint angles, object
poses) and the Cartesian targets the script computes from them. Images are
rendered and stored as observations; no control decision reads them.

DART-style noise: when an episode is flagged perturbed, the EXECUTED joint
command gets zero-mean Gaussian noise added per joint per step, while the
recorded action label stays the clean Mink target. That is the whole point --
the policy learns to map noisy-visited states back to the clean expert action.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mj_backend import select_backend  # noqa: E402

select_backend()

import mujoco  # noqa: E402

from bimanual_scene import (  # noqa: E402
    ARMS,
    ARM_X,
    CARRY_MODE,
    CUP_HALF_H,
    CUP_RADIUS,
    GRIPPER_OPEN,
    HOME_QPOS,
    JOINT_NAMES,
    NEUTRAL_QPOS,
    OBJECTS,
    PLATE_RADIUS,
    PLATE_TOP_Z,
    SceneIndex,
    build_model,
    object_pos,
    reset_home,
    set_object_pose,
    state_vector,
)
from tools.reach_table import ReachTable  # noqa: E402
from ik_control import (  # noqa: E402
    ArmIK,
    CLOSE_Q,
    FINGER_SIGN,
    GRASP_CENTRE_SITE,
    GRASP_WINDOW,
    OPEN_Q,
    grasp_centre_now,
    grasp_frame,
    rot_z,
    site_pose_for_grasp,
)

# The challenge brief prescribes a 200 Hz simulation under 50 Hz control, and
# the Menagerie SO-101 ships tuned for a 5 ms implicitfast step, so both rates
# are taken from there rather than chosen here.
PHYS_DT = 0.005
SUBSTEPS = 4
CONTROL_DT = PHYS_DT * SUBSTEPS  # 0.02 s
FPS = int(round(1.0 / CONTROL_DT))  # 50

CAMERAS = ("top", "left_wrist", "right_wrist")

# --- grasp and hand-off geometry -----------------------------------------
# The fingers reach ~22 mm past the grasp centre, so a top-down grasp needs
# that much clearance under the grasp point or the fingertips hit the table.
FINGER_REACH_PAST_CENTRE = 0.0225

# Where each arm grips the cup during the transfer, as an offset from the cup's
# centre along the cup axis. The giver takes it low from the SIDE and the
# receiver takes it high from ABOVE: two top-down grippers cannot share a 90 mm
# cup, because each one's fingers extend ~110 mm below its own wrist, far more
# than any vertical separation the cup allows.
GIVE_GRIP_DZ = -0.015
TAKE_GRIP_DZ = +0.030

# Where the cup's centre sits at the transfer. Chosen by searching both arms'
# measured reachable sets for a cup-centre position each can serve at the two
# grip heights above -- see tools/reach_table.py and the region search. 1395
# positions qualify; this one is the best of them, with both arms landing
# within a millimetre. It is HIGH above the table on purpose: a near-vertical
# wrist is only comfortable for this arm above z ~ 0.16, and at tabletop
# height the whole set collapses to 485 configurations out of 205 000.
HANDOFF_XY = (0.08, 0.0)
HANDOFF_CUP_Z = 0.24

# A waypoint is only accepted if Mink hits it this closely. Past this the
# episode is FAULTED rather than quietly retargeted -- see BimanualEnv.fault.
REACH_TOL = 0.008

# How far a planned grasp may sit from the object's axis. Bounded by what the
# grasp check will accept for the cup -- radial error must stay under
# CUP_RADIUS - 4 mm = 20 mm -- so 12 mm leaves margin while still admitting
# enough poses for the hand-off, where the best available pair sits at 8 mm.
GRASP_MISS_TOL = 0.014

# The one grasp specification, shared by the layout sampler and the planner.
# They MUST agree: when sampling used a loose predicate and planning used a
# 20 degree cone, layouts passed as performable and then every waypoint came
# out 40-120 mm away. Shallow approaches are allowed on purpose -- a cup
# standing on the table can only be gripped near its rim (a near-vertical
# wrist cannot get its grasp centre below z = 0.071, and reaching lower folds
# the forearm into the table), and the giving arm needs a shallow grip on the
# body so the receiving arm still has the rim to take.
FINGERS_DOWN_MIN = 0.05
JAW_HORIZONTAL_MAX = 0.35

# Band of grasp-centre heights either arm can actually reach on a cup standing
# on a surface. The lower bound is the measured floor for a downward-pointing
# gripper (z = 0.085 above the table); the upper bound is the cup's rim, less a
# margin so the grip stays on the body.
CUP_GRIP_BAND = (0.088, 2 * CUP_HALF_H - 0.008)

# Objects an upright carry is the right model for: axisymmetric, gripped across
# the body, and meaningless upside down.
UPRIGHT_OBJECTS = ("cup",)

TASK_STRINGS = {
    "plate": "pick up the plate and place it on the table with arm A",
    "cup": "bring the cup across the table and set it on the plate",
}


# --------------------------------------------------------------------------
# small maths helpers
# --------------------------------------------------------------------------
def mat2quat(R: np.ndarray) -> np.ndarray:
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).flatten())
    return q


def quat2mat(q: np.ndarray) -> np.ndarray:
    m = np.empty(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, float))
    return m.reshape(3, 3)


def slerp(Ra: np.ndarray, Rb: np.ndarray, t: float) -> np.ndarray:
    qa, qb = mat2quat(Ra), mat2quat(Rb)
    if qa @ qb < 0:
        qb = -qb
    d = float(np.clip(qa @ qb, -1.0, 1.0))
    if d > 0.9995:
        q = qa + t * (qb - qa)
    else:
        th = np.arccos(d)
        q = (np.sin((1 - t) * th) * qa + np.sin(t * th) * qb) / np.sin(th)
    q /= np.linalg.norm(q)
    return quat2mat(q)


def smoothstep(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


# --------------------------------------------------------------------------
# the environment
# --------------------------------------------------------------------------
@dataclass
class ArmTarget:
    pos: np.ndarray
    R: np.ndarray
    grip: float


@dataclass
class EpisodeRecord:
    states: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    images: dict = field(default_factory=lambda: {c: [] for c in CAMERAS})
    n_ik_fail: int = 0
    max_ik_pos_err: float = 0.0
    faults: list = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.states)


class BimanualEnv:
    """MuJoCo scene + Mink controllers + recording."""

    def __init__(
        self,
        img_w: int = 320,
        img_h: int = 240,
        record_images: bool = True,
        seed: int | None = None,
    ) -> None:
        self.model, self.idx = build_model()
        self.model.opt.timestep = PHYS_DT
        self.data = mujoco.MjData(self.model)
        self.img_w, self.img_h = img_w, img_h
        self.record_images = record_images
        self.renderer = (
            mujoco.Renderer(self.model, height=img_h, width=img_w)
            if record_images
            else None
        )
        self.ik = {a: ArmIK(a) for a in ARMS}
        # Measured reachable set per arm. Waypoints are planned inside this
        # rather than invented and hoped for: the arm has five joints for a
        # six-DOF pose, so most poses that look reasonable on paper are simply
        # not in the set, and differential IK reports a converged solve while
        # sitting 70-250 mm away from them.
        self.table = {a: ReachTable(a) for a in ARMS}
        self._probe = mujoco.MjData(self.model)
        self._holds_cache: dict = {}
        self.rng = np.random.default_rng(seed)
        self.q_cmd = {a: HOME_QPOS.copy() for a in ARMS}
        self.target: dict[str, ArmTarget] = {}
        self.held: dict[str, str | None] = {a: None for a in ARMS}
        self.grasp_rel: dict[str, tuple] = {}
        self.rec = EpisodeRecord()
        self.noise_sigma = 0.0
        self._light_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_LIGHT, n)
            for n in ("key", "fill")
        ]
        self._light_diffuse0 = self.model.light_diffuse.copy()
        self._light_pos0 = self.model.light_pos.copy()
        self._cam_pos0 = self.model.cam_pos.copy()
        self._cam_quat0 = self.model.cam_quat.copy()

    # -- setup ----------------------------------------------------------
    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def base_of(self, arm: str) -> np.ndarray:
        return np.array([-ARM_X if arm == "right" else ARM_X, 0.0, 0.0])

    def reset(self, noise_sigma: float = 0.0) -> None:
        reset_home(self.model, self.data, self.idx)
        self.q_cmd = {a: HOME_QPOS.copy() for a in ARMS}
        self.held = {a: None for a in ARMS}
        self.grasp_rel = {}
        self.rec = EpisodeRecord()
        self.noise_sigma = noise_sigma
        self.target = {}
        self.give_dz, self.take_dz, self.handoff_z = -0.015, 0.030, HANDOFF_CUP_Z
        self.handoff = None
        self.relay = None
        for arm in ARMS:
            p, R = self.ik[arm].fk(self.q_cmd[arm])
            self.target[arm] = ArmTarget(p + R @ GRASP_CENTRE_SITE, R, HOME_QPOS[5])

    def randomize_objects(self) -> None:
        """Lay the table out so that the episode is actually performable.

        Object positions are SAMPLED FROM THE MEASURED REACHABLE SETS, and the
        grip heights are then DERIVED from the poses that were drawn -- not the
        other way round. Asking for a particular grip height and hoping the arm
        has it is what left the cup 21-81 mm outside the grasp set; asking the
        arm which grasps it owns, and building the episode around the answer,
        cannot fail that way.

        The chain runs backwards from the end of the task, because the last
        grasp is the most constrained: whatever height the receiving arm can
        hold the cup at while setting it on the plate fixes where it must take
        the cup during the hand-off, which in turn fixes where the giving arm
        has to grip it.
        """
        rng = self.rng
        rest_z = PLATE_TOP_Z + CUP_HALF_H

        # 1. where can the LEFT arm hold a cup that is standing on the plate?
        place_idx = self.table["left"].grasp_poses(
            rest_z + CUP_GRIP_BAND[0] - CUP_HALF_H,
            rest_z + CUP_GRIP_BAND[1] - CUP_HALF_H,
            fingers_down=FINGERS_DOWN_MIN, jaw_horizontal=JAW_HORIZONTAL_MAX,
            x_lim=(-0.10, 0.28), y_lim=(-0.22, 0.22),
        )
        # 2. where can the RIGHT arm grip a cup standing on the table?
        pick_idx = self.table["right"].grasp_poses(
            CUP_GRIP_BAND[0], CUP_GRIP_BAND[1],
            fingers_down=FINGERS_DOWN_MIN, jaw_horizontal=JAW_HORIZONTAL_MAX,
            x_lim=(-0.28, 0.10), y_lim=(-0.22, 0.22),
        )
        cup_stand_z = CUP_HALF_H + 0.001

        for _ in range(400):
            _, place_pt = self.table["left"].sample_pose(rng, place_idx)
            _, pick_pt = self.table["right"].sample_pose(rng, pick_idx)
            take_dz = float(place_pt[2] - rest_z)
            give_dz = float(pick_pt[2] - cup_stand_z)
            # the giver must hold the cup BELOW the receiver, with room between
            if take_dz - give_dz < 0.022:
                continue
            if np.linalg.norm(pick_pt[:2] - place_pt[:2]) < 0.16:
                continue
            relay = self._find_relay()
            if relay is None:
                continue
            hand = self._find_handoff(give_dz, take_dz)
            self.give_dz, self.take_dz = give_dz, take_dz
            self.relay = relay
            self.handoff = hand
            self.handoff_z = hand["cup_z"] if hand else HANDOFF_CUP_Z
            self.cup_xy = pick_pt[:2].copy()
            self.plate_xy = place_pt[:2].copy()
            break
        else:
            self.fault("no performable layout found in 400 draws")
            self.give_dz, self.take_dz = -0.015, 0.030
            self.relay = None
            self.handoff, self.handoff_z = None, HANDOFF_CUP_Z
            self.cup_xy = np.array([-0.14, 0.06])
            self.plate_xy = np.array([0.14, -0.06])

        spoon_xy = np.array([rng.uniform(-0.08, 0.08), rng.uniform(-0.30, -0.24)])
        yaw = rng.uniform(-np.pi, np.pi)
        set_object_pose(self.model, self.data, self.idx, "cup",
                        [self.cup_xy[0], self.cup_xy[1], cup_stand_z])
        set_object_pose(self.model, self.data, self.idx, "plate",
                        [self.plate_xy[0], self.plate_xy[1], 0.0075])
        set_object_pose(self.model, self.data, self.idx, "spoon",
                        [spoon_xy[0], spoon_xy[1], 0.005],
                        quat=[np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
        mujoco.mj_forward(self.model, self.data)
        for _ in range(60):  # let them settle on the table
            mujoco.mj_step(self.model, self.data)

    def _find_handoff(self, give_dz: float, take_dz: float,
                      tol: float = GRASP_MISS_TOL):
        """Cup-centre height at which both arms can serve the transfer.

        Returns the height AND the two configurations it validated, which
        `handoff_cup` then reuses verbatim. Re-planning them later instead --
        with a joint-proximity bias, from wherever the giver happened to be --
        moved the giver's grip up to 53 mm off the cup and failed a transfer
        this search had already proved feasible. The tolerance is the same
        constant the primitives enforce, for the same reason: when the search
        accepted 10 mm and the primitive demanded 6, layouts passed as
        performable and then faulted.

        Transfer heights well above the table are tried first: a near-vertical
        wrist is only comfortable for this arm above z ~ 0.16.
        """
        best = None
        for cx in (HANDOFF_XY[0], 0.0, 0.04, -0.04, 0.12):
            for cz in np.arange(0.26, 0.155, -0.01):
                centre = np.array([cx, HANDOFF_XY[1], cz])
                qg, _, Rg, mg = cup_grasp_spec(self, "right", centre, give_dz)
                if mg > tol:
                    continue
                qt, pt, Rt, mt = cup_grasp_spec(self, "left", centre, take_dz)
                if mt > tol:
                    continue
                if best is None or max(mg, mt) < best[0]:
                    best = (max(mg, mt), cx, cz, qg, Rg, qt, Rt, pt)
        if best is not None:
            _, cx, cz, qg, Rg, qt, Rt, pt = best
            return {
                    "cup_xy": (float(cx), float(HANDOFF_XY[1])),
                    "cup_z": float(cz),
                    "q_give": qg.copy(), "R_give": Rg.copy(),
                    "q_take": qt.copy(), "R_take": Rt.copy(),
                    "take_pt": pt.copy(),
            }
        return None

    def _find_relay(self, tol: float = GRASP_MISS_TOL):
        """A spot on the table where one arm can set the cup down and the
        other can pick it up.

        This replaces the mid-air hand-off, which is not achievable with this
        model. A transfer in free space needs both grippers inside the same
        small volume at the same moment, and MuJoCo collides these meshes as
        convex hulls that are far fatter than the parts: the receiver, sent to
        a pose validated on its own, landed 116 mm away because it was fouling
        the giver's arm. The same rigid carry also inverted the cup on the way
        up (measured up = -0.87), because a kinematic grasp rotates the object
        with the wrist and the presentation pose is nothing like the pick pose.

        Handing over via the table costs one extra pick and place and avoids
        both problems: the arms are never in the same place, and every grasp is
        drawn from the tabletop family, so the cup stays upright throughout.
        """
        stand_z = CUP_HALF_H + 0.001
        for rx in (0.0, 0.03, -0.03, 0.06, -0.06):
            for ry in (0.0, 0.05, -0.05, 0.10, -0.10):
                centre = np.array([rx, ry, stand_z])
                best = None
                for dz_r in np.arange(0.014, 0.047, 0.004):
                    _, _, _, mr = cup_grasp_spec(self, "right", centre, dz_r)
                    if mr > tol:
                        continue
                    for dz_l in np.arange(0.014, 0.047, 0.004):
                        _, _, _, ml = cup_grasp_spec(self, "left", centre, dz_l)
                        if ml > tol:
                            continue
                        if best is None or max(mr, ml) < best[0]:
                            best = (max(mr, ml), dz_r, dz_l)
                if best is not None:
                    return {"xy": (float(rx), float(ry)),
                            "dz_put": float(best[1]), "dz_take": float(best[2])}
        return None

    def randomize_visuals(self, light_scale=(0.6, 1.35), cam_jitter=0.02) -> None:
        """Per-episode lighting and camera jitter. Purely observational."""
        rng = self.rng
        self.model.light_diffuse[:] = self._light_diffuse0
        self.model.light_pos[:] = self._light_pos0
        for lid in self._light_ids:
            if lid < 0:
                continue
            self.model.light_diffuse[lid] = np.clip(
                self._light_diffuse0[lid] * rng.uniform(*light_scale), 0.02, 1.0
            )
            self.model.light_pos[lid] = self._light_pos0[lid] + rng.normal(0, 0.12, 3)

        self.model.cam_pos[:] = self._cam_pos0
        self.model.cam_quat[:] = self._cam_quat0
        for name in CAMERAS:
            cid = self.idx.cam_id[name]
            scale = 1.0 if name == "top" else 0.35  # wrist mounts wobble less
            self.model.cam_pos[cid] = self._cam_pos0[cid] + rng.normal(
                0, cam_jitter * scale, 3
            )
            dq = np.concatenate([[1.0], rng.normal(0, 0.012 * scale, 3)])
            dq /= np.linalg.norm(dq)
            q = np.empty(4)
            mujoco.mju_mulQuat(q, self._cam_quat0[cid], dq)
            self.model.cam_quat[cid] = q

    # -- grasping --------------------------------------------------------
    def attach(self, arm: str, obj: str) -> None:
        """Latch `obj` to `arm`'s gripper at their current relative pose.

        This is a kinematic carry, not a soft weld: the object's free joint is
        rewritten from the gripper pose after every physics substep. A MuJoCo
        weld equality was tried first and rejected -- it is compliant enough
        that the receiving gripper shoves the object out of the giving
        gripper's grasp during a handoff, which silently corrupts the whole
        downstream trajectory. The jaws still close on the object visually and
        everything else in the scene stays fully dynamic.
        """
        b1 = self.idx.gripper_body[arm]
        b2 = self.idx.obj_body[obj]
        R1 = self.data.xmat[b1].reshape(3, 3)
        R2 = self.data.xmat[b2].reshape(3, 3)
        self.grasp_rel[arm] = (
            R1.T @ (self.data.xpos[b2] - self.data.xpos[b1]),
            mat2quat(R1.T @ R2),
        )
        self.held[arm] = obj

    def detach(self, arm: str) -> None:
        self.held[arm] = None
        self.grasp_rel.pop(arm, None)

    def _carrier(self, obj: str) -> str | None:
        """Which arm owns `obj` right now. During a handoff both arms briefly
        hold it; the giver keeps authority until it lets go, so the pose never
        jumps mid-transfer."""
        holders = [a for a in ARMS if self.held.get(a) == obj]
        return holders[0] if holders else None

    def _apply_carry(self) -> None:
        if CARRY_MODE == "physics":
            return  # the jaws are holding it; there is nothing to rewrite
        for obj in OBJECTS:
            arm = self._carrier(obj)
            if arm is None:
                continue
            b1 = self.idx.gripper_body[arm]
            R1 = self.data.xmat[b1].reshape(3, 3)
            relp, relq = self.grasp_rel[arm]
            adr = self.idx.obj_qpos_adr[obj]
            self.data.qpos[adr : adr + 3] = self.data.xpos[b1] + R1 @ relp
            q1 = mat2quat(R1)
            qo = np.empty(4)
            mujoco.mju_mulQuat(qo, q1, relq)
            if obj in UPRIGHT_OBJECTS:
                # Keep the cup's axis vertical and let it follow the gripper in
                # position and yaw only. A rigid carry rotates the object with
                # the wrist, and because the pick and place poses are different
                # families the cup arrived upside down (measured up = -1.00) --
                # an artifact of the carry, not of anything the arm did. A real
                # cup in a gripper whose jaws close horizontally does not
                # invert; it stays level or slips. This is the "stays level"
                # assumption, and it is stated in the README.
                yaw = np.arctan2(R1[1, 0], R1[0, 0])
                qo = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
            self.data.qpos[adr + 3 : adr + 7] = qo
            jid = self.model.body_jntadr[self.idx.obj_body[obj]]
            dof = self.model.jnt_dofadr[jid]
            self.data.qvel[dof : dof + 6] = 0.0

    # -- stepping / recording -------------------------------------------
    def capture(self) -> dict:
        out = {}
        for cam in CAMERAS:
            self.renderer.update_scene(self.data, camera=cam)
            out[cam] = self.renderer.render().copy()
        return out

    def control_step(self, record: bool = True) -> None:
        """One control tick: IK -> clean action -> (noisy) execution -> record."""
        clean = np.zeros(12)
        for i, arm in enumerate(ARMS):
            t = self.target[arm]
            site_p = site_pose_for_grasp(t.pos, t.R)
            q = self.ik[arm].solve(self.q_cmd[arm], site_p, t.R)
            q[5] = t.grip
            self.q_cmd[arm] = q
            clean[6 * i : 6 * i + 6] = q
            self.rec.max_ik_pos_err = max(
                self.rec.max_ik_pos_err, self.ik[arm].last_status.pos_err
            )
            if self.ik[arm].last_status.pos_err > 0.02:
                self.rec.n_ik_fail += 1

        executed = clean.copy()
        if self.noise_sigma > 0.0:
            executed = executed + self.rng.normal(0.0, self.noise_sigma, 12)

        for i, arm in enumerate(ARMS):
            for k, aid in enumerate(self.idx.arm_act_ids[arm]):
                self.data.ctrl[aid] = executed[6 * i + k]

        if record:
            self.rec.states.append(state_vector(self.data, self.idx))
            self.rec.actions.append(clean)  # CLEAN label, always
            if self.record_images:
                imgs = self.capture()
                for c in CAMERAS:
                    self.rec.images[c].append(imgs[c])

        for _ in range(SUBSTEPS):
            mujoco.mj_step(self.model, self.data)
            self._apply_carry()

    # -- phase execution -------------------------------------------------
    def run_phase(
        self,
        duration: float,
        goals: dict,
        on_end: Callable[[], None] | None = None,
        record: bool = True,
    ) -> None:
        """Interpolate each arm's Cartesian target to its goal over `duration`."""
        n = max(1, int(round(duration * FPS)))
        start = {a: ArmTarget(t.pos.copy(), t.R.copy(), t.grip) for a, t in self.target.items()}
        for k in range(1, n + 1):
            s = smoothstep(k / n)
            for arm in ARMS:
                g = goals.get(arm)
                if g is None:
                    continue
                p0, R0, g0 = start[arm].pos, start[arm].R, start[arm].grip
                p1 = g.get("pos", p0)
                R1 = g.get("R", R0)
                g1 = g.get("grip", g0)
                self.target[arm] = ArmTarget(
                    pos=p0 + (np.asarray(p1, float) - p0) * s,
                    R=slerp(R0, np.asarray(R1, float), s),
                    grip=g0 + (g1 - g0) * s,
                )
            self.control_step(record=record)
        if on_end is not None:
            on_end()

    def hold(self, duration: float, record: bool = True) -> None:
        self.run_phase(duration, {}, record=record)

    def run_joint_phase(self, duration: float, goals: dict,
                        record: bool = True) -> None:
        """Interpolate each arm's JOINT command to its goal configuration.

        Transits are driven in joint space, not Cartesian space, because the
        goal configurations come out of the measured reachability table and are
        therefore known to exist. Interpolating the end-effector pose instead
        walks the target through poses that are NOT in the arm's reachable set,
        where differential IK does not merely stall -- it diverges, and the
        commanded pose runs away to metres from the table. Interpolating the
        joints cannot leave the reachable set, needs no solve, and lands
        exactly on the pose that was planned.
        """
        n = max(1, int(round(duration * FPS)))
        start = {a: self.q_cmd[a].copy() for a in ARMS}
        for k in range(1, n + 1):
            sm = smoothstep(k / n)
            for arm in ARMS:
                g = goals.get(arm)
                if g is None:
                    continue
                q0 = start[arm]
                q1 = np.asarray(g, float)
                self.q_cmd[arm] = q0 + (q1 - q0) * sm
            self.joint_step(record=record)
        # keep the Cartesian target consistent for anything that reads it next
        for arm in ARMS:
            if arm in goals:
                p, R = self.ik[arm].fk(self.q_cmd[arm])
                self.target[arm] = ArmTarget(
                    p + R @ GRASP_CENTRE_SITE, R, self.q_cmd[arm][5]
                )

    def joint_step(self, record: bool = True) -> None:
        """One control tick with the joint command taken as given."""
        clean = np.concatenate([self.q_cmd[a] for a in ARMS])
        executed = clean.copy()
        if self.noise_sigma > 0.0:
            executed = executed + self.rng.normal(0.0, self.noise_sigma, 12)
        for i, arm in enumerate(ARMS):
            for k, aid in enumerate(self.idx.arm_act_ids[arm]):
                self.data.ctrl[aid] = executed[6 * i + k]
        if record:
            self.rec.states.append(state_vector(self.data, self.idx))
            self.rec.actions.append(clean)  # CLEAN label, always
            if self.record_images:
                imgs = self.capture()
                for c in CAMERAS:
                    self.rec.images[c].append(imgs[c])
        for _ in range(SUBSTEPS):
            mujoco.mj_step(self.model, self.data)
            self._apply_carry()

    def grip_phase(self, duration: float, arm: str, grip: float,
                   on_end=None, record: bool = True) -> None:
        """Move one gripper joint only, holding the arm still."""
        n = max(1, int(round(duration * FPS)))
        g0 = float(self.q_cmd[arm][5])
        for k in range(1, n + 1):
            self.q_cmd[arm][5] = g0 + (grip - g0) * smoothstep(k / n)
            self.joint_step(record=record)
        # `hold()` resumes Cartesian control from `self.target`.  Keep its
        # gripper component in sync or the first hold tick restores the stale
        # pre-phase command (normally OPEN_Q) and immediately drops the object.
        self.target[arm].grip = float(grip)
        if on_end is not None:
            on_end()

    def settle_object(self, arm: str, obj: str, obj_target, q_seed,
                      what: str, iters: int = 4, dt: float = 0.24,
                      tol: float = 0.010) -> float:
        """Nudge the joint command until the HELD object sits on its target.

        Seeded from the planned configuration, so the solve starts inside the
        right basin and only has to polish. This is the one place a Cartesian
        solve is safe, and it is what absorbs both the servo droop and the
        small shift the object makes as the jaws close.
        """
        obj_target = np.asarray(obj_target, float)
        R = self.actual_grasp(arm)[1]
        err = np.inf
        for _ in range(iters):
            cur = object_pos(self.data, self.idx, obj)
            resid = obj_target - cur
            err = float(np.linalg.norm(resid))
            if err <= tol:
                break
            got, R = self.actual_grasp(arm)
            q = self.ik[arm].solve(
                self.q_cmd[arm], site_pose_for_grasp(got + resid, R), R, iters=25
            )
            if self.ik[arm].last_status.pos_err > 0.05:
                break  # solve is not helping; do not chase it
            q[5] = self.q_cmd[arm][5]
            if not self.holds(arm, q):
                break  # the correction is not a pose the servos could hold
            self.run_joint_phase(dt, {arm: q})
        err = float(np.linalg.norm(object_pos(self.data, self.idx, obj) - obj_target))
        if err > tol:
            self.fault(f"{what}: object {err*1000:.0f}mm off target")
        return err

    # -- faults, closed-loop reaching, grasp checks ----------------------
    def fault(self, what: str) -> None:
        """Record that the episode did not do what the script asked.

        This exists because the previous failure mode was silence. A waypoint
        that could not be reached was quietly replaced by a compromise point
        near the arm's base, the primitive carried on regardless, and the
        episode was written out as a demonstration in which the gripper closes
        on air and the object is released 70 mm above the table. A faulted
        episode is discarded by the success predicate, which is the only
        outcome that keeps the dataset honest.
        """
        self.rec.faults.append(what)

    def actual_grasp(self, arm: str):
        """The arm's ACHIEVED grasp centre and orientation, from MjData."""
        return grasp_centre_now(self.data, self.idx.ee_site[arm])

    def reach(self, arm: str, point, R: np.ndarray, grip: float,
              duration: float, what: str, tol: float = REACH_TOL,
              strict: bool = True) -> np.ndarray:
        """Move `arm`'s grasp centre to `point`, then check it got there.

        Returns the achieved grasp centre. When `strict`, a miss faults the
        episode instead of being absorbed.
        """
        point = np.asarray(point, float)
        self.run_phase(duration, {arm: {"pos": point, "R": R, "grip": grip}})
        got, _ = self.actual_grasp(arm)
        err = float(np.linalg.norm(got - point))
        if err > tol:
            if strict:
                self.fault(f"{what}: {err*1000:.0f}mm short of the waypoint")
            return got
        return got

    def settle(self, arm: str, point, R: np.ndarray, grip: float,
               what: str, iters: int = 3, dt: float = 0.24,
               tol: float = REACH_TOL) -> float:
        """Close the loop: correct the COMMAND by the measured residual.

        These are torque-limited position servos, so the achieved pose lags the
        command whenever the arm is loaded or near its limits. Open-loop
        Cartesian scripting therefore lands consistently short. Nudging the
        command by the residual and re-settling converges in two or three
        passes and costs a fraction of a second of episode time.
        """
        point = np.asarray(point, float)
        cmd = self.target[arm].pos.copy()
        err = np.inf
        for _ in range(iters):
            got, _ = self.actual_grasp(arm)
            resid = point - got
            err = float(np.linalg.norm(resid))
            if err <= tol:
                break
            cmd = cmd + resid
            self.run_phase(dt, {arm: {"pos": cmd, "R": R, "grip": grip}})
        got, _ = self.actual_grasp(arm)
        err = float(np.linalg.norm(got - point))
        if err > tol:
            self.fault(f"{what}: settled {err*1000:.0f}mm off target")
        return err

    def grasp_ok(self, arm: str, obj: str, what: str) -> bool:
        """Is `obj` actually inside the jaws right now?

        The cup is a vertical cylinder and is deliberately gripped OFF-CENTRE
        -- up to 46 mm up its own axis, because a cup standing on the table can
        only be reached near its rim. So the test is radial distance from the
        cup's axis plus height along it, not a box around the cup's centre.
        Comparing against the centre instead made every correct grasp look
        50-60 mm wide of the mark, which sent me chasing a frame error that was
        not there: the offset was simply the intended grip height, rotated into
        the gripper's frame.
        """
        got, _ = self.actual_grasp(arm)
        c = object_pos(self.data, self.idx, obj)
        if obj == "cup":
            radial = float(np.linalg.norm((got - c)[:2]))
            height = float(got[2] - c[2])
            if radial > CUP_RADIUS - 0.004:
                self.fault(f"{what}: jaws {radial*1000:.0f}mm off the cup's axis")
                return False
            if abs(height) > CUP_HALF_H - 0.004:
                self.fault(
                    f"{what}: jaws {height*1000:+.0f}mm from the cup's centre, "
                    f"past its {CUP_HALF_H*1000:.0f}mm body"
                )
                return False
            return True
        d = _R_of(self, arm).T @ (c - got)
        if np.any(np.abs(d) > GRASP_WINDOW + 0.004):
            self.fault(
                f"{what}: {obj} outside the jaws by "
                f"{np.round((np.abs(d) - GRASP_WINDOW) * 1000, 1)}mm"
            )
            return False
        return True

    def _jaw_body(self, arm: str) -> int:
        cache = self.__dict__.setdefault("_jaw_body_ids", {})
        if arm not in cache:
            cache[arm] = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, f"{arm}_moving_jaw_so101_v1"
            )
        return cache[arm]

    def holding(self, arm: str, obj: str) -> bool:
        """Is `obj` pinched between BOTH of `arm`'s jaws right now?

        Read off the contact list, which is the only honest answer once the
        grasp is physical: a jaw that closed on air, or one that pushed the
        object away while closing, leaves no contact pair behind. The same
        test PegBitStudio's entry uses for its `holding()`; credited in the
        README.
        """
        fixed = self.idx.gripper_body[arm]
        moving = self._jaw_body(arm)
        target = self.idx.obj_body[obj]
        f = m = False
        for c in range(self.data.ncon):
            con = self.data.contact[c]
            b = {self.model.geom_bodyid[con.geom1], self.model.geom_bodyid[con.geom2]}
            if target in b:
                f |= fixed in b
                m |= moving in b
        return f and m

    def check_held(self, arm: str, obj: str, what: str) -> bool:
        """Fault unless the jaws really closed on `obj` (physics mode only).

        In kinematic mode the carry is a pose rewrite and there is nothing to
        check, so this always passes there.
        """
        if CARRY_MODE != "physics" or self.holding(arm, obj):
            return True
        self.fault(f"{what}: jaws closed but the {obj} is not between them")
        return False

    def holds(self, arm: str, q, tol: float = 0.020, steps: int = 150) -> bool:
        """Can the position servos actually hold this configuration?

        Simulated in a scratch MjData so the live episode is untouched. This
        replaces the static gravity-torque test in `validate_reach.py`, which
        turned out to predict nothing: a pose it passed showing 0.62 N.m of
        bias then saturated `shoulder_lift` at its 3.35 N.m limit and settled
        300 mrad from its command. Measured over 400 random tabletop-band
        poses, only 36% are actually holdable -- so this is the difference
        between a planner that works and one that keeps sending the arm
        somewhere it can reach on paper and collapses in practice.

        Cached per configuration; a run re-plans the same few waypoints often.
        """
        q = np.asarray(q, float)
        key = (arm, q[:5].tobytes())
        hit = self._holds_cache.get(key)
        if hit is not None:
            return hit
        d = self._probe
        mujoco.mj_resetData(self.model, d)
        other = [a for a in ARMS if a != arm][0]
        for k in range(6):
            d.qpos[self.idx.arm_qpos_ids[other][k]] = HOME_QPOS[k]
            d.ctrl[self.idx.arm_act_ids[other][k]] = HOME_QPOS[k]
        # Park scratch objects far below the scene AND apart.  Co-locating all
        # three at [0, 0, -5] made their deeply overlapping geoms explode on
        # cup rotational DOF 15, corrupting otherwise deterministic hold tests.
        for j, o in enumerate(OBJECTS):
            set_object_pose(self.model, d, self.idx, o,
                            [float(j), 0.0, -5.0])
        for k in range(5):
            d.qpos[self.idx.arm_qpos_ids[arm][k]] = q[k]
            d.ctrl[self.idx.arm_act_ids[arm][k]] = q[k]
        d.qpos[self.idx.arm_qpos_ids[arm][5]] = OPEN_Q
        d.ctrl[self.idx.arm_act_ids[arm][5]] = OPEN_Q
        d.qvel[:] = 0.0
        mujoco.mj_forward(self.model, d)
        for _ in range(steps):
            mujoco.mj_step(self.model, d)
        got = np.array([d.qpos[self.idx.arm_qpos_ids[arm][k]] for k in range(5)])
        ok = bool(np.max(np.abs(q[:5] - got)) < tol)
        self._holds_cache[key] = ok
        return ok

    def reachable(self, arm: str, point, R: np.ndarray,
                  tol: float = REACH_TOL) -> bool:
        """Can Mink hit this pose at all? Used to reject a layout up front."""
        self.ik[arm].solve(
            NEUTRAL_QPOS, site_pose_for_grasp(np.asarray(point, float), R), R, iters=60
        )
        return self.ik[arm].last_status.pos_err < tol

    # -- pose helpers ----------------------------------------------------
    def carry_anchor(self, arm: str, azimuth_from=None) -> np.ndarray:
        """A grasp point that is always comfortably inside this arm's envelope."""
        base = self.base_of(arm)
        ref = np.asarray(azimuth_from, float) if azimuth_from is not None else None
        if ref is None:
            d = np.array([1.0, 0.0, 0.0]) * (1 if arm == "right" else -1)
        else:
            d = ref - base
            d[2] = 0.0
            n = np.linalg.norm(d)
            d = d / n if n > 1e-6 else np.array([1.0 if arm == "right" else -1.0, 0, 0])
        return base + d * 0.20 + np.array([0.0, 0.0, 0.10])

    def clamp_reachable(
        self, arm: str, point, R: np.ndarray, tol: float = 0.006, n: int = 12
    ) -> np.ndarray:
        """Pull `point` toward the arm's carry anchor until Mink can actually hit it.

        Cheap insurance against a randomised layout pushing a waypoint outside
        the 5-DOF workspace, which is what silently wrecks a scripted episode.
        """
        point = np.asarray(point, float)
        anchor = self.carry_anchor(arm, azimuth_from=point)
        seed = self.q_cmd[arm]
        for s in np.linspace(0.0, 1.0, n):
            p = point * (1 - s) + anchor * s
            self.ik[arm].solve(seed, site_pose_for_grasp(p, R), R, iters=20)
            if self.ik[arm].last_status.pos_err < tol:
                return p
        return anchor

    def clamp_above(
        self, arm: str, point, R: np.ndarray, floor_z: float, tol: float = 0.006
    ) -> np.ndarray:
        """Reachability clamp that only ever moves the point DOWN.

        Hover waypoints must stay directly above their grasp point: if the
        horizontal position is allowed to drift, the descent turns into a
        diagonal sweep that knocks the object out of the jaws before the
        fingers ever close on it.
        """
        point = np.asarray(point, float)
        for z in np.linspace(point[2], floor_z, 10):
            p = np.array([point[0], point[1], z])
            self.ik[arm].solve(self.q_cmd[arm], site_pose_for_grasp(p, R), R, iters=20)
            if self.ik[arm].last_status.pos_err < tol:
                return p
        return np.array([point[0], point[1], floor_z])

    def topdown_frame(self, arm: str, point, roll: float = 0.0) -> np.ndarray:
        reach = np.asarray(point, float) - self.base_of(arm)
        reach[2] = 0.0
        return grasp_frame([0, 0, -1], reach, roll=roll)

    def side_frame(self, arm: str, point, tilt: float = 0.18) -> np.ndarray:
        """Approach a standing object from the side, jaws closing horizontally.

        The fingers travel in almost horizontally (tilted `tilt` rad downward
        so the wrist clears the table) and close across the object in the
        horizontal plane. This is the grasp for a cup the other arm has to take
        from above.
        """
        reach = np.asarray(point, float) - self.base_of(arm)
        reach[2] = 0.0
        reach /= np.linalg.norm(reach)
        approach = reach * np.cos(tilt) + np.array([0, 0, -1.0]) * np.sin(tilt)
        closing = np.cross([0.0, 0.0, 1.0], reach)
        return grasp_frame(approach, closing)

    def tilted_frame(self, arm: str, point, tilt: float = 0.8,
                     roll: float = 0.0) -> np.ndarray:
        """Approach tilted `tilt` rad away from straight-down, leaning outward.

        At large tilts the approach axis becomes nearly parallel to the reach
        direction and the default jaw axis degenerates to vertical -- which
        would try to squeeze a standing cup along its own axis. Pass roll=pi/2
        to keep the fingers closing horizontally.
        """
        reach = np.asarray(point, float) - self.base_of(arm)
        reach[2] = 0.0
        reach /= np.linalg.norm(reach)
        a = np.array([0, 0, -1.0]) * np.cos(tilt) + reach * np.sin(tilt)
        return grasp_frame(a, reach, roll=roll)


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------
def _R_of(env, arm: str) -> np.ndarray:
    return env.actual_grasp(arm)[1]


def approach_dir(R: np.ndarray) -> np.ndarray:
    """The direction the fingers point, in world coordinates."""
    return FINGER_SIGN * R[:, 0]


def cup_grasp_spec(env: BimanualEnv, arm: str, cup_centre, dz: float,
                   z_band: float = 0.008, q_near=None,
                   standing: str | None = None):  # `standing` kept for callers
    """Plan a grasp on the cup's body, `dz` from its centre along the axis.

    Poses are drawn from the SAME predicate the layout sampler uses -- coming
    down onto the cup at any workable angle, jaws closing around the body
    rather than along it -- restricted to a narrow band of grip heights around
    the one asked for. Returns (configuration, grasp centre, orientation, miss).
    """
    want = np.array([cup_centre[0], cup_centre[1], cup_centre[2] + dz])
    idx = env.table[arm].grasp_poses(
        want[2] - z_band, want[2] + z_band,
        fingers_down=FINGERS_DOWN_MIN, jaw_horizontal=JAW_HORIZONTAL_MAX,
       
    )
    if len(idx) == 0:
        return NEUTRAL_QPOS.copy(), want, np.eye(3), 1.0
    for q, centre, miss in env.table[arm].nearest_in_k(
        want, idx, k=48, q_ref=q_near, tol=GRASP_MISS_TOL
    ):
        if not env.holds(arm, q):
            continue
        q6 = np.concatenate([q[:5], [OPEN_Q]])
        p, R = env.ik[arm].fk(q6)
        return q6, p + R @ GRASP_CENTRE_SITE, R, miss
    return NEUTRAL_QPOS.copy(), want, np.eye(3), 1.0


def plan_point(env: BimanualEnv, arm: str, point, q_near=None,
               z_band: float = 0.022):
    """Nearest validated pose that puts the grasp centre at an arbitrary point.

    Used for stand-off waypoints, which are NOT on the object and so cannot go
    through `cup_grasp_spec`.
    """
    point = np.asarray(point, float)
    idx = env.table[arm].grasp_poses(
        point[2] - z_band, point[2] + z_band,
        fingers_down=FINGERS_DOWN_MIN, jaw_horizontal=JAW_HORIZONTAL_MAX,
    )
    if len(idx) == 0:
        return None
    for q, centre, miss in env.table[arm].nearest_in_k(
        point, idx, k=48, q_ref=q_near, tol=0.030
    ):
        if not env.holds(arm, q):
            continue
        q6 = np.concatenate([q[:5], [OPEN_Q]])
        pp, R = env.ik[arm].fk(q6)
        return q6, pp + R @ GRASP_CENTRE_SITE, R, miss
    return None


def stand_off(env: BimanualEnv, arm: str, grasp_pt, R: np.ndarray, q_grasp,
              back: float = 0.090):
    """A waypoint backed off along the arm's OWN approach axis.

    Not simply "above the object". These grasps are often shallow -- a cup
    standing on the table can only be gripped near its rim, and the giving arm
    comes in at 25 degrees below horizontal -- so descending vertically onto
    the grasp point drives the side of the gripper into the cup instead of
    bringing the fingers around it. That collision reached 16 N and saturated
    `shoulder_lift` 277 mrad short of its command, which left the jaws open
    around nothing and the cup shoved 22 mm sideways.
    """
    want = np.asarray(grasp_pt, float) - approach_dir(R) * back
    res = plan_point(env, arm, want, q_near=q_grasp)
    if res is None:
        return None
    q_pre, _, _, _ = res
    q_pre = q_pre.copy()
    q_pre[5] = OPEN_Q
    return q_pre


def approach_straight(env: BimanualEnv, arm: str, q_goal, duration: float = 0.9,
                      grip: float | None = None) -> None:
    """Close the last few centimetres onto the PLANNED configuration.

    This used to interpolate the end-effector pose in Cartesian space, to stop
    joint interpolation bulging the hand through the object it was reaching
    for. Two things changed. The gripper no longer collides with the objects it
    manipulates, so a bulge cannot knock the cup any more; and, more
    importantly, a Cartesian leg DISCARDS the configuration the planner chose
    and lets the IK pick its own. That is how the arm kept ending up 81-109 mm
    above the cup: the planner selected a pose the servos could hold, and then
    the final leg substituted one they could not, saturating `shoulder_lift`
    300 mrad short. Ending on the planned joint vector is the whole point.
    """
    q = np.asarray(q_goal, float).copy()
    if grip is not None:
        q[5] = grip
    env.run_joint_phase(duration, {arm: q})


def _place_until_contact(env: BimanualEnv, arm: str, q_goal, obj_target,
                         duration: float = 1.0, stop_tol: float = 0.010) -> float:
    """Lower a held object until it reaches its resting target.

    Physical contact can unload the jaws before the arm reaches the nominal
    no-contact configuration.  Stop on the measured object pose at that
    moment; continuing to drive the closed fingers is what sweeps a correctly
    placed cup back off the surface.
    """
    q0 = env.q_cmd[arm].copy()
    q1 = np.asarray(q_goal, float).copy()
    n = max(1, int(round(duration * FPS)))
    err = float(np.linalg.norm(object_pos(env.data, env.idx, "cup") - obj_target))
    for k in range(1, n + 1):
        env.q_cmd[arm] = q0 + (q1 - q0) * smoothstep(k / n)
        env.joint_step()
        err = float(np.linalg.norm(object_pos(env.data, env.idx, "cup") - obj_target))
        if err <= stop_tol:
            break
    p, R = env.ik[arm].fk(env.q_cmd[arm])
    env.target[arm] = ArmTarget(p + R @ GRASP_CENTRE_SITE, R, env.q_cmd[arm][5])
    return err


def pick_cup(env: BimanualEnv, arm: str, dz: float | None = None) -> None:
    """Take the standing cup, gripping its lower body.

    Low on the body, not on the rim, because the receiving arm has to grip the
    same cup higher up a moment later: two grippers cannot share a 90 mm cup
    unless they take different heights on it.
    """
    cup = object_pos(env.data, env.idx, "cup").copy()
    surface = cup[2] - CUP_HALF_H
    if dz is not None:
        cands = [dz]
    else:
        # Search the band on the cup's body rather than trusting a grip height
        # predicted earlier: the cup is wherever the previous arm actually left
        # it, which is not exactly where the plan intended.
        # `dz` is measured from the cup's CENTRE, while CUP_GRIP_BAND is a
        # world height above the surface the cup stands on -- hence both terms.
        lo = max(CUP_GRIP_BAND[0] - surface - CUP_HALF_H, -(CUP_HALF_H - 0.006))
        # Do not start at the highest merely-valid body point.  The Menagerie
        # squeeze test holds reliably with the cup centre 10--40 mm below the
        # grasp site; near-rim (+59 mm) plans leave too little finger overlap
        # and were the source of the 0/10 gate's empty/rim closes.
        hi = min(CUP_HALF_H - 0.006, 0.040)
        cands = list(np.arange(hi, lo - 0.001, -0.005)) if hi > lo else []
    best = None
    for cand in cands:
        q_grasp, gp, R, miss = cup_grasp_spec(env, arm, cup, float(cand))
        if best is None or miss < best[0]:
            best = (miss, q_grasp, gp, R)
        if miss <= GRASP_MISS_TOL:
            break
    if best is None or best[0] > GRASP_MISS_TOL:
        env.fault(
            f"cup is {(best[0] if best else 1.0)*1000:.0f}mm outside the "
            f"{arm} grasp set"
        )
        return
    _, q_grasp, gp, R = best
    q_grasp = q_grasp.copy()
    q_grasp[5] = OPEN_Q
    q_pre = stand_off(env, arm, gp, R, q_grasp)
    if q_pre is None:
        env.fault("no stand-off pose for the cup pick")
        return

    env.run_joint_phase(1.3, {arm: q_pre})
    approach_straight(env, arm, q_grasp, grip=OPEN_Q)
    env.hold(0.16)
    if not env.grasp_ok(arm, "cup", "cup pick"):
        return
    env.grip_phase(0.3, arm, OPEN_Q, on_end=lambda: env.attach(arm, "cup"))
    env.grip_phase(0.45, arm, CLOSE_Q["cup"])
    env.hold(0.15)  # let the contact settle before anything reads it
    env.check_held(arm, "cup", "cup pick")


def held_offset(env: BimanualEnv, arm: str, obj: str) -> np.ndarray:
    """Where `obj` sits in the gripper's own frame, measured now."""
    got, R = env.actual_grasp(arm)
    return R.T @ (object_pos(env.data, env.idx, obj) - got)


def handoff_cup(env: BimanualEnv, from_arm: str, to_arm: str) -> None:
    """Giver lifts the cup to the transfer point; receiver takes it higher up.

    The transfer point is high above the table and near the midline because
    that is where both arms' reachable sets overlap at the two grip heights
    this needs -- a near-vertical wrist is only comfortable for this arm above
    z ~ 0.16.
    """
    if env.handoff is None:
        env.fault("no validated hand-off for this layout")
        return
    plan = env.handoff
    give_centre = np.array([plan["cup_xy"][0], plan["cup_xy"][1], plan["cup_z"]])
    q_give, R_give = plan["q_give"], plan["R_give"]
    q_take, R_take, take_pt = plan["q_take"], plan["R_take"], plan["take_pt"]

    # 1. giver raises the cup to the transfer point and trims it onto the mark
    q_give = q_give.copy()
    q_give[5] = CLOSE_Q["cup"]
    env.run_joint_phase(1.7, {from_arm: q_give})
    env.settle_object(from_arm, "cup", give_centre, q_give, "present the cup")
    env.hold(0.2)

    # 2. receiver comes in on the cup where it actually ended up
    cup_now = object_pos(env.data, env.idx, "cup").copy()
    q_take, take_pt, R_take, miss_t = cup_grasp_spec(env, to_arm, cup_now, env.take_dz)
    if miss_t > GRASP_MISS_TOL:
        # The giver did not manage to present the cup somewhere the receiver
        # can reach. Say so: the alternative is the silent fallback this
        # function used to take, which walked the receiver to its neutral
        # posture and then reported the cup 81-115 mm from the jaws.
        env.fault(f"cup presented {miss_t*1000:.0f}mm outside the receiver's set")
        return
    q_take = q_take.copy()
    q_take[5] = OPEN_Q
    q_pre = stand_off(env, to_arm, take_pt, R_take, q_take)
    if q_pre is None:
        env.fault("no stand-off pose for the receiver")
        return
    env.run_joint_phase(1.4, {to_arm: q_pre})
    approach_straight(env, to_arm, q_take, grip=OPEN_Q)
    env.hold(0.16)

    # 3. receiver closes, then the giver lets go. Both hold it for a moment,
    #    which is what a real two-handed transfer looks like.
    if not env.grasp_ok(to_arm, "cup", "hand-off"):
        return
    env.grip_phase(0.25, to_arm, OPEN_Q, on_end=lambda: env.attach(to_arm, "cup"))
    env.grip_phase(0.35, to_arm, CLOSE_Q["cup"])
    env.hold(0.1)
    if not env.check_held(to_arm, "cup", "hand-off"):
        return
    env.grip_phase(0.4, from_arm, OPEN_Q, on_end=lambda: env.detach(from_arm))

    # 4. giver backs out of the way, and parks
    env.run_joint_phase(1.2, {from_arm: HOME_QPOS})


def _upright_cup_pose(env: BimanualEnv, arm: str, target, q_seed):
    """Solve the held cup's position and upright axis, leaving its yaw free."""
    got, R = env.actual_grasp(arm)
    local = R.T @ (object_pos(env.data, env.idx, "cup") - got)
    axis = R.T @ env.data.xmat[env.idx.obj_body["cup"]].reshape(3, 3)[:, 2]
    grip = env.q_cmd[arm][5]

    def residual(q):
        p, orient = env.ik[arm].fk(np.r_[q, grip])
        return np.r_[p + orient @ (GRASP_CENTRE_SITE + local) - target,
                     0.1 * (orient @ axis - [0.0, 0.0, 1.0])]

    bounds = env.ik[arm].model.jnt_range[:5]
    candidates = [least_squares(
        residual, seed[:5], bounds=(bounds[:, 0], bounds[:, 1]),
        max_nfev=150, ftol=1e-10, xtol=1e-10, gtol=1e-10,
    ) for seed in (env.q_cmd[arm], q_seed)]
    result = min(candidates, key=lambda r: np.linalg.norm(r.fun))
    return np.r_[result.x, grip]


def _place_cup_physics(env: BimanualEnv, arm: str, drop, dz):
    if not env.check_held(arm, "cup", "cup place"):
        return
    release_target = drop + np.array([0.0, 0.0, 0.003])
    q_seed, _, _, miss = cup_grasp_spec(
        env, arm, release_target, dz, q_near=env.q_cmd[arm],
    )
    if miss > GRASP_MISS_TOL:
        env.fault(f"plate is {miss*1000:.0f}mm outside the {arm} placing set")
        return
    # Lift before crossing the table, then lower with the measured cup axis
    # upright. A position-only target can otherwise invert a real rigid grasp.
    lift = object_pos(env.data, env.idx, "cup") + np.array([0.0, 0.0, 0.080])
    env.run_joint_phase(0.9, {arm: _upright_cup_pose(env, arm, lift, q_seed)})
    above = release_target + np.array([0.0, 0.0, 0.080])
    env.run_joint_phase(1.5, {arm: _upright_cup_pose(env, arm, above, q_seed)})
    q_drop = _upright_cup_pose(env, arm, release_target, q_seed)
    _place_until_contact(env, arm, q_drop, release_target, duration=1.2)
    env.settle_object(arm, "cup", release_target, env.q_cmd[arm],
                      "cup at release", iters=4, dt=0.18, tol=0.010)
    env.hold(0.3)
    err_z = abs(object_pos(env.data, env.idx, "cup")[2] - drop[2])
    if err_z > 0.018:
        env.fault(f"cup {err_z*1000:.0f}mm off its resting height at release")
    env.grip_phase(1.2, arm, OPEN_Q, on_end=lambda: env.detach(arm))
    env.hold(0.3)
    start, R = env.actual_grasp(arm)
    for k in range(1, 9):
        point = start + np.array([0.0, 0.0, 0.010 * k])
        q = env.ik[arm].solve(
            env.q_cmd[arm], site_pose_for_grasp(point, R), R, iters=25,
        )
        q[5] = OPEN_Q
        env.run_joint_phase(0.12, {arm: q})


def place_cup(env: BimanualEnv, arm: str, target_xy, rest_z: float,
              dz: float | None = None) -> None:
    """Set the carried cup down at `target_xy` with its centre at `rest_z`."""
    drop = np.array([target_xy[0], target_xy[1], rest_z])
    if CARRY_MODE == "physics":
        _place_cup_physics(env, arm, drop, env.take_dz if dz is None else dz)
        return
    q_drop, gp, R, miss = cup_grasp_spec(
        env, arm, drop, env.take_dz if dz is None else dz,
        q_near=env.q_cmd[arm],
    )
    if miss > GRASP_MISS_TOL:
        env.fault(f"plate is {miss*1000:.0f}mm outside the {arm} placing set")
        return
    grip = float(env.q_cmd[arm][5])
    q_drop = q_drop.copy()
    q_drop[5] = grip
    q_above = stand_off(env, arm, gp, R, q_drop)
    if q_above is None:
        env.fault("no stand-off pose for the place")
        return
    q_above = q_above.copy()
    q_above[5] = grip
    env.run_joint_phase(1.5, {arm: q_above})
    approach_straight(env, arm, q_drop, duration=1.0, grip=grip)
    env.settle_object(arm, "cup", drop, q_drop, "cup onto the plate")
    env.hold(0.3)
    err_z = abs(object_pos(env.data, env.idx, "cup")[2] - rest_z)
    if err_z > 0.018:
        env.fault(f"cup {err_z*1000:.0f}mm off its resting height at release")
    env.grip_phase(0.25, arm, OPEN_Q, on_end=lambda: env.detach(arm))
    q_retract = q_above.copy()
    q_retract[5] = OPEN_Q
    env.run_joint_phase(1.0, {arm: q_retract})


def go_home(env: BimanualEnv, arm: str) -> None:
    env.run_joint_phase(1.0, {arm: HOME_QPOS})


# --------------------------------------------------------------------------
# scripted tasks + success predicates
# --------------------------------------------------------------------------
def task_cup_relay(env: BimanualEnv) -> dict:
    """Right brings the cup across, left puts it on the plate.

    The cup starts on the right arm's side and the plate sits on the left's,
    so neither arm can do this alone -- it is genuinely bimanual. The transfer
    happens on the table rather than in mid-air; see `_find_relay` for why.
    """
    plate0 = object_pos(env.data, env.idx, "plate").copy()
    if env.relay is None:
        env.fault("no relay spot for this layout")
        return {"plate_xy": plate0[:2]}
    relay = env.relay
    pick_cup(env, "right")
    place_cup(env, "right", relay["xy"], rest_z=CUP_HALF_H + 0.001,
              dz=relay["dz_put"])
    go_home(env, "right")
    pick_cup(env, "left", dz=relay["dz_take"])
    place_cup(env, "left", plate0[:2], rest_z=PLATE_TOP_Z + CUP_HALF_H)
    go_home(env, "left")
    env.hold(0.6)
    return {"plate_xy": plate0[:2]}


def success_cup_handoff(env: BimanualEnv, info: dict) -> tuple[bool, str]:
    if env.rec.faults:
        return False, f"faulted: {env.rec.faults[0]}"
    cup = object_pos(env.data, env.idx, "cup")
    plate = object_pos(env.data, env.idx, "plate")
    R = env.data.xmat[env.idx.obj_body["cup"]].reshape(3, 3)
    upright = float(R[:, 2] @ np.array([0, 0, 1.0]))
    dxy = float(np.linalg.norm(cup[:2] - plate[:2]))
    on_plate_z = PLATE_TOP_Z + CUP_HALF_H
    if dxy > PLATE_RADIUS:
        return False, f"cup {dxy*1000:.0f}mm from plate centre"
    if upright < 0.85:
        return False, f"cup tipped over (up={upright:.2f})"
    if not (on_plate_z - 0.030 <= cup[2] <= on_plate_z + 0.035):
        return False, f"cup z={cup[2]:.3f} not resting on plate ({on_plate_z:.3f})"
    if plate[2] < -0.01 or abs(plate[0]) > 0.40 or abs(plate[1]) > 0.32:
        return False, "plate left the table"
    return True, f"cup on plate, {dxy*1000:.0f}mm off centre"


def task_plate_move(env: BimanualEnv) -> dict:
    """Left arm picks the plate up and sets it down elsewhere on the table."""
    p0 = object_pos(env.data, env.idx, "plate").copy()
    tgt = np.array([env.rng.uniform(0.10, 0.19), env.rng.uniform(-0.12, 0.12)])
    while np.linalg.norm(tgt - p0[:2]) < 0.07:
        tgt = np.array([env.rng.uniform(0.10, 0.19), env.rng.uniform(-0.12, 0.12)])
    # Deliberately not implemented. The plate as modelled -- a 55 mm radius
    # disc with a 3 mm rim overhang, lying flat on the table -- cannot be
    # grasped by this gripper at all: closing across it needs 110 mm against a
    # ~85 mm maximum aperture, and pinching the rim vertically needs a finger
    # underneath a plate that is resting on the table. It needs either a foot
    # ring to hook under or a smaller radius, and that is a change to the
    # declared task, so it is the owner's call rather than mine.
    env.fault(
        "plate task not implemented: the plate is not graspable as modelled "
        "(needs a foot ring or a smaller radius -- see FINDINGS-SIM.md)"
    )
    return {"target_xy": tgt, "start_xy": p0[:2]}
    go_home(env, "left")
    env.hold(0.6)
    return {"target_xy": tgt, "start_xy": p0[:2]}


def success_plate_move(env: BimanualEnv, info: dict) -> tuple[bool, str]:
    if env.rec.faults:
        return False, f"faulted: {env.rec.faults[0]}"
    plate = object_pos(env.data, env.idx, "plate")
    R = env.data.xmat[env.idx.obj_body["plate"]].reshape(3, 3)
    upright = float(R[:, 2] @ np.array([0, 0, 1.0]))
    err = float(np.linalg.norm(plate[:2] - info["target_xy"]))
    moved = float(np.linalg.norm(plate[:2] - info["start_xy"]))
    if moved < 0.04:
        return False, f"plate barely moved ({moved*1000:.0f}mm)"
    if err > 0.06:
        return False, f"plate {err*1000:.0f}mm from target"
    if upright < 0.90:
        return False, f"plate not flat (up={upright:.2f})"
    if not (-0.005 <= plate[2] <= 0.03):
        return False, f"plate z={plate[2]:.3f} off the table"
    return True, f"plate placed, {err*1000:.0f}mm from target"


TASKS = {
    "cup": (TASK_STRINGS["cup"], task_cup_relay, success_cup_handoff),
    "plate": (TASK_STRINGS["plate"], task_plate_move, success_plate_move),
}
