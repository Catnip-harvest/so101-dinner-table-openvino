"""Mink inverse kinematics + scripted manipulation primitives.

The IK is fed ONLY ground-truth simulator state (joint angles, object poses
read out of MjData) plus a target end-effector pose. It never sees an image.
Images are recorded as observations alongside, and nothing in this file reads
them.

Each arm gets its own single-arm mink Configuration built from the stock
so101_new_calib.xml. Targets are given in world coordinates and transformed
into that arm's base frame using the known mount transform, so the two arms
never fight over one solver.

Geometry note -- MEASURED, not read off the drawing. An earlier version of
this file had the approach axis inverted and the grasp centre 58 mm out, which
meant every "top-down" grasp was executed with the gripper upside down and the
jaws closed on air; objects only appeared to be carried because the episode
runner latches them kinematically. The numbers below come from
`tools/measure_grasp.py`, which parks a free object at candidate site-frame
offsets, closes the jaws and checks that BOTH fingers make contact:

    fingers point along site -x   (so site +x points back up the wrist)
    jaws close along site  z
    finger width spans site y
    grasp centre  = (-0.0875, +0.005, +0.005)
    holding window: x +/-22 mm, z +/-10 mm, y +/-5 mm
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bimanual_scene import (
    ARM_MODEL,
    ARM_X,
    ARM_XML,
    ARMS,
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    HOME_QPOS,
    NEUTRAL_QPOS,
    SceneIndex,
)

import mujoco  # noqa: E402
import mink  # noqa: E402

# Which way the fingers run along the site's x axis: +1 means site +x goes from
# the finger root to the fingertip. This differs between the two arm files and
# getting it wrong points the gripper at the ceiling, so it is measured per
# model (tools/measure_grasp.py for menagerie, tools/measure_jaw_gap.py for trs).
FINGER_SIGN = {"menagerie": 1.0, "trs": -1.0}[ARM_MODEL]

if ARM_MODEL == "menagerie":
    # Menagerie robotstudio_so101. The `gripperframe` site sits AT the fixed
    # fingertip; the fingers run along site +x toward it; the moving jaw comes
    # down along site z. Squeeze test on a 10 mm grid (tools/measure_grasp.py,
    # 15 Sep): a 48 mm cup whose centre starts at x in [-0.03,+0.01],
    # z in [+0.01,+0.04] before the jaws close is then held against gravity by
    # BOTH jaws, ends up centred on the tips (settled x,z within 4 mm of the
    # site origin) and the servo settles at 0.38-0.40 rad. Anything further
    # back along the finger starts inside the fixed jaw and is never gripped.
    GRASP_CENTRE_SITE = np.array([-0.010, 0.0, 0.025])
    GRASP_WINDOW = np.array([0.020, 0.010, 0.015])
    # Command well below the 0.38-0.40 rad contact angle so the servo squeezes
    # at its 2.94 N.m force limit; both 0.0 and 0.3 held in the test.
    CLOSE_Q = {"cup": 0.20, "plate": 0.20}
else:
    # Raw TheRobotStudio export with this project's replacement jaw plates.
    # Grasp centre in the gripperframe site frame: where an object's centre
    # must sit to be held. Measured, see the module docstring.
    GRASP_CENTRE_SITE = np.array([-0.075, 0.0, -0.005])
    # Half-widths of the jaw volume an object must sit inside to count as
    # grasped, metres, in site axes, from the jaw plates in bimanual_scene.py.
    GRASP_WINDOW = np.array([0.022, 0.014, 0.024])
    # Gripper joint angle that actually clamps each object, measured the same
    # way. The spoon is NOT grippable at the jaw centre; it is a distractor.
    CLOSE_Q = {"cup": 0.325, "plate": 0.350}
OPEN_Q = 1.20

# Jaw aperture (metres) as a function of the `gripper` joint angle, measured
# off the meshes. Used to pick an opening wide enough for a given object.
APERTURE_TABLE = np.array(
    [
        [-0.17, 0.0023],
        [-0.10, 0.0078],
        [0.00, 0.0158],
        [0.10, 0.0223],
        [0.20, 0.0246],
        [0.30, 0.0271],
        [0.40, 0.0296],
        [0.55, 0.0336],
        [0.70, 0.0394],
        [0.85, 0.0447],
        [1.00, 0.0513],
        [1.15, 0.0633],
        [1.30, 0.0851],
    ]
)

MOUNTS = {
    "right": (np.array([-ARM_X, 0.0, 0.0]), 0.0),
    "left": (np.array([ARM_X, 0.0, 0.0]), np.pi),
}


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def gripper_q_for_width(width: float) -> float:
    """Smallest `gripper` joint angle whose aperture clears `width`."""
    q = np.interp(width, APERTURE_TABLE[:, 1], APERTURE_TABLE[:, 0])
    return float(np.clip(q, -0.17, 1.6))


def grasp_frame(approach: np.ndarray, reach_dir: np.ndarray, roll: float = 0.0) -> np.ndarray:
    """Build a target site rotation from an approach axis and a reach direction.

    `approach` is the direction the FINGERS POINT -- the way they travel onto
    the object. Site x = FINGER_SIGN * approach: the fingers lie along site -x
    in the TRS file and along site +x in the Menagerie file. Getting this sign
    wrong inverts the gripper: a top-down grasp comes out pointing at the
    ceiling, which is the bug this convention exists to prevent.

    site z ~ reach_dir is the jaw closing direction; `roll` spins the jaws
    about the approach axis, which chooses which way the fingers straddle.
    """
    x = FINGER_SIGN * np.asarray(approach, float)
    x /= np.linalg.norm(x)
    z = np.asarray(reach_dir, float)
    z = z - x * (z @ x)
    n = np.linalg.norm(z)
    if n < 1e-8:  # degenerate: pick any perpendicular
        z = np.cross(x, [0.0, 0.0, 1.0])
        if np.linalg.norm(z) < 1e-8:
            z = np.cross(x, [1.0, 0.0, 0.0])
        n = np.linalg.norm(z)
    z /= n
    y = np.cross(z, x)
    R = np.column_stack([x, y, z])
    if abs(roll) > 1e-9:
        c, s = np.cos(roll), np.sin(roll)
        # rotate about the local x axis
        Rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)
        R = R @ Rx
    return R


def site_pose_for_grasp(grasp_point: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Where the site must sit so the grasp centre lands on `grasp_point`.

    `grasp_point` is where the OBJECT's centre should end up. There is no
    per-object offset to apply on top of this: the measured holding window is
    +/-5 mm across the fingers, so deliberately biasing the object toward one
    jaw face -- which the old `jaw_point_for_object` did by 28 mm -- pushes it
    straight out of the window.
    """
    return np.asarray(grasp_point, float) - R @ GRASP_CENTRE_SITE


def grasp_centre_now(data, site_id: int) -> tuple[np.ndarray, np.ndarray]:
    """The ACTUAL grasp centre and orientation, read out of MjData.

    The commanded Cartesian target and the achieved pose are not the same
    thing: these are torque-limited position servos, and a blocked or
    over-extended arm settles short of its command while the IK still reports
    a millimetre solution. Anything that decides whether a grasp happened must
    use this, not the target it asked for.
    """
    p = data.site_xpos[site_id]
    R = data.site_xmat[site_id].reshape(3, 3)
    return p + R @ GRASP_CENTRE_SITE, R.copy()


@dataclass
class IKStatus:
    pos_err: float
    ori_err: float


class ArmIK:
    """Mink IK for one SO-101, solved on a standalone single-arm model."""

    def __init__(
        self,
        arm: str,
        position_cost: float = 1.0,
        orientation_cost: float = 0.30,
        posture_cost: float = 3e-3,
    ) -> None:
        if arm not in MOUNTS:
            raise ValueError(arm)
        self.arm = arm
        self.model = mujoco.MjModel.from_xml_path(str(ARM_XML))
        self.data = mujoco.MjData(self.model)
        self.cfg = mink.Configuration(self.model)
        self.frame_task = mink.FrameTask(
            frame_name="gripperframe",
            frame_type="site",
            position_cost=position_cost,
            orientation_cost=orientation_cost,
            lm_damping=1.0,
        )
        self.posture_task = mink.PostureTask(self.model, cost=posture_cost)
        self.posture_task.set_target(NEUTRAL_QPOS.copy())
        self.limits = [mink.ConfigurationLimit(self.model)]
        p, yaw = MOUNTS[arm]
        self.p_mount = p
        self.R_mount = rot_z(yaw)
        self.site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe"
        )
        self.last_status = IKStatus(0.0, 0.0)

    # -- frame helpers ---------------------------------------------------
    def world_to_base(self, pos: np.ndarray, R: np.ndarray):
        return self.R_mount.T @ (np.asarray(pos, float) - self.p_mount), self.R_mount.T @ R

    def base_to_world(self, pos: np.ndarray, R: np.ndarray):
        return self.R_mount @ np.asarray(pos, float) + self.p_mount, self.R_mount @ R

    def fk(self, q: np.ndarray):
        """World-frame site pose for a joint vector (ground truth, no images)."""
        self.data.qpos[:] = q
        mujoco.mj_kinematics(self.model, self.data)
        p = self.data.site_xpos[self.site_id].copy()
        R = self.data.site_xmat[self.site_id].reshape(3, 3).copy()
        return self.base_to_world(p, R)

    # -- the solve -------------------------------------------------------
    def solve(
        self,
        q_seed: np.ndarray,
        target_pos: np.ndarray,
        target_R: np.ndarray,
        iters: int = 12,
        dt: float = 0.02,
        damping: float = 1e-3,
    ) -> np.ndarray:
        """Return a 6-vector joint command reaching the world-frame site pose.

        The `gripper` joint is left at its seed value; it has no influence on
        the site, so it is commanded separately by the primitives.
        """
        pb, Rb = self.world_to_base(target_pos, target_R)
        target = mink.SE3.from_rotation_and_translation(
            mink.SO3.from_matrix(Rb), np.asarray(pb, float)
        )
        self.frame_task.set_target(target)
        self.cfg.update(np.asarray(q_seed, float).copy())
        for _ in range(iters):
            v = mink.solve_ik(
                self.cfg,
                [self.frame_task, self.posture_task],
                dt,
                "daqp",
                damping,
                limits=self.limits,
            )
            self.cfg.integrate_inplace(v, dt)
        q = self.cfg.q.copy()
        q[5] = q_seed[5]

        err = self.frame_task.compute_error(self.cfg)
        self.last_status = IKStatus(
            float(np.linalg.norm(err[:3])), float(np.linalg.norm(err[3:]))
        )
        return q
