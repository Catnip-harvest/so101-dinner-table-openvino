"""Bimanual SO-101 dinner-table scene.

Two TheRobotStudio SO-101 arms face each other across a table. Three free
bodies sit between them: a plate, a cup and a spoon. Three cameras are named
exactly `top`, `left_wrist` and `right_wrist`.

The two arms are one SO-101 MJCF attached twice with MjSpec, under the
prefixes `left_` and `right_`. By default that is the Menagerie
`robotstudio_so101` model (real, graspable jaws); `BIMANUAL_ARM=trs` selects
the raw TheRobotStudio export, and `BIMANUAL_CARRY=kinematic|physics` picks
how a grasped object is carried (see ARM_MODEL / CARRY_MODE below).

Nothing in here is Windows-specific; the only platform-sensitive bit is the GL
backend, which lives in mj_backend.py.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mj_backend import select_backend

select_backend()

import mujoco  # noqa: E402

HERE = Path(__file__).resolve().parent

# Which SO-101 model file to attach. Default is DeepMind's Menagerie release
# (assets/menagerie_so101, Apache-2.0, origin in UPSTREAM.txt), which ships the
# gripper with its collision geometry split into parts plus fingertip
# primitives, so the jaws can actually close on something. `trs` is the raw
# TheRobotStudio export this project started with, whose fixed jaw collides as
# one convex hull that fills its own opening (FINDINGS-SIM.md section 5).
ARM_MODEL = os.environ.get("BIMANUAL_ARM", "menagerie")
ARM_XML = {
    "menagerie": HERE / "assets" / "menagerie_so101" / "so101.xml",
    "trs": HERE / "assets" / "so101" / "so101_new_calib.xml",
}[ARM_MODEL]

# How a grasped object travels with the arm. `physics` is real contact: the
# jaws squeeze the object and friction carries it. `kinematic` rewrites the
# object's pose from the gripper after every substep and switches the arms'
# contact with the objects off -- the disclosed fallback the TRS model needs.
CARRY_MODE = os.environ.get(
    "BIMANUAL_CARRY", "physics"  # WP1b: physics 7/10 and 14/20, 15 Sep; see briefs/WP1b-REPORT.md
)

JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
ARMS = ("left", "right")
OBJECTS = ("plate", "cup", "spoon")

# Home posture, radians, in JOINT_NAMES order. Gripper open.
#
# This is a PARK pose, and the retraction is the point. The obvious "elbow
# tucked, leaning over the table" home put both jaws at x = -/+0.083, a 166 mm
# gap across the midline -- so the moment either arm reached for the centre it
# drove its moving jaw into the idle arm's moving jaw at 10 N. That contact
# stalled `shoulder_pan` against its 3.35 N.m torque limit 636 mrad short of
# its command, while the IK still reported a 0.1 mm solution. Every downstream
# symptom (grasping air, releasing 7 cm high, the cup carried floating beside
# the gripper) came from that one collision.
#
# Measured clearance for the pose below: jaw at [-/+0.156, 0, 0.249], a 313 mm
# midline gap, high enough to clear the table and every object on it.
HOME_QPOS = np.array([0.0, -1.60, 0.30, 1.50, 0.0, 1.20], dtype=np.float64)

# Mid-workspace posture. Separate from HOME_QPOS on purpose: the park pose is
# folded up and away, which makes it a bad IK seed and a worse posture-task
# target -- seeding a reach from it leaves the solver 255 mm short where
# seeding from here leaves it 88 mm. Park is where the arm waits; this is what
# the solver should regard as a neutral, unstretched arm.
NEUTRAL_QPOS = np.array([0.0, -0.90, 0.65, 1.45, 0.0, 1.20], dtype=np.float64)

# Table top sits at z = 0. Everything above it is the workspace.
TABLE_HALF = (0.42, 0.34, 0.02)
TABLE_TOP_Z = 0.0
ARM_X = 0.25  # each base this far from the table centre, along +/- x

# Height of each arm's base above the table top. Overridable for experiments
# via BIMANUAL_MOUNT_Z, because it is the single most consequential number in
# the scene: mounted flush at 0.0, both arms have to reach DOWN and OUT to
# touch the table, which is their weakest configuration -- `shoulder_lift`
# saturates at its 3.35 N.m limit and settles 300 mrad from its command, and a
# near-vertical wrist cannot get its grasp centre below z = 0.085 at all.
MOUNT_Z = float(os.environ.get("BIMANUAL_MOUNT_Z", "0.0"))

# Wrist camera mount, taken verbatim from TheRobotStudio's own
# so101_new_calib_camera.xml (the `wrist_camera` body, expressed in the
# `gripper` body frame). MuJoCo cameras look down their own -z, so the camera
# z-axis is the negation of the module's optical axis.
WRIST_CAM_POS = (0.0, 0.0, -0.025)
WRIST_CAM_Z = (0.0, 0.0, 1.0)
WRIST_CAM_X = (1.0, 0.0, 0.0)
WRIST_CAM_Y = (0.0, 1.0, 0.0)

CUP_RADIUS = 0.024
# A tumbler, not a teacup, and the height is forced by the arm rather than
# chosen for looks. This arm cannot put its grasp centre below z = 0.085 with
# the fingers pointing downward -- measured over 200k sampled configurations --
# so a 90 mm cup standing on the table offers a 6 mm band between that floor
# and its own rim, and the grasp check needs the grip to be on the body, which
# leaves about 2 mm. Nothing can be picked up reliably in 2 mm. At 130 mm tall
# the usable band is 45 mm, and a relay spot both arms can reach exists at the
# midline with a 10.6 mm worst-case miss. Measured alternatives: 90 mm gives no
# mutual spot at any arm spacing from 34 to 50 cm; 110 mm gives 14.2 mm.
CUP_HALF_H = 0.065
PLATE_RADIUS = 0.055
PLATE_TOP_Z = 0.014

GRIPPER_OPEN = 1.35
GRIPPER_CLOSED = 0.05


def _quat_from_mat(m: np.ndarray) -> np.ndarray:
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, m.flatten())
    return q


def _cam_quat(x_axis, y_axis, z_axis) -> np.ndarray:
    """MuJoCo camera frame: looks down -z, image up is +y."""
    m = np.column_stack(
        [np.asarray(x_axis, float), np.asarray(y_axis, float), np.asarray(z_axis, float)]
    )
    # orthonormalise defensively
    m[:, 2] /= np.linalg.norm(m[:, 2])
    m[:, 0] -= m[:, 2] * (m[:, 0] @ m[:, 2])
    m[:, 0] /= np.linalg.norm(m[:, 0])
    m[:, 1] = np.cross(m[:, 2], m[:, 0])
    return _quat_from_mat(m)


BASE_XML = f"""
<mujoco model="bimanual_dinner_table">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.005" integrator="implicitfast" cone="elliptic" impratio="10"
          noslip_iterations="3"/>
  <size njmax="600" nconmax="300"/>

  <visual>
    <global offwidth="1280" offheight="960"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.55 0.55 0.55" specular="0.1 0.1 0.1"/>
    <quality shadowsize="2048"/>
  </visual>

  <asset>
    <texture name="skybox" type="skybox" builtin="gradient" rgb1="0.35 0.45 0.6"
             rgb2="0.08 0.10 0.15" width="256" height="256"/>
    <texture name="floor_tex" type="2d" builtin="checker" rgb1="0.22 0.24 0.27"
             rgb2="0.28 0.30 0.34" width="300" height="300"/>
    <material name="floor_mat" texture="floor_tex" texrepeat="6 6" reflectance="0.05"/>
    <material name="table_mat" rgba="0.78 0.62 0.42 1" reflectance="0.02"/>
    <material name="plate_mat" rgba="0.94 0.94 0.96 1" reflectance="0.15"/>
    <material name="cup_mat" rgba="0.20 0.45 0.80 1" reflectance="0.10"/>
    <material name="spoon_mat" rgba="0.80 0.80 0.85 1" reflectance="0.35"/>
  </asset>

  <default>
    <default class="obj">
      <geom condim="4" friction="1.2 0.02 0.001" solref="0.004 1" solimp="0.95 0.99 0.001"
            density="600"/>
    </default>
  </default>

  <worldbody>
    <light name="key" pos="0.0 -0.6 1.6" dir="0 0.35 -1" directional="false"
           diffuse="0.6 0.6 0.6" specular="0.15 0.15 0.15"/>
    <light name="fill" pos="0.0 0.6 1.4" dir="0 -0.35 -1" directional="false"
           diffuse="0.35 0.35 0.38"/>
    <geom name="floor" type="plane" size="3 3 0.05" pos="0 0 -0.72" material="floor_mat"/>

    <body name="table" pos="0 0 {TABLE_TOP_Z - TABLE_HALF[2]}">
      <geom name="table_top" type="box"
            size="{TABLE_HALF[0]} {TABLE_HALF[1]} {TABLE_HALF[2]}"
            material="table_mat" condim="4" friction="0.9 0.01 0.001"/>
      <geom name="leg_a" type="box" size="0.025 0.025 0.35" pos="0.36 0.28 -0.37"
            material="table_mat"/>
      <geom name="leg_b" type="box" size="0.025 0.025 0.35" pos="-0.36 0.28 -0.37"
            material="table_mat"/>
      <geom name="leg_c" type="box" size="0.025 0.025 0.35" pos="0.36 -0.28 -0.37"
            material="table_mat"/>
      <geom name="leg_d" type="box" size="0.025 0.025 0.35" pos="-0.36 -0.28 -0.37"
            material="table_mat"/>
    </body>

    <!-- Free objects. Poses are overwritten per-episode by the randomiser. -->
    <body name="plate" pos="0.14 -0.06 0.008">
      <freejoint name="plate_free"/>
      <geom name="plate_g" class="obj" type="cylinder" size="0.055 0.006"
            material="plate_mat" mass="0.09"/>
      <geom name="plate_rim" class="obj" type="cylinder" size="0.058 0.002" pos="0 0 0.008"
            material="plate_mat" mass="0.01"/>
      <site name="plate_site" pos="0 0 0.008" size="0.005" group="4"/>
    </body>

    <body name="cup" pos="-0.14 0.06 0.066">
      <freejoint name="cup_free"/>
      <geom name="cup_g" class="obj" type="cylinder" size="0.024 0.065"
            material="cup_mat" mass="0.06"/>
      <site name="cup_site" pos="0 0 0" size="0.005" group="4"/>
    </body>

    <body name="spoon" pos="0.0 -0.22 0.006" euler="0 0 0.3">
      <freejoint name="spoon_free"/>
      <geom name="spoon_handle" class="obj" type="box" size="0.055 0.006 0.003"
            material="spoon_mat" mass="0.012"/>
      <geom name="spoon_bowl" class="obj" type="ellipsoid" size="0.022 0.014 0.005"
            pos="-0.072 0 0.001" material="spoon_mat" mass="0.010"/>
      <site name="spoon_site" pos="0 0 0" size="0.005" group="4"/>
    </body>

    <!-- Overhead camera, tilted a little so both arms read as 3D. -->
    <camera name="top" pos="0 -0.62 0.56" xyaxes="1 0 0 0 0.610 0.792" fovy="52"/>

    <!-- Placement targets, visual only, so a human can see where "on the plate" is. -->
    <site name="table_centre" pos="0 0 0.001" size="0.004" rgba="0 1 0 0.0" group="4"/>
  </worldbody>
</mujoco>
"""


@dataclass
class SceneIndex:
    """Everything the controllers need to address the model by id."""

    model: mujoco.MjModel
    arm_joint_ids: dict = field(default_factory=dict)
    arm_dof_ids: dict = field(default_factory=dict)
    arm_qpos_ids: dict = field(default_factory=dict)
    arm_act_ids: dict = field(default_factory=dict)
    ee_site: dict = field(default_factory=dict)
    gripper_body: dict = field(default_factory=dict)
    obj_body: dict = field(default_factory=dict)
    obj_qpos_adr: dict = field(default_factory=dict)
    weld_id: dict = field(default_factory=dict)
    cam_id: dict = field(default_factory=dict)


# Collision primitives that replace the gripper's convex mesh hulls.
#
# Why this is necessary: MuJoCo collides meshes as their CONVEX HULL, and the
# SO-101's static gripper half (`wrist_roll_follower_so101_v1`) is C-shaped.
# Shrink-wrapping a C fills in the mouth of the C, so the hull swallows the
# whole jaw opening. Measured with a 4 mm probe sphere walked through the
# gripper's own frame (tools/measure_jaw_gap.py --raw): every point from
# z = -0.030 to +0.025 across the entire length of the gripper is inside the
# static hull, and the moving jaw's hull permanently interpenetrates it. There
# is no point that is clear with the jaws open and pinched with them closed.
#
# So as shipped this is not a gripper, it is a solid block, and no grasp is
# possible by construction. That is why the episode runner was carrying objects
# by rewriting their pose instead of gripping them, and why two arms parked
# near each other registered a permanent 10 N contact.
#
# The fix: disable collision on the two mesh hulls and replace them with
# primitives -- one plate per jaw face plus a housing block that sits BEHIND
# the jaw band in x, so it keeps the wrist solid against the table and the
# other arm without ever reaching into the gap.
#
# Boxes are given as (centre, half-size) in the `gripperframe` SITE frame,
# because that is the frame every measurement in this project is expressed in:
# fingers along -x, jaws closing along z, finger width along y. They are
# converted to their body frames at build time using the transforms below.
#
# This changes what the collision engine sees. It does NOT touch the vendored
# MJCF, which stays byte-identical to upstream, and it does not touch a single
# visual mesh -- the robot still looks exactly like an SO-101.

# Static side, on the `gripper` body.
GRIPPER_COLLISION_BOXES = (
    # fixed jaw face: top surface at z = -0.029
    ((-0.074, 0.0, -0.035), (0.024, 0.016, 0.006)),
    # housing: behind the jaw band, keeps the wrist solid against the world
    ((-0.0195, -0.001, 0.005), (0.0255, 0.025, 0.033)),
)

# Moving jaw, on the `moving_jaw_so101_v1` body. Defined at the CLOSED pose
# with its underside 2 mm above the fixed face, so closing the joint brings the
# two together and opening it swings this plate away through the jaw's own
# ~115 degree rotation.
MOVING_JAW_COLLISION_BOXES = (
    ((-0.074, 0.0, -0.021), (0.024, 0.016, 0.006)),
)

# Pose of the `gripperframe` site in each body's own frame, measured off the
# compiled model (see the derivation in tools/measure_jaw_gap.py). The moving
# jaw's entry is taken at the CLOSED joint angle, which is the pose its box is
# defined in.
SITE_IN_BODY = {
    "gripper": (
        (-0.0079, -0.00022, -0.09813),
        (0.707107, 0.0, 0.707107, 0.0),
    ),
    "moving_jaw": (
        (-0.03542, -0.07155, 0.01902),
        (0.474386, -0.474386, 0.524365, -0.524365),
    ),
}


def _site_to_body(key: str, centre):
    """Box centre and orientation in the body frame, from site coordinates."""
    p_bs, q_bs = SITE_IN_BODY[key]
    R = np.empty(9)
    mujoco.mju_quat2Mat(R, np.array(q_bs, dtype=float))
    R = R.reshape(3, 3)
    pos = np.array(p_bs, float) + R @ np.asarray(centre, float)
    return pos, np.array(q_bs, float)


def replace_gripper_collision(
    spec: mujoco.MjSpec, gripper_body: str, jaw_body: str
) -> None:
    """Swap the gripper's mesh collision hulls for primitives."""
    for body_name, key, boxes in (
        (gripper_body, "gripper", GRIPPER_COLLISION_BOXES),
        (jaw_body, "moving_jaw", MOVING_JAW_COLLISION_BOXES),
    ):
        body = spec.body(body_name)
        for geom in body.geoms:
            if geom.contype or geom.conaffinity:
                geom.contype = 0
                geom.conaffinity = 0
        for i, (centre, half) in enumerate(boxes):
            pos, quat = _site_to_body(key, centre)
            g = body.add_geom(
                name=f"{body_name}_col{i}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=pos.tolist(),
                quat=quat.tolist(),
                size=list(half),
                contype=1,
                conaffinity=1,
                group=3,
            )
            g.rgba = [0.9, 0.2, 0.2, 0.0]  # group 3 is collision-only anyway


def build_spec() -> mujoco.MjSpec:
    if not ARM_XML.exists():
        raise FileNotFoundError(
            f"SO-101 MJCF missing at {ARM_XML}. Run tools/fetch_so101.py first."
        )
    spec = mujoco.MjSpec.from_string(BASE_XML)

    placements = {
        # (x, y, yaw) -- right arm at -x faces +x, left arm at +x faces -x.
        "right": (-ARM_X, 0.0, 0.0),
        "left": (ARM_X, 0.0, np.pi),
    }
    for arm, (x, y, yaw) in placements.items():
        child = mujoco.MjSpec.from_file(str(ARM_XML))
        frame = spec.worldbody.add_frame(
            name=f"{arm}_mount", pos=[x, y, TABLE_TOP_Z + MOUNT_Z], euler=[0, 0, yaw]
        )
        spec.attach(child, prefix=f"{arm}_", frame=frame)

    if ARM_MODEL == "trs":
        for arm in ARMS:
            replace_gripper_collision(
                spec, f"{arm}_gripper", f"{arm}_moving_jaw_so101_v1"
            )

    # In kinematic carry mode the ARMS do not collide with the objects they
    # manipulate, and the episode runner carries them by rewriting their pose.
    # In physics mode (the default with the Menagerie model) this whole block
    # is skipped and the jaws really squeeze the object.
    #
    # This is a deliberate simplification and it must be stated in the README,
    # because it is the difference between "the policy learned to grasp" and
    # "the policy learned to move to a grasp pose".
    #
    # Why it is necessary rather than merely convenient: MuJoCo collides meshes
    # as convex hulls, and this arm's hulls are far fatter than the parts they
    # stand for. The static gripper half is C-shaped, so its hull swallows its
    # own jaw opening -- measured with a probe sphere, no point anywhere in the
    # gripper is clear with the jaws open and pinched with them closed, which
    # makes it a solid block rather than a gripper. The wrist hull sits 20 mm
    # inside the cup at a legitimate grasp pose and the forearm hull 2 mm in.
    # Left in, they fight every object the arm is sent to pick up: 16 N of
    # contact, three actuators saturated, the cup shoved 28 mm aside and the
    # gripper stalling 58-94 mm short of its target.
    #
    # Excluding only the gripper was tried first and is worse than excluding
    # everything: the jaws pass through the cup while the forearm still knocks
    # it over, which is incoherent rather than merely simplified.
    #
    # What stays real: the objects collide with the table, with the plate, with
    # the spoon and with each other, so setting the cup down and having it
    # rest, tip or roll is genuine physics. The arms collide with the table,
    # with themselves and with each other, so self-collisions and unreachable
    # poses are still rejected -- and `BimanualEnv.holds` additionally checks
    # that the servos can hold a pose against gravity before it is planned.
    # `BimanualEnv.grasp_ok` gates the carry: the object must sit inside the
    # measured jaw volume before it is allowed to start.
    if CARRY_MODE == "kinematic":
        arm_parts = [
            b.name
            for b in spec.bodies
            if any(b.name.startswith(f"{a}_") for a in ARMS)
        ]
        for part in arm_parts:
            for obj in OBJECTS:
                spec.add_exclude(
                    name=f"nc_{part}_{obj}", bodyname1=part, bodyname2=obj
                )

    # Wrist cameras, mounted on each gripper body looking down the tool axis.
    # The gripper body's tool direction is its local -z (the `gripperframe` site
    # sits at z = -0.098 in that body).
    cam_quat = _cam_quat(
        x_axis=WRIST_CAM_X, y_axis=WRIST_CAM_Y, z_axis=WRIST_CAM_Z
    )
    for arm in ARMS:
        body = spec.body(f"{arm}_gripper")
        body.add_camera(
            name=f"{arm}_wrist",
            pos=list(WRIST_CAM_POS),
            quat=cam_quat,
            fovy=58.0,
        )

    # Grasp-assist welds: one per (arm, object), all inactive at compile time.
    for arm in ARMS:
        for obj in OBJECTS:
            eq = spec.add_equality(
                name=f"weld_{arm}_{obj}",
                type=mujoco.mjtEq.mjEQ_WELD,
                objtype=mujoco.mjtObj.mjOBJ_BODY,
                name1=f"{arm}_gripper",
                name2=obj,
                active=0,
            )
            eq.solref = [0.008, 1.0]
            eq.solimp = [0.95, 0.99, 0.001, 0.5, 2]
            data = np.zeros(11)
            data[10] = 1.0  # torquescale
            eq.data = data

    return spec


def build_model() -> tuple[mujoco.MjModel, SceneIndex]:
    spec = build_spec()
    model = spec.compile()
    idx = SceneIndex(model=model)

    for arm in ARMS:
        idx.arm_joint_ids[arm] = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{arm}_{j}")
            for j in JOINT_NAMES
        ]
        idx.arm_dof_ids[arm] = [model.jnt_dofadr[j] for j in idx.arm_joint_ids[arm]]
        idx.arm_qpos_ids[arm] = [model.jnt_qposadr[j] for j in idx.arm_joint_ids[arm]]
        idx.arm_act_ids[arm] = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{arm}_{j}")
            for j in JOINT_NAMES
        ]
        idx.ee_site[arm] = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, f"{arm}_gripperframe"
        )
        idx.gripper_body[arm] = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"{arm}_gripper"
        )
        idx.cam_id[f"{arm}_wrist"] = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, f"{arm}_wrist"
        )
    idx.cam_id["top"] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "top")

    for obj in OBJECTS:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj)
        idx.obj_body[obj] = bid
        jid = model.body_jntadr[bid]
        idx.obj_qpos_adr[obj] = model.jnt_qposadr[jid]
        for arm in ARMS:
            idx.weld_id[(arm, obj)] = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_EQUALITY, f"weld_{arm}_{obj}"
            )

    for name, val in idx.cam_id.items():
        if val < 0:
            raise RuntimeError(f"camera {name!r} not found in compiled model")
    return model, idx


def reset_home(model: mujoco.MjModel, data: mujoco.MjData, idx: SceneIndex) -> None:
    mujoco.mj_resetData(model, data)
    for arm in ARMS:
        for k, qadr in enumerate(idx.arm_qpos_ids[arm]):
            data.qpos[qadr] = HOME_QPOS[k]
        for k, aid in enumerate(idx.arm_act_ids[arm]):
            data.ctrl[aid] = HOME_QPOS[k]
    data.eq_active[:] = 0
    mujoco.mj_forward(model, data)


def arm_qpos(data: mujoco.MjData, idx: SceneIndex, arm: str) -> np.ndarray:
    return np.array([data.qpos[a] for a in idx.arm_qpos_ids[arm]])


def state_vector(data: mujoco.MjData, idx: SceneIndex) -> np.ndarray:
    """12-D proprioceptive state: left 6 joints then right 6 joints."""
    return np.concatenate([arm_qpos(data, idx, a) for a in ARMS])


def set_object_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    idx: SceneIndex,
    obj: str,
    pos,
    quat=(1.0, 0.0, 0.0, 0.0),
) -> None:
    a = idx.obj_qpos_adr[obj]
    data.qpos[a : a + 3] = pos
    data.qpos[a + 3 : a + 7] = quat
    jid = model.body_jntadr[idx.obj_body[obj]]
    dof = model.jnt_dofadr[jid]
    data.qvel[dof : dof + 6] = 0.0


def object_pos(data: mujoco.MjData, idx: SceneIndex, obj: str) -> np.ndarray:
    return data.xpos[idx.obj_body[obj]].copy()


if __name__ == "__main__":
    m, i = build_model()
    print(f"nq={m.nq} nv={m.nv} nu={m.nu} nbody={m.nbody} ngeom={m.ngeom} ncam={m.ncam}")
    print("cameras:", {k: v for k, v in i.cam_id.items()})
    d = mujoco.MjData(m)
    reset_home(m, d, i)
    print("state (12D):", np.round(state_vector(d, i), 4))
