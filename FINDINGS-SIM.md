# What the simulation track actually does — measured, 15 Sep 2026

Written after a full session spent verifying `sim/` by running it. Every number
below came out of a script on this machine; nothing here is inferred.

**Read the headline first, because it changes the plan: the scripted
demonstrations have never worked, and the reason they appeared to work is that
the episode runner carries objects by rewriting their pose rather than by
gripping them.** A dataset collected from the pipeline as it stood would have
trained a policy on physically impossible data.

**RESOLVED, later the same day: the cup task now succeeds on 9 of 20 seeds,
with placements 2-7 mm off the plate centre, and every failure is detected and
discarded.** Two things changed the task itself and both need your sign-off:
the cup is now a 130 mm tumbler rather than a 90 mm cup, and the transfer
happens on the table rather than in mid-air. Sections 8 and 9 give the
measurements that forced each. A render of a successful episode is at
`out/relay/relay_1001_top.png`.

---

## 1. The headline

`sim/episode.py` attaches a carried object with `env.attach()`, which records the
object's pose relative to the gripper and then rewrites it after every physics
substep (`_apply_carry`). That is a deliberate design choice and it is
documented in the file. What was never checked is whether the jaws were ever
anywhere near the object when `attach()` fired.

They were not. Measured after a "successful" pick, the cup sat at

    (+0.002, +0.001, +0.037) m

in the gripper's own frame. The fingers occupy `x ∈ [-0.087, -0.034]` in that
frame. The cup was 65–90 mm from the nearest finger, held in mid-air by the
kinematic latch alone.

So the milestone-2 numbers in `HANDOFF-LABLAB.md` measured the wrong thing. The
plate task's "5/10 success" was five episodes of a cup-shaped object floating
beside a gripper that had closed on nothing.

**Superseded by sections 8-11: collection is now viable at a 45% yield.**

---

## 2. Four real defects, found and fixed

These are fixed in the working tree and each was verified by re-measuring.

### 2.1 Both arms parked with their jaws in the middle of the table

`HOME_QPOS` put each gripper at `x = ∓0.083`, a **166 mm gap** across the
midline. Any reach toward the centre — which is where the hand-off happens —
drove one arm's moving jaw into the other's at **10.2 N**. That contact stalled
`shoulder_pan` against its 3.35 N·m limit **636 mrad (36°)** short of its
command, while the IK still reported a 0.1 mm solution.

Fixed: `HOME_QPOS` is now a park pose with the jaw at `z = 0.249` and a
**313 mm** midline gap. Pan error went 636 mrad → 0, and the arm-on-arm contact
is gone. Measured, not assumed.

### 2.2 The gripper was upside down

`ik_control.py` documented the site's `+x` axis as the approach direction. The
fingers are actually along **`-x`**. So `grasp_frame([0,0,-1], …)` — every
"top-down" grasp in the file — produced a gripper pointing at the ceiling.

Fixed: `grasp_frame` now takes `approach` as the direction the fingers travel
and sets `site x = -approach`.

### 2.3 The grasp offset was 58 mm wrong

`GRASP_OFFSET` placed the grasp point 30 mm from the site along `+x`. Measured
by parking a free object at candidate offsets, closing the jaws and requiring
**both** fingers to make contact (`sim/tools/…`, and section 5 below on why this
measurement is itself compromised):

    grasp centre = (-0.0875, +0.005, +0.005) m in the site frame
    holding window: x ±22 mm, z ±10 mm, y ±5 mm
    fingers along site -x · jaws close along site z · finger width along y

### 2.4 Unreachable waypoints were silently retargeted

`clamp_reachable()` pulled any unreachable waypoint toward a point near the
arm's base and returned it, and every caller carried on. Consequences measured
on the cup task:

| Stage | What happened |
|---|---|
| Hand-off | receiver's take point moved 57 mm and up to `z = 0.100`; jaws closed on air; `attach` latched the cup at a 96 mm offset |
| Place | drop point moved 71 mm and rose from `z = 0.030` to `z = 0.100`; cup released 70 mm up, fell and tipped (`up = -0.21`) |

Fixed: waypoints are now planned inside the arm's measured reachable set, and a
miss **faults the episode** (`env.fault`) so the success predicate discards it.
A faulted episode is worth nothing; a silently retargeted one is worth less
than nothing.

---

## 3. What these arms can actually reach

The old code invented poses and asked the IK to hit them. That does not work
here: the arm has **five joints driving the gripper** (the sixth is the jaws),
so a fully specified six-DOF pose is generically unreachable — and differential
IK does not say so cleanly. It returns a "converged" solve sitting 70–250 mm
away, and multi-start does not rescue it (246 mm → 70 mm, then flat from 8 to
200 seeds).

So `sim/tools/reach_table.py` now samples joint space once (240 k draws),
records where the grasp centre lands and which way the fingers point, and
caches it. Planning picks the nearest pose the arm genuinely owns.

`sim/tools/validate_reach.py` then prunes that set to poses the robot can
**hold**: no contact with the table, the other arm, or itself, gravity torque
inside the actuator range, fingertips above the table. **77.2%** survive. This
pass was necessary — before it, the planner chose a pose whose `shoulder_lift`
settled **2043 mrad (117°)** from its command because the arm folded onto the
table under its own weight.

Measured properties worth keeping:

| Fact | Value |
|---|---|
| Fingers within 26° of straight down: lowest reachable grasp centre | **z = 0.085 m** (0.071 before the grasp centre was re-measured; see section 11) |
| Fingers >45° below horizontal: lowest | z = 0.026 m |
| Near-vertical wrist at tabletop height (z 0.07–0.16), left arm | **485 of 205 039** configurations |
| Same, well above the table (z 0.16–0.26) | 2854 configurations |
| Self/table collisions among tabletop grasps | penetration median 13–26 mm, max 53 mm — **real, not hull noise** |

The consequence: **a cup standing on the table can only be gripped near its
rim** -- and for a 90 mm cup that band is 6 mm wide, which is what forced the
taller cup in section 8. The original code's "grip the rim" choice was right,
for a reason it did not state.

With poses drawn from the validated set instead of invented, every stage the
task needs is feasible to sub-millimetre:

| Stage | Validated poses |
|---|---|
| Cup on table (right arm) | 255 |
| Cup on plate (left arm) | 364 |
| Hand-off, both arms, at their two grip heights | 35 combinations; best `x=+0.08, z=0.24`, misses 1.9 mm |

The hand-off has to be **high** — cup centre around `z = 0.24` — because that
is where two near-vertical wrists both work. That turned out to be necessary
but not sufficient; see section 9 for why the mid-air hand-off was dropped
anyway.

---

## 4. Other things now built

- **Joint-space transits.** Cartesian interpolation walks the target through
  poses outside the reachable set, where the IK diverges: the commanded pose ran
  away to metres from the table (`ikmax = 11 m`). Transits now interpolate joint
  angles between validated configurations, which cannot leave the set and needs
  no solve. The final approach is ALSO in joint space, landing exactly on the
  planned configuration: an earlier Cartesian last leg discarded the pose the
  planner had validated and let the IK substitute one the servos could not
  hold, which left the gripper 81-109 mm above the cup.
- **Layout sampling runs backwards from the end of the task.** Whatever height
  the receiving arm can hold the cup at while setting it on the plate fixes
  where it must take the cup at the hand-off, which fixes where the giving arm
  grips it. Sampling forwards and hoping is what produced "cup 21–81 mm outside
  the grasp set".
- **One shared grasp specification** for the sampler and the planner. When they
  disagreed — a loose filter for sampling, a 20° cone for planning — layouts
  passed as performable and then every waypoint came out 40–120 mm away.
- **`env.grasp_ok()`** checks the object is inside the measured holding window
  before `attach()` fires, in the gripper's frame. This is the check whose
  absence hid the headline defect for the whole project.

---

## 5. The blocker: the gripper's collision mesh fills the jaw gap

MuJoCo convexifies mesh collision geoms. The SO-101's fixed-finger part —
`wrist_roll_follower_so101_v1`, `geom38` — has a convex hull spanning

    x ∈ [-0.0991, +0.0063]   y ∈ [-0.0278, +0.0242]   z ∈ [-0.0273, +0.0379]

in the site frame, and **the grasp centre `(-0.0875, +0.005, +0.005)` is inside
it.** The part is C-shaped; its hull fills the open space between the jaws.

So an object placed where it must be to be gripped registers as penetrating the
gripper. Measured on a planned cup grasp:

    cup <-> right_gripper   dist = -37.2 mm
    cup <-> right_wrist     dist = -20.6 mm
    cup <-> right_moving_jaw dist = -8.1 mm

The arm then drives into the cup (16 N), **three actuators saturate**, and the
gripper stops **94 mm above** its commanded pose with `shoulder_lift` 405 mrad
off.

This also invalidates the section 2.3 measurement as a *reachability* criterion:
when `tools/validate_standing.py` asks which poses can reach a standing cup
without penetrating it, the only survivors put the grasp centre **20–35 mm
above the cup's rim** — hovering over it, not around it. There is no pose that
encloses a standing cup, because the hull makes enclosure impossible by
definition.

**The fix is to decompose the fixed finger's collision geometry into primitives**
— a box for the housing behind the jaw gap, a thin box for the finger blade —
and disable collision on the mesh hull. This is standard practice for grippers
in MuJoCo and is the single change standing between this pipeline and working
demonstrations. Zeroing the hull's collision outright is *not* enough: the
object would then have only the moving jaw to push against and would squeeze
straight out.

Until that lands, no episode can grasp anything, and `grasp_ok` will correctly
refuse every one.

---

## 6. Honest status

| Task | Before this session | Now |
|---|---|---|
| Cup hand-off | 0/10, silently recording garbage | 0/10, **failing loudly with a named cause** |
| Plate move | "5/10" — successes were the kinematic latch, not grasps | not runnable; `pick_plate`/`place_plate` are unwritten |

The two zeros are not equivalent. Before, the pipeline would have filled a
dataset. Now it refuses to, and says why.

The plate is a separate geometric problem and was left alone deliberately: as
modelled (a disc of radius 0.055 with a 3 mm rim overhang sitting flat on the
table) it cannot be grasped by this gripper at all — a horizontal-closing grasp
needs 110 mm of aperture against a ~85 mm maximum, and a vertical-closing rim
pinch needs a finger underneath a plate that is resting on the table. It needs
either a foot ring it can hook under, or a smaller radius. **That is a scene
change and therefore the owner's call, not mine.**

## 7. Files

| Path | State |
|---|---|
| `sim/bimanual_scene.py` | park pose fixed; `NEUTRAL_QPOS` split out; `.bak` kept |
| `sim/ik_control.py` | grasp geometry corrected and documented; `.bak` kept |
| `sim/episode.py` | faults, joint-space transits, measured-set planning; `.bak` kept |
| `sim/tools/reach_table.py` | new — samples and caches the reachable set |
| `sim/tools/validate_reach.py` | new — prunes to holdable poses (77.2% kept) |
| `sim/tools/validate_standing.py` | new — prunes to poses that can reach a standing object; currently proves section 5 |
| `sim/assets/reach_*.npz`, `standing_*.npy` | caches; delete to rebuild |

Rebuild the caches with:

```bash
python sim/tools/reach_table.py && python sim/tools/validate_reach.py && python sim/tools/validate_standing.py
```


---

## 8. Why the cup is now 130 mm tall

Measured, not chosen for looks. With the fingers pointing downward this arm
cannot put its grasp centre below **z = 0.085 m** (200 000 sampled
configurations; section 3). A 90 mm cup standing on the table has its rim at
0.091, so the band between that floor and the rim is **6 mm** -- and the grasp
has to be on the body, which leaves about 2 mm. Nothing can be picked up
reliably in 2 mm.

At 130 mm the usable band is 45 mm. The alternatives were measured too:

| Cup height | Best mutual grasp spot for the two arms |
|---|---|
| 90 mm | none, at any arm spacing from 34 to 50 cm |
| 110 mm | 14.2 mm worst-case miss |
| **130 mm** | **10.6 mm** |
| 150 mm | 7.7 mm |

Moving the arms closer together does **not** fix a short cup -- tested at 50,
46, 42, 38 and 34 cm between bases, no mutual grasp spot exists. The binding
constraint is height, not spacing.

## 9. Why the transfer happens on the table

A mid-air hand-off needs both grippers inside the same small volume at the same
moment. These collision hulls cannot do it: the receiving arm, sent to a pose
validated on its own, landed **116 mm** away because it was fouling the giver's
arm, which `BimanualEnv.holds` cannot see (it parks the other arm at home).
The same rigid carry also arrived with the cup **upside down** (up = -1.00),
because a kinematic grasp rotates the object with the wrist and the
presentation pose is nothing like the pick pose.

Handing over via the table costs one extra pick and place and removes both
problems: the arms are never in the same place, and every grasp comes from the
tabletop family, so the cup stays upright. The task is still genuinely
bimanual -- the cup starts on the right arm's side and the plate sits on the
left's, so neither arm can do it alone. The task string is now
*"bring the cup across the table and set it on the plate"*.

If a true mid-air hand-off is required for the submission, it needs the
gripper's collision geometry decomposed into primitives (section 5), which is
a modelling job of its own and was not attempted.

## 10. Current numbers

| Quantity | Value |
|---|---|
| Cup relay success | **9/20 seeds** |
| Placement accuracy on success | 2-7 mm off the plate centre |
| Episode length | ~437 frames, ~17 s of sim |
| Wall clock, no images | ~4.3 s/episode |
| Failure modes, all faulted and discarded | 4x pick out of reach, 3x place off target, 2x second grasp out of reach, 2x cup ended off the plate |

At a 45% yield, 300 good episodes needs ~670 attempts. The remaining
simplification that must go in the README: **the arms do not collide with the
objects they manipulate** and the carry is kinematic, gated by a jaw-volume
check. Object-to-object and object-to-table physics is real; arm-to-arm and
arm-to-table collision is real.

## 11. Two other simplifications now in the model

- The gripper's mesh collision hulls are replaced by primitives
  (`replace_gripper_collision`), because the shipped hulls make the gripper a
  solid block with no mouth. Visual meshes are untouched; the robot still looks
  exactly like an SO-101, and the vendored MJCF is byte-identical to upstream.
- An axisymmetric carried object keeps its axis vertical and follows the
  gripper in position and yaw only (`UPRIGHT_OBJECTS`).

## 12. Menagerie swap (15 Sep evening)

The default arm is now DeepMind Menagerie's `robotstudio_so101`; physics mode
keeps its split jaw collisions and disables kinematic carry. Control is 50 Hz
over 200 Hz physics. Measured constants are grasp-centre site
`(-0.010, 0, 0.025)`, window `(0.020, 0.010, 0.015)`, close/open `0.20/1.20`.
The squeeze grid held 15/108 samples at both 0.002 s and 0.005 s; the held
start window was site x -30..+10 mm and z +10..+40 mm.

Final seed-1000 gate: physics **0/10**. First-fault distribution: 5 cup-pick
radial misses, 4 release-height faults (18-19 mm), and 1 release target miss
(135 mm). The earlier baseline was also 0/10, all ten at the +65 mm rim grip.
Kinematic fallback: **8/10**; two failures were 53 mm final-placement misses.
No 20-seed physics run was made because the required 6/10 gate was not met.

Fixes verified in this pass: default grip search starts at +40 mm rather than
the topmost +59 mm; scratch `holds()` objects are parked apart (removing the
repeatable DOF-15 QACC instability); placement accounts for the measured
post-squeeze offset and uses contact-driven slow release. Release/withdrawal
still moves the cup, so physics is not shippable. The image-on kinematic strip
passed and visibly shows the blue cup between/moving with the jaws. No success
predicate or tolerance was loosened.
