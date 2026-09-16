# RESUME NOTICE 2 (17:30)

The previous run of this brief stopped at ~17:25 with "You have hit your usage limit" on the ChatGPT account. You now run on an OpenAI API key with a small dollar cap: be economical -- do not re-read large files you already know from what is on disk, do not repeat long commands. State on disk: the first gate run finished 0/10 -- every rep faulted with "cup pick: jaws +65mm from the cup's centre, past its 65mm body" (the planner picks the top-most grip height, 59 mm above the cup centre, and the fingertip-centred grasp lands ~65 mm, just over the CUP_HALF_H-0.004 limit in grasp_ok). A debug snapshot script was being written when the run stopped (find it with: ls -t sim/*.py sim/tools/*.py | head -3). Continue from there.

---

# RESUME NOTICE

A previous run of this brief was cut short: the Codex sandbox could not start python.exe ("Access is denied"), so nothing was ever executed. This run has NO sandbox. Files the earlier run wrote are in place (see briefs/WP1-codex-last.md if present) -- read them, keep what is right, and now actually RUN every step in the brief and paste the real output. Do not start over. Both reach tables are built (out/reach_right.log, out/reach_left.log say kept ... written to), so start the gate immediately.

---

# WP1 — Menagerie SO-101 swap: run the gate, fix what fails, report numbers

You are working in `C:\Users\vieth\Documents\lablab hackathon` (Windows 11, Git Bash and
PowerShell, system Python 3.13 with mujoco 3.11.0 + mink + numpy + Pillow). Hard stop for
this package: **21:15 Hanoi time tonight** (check the clock with `date`). Report with the
actual command output pasted, never with claims. Work in `sim/` only.

## Hard rules

* **Never open, print, grep, diff or copy any `.env*` file.** `.env.local` exists in the
  project root and is out of bounds.
* **Do not touch** `sim/collect.py`, `sim/to_lerobot.py`, `intel_eval.py`,
  `run_intel_demo.ps1`, `k1_kaggle_autopilot.py` — two other workers own those right now.
  You MAY edit `sim/episode.py`, `sim/ik_control.py`, `sim/bimanual_scene.py`, `sim/tools/*`.
  Do not change public function names or signatures in those files (the other workers
  import `BimanualEnv`, `task_cup_relay`, `pick_cup`, `place_cup`, `go_home`, `object_pos`,
  `success_cup_handoff`, `TASK_STRINGS`, `CAMERAS`, `ArmTarget`, `smoothstep`,
  `site_pose_for_grasp`, `GRASP_CENTRE_SITE`).
* Nothing is pushed anywhere. No new infrastructure.
* Do not "fix" a bad number by loosening a success predicate or a tolerance. If a
  tolerance must change, say why with a measurement.

## What was done in the last hour (already in the working tree)

The arm model was swapped from the raw TheRobotStudio export to DeepMind Menagerie's
`robotstudio_so101` (vendored at `sim/assets/menagerie_so101/`, commit in `UPSTREAM.txt`,
files unmodified). Reason: the old file's fixed jaw collides as one convex hull that fills
its own opening, so no grasp was physically possible and objects were carried by rewriting
their pose ("kinematic carry"). The Menagerie file ships split jaw hulls plus fingertip
primitives and a tuned gripper contact class. Read `FINDINGS-SIM.md` sections 1, 2, 5, 8–11
for the history; it is long, skim it.

Changes made, all measured:

* `sim/bimanual_scene.py`: `ARM_MODEL` (`BIMANUAL_ARM`, default `menagerie`), `ARM_XML`
  switch, `CARRY_MODE` (`BIMANUAL_CARRY`, default `physics` for menagerie). In physics mode
  the old `replace_gripper_collision` hack and the arm–object contact exclusions are
  skipped. `<option timestep>` is now 0.005 (200 Hz, what the brief prescribes and what
  Menagerie tuned for; integrator implicitfast, elliptic cone, impratio 10).
* `sim/episode.py`: `PHYS_DT=0.005, SUBSTEPS=4` → 50 Hz control. `_apply_carry()` is a
  no-op in physics mode. New `holding(arm, obj)` (both jaw bodies in `data.contact` with
  the object) and `check_held()` which faults in physics mode when the jaws close on
  nothing; called after the close in `pick_cup` and `handoff_cup`. `approach_dir` uses
  `FINGER_SIGN`.
* `sim/ik_control.py`: `FINGER_SIGN = +1` for menagerie (fingers run along site **+x**,
  the `gripperframe` site sits AT the fixed fingertip; in the old file they ran along −x).
  `grasp_frame` uses it. Measured constants for menagerie:
  `GRASP_CENTRE_SITE = (-0.010, 0, 0.025)`, `GRASP_WINDOW = (0.020, 0.010, 0.015)`,
  `CLOSE_Q = 0.20`, `OPEN_Q = 1.20`.
* `sim/tools/measure_grasp.py` (new): the squeeze test that produced those constants. Run it
  (`python tools/measure_grasp.py`, 6 s) to see the held map: a 48 mm cup whose centre
  starts at site x ∈ [−0.03, +0.01], z ∈ [+0.01, +0.04] is held against gravity by both
  jaws, settles centred on the tips, servo at 0.38–0.40 rad. 15/108 grid points held at
  both the 0.002 and 0.005 timesteps.
* `sim/tools/reach_table.py` / `validate_reach.py`: finger sign and fingertip clearance
  made model-aware. The tables are being rebuilt right now in the background: wait until
  BOTH `out/reach_right.log` and `out/reach_left.log` contain a line matching
  `kept .* written to` (and `exit 0`) before running anything that constructs
  `BimanualEnv`. Expect ~5 more minutes. Do not start a second build.
* Renders at `out/menagerie_cams/{top,right_wrist,left_wrist}.png` were produced with the
  right arm at `NEUTRAL_QPOS` and a cup at (−0.12, 0.05). Nobody has looked at them yet.

## The gate (this is the deliverable)

```
cd sim
python milestone2_check.py --tasks cup --reps 10 --no-images --seed 1000
```

**Pass = at least 6 of 10 `SUCCESS` with `BIMANUAL_CARRY` left at its default (physics),**
i.e. the cup is really squeezed, carried and released twice (right arm → relay spot on
the table → left arm → plate). The predicate is `success_cup_handoff`. Every failure prints
the first fault string; those strings are the diagnosis. Then:

1. Look at the three PNGs in `out/menagerie_cams/` (open them; describe in one line each
   what they show). The wrist cameras must look down the fingers toward the fingertips
   with the jaws visible at the bottom of the frame. If they look elsewhere, fix
   `WRIST_CAM_POS/X/Y/Z` in `bimanual_scene.py` against the Menagerie gripper body frame
   (the `gripperframe` site is at body `(0.012, -0.000218, -0.098127)`, quat `1 0 1 0`) and
   re-render.
2. If the gate fails, fix the actual cause in the sim files. Likely suspects, in order:
   * `approach_straight` interpolates in joint space for the last 90 mm; with real contact
     a joint-space bulge can knock the cup. Check the fault strings for "off the cup's
     axis" / "not between them" and, if that is it, shorten `stand_off(back=...)` or add
     one intermediate waypoint along the approach axis.
   * `place_cup` release: the cup is now physically held and drops 2–18 mm at release; the
     `err_z > 0.018` check and the plate-height predicate may need the measured drop.
   * `randomize_objects` / `_find_relay` still assume the old grasp geometry (grip
     heights `CUP_GRIP_BAND`, `take_dz`/`give_dz`). With the new gripper the cup is held
     at the fingertips, so the required height difference between the two arms' grips may
     be different or unnecessary.
   * `holds()` and the reach tables use `OPEN_Q`; fine. The `handoff` plan is unused by
     the relay task; ignore faults from `_find_handoff` unless they abort the layout.
3. After it passes, run 20 more with `--seed 2000` for the number, and run once with
   `--strip` (images on, default size) so `out/milestone2/*_strip.png` exists; look at the
   strips and confirm the cup is visibly between the jaws during the carry.
4. Also measure the fallback so both numbers exist: `BIMANUAL_CARRY=kinematic` for
   `--reps 10 --seed 1000` (set the env var in the shell for that one command only).
5. Record how long one episode takes (the `wall=` column) — the collector is sized from it.

## Docs, same pass as the code

* Append a section **12. Menagerie swap (15 Sep evening)** to `FINDINGS-SIM.md`: what
  changed, the measured squeeze window, the gate numbers (physics and kinematic), the
  fault distribution, what you fixed and how you verified it. Numbers, not adjectives.
* In `HANDOFF-LABLAB.md`, update the status table rows for the sim files.

## Report

Write `briefs/WP1-REPORT.md`: the gate output verbatim (both runs), the 20-seed number,
the kinematic number, per-fault counts, the camera check, every file you changed with a
one-line reason, wall time per episode, and — if the gate did not pass by 21:15 — your
recommendation between (a) ship physics with the measured success rate, (b) fall back to
`BIMANUAL_CARRY=kinematic`, with the evidence for each. Stop at 21:15 regardless.
