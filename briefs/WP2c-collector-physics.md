# WP2c — the collector gets 0 % under physics while the gate runner gets 70 %. Find why, fix it.

Project `C:\Users\vieth\Documents\lablab hackathon` (Windows 11; Git Bash/PowerShell; system
Python 3.13 with mujoco + mink). Hard stop **20:30 Hanoi** (`date`). Rules: never open any
`.env*`; nothing public; no loosening of predicates/tolerances; paste real output; be
economical (API key, small cap) — read only what you need, short outputs.

## The discrepancy (measured 19:20–19:24)
* `cd sim && BIMANUAL_CARRY=physics python milestone2_check.py --tasks cup --reps 10 --no-images --seed 1000`
  → **7/10 SUCCESS** (WP1b's patch, now in `sim/episode.py`; see `briefs/WP1b-REPORT.md`
  for what it changed: upright-cup solve at release, slower jaw opening, vertical
  withdrawal).
* `BIMANUAL_CARRY=physics python sim/collect.py --out out/episodes_v2_chunk1 --target-kept 100 --instruction both --workers 3 --seed 100000`
  → **0 kept of 93 attempts**. `out/episodes_v2_chunk1/manifest.json` `per_fault_counts`:
  `cup place: jaws closed but the cup is not between them` 140,
  `cup pick: jaws closed but the cup is not between them` 101,
  `cup pick: jaws NNmm off the cup's axis` (37–94 mm) several, a few
  `cup is NNmm outside the left grasp set`. Both instructions, perturbed and unperturbed
  episodes alike.

## Where to look first
`sim/collect.py` defines `RecordingEnv(BimanualEnv)` and **overrides `joint_step` and
`control_step`** with its own `_record_and_step` (writes `data.ctrl` from `self.q_cmd`,
steps `mj_step` `substeps` times, calls `_apply_carry`). It was written against the
pre-patch simulator. Diff its stepping against the current `BimanualEnv.joint_step` /
`control_step` / `grip_phase` in `sim/episode.py` — anything the patch added there (a
gripper command path, a per-step hook, the release solve updating `q_cmd`, a hold/settle
that reads state between substeps) is bypassed by the override. Also check the collector's
environment default (`WP2-VOLUME.md` says it had its own kinematic default independent of
the simulator's) and that the env var actually reaches the spawned workers (Windows spawn:
`os.environ` is inherited, but verify `CARRY_MODE` inside a worker by printing it once).

Preferred fix: make `RecordingEnv` **delegate** stepping to the parent class and only wrap
it for recording and DART (e.g. override the single lowest-level place where `data.ctrl`
is written, add noise there, record before the substeps), instead of re-implementing the
loop. Do not edit `sim/episode.py` unless the bug is provably there; if you must, keep the
7/10 gate result and re-run it to prove it.

## Acceptance (paste output)
1. `BIMANUAL_CARRY=physics python sim/collect.py --out out/probe_phys --target-kept 10 --workers 2 --instruction both --seed 7000`
   reaches 10 kept with kept/attempts ≥ 50 %, manifest fault histogram printed.
2. One kept episode's npz: `state.shape`, `action.shape`, frame counts of the three mp4s,
   and `is_perturbed` values across the kept set (some True, some False).
3. `C:\pai\ev\Scripts\python.exe sim/to_lerobot.py --raw out/probe_phys --repo VietHwang/dinner-table-probe-phys --root out/lerobot/probe_phys --append --chunk-id probe --push`
   succeeds, private, prints URL + counts.
4. Write `briefs/WP2c-REPORT.md`: root cause (file:line), the fix, the numbers.
