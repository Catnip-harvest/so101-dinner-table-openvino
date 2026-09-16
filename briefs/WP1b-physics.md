# WP1b — make the PHYSICS grasp relay pass, in a private copy of the simulator

Project: `C:\Users\vieth\Documents\lablab hackathon` (Windows 11, Git Bash/PowerShell, system
Python 3.13 with mujoco 3.11 + mink). **Work ONLY inside `sim_phys/`** — a verbatim copy of
`sim/` made at 19:00. `sim/` itself is in use by a data-collection run and must not be
touched. Hard stop **21:00 Hanoi** (`date`), then write the report whatever the state.

## Rules
Never open any `.env*` file. Nothing public. Do not loosen `success_cup_handoff`, `grasp_ok`
tolerances, `REACH_TOL` or `GRASP_MISS_TOL` to make numbers pass — fix causes. Paste real
command output in the report. Be economical (API key, small cap): no re-reading of files
you already know; short outputs (`tail`, `grep`).

## Context — read first, in this order
`briefs/WP1-REPORT.md` (the previous worker's diagnosis: physics 0/10 with faults radial
cup-pick miss ×5, release height ×4, release-target miss ×1; kinematic 8/10),
`briefs/WP1-gate.md` (what changed in the swap and the measured grasp constants),
`FINDINGS-SIM.md` §12 (written by that worker). Then `sim_phys/episode.py` primitives
`pick_cup`, `place_cup`, `stand_off`, `approach_straight`, `settle_object`, `grasp_ok`,
`holding`, and `sim_phys/tools/measure_grasp.py` (the squeeze test: run it, 6 s).

## Gate command (physics mode is forced by the env var)
```bash
cd sim_phys && BIMANUAL_CARRY=physics python milestone2_check.py --tasks cup --reps 10 --no-images --seed 1000
```
Pass = **≥ 6/10 SUCCESS**. Every failure prints its first fault; that string is the diagnosis.

## Where to look (from the previous worker's evidence, verify before acting)
* **Release**: with a real grasp the cup drops when the jaws open; `place_cup` checks the cup
  height before release (`err_z > 0.018`) and the plate predicate wants the cup resting on
  the plate. Measure the actual settle height of a held cup vs `rest_z`; likely the grasp
  holds the cup a fixed offset from the fingertip that `place_cup`'s target does not account
  for. Also open the jaws slower / lift straight up after release so withdrawal does not
  drag the cup sideways (the "53 mm off target" and "release height" faults).
* **Radial pick miss**: the pre-close check compares the grasp centre to the cup axis. With
  the fingertip-centred grasp the approach may push the cup before the jaws close (arm now
  collides with objects). Try a shorter final approach (`stand_off(back=...)` smaller, or a
  straight-line Cartesian last 30 mm), and closing earlier. Check `holding()` after close.
* Keep DART/recording untouched; only the primitives and constants.

## Deliverables
1. Gate output verbatim (10 reps), then if passing, 20 reps `--seed 2000`.
2. `briefs/WP1b-REPORT.md`: numbers, per-fault counts before/after, exactly which lines in
   `sim_phys/*.py` changed and why, and a unified diff `sim` → `sim_phys` for the changed
   files (`diff -u sim/episode.py sim_phys/episode.py > briefs/WP1b.patch`, etc.).
   If not passing by 21:00: the best number reached and the remaining fault class.
