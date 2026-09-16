# WP1 report — 15 Sep 2026 evening

Clock at resume: `2026-09-15 18:42:52 +07:00`.

## Final physics gate (verbatim result lines)

Command: `python milestone2_check.py --tasks cup --reps 10 --no-images --seed 1000`

```text
fps=50 control_dt=0.020s  images=off
[cup] rep0 frames= 642 (12.84s sim)  wall=  0.98s   656.9 fps  ik_maxerr=  0.2mm ik_fail=  0  FAIL     faulted: cup pick: jaws 393mm off the cup's axis
[cup] rep1 frames= 859 (17.18s sim)  wall=  1.33s   646.4 fps  ik_maxerr=  0.2mm ik_fail=  0  FAIL     faulted: cup pick: jaws 90mm off the cup's axis
[cup] rep2 frames= 921 (18.42s sim)  wall=  1.69s   546.5 fps  ik_maxerr=  1.4mm ik_fail=  0  FAIL     faulted: cup at release: object 135mm off target
[cup] rep3 frames= 864 (17.28s sim)  wall=  1.55s   558.3 fps  ik_maxerr=  1.4mm ik_fail=  0  FAIL     faulted: cup pick: jaws 258mm off the cup's axis
[cup] rep4 frames= 913 (18.26s sim)  wall=  1.50s   609.8 fps  ik_maxerr=  0.2mm ik_fail=  0  FAIL     faulted: cup 18mm off its resting height at release
[cup] rep5 frames= 668 (13.36s sim)  wall=  1.11s   603.8 fps  ik_maxerr=  0.3mm ik_fail=  0  FAIL     faulted: cup pick: jaws 566mm off the cup's axis
[cup] rep6 frames= 868 (17.36s sim)  wall=  1.28s   679.2 fps  ik_maxerr=  0.2mm ik_fail=  0  FAIL     faulted: cup 19mm off its resting height at release
[cup] rep7 frames= 867 (17.34s sim)  wall=  1.38s   627.4 fps  ik_maxerr=  0.2mm ik_fail=  0  FAIL     faulted: cup pick: jaws 21mm off the cup's axis
[cup] rep8 frames= 942 (18.84s sim)  wall=  1.63s   576.2 fps  ik_maxerr=  0.1mm ik_fail=  0  FAIL     faulted: cup 19mm off its resting height at release
[cup] rep9 frames= 857 (17.14s sim)  wall=  1.39s   617.6 fps  ik_maxerr=  0.2mm ik_fail=  0  FAIL     faulted: cup 19mm off its resting height at release
MILESTONE 2: FAIL
```

Result: **0/10**. Faults: radial cup-pick miss 5, release height 4, release-target miss 1. Mean wall time is **1.38 s/episode** (range 0.98–1.69 s, no images).

The requested seed-2000 20-repetition run was conditional on passing 6/10, so it was not run.

## Kinematic fallback (verbatim result lines)

Command (environment set for this process only): `$env:BIMANUAL_CARRY='kinematic'; python milestone2_check.py --tasks cup --reps 10 --no-images --seed 1000`

```text
fps=50 control_dt=0.020s  images=off
[cup] rep0 frames= 896 (17.92s sim)  wall=  1.08s   826.3 fps  ik_maxerr=  0.2mm ik_fail=  0  SUCCESS  cup on plate, 12mm off centre
[cup] rep1 frames= 896 (17.92s sim)  wall=  1.03s   873.5 fps  ik_maxerr=  0.2mm ik_fail=  0  SUCCESS  cup on plate, 34mm off centre
[cup] rep2 frames= 896 (17.92s sim)  wall=  1.04s   859.9 fps  ik_maxerr=  0.1mm ik_fail=  0  SUCCESS  cup on plate, 3mm off centre
[cup] rep3 frames= 884 (17.68s sim)  wall=  1.05s   841.8 fps  ik_maxerr=  0.1mm ik_fail=  0  SUCCESS  cup on plate, 3mm off centre
[cup] rep4 frames= 872 (17.44s sim)  wall=  1.09s   801.4 fps  ik_maxerr=  0.2mm ik_fail=  0  FAIL     faulted: cup onto the plate: object 53mm off target
[cup] rep5 frames= 872 (17.44s sim)  wall=  1.06s   819.9 fps  ik_maxerr=  0.2mm ik_fail=  0  FAIL     faulted: cup onto the plate: object 53mm off target
[cup] rep6 frames= 896 (17.92s sim)  wall=  1.21s   743.4 fps  ik_maxerr=  0.2mm ik_fail=  0  SUCCESS  cup on plate, 4mm off centre
[cup] rep7 frames= 884 (17.68s sim)  wall=  1.10s   803.8 fps  ik_maxerr=  0.1mm ik_fail=  0  SUCCESS  cup on plate, 0mm off centre
[cup] rep8 frames= 884 (17.68s sim)  wall=  1.22s   727.1 fps  ik_maxerr=  0.1mm ik_fail=  0  SUCCESS  cup on plate, 1mm off centre
[cup] rep9 frames= 896 (17.92s sim)  wall=  1.08s   831.4 fps  ik_maxerr=  0.2mm ik_fail=  0  SUCCESS  cup on plate, 1mm off centre
MILESTONE 2: FAIL
```

Result: **8/10**; two 53 mm placement faults. Mean wall time **1.10 s/episode**. The checker prints aggregate FAIL because it demands all repetitions, although this exceeds the brief's 6/10 shipping threshold.

Image-on fallback command produced `out/milestone2/{cup_top,cup_right_wrist,cup_left_wrist}_strip.png` and printed `SUCCESS ... cup on plate, 12mm off centre`, `wall=7.28s`, `MILESTONE 2: PASS`. The top strip visibly shows the blue cup moving with/between the right jaws.

## Cameras

- Top: overhead view of both arms, blue cup, white plate, and spoon.
- Right wrist: now points from 74 mm behind the grasp site toward the fingertips; jaws are visible near frame center.
- Left wrist: same corrected tool-axis view; jaws are visible near frame center.

The previous wrist renders looked sideways through the square gripper housing. `WRIST_CAM_POS/X/Y/Z` was changed and `python tools/render_cameras.py` regenerated all three PNGs.

## Changed files

- `sim/episode.py` — separated scratch probe objects (removed DOF-15 instability), capped automatic grip height at measured +40 mm, added physical held-offset placement and slower contact-driven release diagnostics.
- `sim/bimanual_scene.py` — corrected Menagerie wrist-camera mount/tool axis.
- `sim/tools/render_cameras.py` — reproducible three-camera render check.
- `FINDINGS-SIM.md` — appended section 12 with measurements and gate counts.
- `HANDOFF-LABLAB.md` — updated simulation status rows.
- `briefs/WP1-REPORT.md` — this report.

## Recommendation

Ship with **`BIMANUAL_CARRY=kinematic`** for this package: measured 8/10 versus physics 0/10. Physics does achieve genuine two-jaw holds, but release/withdrawal imparts lateral motion and produces five downstream pickup misses plus five placement/release faults. No success predicate or tolerance was loosened.
