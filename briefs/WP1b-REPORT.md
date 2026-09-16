# WP1b — physics grasp relay in sim_phys

Completed 2026-09-15T19:09:24+07:00, before the 21:00 Hanoi hard stop.

**PASS against the requested threshold:** physics seed 1000 **7/10**, seed 2000 **14/20**. Reproduced baseline: **0/10**. The checker prints `MILESTONE 2: FAIL` and returns 1 whenever any repetition fails; the WP1b acceptance rule is at least 6/10.

Only `sim_phys/episode.py` was edited among simulator modules. `sim/` received no writes. DART, recording, task scripts, success predicates and all existing module constants are unchanged. No `.env*` file was opened or copied, and nothing was published.

`sim_phys/` was absent when work began at 18:57:13, earlier than the stated 19:00 copy. It was created with `robocopy sim sim_phys /E /XF .env* /XD .env* /NFL /NDL /NJH /NJS /NP` (exit 1 means files copied). Diagnostic scripts, snapshots, logs and images are under `sim_phys/out/wp1b/`; only the explicitly requested report and patch were written under `briefs/`. The live `sim/collect.py` and `sim/to_lerobot.py` subsequently changed at 18:58:40 and 19:05:45; their private copies retain the original 17:21:25 and 17:27:38 timestamps. They were not edited or synchronized.

Runtime: Python 3.13.14, MuJoCo 3.11.0, SciPy 1.18.0. `BIMANUAL_CARRY=physics` was explicitly set for every physics command. The new solve uses the already installed SciPy; nothing was installed.

## Counts and timing

| Run | Success | Mean wall seconds/episode | Range |
|---|---:|---:|---:|
| Baseline seed 1000 | 0/10 | 1.386 | 1.04–1.65 |
| Final seed 1000 | 7/10 | 1.866 | 1.16–2.36 |
| Final seed 2000 | 14/20 | 1.872 | 0.52–2.50 |

Counts use the **first** fault printed by each repetition, not downstream faults.

| First failure class | Before, 10 | After, 10 | After, 20 |
|---|---:|---:|---:|
| Radial pick miss | 5 | 0 | 2 |
| Release height | 4 | 0 | 0 |
| Release target miss | 1 | 2 | 2 |
| Empty pickup | 0 | 1 | 1 |
| Final cup/plate distance | 0 | 0 | 1 |

Remaining seed-1000 failures: rep0 empty pickup; rep1 and rep9 release targets missed by 16 and 21 mm. Seed-2000 failures: radial pickup misses of 41 and 24 mm, one empty pickup, release misses of 41 and 26 mm, and one final cup/plate distance of 85 mm. These are retained failures, not relaxed checks.

## Measurements and cause

The baseline was reproduced with the exact 5 radial / 4 height / 1 release-target first-fault distribution. In seed-1000 rep0, the right pickup had both-jaw contact. After the old release loop opened to 0.4817 rad, the following `hold()` restored the stale 0.2000-rad Cartesian target: actual jaw angle became 0.4117 rad and both-jaw contact returned. The cup later reached the left pickup tipped over, and was knocked away. This is direct evidence that at least that radial pickup fault originated in the preceding release.

The old target was `rest_z + 10 mm` with a separate 10 mm object-settling tolerance. After fixing only target synchronization, measured right-relay heights included 84.60 mm against `rest_z = 66 mm`, an 18.60 mm excess. This accounts for the unchanged 18 mm release-height check firing. The new commanded clearance is 3 mm; the 10 mm object check and 18 mm height check retain their original limits.

Position-only placement also changes the cup orientation with the wrist: the baseline rep0 cup axis dot world-up was 0.845 before release, and the sync-only rep3 left placement reached -0.5985 (inverted). The new five-joint solve targets the measured held cup position and upright axis, leaving yaw free. It tries the current and table-planned joint configurations to avoid a wrong joint-solution branch. It changes arm commands only; `_apply_carry()` remains a no-op in physics mode.

Actual experiment sequence: synchronization alone 0/10; full slow opening plus lower target and vertical withdrawal 0/10; partial opening plus vertical withdrawal 0/10; adding the measured upright cup solve 7/10. After removing superseded code and preventing an empty placement from proceeding, the final gate remained 7/10.

An initial seed-2000 run stopped during rep17 with `numpy._core._exceptions._ArrayMemoryError: Unable to allocate 1.20 MiB`, while a separate visual check was running. That incomplete run is not counted as validation. The final 20-repetition command ran alone and completed. Its complete output follows. The allocation failure cause was not established.

## Exact source changes

| Final sim_phys/episode.py lines | Change and reason |
|---|---|
| 29 | Import the installed `scipy.optimize.least_squares` for the five-joint held-cup pose solve. |
| 1222–1240 | Add `_upright_cup_pose`: measure cup offset and axis in the grasp frame, solve position and upright axis with existing joint bounds, leave cup yaw unconstrained, and choose between current/table seeds. |
| 1243–1276 | Add `_place_cup_physics`: require a real hold, lift 80 mm, transfer above the target, lower with the cup upright, use a 3 mm commanded clearance, retain actual object/height checks, open over 1.2 s through `grip_phase` (which synchronizes the target), then withdraw upward in eight 10 mm waypoints. |
| 1279–1312 | Dispatch physics placement to the new primitive; retain the previous kinematic sequence and its tolerances. Remove the superseded position-only physics branch and manually opened/stale gripper target. |

The complete unified diff, including every added/deleted line, is [WP1b.patch](WP1b.patch), with `sim/episode.py` as the old side and `sim_phys/episode.py` as the new side. It was generated with Python `difflib.unified_diff`, equivalent to `diff -u sim/episode.py sim_phys/episode.py`. No other simulator source file was edited by WP1b. Hunk headers:

```diff
@@ -26,6 +26,7 @@
@@ -1218,21 +1219,73 @@
@@ -1240,60 +1293,23 @@
```

Final episode.py SHA-256: `8d4709f8a4904a23e1699131f9c5b69ff87f7517b14caba1bfae7d13189d4f64`.

AST comparison output:

```text
Changed existing functions: ['place_cup']
Added functions: _place_cup_physics, _upright_cup_pose, residual
All other existing function ASTs unchanged, including predicates, pickup, DART, recording, carry, and task script.
All existing module constants unchanged.
```

The axis residual weight of 0.1 in the new optimizer is a planning weight, not a success tolerance. The reported `ik_maxerr` still measures the existing Mink calls; placement acceptance continues to use the actual simulated cup position.

## Squeeze test — complete captured output

Command in `sim_phys/`: `$env:BIMANUAL_CARRY='physics'; python -u tools/measure_grasp.py`.

Held samples: 15/108 at close command 0.00; 14/108 at 0.30. The current run is reported as measured; the earlier report’s 15/108 for both should not be substituted.

```text
python : WARNING: Attach conflict when attaching 'so101' to 'bimanual_dinner_table', policy is 'warning'
At line:2 char:97
+ ... ='physics'; python -u tools/measure_grasp.py *> out/wp1b/squeeze.log; ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : NotSpecified: (WARNING: Attach...cy is 'warning':String) [], RemoteException
    + FullyQualifiedErrorId : NativeCommandError
 
iterations: parent has 100 (default), child has 10, keeping parent value
ls_iterations: parent has 50 (default), child has 20, keeping parent value

WARNING: Attach conflict when attaching 'so101' to 'bimanual_dinner_table', policy is 'warning'
iterations: parent has 100 (default), child has 10, keeping parent value
ls_iterations: parent has 50 (default), child has 20, keeping parent value

C:\Users\vieth\Documents\lablab hackathon\sim_phys\bimanual_scene.py:457: UserWarning: Attach conflict when attaching 
'so101' to 'bimanual_dinner_table', policy is 'warning'
iterations: parent has 100 (default), child has 10, keeping parent value
ls_iterations: parent has 50 (default), child has 20, keeping parent value
  model = spec.compile()
arm model: menagerie   grid step 10 mm   x -0.100..+0.010  z -0.020..+0.060

close command q = +0.00   ('H' held, 'd' pinched then lost, 's'/'m' one jaw only, '.' nothing, 'x' started inside)
  z=+0.060  . . x x x x x x x . . .
  z=+0.050  . x x x x x x x . . . .
  z=+0.040  x x x x x x x x H H . .
  z=+0.030  x x x x x x x H H H H .
  z=+0.020  x x x x x x x H H H H .
  z=+0.010  x x x x x x x H H H H H
  z=+0.000  x x x x x x x x x x x x
  z=-0.010  x x x x x x x x x x x x
  z=-0.020  x x x x x x x x x x x x
           x: -0.10  -0.06  -0.02
  held 15 of 108: centre (settled, site frame) [ 0.0016 -0.0007  0.0021]
  extent x [-0.0164,+0.0206]  y [-0.0103,+0.0000]  z [-0.0027,+0.0037]
  jaw angle reached while holding: 0.386 +/- 0.051 rad

close command q = +0.30   ('H' held, 'd' pinched then lost, 's'/'m' one jaw only, '.' nothing, 'x' started inside)
  z=+0.060  . . x x x x x x x . . .
  z=+0.050  . x x x x x x x . . . .
  z=+0.040  x x x x x x x x H H . .
  z=+0.030  x x x x x x x H H H . .
  z=+0.020  x x x x x x x H H H H .
  z=+0.010  x x x x x x x H H H H H
  z=+0.000  x x x x x x x x x x x x
  z=-0.010  x x x x x x x x x x x x
  z=-0.020  x x x x x x x x x x x x
           x: -0.10  -0.06  -0.02
  held 14 of 108: centre (settled, site frame) [ 0.0002 -0.0008  0.0024]
  extent x [-0.0164,+0.0160]  y [-0.0103,+0.0000]  z [-0.0000,+0.0037]
  jaw angle reached while holding: 0.397 +/- 0.033 rad

HOME site position, left: [0.105 0.001 0.253]

HOME site position, right: [-0.105 -0.001  0.253]
```

## Reproduced baseline — complete captured output

Command in `sim_phys/`: `$env:BIMANUAL_CARRY='physics'; python -u milestone2_check.py --tasks cup --reps 10 --no-images --seed 1000`.

```text
python : WARNING: Attach conflict when attaching 'so101' to 'bimanual_dinner_table', policy is 'warning'
At line:2 char:32
+ ... ='physics'; python -u milestone2_check.py --tasks cup --reps 10 --no- ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : NotSpecified: (WARNING: Attach...cy is 'warning':String) [], RemoteException
    + FullyQualifiedErrorId : NativeCommandError
 
iterations: parent has 100 (default), child has 10, keeping parent value
ls_iterations: parent has 50 (default), child has 20, keeping parent value

WARNING: Attach conflict when attaching 'so101' to 'bimanual_dinner_table', policy is 'warning'
iterations: parent has 100 (default), child has 10, keeping parent value
ls_iterations: parent has 50 (default), child has 20, keeping parent value

C:\Users\vieth\Documents\lablab hackathon\sim_phys\bimanual_scene.py:457: UserWarning: Attach conflict when attaching 
'so101' to 'bimanual_dinner_table', policy is 'warning'
iterations: parent has 100 (default), child has 10, keeping parent value
ls_iterations: parent has 50 (default), child has 20, keeping parent value
  model = spec.compile()
fps=50 control_dt=0.020s  images=off
[cup] rep0 frames= 642 (12.84s sim)  wall=  1.04s   617.1 fps  ik_maxerr=  0.2mm ik_fail=  0  FAIL     faulted: cup pick: jaws 393mm off the cup's axis
        task string: "bring the cup across the table and set it on the plate"
[cup] rep1 frames= 859 (17.18s sim)  wall=  1.39s   616.9 fps  ik_maxerr=  0.2mm ik_fail=  0  FAIL     faulted: cup pick: jaws 90mm off the cup's axis
        task string: "bring the cup across the table and set it on the plate"
[cup] rep2 frames= 921 (18.42s sim)  wall=  1.63s   565.3 fps  ik_maxerr=  1.4mm ik_fail=  0  FAIL     faulted: cup at release: object 135mm off target
        task string: "bring the cup across the table and set it on the plate"
[cup] rep3 frames= 864 (17.28s sim)  wall=  1.46s   590.3 fps  ik_maxerr=  1.4mm ik_fail=  0  FAIL     faulted: cup pick: jaws 258mm off the cup's axis
        task string: "bring the cup across the table and set it on the plate"
[cup] rep4 frames= 913 (18.26s sim)  wall=  1.60s   569.1 fps  ik_maxerr=  0.2mm ik_fail=  0  FAIL     faulted: cup 18mm off its resting height at release
        task string: "bring the cup across the table and set it on the plate"
[cup] rep5 frames= 668 (13.36s sim)  wall=  1.20s   558.0 fps  ik_maxerr=  0.3mm ik_fail=  0  FAIL     faulted: cup pick: jaws 566mm off the cup's axis
        task string: "bring the cup across the table and set it on the plate"
[cup] rep6 frames= 868 (17.36s sim)  wall=  1.08s   800.3 fps  ik_maxerr=  0.2mm ik_fail=  0  FAIL     faulted: cup 19mm off its resting height at release
        task string: "bring the cup across the table and set it on the plate"
[cup] rep7 frames= 867 (17.34s sim)  wall=  1.45s   599.5 fps  ik_maxerr=  0.2mm ik_fail=  0  FAIL     faulted: cup pick: jaws 21mm off the cup's axis
        task string: "bring the cup across the table and set it on the plate"
[cup] rep8 frames= 942 (18.84s sim)  wall=  1.65s   569.7 fps  ik_maxerr=  0.1mm ik_fail=  0  FAIL     faulted: cup 19mm off its resting height at release
        task string: "bring the cup across the table and set it on the plate"
[cup] rep9 frames= 857 (17.14s sim)  wall=  1.36s   628.4 fps  ik_maxerr=  0.2mm ik_fail=  0  FAIL     faulted: cup 19mm off its resting height at release
        task string: "bring the cup across the table and set it on the plate"
MILESTONE 2: FAIL
```

## Final 10-repetition gate — complete captured output

Command in `sim_phys/`: `$env:BIMANUAL_CARRY='physics'; python -u milestone2_check.py --tasks cup --reps 10 --no-images --seed 1000`.

```text
python : WARNING: Attach conflict when attaching 'so101' to 'bimanual_dinner_table', policy is 'warning'
At line:2 char:32
+ ... ='physics'; python -u milestone2_check.py --tasks cup --reps 10 --no- ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : NotSpecified: (WARNING: Attach...cy is 'warning':String) [], RemoteException
    + FullyQualifiedErrorId : NativeCommandError
 
iterations: parent has 100 (default), child has 10, keeping parent value
ls_iterations: parent has 50 (default), child has 20, keeping parent value

WARNING: Attach conflict when attaching 'so101' to 'bimanual_dinner_table', policy is 'warning'
iterations: parent has 100 (default), child has 10, keeping parent value
ls_iterations: parent has 50 (default), child has 20, keeping parent value

C:\Users\vieth\Documents\lablab hackathon\sim_phys\bimanual_scene.py:457: UserWarning: Attach conflict when attaching 
'so101' to 'bimanual_dinner_table', policy is 'warning'
iterations: parent has 100 (default), child has 10, keeping parent value
ls_iterations: parent has 50 (default), child has 20, keeping parent value
  model = spec.compile()
fps=50 control_dt=0.020s  images=off
[cup] rep0 frames= 758 (15.16s sim)  wall=  1.16s   655.5 fps  ik_maxerr=  0.2mm ik_fail=  0  FAIL     faulted: cup pick: jaws closed but the cup is not between them
        task string: "bring the cup across the table and set it on the plate"
[cup] rep1 frames=1137 (22.74s sim)  wall=  1.88s   604.4 fps  ik_maxerr=  0.2mm ik_fail=  0  FAIL     faulted: cup at release: object 16mm off target
        task string: "bring the cup across the table and set it on the plate"
[cup] rep2 frames=1058 (21.16s sim)  wall=  1.90s   557.6 fps  ik_maxerr=  0.2mm ik_fail=  0  SUCCESS  cup on plate, 11mm off centre
        task string: "bring the cup across the table and set it on the plate"
[cup] rep3 frames=1061 (21.22s sim)  wall=  1.79s   591.8 fps  ik_maxerr=  0.1mm ik_fail=  0  SUCCESS  cup on plate, 13mm off centre
        task string: "bring the cup across the table and set it on the plate"
[cup] rep4 frames=1060 (21.20s sim)  wall=  1.68s   631.8 fps  ik_maxerr=  0.2mm ik_fail=  0  SUCCESS  cup on plate, 10mm off centre
        task string: "bring the cup across the table and set it on the plate"
[cup] rep5 frames=1061 (21.22s sim)  wall=  1.71s   619.9 fps  ik_maxerr=  0.2mm ik_fail=  0  SUCCESS  cup on plate, 10mm off centre
        task string: "bring the cup across the table and set it on the plate"
[cup] rep6 frames=1090 (21.80s sim)  wall=  2.10s   519.3 fps  ik_maxerr=  0.2mm ik_fail=  0  SUCCESS  cup on plate, 26mm off centre
        task string: "bring the cup across the table and set it on the plate"
[cup] rep7 frames=1062 (21.24s sim)  wall=  1.92s   552.4 fps  ik_maxerr=  0.2mm ik_fail=  0  SUCCESS  cup on plate, 10mm off centre
        task string: "bring the cup across the table and set it on the plate"
[cup] rep8 frames=1061 (21.22s sim)  wall=  2.16s   492.0 fps  ik_maxerr=  0.3mm ik_fail=  0  SUCCESS  cup on plate, 9mm off centre
        task string: "bring the cup across the table and set it on the plate"
[cup] rep9 frames=1076 (21.52s sim)  wall=  2.36s   455.3 fps  ik_maxerr=  0.5mm ik_fail=  0  FAIL     faulted: cup at release: object 21mm off target
        task string: "bring the cup across the table and set it on the plate"
MILESTONE 2: FAIL
```

## Final 20-repetition validation — complete captured output

Command in `sim_phys/`: `$env:BIMANUAL_CARRY='physics'; python -u milestone2_check.py --tasks cup --reps 20 --no-images --seed 2000`.

```text
python : WARNING: Attach conflict when attaching 'so101' to 'bimanual_dinner_table', policy is 'warning'
At line:2 char:32
+ ... ='physics'; python -u milestone2_check.py --tasks cup --reps 20 --no- ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : NotSpecified: (WARNING: Attach...cy is 'warning':String) [], RemoteException
    + FullyQualifiedErrorId : NativeCommandError
 
iterations: parent has 100 (default), child has 10, keeping parent value
ls_iterations: parent has 50 (default), child has 20, keeping parent value

WARNING: Attach conflict when attaching 'so101' to 'bimanual_dinner_table', policy is 'warning'
iterations: parent has 100 (default), child has 10, keeping parent value
ls_iterations: parent has 50 (default), child has 20, keeping parent value

C:\Users\vieth\Documents\lablab hackathon\sim_phys\bimanual_scene.py:457: UserWarning: Attach conflict when attaching 
'so101' to 'bimanual_dinner_table', policy is 'warning'
iterations: parent has 100 (default), child has 10, keeping parent value
ls_iterations: parent has 50 (default), child has 20, keeping parent value
  model = spec.compile()
fps=50 control_dt=0.020s  images=off
[cup] rep0 frames= 718 (14.36s sim)  wall=  1.11s   644.8 fps  ik_maxerr=  0.1mm ik_fail=  0  FAIL     faulted: cup pick: jaws 41mm off the cup's axis
        task string: "bring the cup across the table and set it on the plate"
[cup] rep1 frames=1059 (21.18s sim)  wall=  2.03s   522.7 fps  ik_maxerr=  0.2mm ik_fail=  0  SUCCESS  cup on plate, 6mm off centre
        task string: "bring the cup across the table and set it on the plate"
[cup] rep2 frames= 411 ( 8.22s sim)  wall=  0.52s   797.7 fps  ik_maxerr=  0.2mm ik_fail=  0  FAIL     faulted: cup pick: jaws closed but the cup is not between them
        task string: "bring the cup across the table and set it on the plate"
[cup] rep3 frames=1087 (21.74s sim)  wall=  2.00s   544.1 fps  ik_maxerr=  0.1mm ik_fail=  0  SUCCESS  cup on plate, 21mm off centre
        task string: "bring the cup across the table and set it on the plate"
[cup] rep4 frames=1060 (21.20s sim)  wall=  2.11s   502.6 fps  ik_maxerr=  0.1mm ik_fail=  0  SUCCESS  cup on plate, 14mm off centre
        task string: "bring the cup across the table and set it on the plate"
[cup] rep5 frames=1062 (21.24s sim)  wall=  1.94s   547.9 fps  ik_maxerr=  0.2mm ik_fail=  0  SUCCESS  cup on plate, 10mm off centre
        task string: "bring the cup across the table and set it on the plate"
[cup] rep6 frames=1062 (21.24s sim)  wall=  1.83s   580.0 fps  ik_maxerr=  0.2mm ik_fail=  0  SUCCESS  cup on plate, 8mm off centre
        task string: "bring the cup across the table and set it on the plate"
[cup] rep7 frames=1061 (21.22s sim)  wall=  2.07s   512.0 fps  ik_maxerr=  0.2mm ik_fail=  0  SUCCESS  cup on plate, 2mm off centre
        task string: "bring the cup across the table and set it on the plate"
[cup] rep8 frames=1098 (21.96s sim)  wall=  2.50s   439.4 fps  ik_maxerr=  0.2mm ik_fail=  0  SUCCESS  cup on plate, 25mm off centre
        task string: "bring the cup across the table and set it on the plate"
[cup] rep9 frames=1061 (21.22s sim)  wall=  1.82s   582.4 fps  ik_maxerr=  0.2mm ik_fail=  0  SUCCESS  cup on plate, 13mm off centre
        task string: "bring the cup across the table and set it on the plate"
[cup] rep10 frames=1060 (21.20s sim)  wall=  1.99s   533.2 fps  ik_maxerr=  0.2mm ik_fail=  0  SUCCESS  cup on plate, 3mm off centre
        task string: "bring the cup across the table and set it on the plate"
[cup] rep11 frames=1060 (21.20s sim)  wall=  1.89s   561.3 fps  ik_maxerr=  0.2mm ik_fail=  0  SUCCESS  cup on plate, 3mm off centre
        task string: "bring the cup across the table and set it on the plate"
[cup] rep12 frames=1077 (21.54s sim)  wall=  2.01s   536.0 fps  ik_maxerr=  1.3mm ik_fail=  0  FAIL     faulted: cup at release: object 41mm off target
        task string: "bring the cup across the table and set it on the plate"
[cup] rep13 frames=1077 (21.54s sim)  wall=  2.31s   465.9 fps  ik_maxerr=  1.3mm ik_fail=  0  FAIL     faulted: cup at release: object 26mm off target
        task string: "bring the cup across the table and set it on the plate"
[cup] rep14 frames=1062 (21.24s sim)  wall=  2.07s   514.1 fps  ik_maxerr=  0.2mm ik_fail=  0  SUCCESS  cup on plate, 4mm off centre
        task string: "bring the cup across the table and set it on the plate"
[cup] rep15 frames=1057 (21.14s sim)  wall=  1.97s   537.4 fps  ik_maxerr=  0.5mm ik_fail=  0  SUCCESS  cup on plate, 9mm off centre
        task string: "bring the cup across the table and set it on the plate"
[cup] rep16 frames=1059 (21.18s sim)  wall=  2.08s   507.9 fps  ik_maxerr=  0.3mm ik_fail=  0  SUCCESS  cup on plate, 8mm off centre
        task string: "bring the cup across the table and set it on the plate"
[cup] rep17 frames=1061 (21.22s sim)  wall=  1.89s   561.7 fps  ik_maxerr=  0.2mm ik_fail=  0  SUCCESS  cup on plate, 10mm off centre
        task string: "bring the cup across the table and set it on the plate"
[cup] rep18 frames=1104 (22.08s sim)  wall=  2.07s   534.0 fps  ik_maxerr=  0.2mm ik_fail=  0  FAIL     cup 85mm from plate centre
        task string: "bring the cup across the table and set it on the plate"
[cup] rep19 frames= 716 (14.32s sim)  wall=  1.24s   576.4 fps  ik_maxerr=  0.2mm ik_fail=  0  FAIL     faulted: cup pick: jaws 24mm off the cup's axis
        task string: "bring the cup across the table and set it on the plate"
MILESTONE 2: FAIL
```

## Physical carry visual check

Command: `python -u out/wp1b/capture_relay.py` with `BIMANUAL_CARRY=physics`. This reproduces seed-1000 rep2 and renders selected stages without changing episode recording. Both wrist strips visibly show the blue cup at the jaws during the corresponding arm’s lift/carry; the top strip shows the cup released on the relay and finally on the plate. Contact checks confirm both jaws on the cup during both lifts and carries.

- [Top strip](../sim_phys/out/wp1b/top-relay.png)
- [Right wrist strip](../sim_phys/out/wp1b/right_wrist-relay.png)
- [Left wrist strip](../sim_phys/out/wp1b/left_wrist-relay.png)

Verbatim phase/result lines from the captured command:

```text
start cup [-0.2437  0.1901  0.065 ] held {'left': False, 'right': False}
right lift cup [-0.2266  0.201   0.1459] held {'left': False, 'right': True}
right carry cup [ 0.0023 -0.0008  0.1491] held {'left': False, 'right': True}
right released cup [-0.0051 -0.001   0.065 ] held {'left': False, 'right': False}
left lift cup [-0.0084 -0.0055  0.1282] held {'left': True, 'right': False}
left carry cup [ 0.2235 -0.2177  0.1409] held {'left': True, 'right': False}
left released cup [ 0.2232 -0.2241  0.0809] held {'left': False, 'right': False}
final cup [ 0.2232 -0.2241  0.0809] held {'left': False, 'right': False}
(True, 'cup on plate, 11mm off centre')
```

The visual run preceded removal of unreachable legacy code and addition of the empty-placement guard; the final gate reproduced that successful repetition. The complete gate logs and diagnostic artifacts are retained under `sim_phys/out/wp1b/`. No kinematic rerun was needed for WP1b; its execution path was preserved and the previous worker’s 8/10 result is not presented as a new measurement.
