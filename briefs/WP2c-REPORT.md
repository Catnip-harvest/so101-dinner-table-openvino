# WP2c — collector physics parity

Completed 2026-09-15T19:37:14+07:00, before 20:30 Hanoi.
**PASS: 10/18 kept (55.6%), 7,005 frames, both instructions, one perturbed and nine unperturbed episodes; private upload verified.**
The existing failed-volume manifest records 0/93 kept. This is a small probe, not a claim of a 55.6% long-run rate.

The root cause was the collector's stale `grip_phase` override, formerly `sim/collect.py:176–185`,
preserved at `out/wp2c/collect.before.py:176–185`. It updated `q_cmd[arm][5]` but omitted
`self.target[arm].grip = float(grip)` from `sim/episode.py:652`. The next `hold()`
(`sim/episode.py:586`) resumed Cartesian control (`sim/episode.py:527`) with the stale jaw target.
The real simulator reproduction below shows a 0.2-rad close immediately restored to 1.2 rad.
The same omission can reclose jaws after release. The duplicated substep loop itself had no
missing physics hook; the stale phase state was the demonstrated failure.

`sim/collect.py:96–117` now delegates joint and Cartesian steps to `BimanualEnv` and inherits
all three phase methods, including the gripper target synchronization. The parent owns IK,
clean labels, actuator writes, all substeps, carry hooks and state-dependent holds/settles.
`sim/collect.py:50` reorders recorded samples to right-then-left; `sim/collect.py:77` streams
the parent's pre-substep images to the existing video writers without accumulating images.
`sim/collect.py:60` adapts only the assignment of seeded Gaussian draws: the collector's
original right-first noise is mapped into the parent's left-first arm order. DART remains
30% selection at sigma 0.02 rad, applied by the parent; labels stay clean.

The first delegation-only trial reached 10/19 but retained no perturbed episodes. It exposed
the changed seeded noise assignment. Restoring the original assignment made seed 7013 a real
perturbed success; the final probe below uses the original scheduling, seeds, rate and sigma.
The interim artifacts remain under `out/wp2c/probe_parent_noise`; they were not uploaded.

The current pre-fix collector already defaulted to physics at line 20; its kinematic comment
and `briefs/WP2-VOLUME.md` were stale. The collector-local default is removed, so it now shares
the simulator's default. Each spawned worker prints the inherited environment and resolved
mode once, and the manifest stores `carry_mode`.

Only `sim/collect.py` changed among simulator modules. `sim/episode.py`, `sim/bimanual_scene.py`
and `sim/to_lerobot.py` match their saved pre-change SHA-256 hashes. Episode hash:
`791c315d66e1ffb46eaddb1e4d69459883a1f4e9ef48aaca1256a5b6b0f399f9`.
AST checks confirm unchanged task B, its predicate, attempt execution, scheduling, DART
constants and label schema. No predicate/tolerance was loosened. The episode module was not
edited; the earlier 7/10 gate result is not presented as a new run.

Regression command: `BIMANUAL_CARRY=physics python out/wp2c/verify_collect.py` (exit 0).
For noisy parent parity, the reference parent uses the same right-first draw adapter.

```text
before close: q_cmd=0.2000 target.grip=1.2000
before hold: q_cmd=1.2000
PASS original collector DART tick: ctrl/qpos/RNG bit-identical
after close: q_cmd=0.2000 target.grip=0.2000
after hold: q_cmd=0.2000
PASS parent parity: seed=7000 instruction=A sigma=0.0 frames=1113 outcome=(False, 'faulted: cup at release: object 22mm off target') qpos/qvel/ctrl/state/action/RNG bit-identical (labels reordered)
PASS parent parity: seed=7013 instruction=B sigma=0.02 frames=548 outcome=(True, 'success') qpos/qvel/ctrl/state/action/RNG bit-identical (labels reordered)
```

Acceptance command (PowerShell equivalent used, exit 0):

```powershell
$env:BIMANUAL_CARRY='physics'
python sim/collect.py --out out/probe_phys --target-kept 10 --workers 2 --instruction both --seed 7000
```

```text
worker pid=41176 BIMANUAL_CARRY=physics CARRY_MODE=physics
worker pid=34368 BIMANUAL_CARRY=physics CARRY_MODE=physics
```

Manifest fault histogram (all faults from failed attempts, including downstream faults):

```json
{
  "cup at release: object 16mm off target": 1,
  "cup at release: object 22mm off target": 1,
  "cup pick: jaws 103mm off the cup's axis": 1,
  "cup pick: jaws 20mm off the cup's axis": 1,
  "cup pick: jaws 29mm off the cup's axis": 1,
  "cup pick: jaws 52mm off the cup's axis": 1,
  "cup pick: jaws 66mm off the cup's axis": 1,
  "cup pick: jaws 85mm off the cup's axis": 1,
  "cup pick: jaws closed but the cup is not between them": 3,
  "cup place: jaws closed but the cup is not between them": 9
}
```

Raw artifact checker output, `python out/wp2c/check_probe.py` (exit 0):

```text
kept=10 attempts=18 yield=55.6%
per_instruction_counts={"A": {"attempts": 9, "kept": 3}, "B": {"attempts": 9, "kept": 7}}
npz=episode_000001.npz state.shape=(546, 12) action.shape=(546, 12)
episode_000001_top.mp4 frames=546 seconds=10.92
episode_000001_left_wrist.mp4 frames=546 seconds=10.92
episode_000001_right_wrist.mp4 frames=546 seconds=10.92
is_perturbed=[False, False, False, False, False, False, True, False, False, False]
kept attempts=[1, 4, 5, 7, 9, 10, 13, 14, 15, 17] total_frames=7005
PASS raw arrays, labels, timestamps, flags, video counts, and acceptance yield
```

Collection wall time was 105.861 seconds;
11.762 seconds per attempted episode per worker.
The checker validates every kept NPZ, its flags and timestamps, and counts all three videos
for the displayed episode. Conversion decodes and checks all three cameras for every episode.

Requested converter command (exit 0):

```powershell
& C:\pai\ev\Scripts\python.exe sim/to_lerobot.py --raw out/probe_phys --repo VietHwang/dinner-table-probe-phys --root out/lerobot/probe_phys --append --chunk-id probe --push
```

Selected actual converter output:

```text
lerobot package=0.5.1 codebase=v3.0
saved episode attempt=1 frames=546 task='slide the cup to the middle of the table'
saved episode attempt=4 frames=1062 task='bring the cup across the table and set it on the plate'
saved episode attempt=5 frames=545 task='slide the cup to the middle of the table'
saved episode attempt=7 frames=546 task='slide the cup to the middle of the table'
saved episode attempt=9 frames=545 task='slide the cup to the middle of the table'
saved episode attempt=10 frames=1059 task='bring the cup across the table and set it on the plate'
saved episode attempt=13 frames=548 task='slide the cup to the middle of the table'
saved episode attempt=14 frames=1061 task='bring the cup across the table and set it on the plate'
saved episode attempt=15 frames=548 task='slide the cup to the middle of the table'
saved episode attempt=17 frames=545 task='slide the cup to the middle of the table'
loadback episodes=10 frames=7005
sample shapes={"action": [12], "episode_index": [], "frame_index": [], "index": [], "observation.images.left_wrist": [3, 256, 256], "observation.images.right_wrist": [3, 256, 256], "observation.images.top": [3, 256, 256], "observation.state": [12], "task": [], "task_index": [], "timestamp": []}
pushed private dataset: https://huggingface.co/datasets/VietHwang/dinner-table-probe-phys
```

Independent authenticated Hub verification, including downloaded `meta/info.json`:

```text
remote private=True episodes=10 frames=7005 fps=50
url=https://huggingface.co/datasets/VietHwang/dinner-table-probe-phys
chunk probe: complete
```

Dataset: [VietHwang/dinner-table-probe-phys](https://huggingface.co/datasets/VietHwang/dinner-table-probe-phys), private.
No `.env*` files were opened; nothing was made public. Existing cached Hub authentication
was used. The local HF CLI executable was blocked by Windows Application Control; the
requested converter and Hub verification used the installed Python SDK successfully.

Evidence: [collector diff](../out/wp2c/collect.patch), [regression](../out/wp2c/verify.log),
[collection](../out/wp2c/probe.log), [raw checks](../out/wp2c/artifacts.log),
[conversion](../out/wp2c/convert.log), [remote checks](../out/wp2c/remote.log).
