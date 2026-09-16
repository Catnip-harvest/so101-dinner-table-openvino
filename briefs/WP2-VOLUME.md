# WP2 volume follow-up — 15 Sep 2026

Collector now supports `--target-kept`, mutually exclusive with legacy `--episodes`
(which still counts attempts). Six-worker batches are bounded by remaining successes;
there is no surplus beyond the target. The first yield gate runs at exactly 200
completed attempts, then after each batch: below 30% kept stops with exit 2.
Every completed result updates `manifest.json` by atomic replacement, retaining its
existing schema, including failed attempts, faults, instruction counts and timing.
DART remains enabled at the existing 30% selection rate and 0.02 rad sigma.

Removed the stale collector-local `grasp_ok` override on orchestrator instruction:
collection now inherits the simulator's predicate, without tolerance relaxation.

Volume command (PowerShell, project root; output must be empty):

```powershell
$env:BIMANUAL_CARRY = 'kinematic'
python sim/collect.py --out out/episodes --target-kept 600 --instruction both --workers 6 --seed 0
```

Exact converter flags (conversion includes decoded load-back):

```powershell
& C:\pai\ev\Scripts\python.exe sim/to_lerobot.py --raw out/episodes --repo VietHwang/dinner-table-v1 --root out/lerobot/VietHwang__dinner-table-v1 --push
```

The converter pushes with `private=True` and asserts privacy with `dataset_info()`
afterward; it prints episode/frame counts and the private dataset URL. Without
`--push` it converts and load-checks locally; a later invocation with `--push` reuses
the finalized dataset. Ensure sufficient disk space before starting collection.

For an approved WP1b physics recollection, explicitly set
`$env:BIMANUAL_CARRY = 'physics'`: the collector still has a kinematic environment
default independent of the simulator default. Use a distinct empty raw directory,
distinct converted root and `--repo VietHwang/dinner-table-v1-physics`.

Validation (no production episodes launched): dependency-isolated execution of the
actual scheduling/accounting functions checked exact target completion, all-failure
stop at 200, the 30% boundary, legacy attempt mode, chronological manifest records,
fault counts and atomic replacement. Source compilation passed.

```text
PASS: target completion, 200-attempt gate, 30% boundary, legacy mode, atomic accounting, compile
```
