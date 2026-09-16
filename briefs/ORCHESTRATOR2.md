# ORCHESTRATOR, round 2 (19:00) — gate decided, run the data path now

Same role and rules as `briefs/ORCHESTRATOR.md` (read it first, plus `HANDOFF-15SEP-1720.md`
and `briefs/ORCHESTRATOR-LOG.md` — continue that log). Hard stop 16 Sep 07:00 Hanoi or when
blocked. Be economical: API key, small cap; one long `sleep` per check; short outputs.

## What changed since round 1
* WP1 reported: physics 0/10, kinematic **8/10** (`briefs/WP1-REPORT.md`). The default
  `BIMANUAL_CARRY` in `sim/bimanual_scene.py` is now **`kinematic`** (done). Disclose it in
  the README later as "objects are carried kinematically once `grasp_ok` confirms the jaws
  enclose them; the Menagerie gripper does close on the cup physically (measured), but the
  physical release/withdrawal was not made reliable in time".
* A separate worker (WP1b, log `briefs/WP1b-codex.log`, report `briefs/WP1b-REPORT.md`,
  stop 21:00) is trying to make physics pass **inside `sim_phys/`** (a copy). Ignore it
  unless its report shows ≥ 6/10; then, after the current collection has finished, apply
  `briefs/WP1b.patch` to `sim/`, flip the default back to physics, and re-collect (it is
  ~12 min) into a second dataset repo `VietHwang/dinner-table-v1-physics` (private).
* WP2 is done (`briefs/WP2-REPORT.md`): collector + converter verified on a probe, 4 episodes
  pushed private. WP3 is still running (leave its files alone).
* Kaggle token still absent → training start is BLOCKED ON OWNER (Save & Run All). Keep
  checking `Test-Path C:\Users\vieth\.kaggle\kaggle.json` at each poll; if it appears, run
  step 2's Kaggle part.

## Do now, in order
1. **Volume collection** with the command documented in `briefs/WP2-REPORT.md`: target
   600 kept episodes, `--instruction both`, DART on, 6 workers, into `out/episodes`. WP1
   measured 1.1–1.4 s/episode without images; with 256×256×3 cameras expect several
   seconds. Log attempts/kept/fault counts from `out/episodes/manifest.json`. If kept/attempt
   < 30 % after 200 attempts, stop and log the fault distribution.
2. Convert with `C:\pai\ev\Scripts\python.exe sim/to_lerobot.py ...` (WP2's exact flags),
   load-back check, push **private** to `VietHwang/dinner-table-v1`; verify `private` after
   the push; log the URL and frame count.
3. Kaggle: if the token file exists → `kaggle kernels` push+run of `k1_kaggle_autopilot.py`
   (GPU T4, internet on) and watch for STATUS.md in `VietHwang/smolvla-dinner-table` within
   30 min; else log BLOCKED ON OWNER with the exact click.
4. When WP3's log ends with `codex WP3 exit`, verify its rehearsal outputs and build the
   Tiber bundle as in round 1 step 3.
5. Spawn WP4 (docs) as in round 1 step 4, with the kinematic disclosure above.
6. Training watch / WP5 as in round 1 step 5.
7. End with "STATE FOR THE OWNER".
