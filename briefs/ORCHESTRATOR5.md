# ORCHESTRATOR, round 5 — babysit the physics collection, QA the data, start training when possible

Same role and rules as `briefs/ORCHESTRATOR.md` (read it, then the tail of
`briefs/ORCHESTRATOR-LOG.md`; continue that log). Hard stop 16 Sep 07:00 Hanoi or when
blocked. You run on the owner's Codex subscription; no dollar cap, but do not waste: one
long `sleep` per check, short outputs. **Do not run shell deletions** (a guard refuses
them); anything that must be removed is removed by program code or left for the owner.

## Facts now (19:45 Hanoi)
* `sim/run_volume.sh` is RUNNING (a plain script, no model). It collects PHYSICS episodes in
  chunks of 100 kept into `out/episodes_v2_chunkN` with 3 workers, converts each chunk with
  `sim/to_lerobot.py --append --chunk-id chunkN` into `out/lerobot/VietHwang__dinner-table-v2`,
  pushes PRIVATE to `VietHwang/dinner-table-v2`, deletes the raw chunk, repeats up to 6 chunks;
  it stops on disk < 1.5 GB, yield < 30 %, or the clock (23:30 after chunk 4). Progress:
  `out/volume_v2.log` and each chunk's `manifest.json`. At 19:44 chunk 1 was at 21 kept / 46.
  **Never touch its files or start a second collection.**
* Collector fix and probe evidence: `briefs/WP2c-REPORT.md` (10/18 kept, 7,005 frames).
* `VietHwang/dinner-table-v1` (private) = 24 older KINEMATIC episodes; keep separate.
* Disk is tight (≈3.5 GB free at 19:45); the driver's own floor check handles it.
* Kaggle token: `Test-Path C:\Users\vieth\.kaggle\kaggle.json` was False all evening.
* `k1_kaggle_autopilot.py` `DATASET` must point at `VietHwang/dinner-table-v2` before
  training (check; edit the one line if not; run `python k1_kaggle_autopilot.py --selftest`).

## Do, in order
1. **Dataset QA after chunk 1 is on the Hub** (watch `out/volume_v2.log` for
   `convert/push chunk 1 rc=0`): with `C:\pai\ev\Scripts\python.exe`, load
   `VietHwang/dinner-table-v2` (private, token in cache), print episode count, fps, feature
   keys and shapes, `task` strings and their counts, `is_perturbed` share if present,
   min/max of `observation.state` and `action` per joint, and write a 3×4 grid of `top` /
   `left_wrist` / `right_wrist` frames from two episodes to `out/dataset_qa/grid.png` — then
   LOOK at it and state in the log whether the wrist cameras look down the fingers with the
   jaws visible and whether the cup is in the jaws mid-episode. Anything wrong here is a
   stop-the-line finding: log it under "STOP THE LINE" and do not start training.
2. **Kaggle**: at every poll test for the token file. If it appears and the dataset has
   ≥ 200 episodes: fix `DATASET`, selftest, push+run the notebook with the `kaggle kernels`
   CLI (GPU T4, internet on; the HF_TOKEN secret is attached to the notebook from 13 Sep),
   then watch `VietHwang/smolvla-dinner-table` for STATUS.md within 30 min. If the token is
   absent: keep the exact owner click in the log ("Save & Run All on the notebook with the
   current `k1_kaggle_autopilot.py` cell, GPU T4, internet on") and carry on.
3. **When the driver prints `DONE`** (or has stopped): log final attempts/kept/frames per
   chunk and the fault histogram; hand a fresh worker `briefs/WP4c-docs.md` to update
   README.md / RESULTS.md / ARCHITECTURE.md with the final dataset numbers, both carry
   modes' gate numbers (kinematic 8/10; physics 7/10 and 14/20), the collector yield, and
   the honest-limitations section; every number cited to its log/report.
4. **Bundle**: only when ≥ 3 GB are free, rebuild `C:\pai\tiber_bundle` from the copy list
   in `TIBER-RUNBOOK.md` (with the current `sim/`), and log its size.
5. **Training watch / WP5** as in round 1: when a checkpoint exists, a fresh worker exports
   it (`intel_export.py`) to `C:\pai\policy_ov_trained\{fp16,int8}`, runs `intel_eval.py`
   20 seeds + `--swap` locally, and writes the numbers into RESULTS.md.
6. End with "STATE FOR THE OWNER".
