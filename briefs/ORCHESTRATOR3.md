# ORCHESTRATOR, round 3 (19:03) — disk is what it is; run the data path within it

Same role and rules as `briefs/ORCHESTRATOR.md` and `ORCHESTRATOR2.md` (read both, plus
`briefs/ORCHESTRATOR-LOG.md` and `briefs/WP2-VOLUME.md` — continue the log). Hard stop
16 Sep 07:00 Hanoi or when blocked. Economical: API key, small cap; one long sleep per check.

## Disk situation (measured just now)
C: has **6.8 GB free** after purging npm/pip caches and the regenerable FP32 IR. Nothing
else may be deleted without the owner (the remaining big items are the owner's files, the
HF model cache the export needs, and Codex's own data). **Do not wait for more space; work
inside it:**

* Collect in **chunks of 100 kept episodes** into `out/episodes_chunkN`; after each chunk,
  convert it with `sim/to_lerobot.py` into ONE growing local LeRobot dataset (or one dataset
  per chunk if the writer cannot append — then push each chunk as
  `VietHwang/dinner-table-v1` with a per-chunk subfolder/branch, or as separate private
  repos `dinner-table-v1-partN`; say which in the log), push private, verify privacy, then
  **delete the chunk's raw npz/mp4** before starting the next chunk.
* Check free space before each chunk (`(Get-PSDrive C).Free/1GB`). **Abort the collection
  and log the state if free space drops below 1.5 GB.** Target 600 kept; **400 kept is an
  acceptable stop** if space or time forces it — log the number honestly.
* Keep `out/probe` and everything under `C:\pai` untouched (WP3 is using them).

## Then, in order (as round 2)
1. Kaggle: token file present → push+run; absent → BLOCKED ON OWNER (Save & Run All).
   The autopilot reads `VietHwang/dinner-table-v1`; if you had to split into part repos,
   edit the `DATASET` constant in `k1_kaggle_autopilot.py` to a comma list only if the
   script supports it — otherwise merge locally before the final push and log it.
2. WP3 verification + Tiber bundle when `codex WP3 exit` appears in its log.
3. WP1b: if `briefs/WP1b-REPORT.md` shows >= 6/10 physics before 21:30, apply
   `briefs/WP1b.patch` to `sim/`, flip the default back to physics, and collect a second
   dataset the same chunked way into `VietHwang/dinner-table-v1-physics`.
4. Spawn WP4 (docs) with the kinematic disclosure; WP5 when a trained checkpoint exists.
5. End with "STATE FOR THE OWNER".
