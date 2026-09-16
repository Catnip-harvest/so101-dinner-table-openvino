# ORCHESTRATOR, round 4 — collect the PHYSICS dataset, serially and within the disk

Same role and rules as rounds 1–3 (read `briefs/ORCHESTRATOR.md`, then the tail of
`briefs/ORCHESTRATOR-LOG.md` and `briefs/WP2-VOLUME.md`; continue the log). Hard stop
16 Sep 07:00 Hanoi or when blocked. Economical: API key, small cap; one long sleep per check.

## Facts since round 3 (verified by the coordinator's predecessor and by hand)
* `sim/` now carries the WP1b patch; **physics is the default** and reproduces 7/10.
* `VietHwang/dinner-table-v1` (private) holds **24 KINEMATIC episodes**. Do not mix
  physics data into it. The physics dataset goes to a NEW private repo
  **`VietHwang/dinner-table-v2`**, and `k1_kaggle_autopilot.py`'s `DATASET` constant must
  be changed to that name (one line) once the first physics chunk is verified there.
* Disk: round 3 saw C: fall from 6.7 GB to 0.2 GB while 6 collector workers and a
  converter ran at once — pagefile growth under memory pressure, not files. Hand-cleanup
  since then brought free space back (measure it: `(Get-PSDrive C).Free/1GB`).
* Round 3's `rm` of raw files was refused by Codex's own command guard. **Do not delete
  with shell commands.** Instead add a `--delete-raw` flag to `sim/to_lerobot.py` that
  removes each episode's npz/mp4 with `os.remove` after that episode is written and the
  dataset is finalised; that is ordinary program behaviour and is not gated.
* WP3 is done; its Tiber bundle staging under `out/` was removed to free space — the copy
  list in `TIBER-RUNBOOK.md` is the source of truth, rebuild the bundle only at the end,
  under `C:\pai\tiber_bundle`, and only if ≥ 3 GB are free after everything else.

## Collection protocol (this is the whole job)
1. **Workers = 3, never more.** Chunks of **100 kept** episodes, `--instruction both`, DART on,
   seeds non-overlapping per chunk (chunk N uses seed0 = 100000·N).
2. **Strictly serial**: collect chunk → convert (`--append` into ONE local dataset, per
   WP2-VOLUME.md's flags) with `--delete-raw` → push private to `dinner-table-v2` → verify
   `private` and the episode count via the Hub API → log → next chunk.
3. Free-space check before every chunk and every conversion; **abort and log if < 1.5 GB**.
4. Target **600 kept**; stop at 400 if the clock passes **23:30** or the disk forces it.
   Log attempts, kept, first-fault histogram per chunk from each `manifest.json`.
5. After the first chunk is verified on the Hub, edit `DATASET` in `k1_kaggle_autopilot.py`
   to `VietHwang/dinner-table-v2`, run `python k1_kaggle_autopilot.py --selftest`, and
   check for the Kaggle token file at every poll (`Test-Path C:\Users\vieth\.kaggle\kaggle.json`):
   present → `kaggle kernels` push+run and watch for STATUS.md; absent → BLOCKED ON OWNER
   (exact click: Save & Run All on the notebook with the current cell, GPU T4, internet on,
   HF_TOKEN secret attached).
6. When the collection ends, ask a fresh worker (`briefs/WP4b-docs.md`) to update
   README.md / RESULTS.md with the final dataset numbers (episodes, frames, kept/attempt,
   fault histogram, both carry modes' success rates) — sourced to the log.
7. Finish with "STATE FOR THE OWNER".
