# ORCHESTRATOR — run the plan from here without the Claude session

You are the coordinator for the last ~30 hours of the Intel Physical AI Online Challenge
entry in `C:\Users\vieth\Documents\lablab hackathon` (Windows 11; Git Bash + PowerShell;
system Python 3.13 with mujoco/mink; `C:\pai\ev\Scripts\python.exe` has lerobot 0.5.1 +
openvino; `C:\pai\dv` is a deploy venv). The owner's Claude budget is gone, so YOU run the
plan. Read, in order: `HANDOFF-15SEP-1720.md`, `PLAN.md`, then the three worker briefs in
`briefs/WP1-gate.md`, `WP2-collector.md`, `WP3-intel-eval.md`.

Deadline: **17 Sep 01:30 Hanoi** (16 Sep 18:30 UTC). Run `date` at the start of every step.
Your own hard stop: **16 Sep 07:00 Hanoi**, or earlier when blocked on the owner.

## Hard rules (owner's, non-negotiable)

1. Never open, print, grep, diff or copy any `.env*` file. A missing credential = stop and
   write it in `briefs/ORCHESTRATOR-LOG.md` under "BLOCKED ON OWNER"; never a placeholder.
2. Nothing public: no GitHub push, no public HF repo, no video upload. Private HF repos are
   fine (`private=True`, verify after push). Public requires the owner's explicit "go",
   which you cannot obtain — leave it as a blocked item.
3. Never claim a result you did not see. Paste actual command output into the log.
   A number that looks too good is a failure until re-measured.
4. Do not loosen success predicates or tolerances to make numbers pass.
5. Be economical with tokens: you run on the owner's OpenAI API key with a ~$30 cap. Do not
   re-read large files you already know. Poll with a single long `sleep` per check (300 s),
   and keep command output short (`tail`, `grep`).

## Three workers are running right now (same Codex setup, launched 18:42)

| WP | Log | Report when done | Owns these files (do not edit while it runs) |
|---|---|---|---|
| 1 gate | `briefs/WP1-codex.log` | `briefs/WP1-REPORT.md` | `sim/episode.py`, `sim/ik_control.py`, `sim/bimanual_scene.py`, `sim/tools/*` |
| 2 data | `briefs/WP2-codex.log` | `briefs/WP2-REPORT.md` | `sim/collect.py`, `sim/to_lerobot.py`, `k1_kaggle_autopilot.py` |
| 3 intel | `briefs/WP3-codex.log` | `briefs/WP3-REPORT.md` | `intel_eval.py`, `run_intel_demo.ps1`, `TIBER-RUNBOOK.md` |

A worker is finished when its log ends with `codex WPn exit <code>`. Exit 1 with
"usage limit" or a 401 means the run died, not that the work is wrong: relaunch it with
the recipe below, prepending a one-paragraph resume notice to its `-resume2.md` brief
(what is on disk, what is left). Relaunch recipe (Git Bash, project root):

```bash
export CODEX_HOME=/c/pai/codex_api
codex exec --dangerously-bypass-approvals-and-sandbox --skip-git-repo-check -C "C:/Users/vieth/Documents/lablab hackathon" -o "briefs/WP1-codex-last.md" - < briefs/WP1-resume2.md > briefs/WP1-codex.log 2>&1 &
```

(`codex exec` reads stdin: always feed the brief on stdin or `< /dev/null`.) After 21:11
Hanoi the ChatGPT quota is back and you may drop `export CODEX_HOME=...` to use it.

## What to do, in order

0. Start `briefs/ORCHESTRATOR-LOG.md` (timestamped entries; keep it terse). Wait for the
   workers; check every 5 min with one command, e.g.
   `sleep 300; date; for w in 1 2 3; do tail -n 2 briefs/WP$w-codex.log | cut -c1-160; done; ls briefs/*REPORT*`.
1. **When WP1 reports:** read `briefs/WP1-REPORT.md`. Gate = ≥6/10 physics-only successes.
   Pass → keep defaults. Fail → set the default `BIMANUAL_CARRY` to `kinematic` in
   `sim/bimanual_scene.py` (one line) and note the disclosure for the README. Log it.
2. **When WP2 reports** (and WP1 is finished, since collection imports the sim): run the
   volume collection with the exact command WP2 documents — target **600 kept** episodes,
   both instructions, DART on, 6 workers; if kept/attempt is below 30 % stop after 200
   attempts and log the fault distribution instead of grinding. Convert with
   `sim/to_lerobot.py` under `C:\pai\ev`, load-back check, push **private** to
   `VietHwang/dinner-table-v1`. Then Kaggle: if `C:\Users\vieth\.kaggle\kaggle.json`
   exists (test with `Test-Path`, never read it), push and run the notebook with the
   `kaggle kernels` CLI (the notebook is `k1_kaggle_autopilot.py`, single cell, GPU T4,
   internet on; the HF_TOKEN secret must already be attached from the 13 Sep smoke test —
   if the run's STATUS.md never appears within 30 min, log it as BLOCKED). If the token
   file is absent, write BLOCKED ON OWNER: "press Save & Run All on the Kaggle notebook".
3. **When WP3 reports:** confirm `out/rehearsal/` holds the JSON (with
   `NOT_THE_REQUIRED_SILICON: true`), the render-test PNG and a video; confirm
   `TIBER-RUNBOOK.md` exists. Then build the Tiber copy list into `C:\pai\tiber_bundle\`
   (the IR dirs, `intel_eval.py`, `intel_bench.py`, `run_intel_demo.ps1`, `sim/` without
   `__pycache__`, `TIBER-RUNBOOK.md`) and record its size.
4. **Then, with a fresh worker (not yourself):** README.md, ARCHITECTURE.md, RESULTS.md
   skeletons in the project root — problem, standard, procedure; every number cited to the
   log/report it came from; credits (Menagerie, PegBitStudio's holding check, peacestate's
   instruction-swap test); honest-limitations section (cup only, 130 mm cup, table relay,
   kinematic fallback if used). Write that worker's brief to `briefs/WP4-docs.md`.
5. **If training is running:** every 20 min (`sleep 1200`) note the latest loss from the
   HF model repo's STATUS.md (`VietHwang/smolvla-dinner-table`, private) in the log. When a
   checkpoint exists, hand a fresh worker `briefs/WP5-eval.md`: export it with
   `intel_export.py` to FP16/INT8 under `C:\pai\policy_ov_trained\`, run `intel_eval.py`
   20 seeds + `--swap` locally, write `RESULTS.md` numbers.
6. Stop when everything above is done or blocked, or at 07:00. End the log with a
   "STATE FOR THE OWNER" section: what is done (with numbers), what is blocked and the
   exact click the owner must make, and what a next session should do first.
