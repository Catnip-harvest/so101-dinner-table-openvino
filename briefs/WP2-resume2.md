# RESUME NOTICE 2 (17:30)

The previous run of this brief stopped at ~17:25 with "You have hit your usage limit" on the ChatGPT account. You now run on an OpenAI API key with a small dollar cap: be economical -- do not re-read large files you already know from what is on disk, do not repeat long commands. State on disk: out/probe/ already holds the 5 probe episodes from the collector (npz + three mp4 each) -- check its manifest.json, then continue: convert with C:/pai/ev/Scripts/python.exe sim/to_lerobot.py, load back, push PRIVATE to the probe repo, run the autopilot --selftest, and rewrite briefs/WP2-REPORT.md with real output.

---

# RESUME NOTICE

A previous run of this brief wrote sim/collect.py, sim/to_lerobot.py, edited k1_kaggle_autopilot.py and wrote briefs/WP2-REPORT.md, but its sandbox could not start python.exe ("Access is denied"), so NOTHING was executed and the report explicitly says the acceptance output is missing. This run has NO sandbox. Read the report and the files, keep what is right, then actually RUN every acceptance step in the brief (collect 5 probe episodes, convert with C:/pai/ev/Scripts/python.exe, load back, push PRIVATE to the probe repo, run --selftest) and rewrite briefs/WP2-REPORT.md with the real output. Note the simulator has since moved to a 0.005 s timestep / 50 Hz control and a new arm model; the other worker (WP1) may be editing sim/episode.py concurrently -- if an import breaks mid-run, wait a minute and retry once before diagnosing.

---

# WP2 — self-recording demo collector → LeRobot dataset → private Hugging Face push

You are working in `C:\Users\vieth\Documents\lablab hackathon` on Windows 11 (Git Bash and
PowerShell available). Deadline for the whole project is in ~32 h; this package must be
done and verified in **3 h**. Work only in NEW files unless told otherwise. Report with
measured output (paste the actual command output), never with claims.

## Hard rules

* **Never open, print, grep, diff or copy any `.env*` file.** `.env.local` exists in the
  project root; it is out of bounds. A missing credential means stop and write it in the
  report — never invent a placeholder.
* **Do not edit** `sim/episode.py`, `sim/bimanual_scene.py`, `sim/ik_control.py` or anything
  under `sim/tools/`. Another engineer is changing those files right now (swapping the arm
  model). Read them freely; extend them by subclassing or wrapping from your own files.
* Nothing goes public. A Hugging Face push must be to a **private** repo. No GitHub pushes.
* No new infrastructure. Python only.

## What exists (read these first)

* `sim/episode.py` (1266 lines): `BimanualEnv` (builds the two-arm MuJoCo scene, has
  `randomize_objects()`, `rec: EpisodeRecord` with `.faults`, joint-space stepping via
  `run_joint_phase` / `joint_step`, cameras), `task_cup_relay(env)` — the scripted controller
  that picks a cup with the right arm, sets it at a relay point, picks it with the left arm
  and places it on the plate — and `TASK_STRINGS`. Success predicates are simulator-truth
  functions in the same file. Look at how `out/relay/relay_1001_top.png` was rendered to find
  the camera/render path. Cameras: `top`, `left_wrist`, `right_wrist`.
* `sim/bimanual_scene.py`: scene builder, `ARMS`, `OBJECTS`, `HOME_QPOS`, index helpers.
* `sim/mj_backend.py`: `select_backend()` must be called before `import mujoco`.
* `k1_kaggle_autopilot.py` (355 lines): the one-cell Kaggle notebook that waits for the
  dataset `VietHwang/dinner-table-v1` on the Hub, then fine-tunes `lerobot/smolvla_base`.
  Its `RENAME` dict expects image keys `observation.images.top`, `observation.images.left_wrist`,
  `observation.images.right_wrist`. Keep those key names.
* Python: system Python 3.13 (`%LOCALAPPDATA%\Programs\Python\Python313\python.exe`) has
  `mujoco 3.11.0`, `mink`, `numpy 2.5.2`, **no lerobot**. The venv `C:\pai\ev` has `lerobot`
  (pinned for the OpenVINO export; check its version with `C:\pai\ev\Scripts\python.exe -c
  "import lerobot; print(lerobot.__version__)"`) and may or may not have mujoco. A Hugging
  Face token is already in the standard cache location, so `huggingface_hub` works without
  you touching any credential.

## Deliverables

### 1. `sim/collect.py` — parallel scripted collection to raw episode files

* `python sim/collect.py --out out/episodes --episodes N --workers W --seed S --instruction A|B|both`
* Uses `multiprocessing` (Windows → spawn; each worker builds its own env). Default workers 6.
* Runs `task_cup_relay` for instruction **A** = the existing `TASK_STRINGS["cup"]` (full relay onto the plate).
  Instruction **B** = `"slide the cup to the middle of the table"`: the right arm picks the cup
  and places it at the relay point, then goes home — the FIRST HALF of the relay only. If
  `task_cup_relay` cannot be split without editing `episode.py`, implement B in `collect.py` by
  calling the module-level primitives it uses (`pick_cup`, `place_cup`, `go_home`, and
  `env.relay`/`_find_relay` output) in the same order the relay does. Success for B: cup
  upright within 20 mm of the relay point, right arm home, no faults.
* Records at **50 Hz control** (the env's control step; the sim runs at 200 Hz):
  `observation.state` (12-D: right arm 6 qpos then left arm 6 qpos — state the order in a
  constant and in the report), `action` (12-D commanded joint targets, same order), three
  camera images 256×256 RGB, `task` string, `seed`, `success`, `faults`, `is_perturbed`.
* **Failures are discarded, not saved**, but counted: the run prints and writes
  `out/episodes/manifest.json` with attempts, kept, per-fault counts, per-instruction counts,
  wall time per episode.
* **DART noise** (Laskey et al. 2017): on 30 % of episodes, the *executed* joint command gets
  Gaussian noise σ = 0.02 rad per joint per control step, while the *recorded* action is the
  clean command. Implement by wrapping/subclassing the env's stepping method from your file
  (not by editing episode.py). Flag `is_perturbed=True` on those episodes.
* Raw format per episode: one `.npz` (state, action, timestamps, flags) plus one `.mp4` per
  camera (imageio-ffmpeg; install into the venv you use if missing, `pip install imageio
  imageio-ffmpeg`), or PNG frames if mp4 encoding is not available — say which.

### 2. `sim/to_lerobot.py` — raw episodes → LeRobotDataset → private HF push

* Runs under `C:\pai\ev\Scripts\python.exe` (that is where lerobot lives).
* Builds a LeRobotDataset with the installed lerobot's **current** dataset API
  (`LeRobotDataset.create(repo_id, fps=50, features=...)`, `add_frame`, `save_episode`, and
  whatever finalisation the installed version requires). Features: the three image keys above
  (video), `observation.state` float32[12], `action` float32[12], and `task` per episode.
* `--push` uploads to `VietHwang/dinner-table-v1` as **private** (`private=True`). A
  `--repo` flag allows the 5-episode probe to go to `VietHwang/dinner-table-probe` first.
* **Version alignment**: read which lerobot version `k1_kaggle_autopilot.py` installs on
  Kaggle. If it differs from `C:\pai\ev`'s, say so in the report and, if the dataset codebase
  version they read differs, make the writer match Kaggle's (that is the consumer).

### 3. Re-audit and fix `k1_kaggle_autopilot.py`

This file WAS allowed to be edited (it is not under `sim/`). An audit on 15 Sep found four
defects that would waste an unattended 6–10 h GPU run; the list was not preserved, so audit
it fresh. Look specifically for anything that: makes the run appear to work while producing
nothing; exits early; loses checkpoints; hits the 20 GB working-disk cap; trains on the wrong
data; crashes only at the first checkpoint push or at the end; or makes the run unresumable
when the 12 h cap kills it. Also: the dataset image keys and the `RENAME` step must match
what `to_lerobot.py` writes; `WAIT_H` should become 8.0 (the owner may start the notebook
early). Fix what you find, keep the file a single pasteable cell, and add a
`--selftest` path at the bottom guarded by `if __name__ == "__main__" and "--selftest" in
sys.argv` that exercises the pure-Python helpers (step budgeting, checkpoint pruning
decision, status writing to a temp dir) without GPU, network or lerobot.

## Acceptance (all must be shown in the report with real output)

1. `python sim/collect.py --out out/probe --episodes 5 --workers 2 --instruction both` finishes,
   `manifest.json` shows kept ≥ 3, and one kept episode's `.npz` has `state.shape == (T, 12)`,
   `action.shape == (T, 12)`, and each camera video has T frames (print T and the values).
2. `C:\pai\ev\Scripts\python.exe sim/to_lerobot.py --raw out/probe --repo VietHwang/dinner-table-probe`
   builds a dataset that **loads back** with `LeRobotDataset(...)` and iterates one sample
   whose keys and shapes you print. Then `--push` it as private and print the returned URL.
3. `python k1_kaggle_autopilot.py --selftest` passes; `python -m py_compile k1_kaggle_autopilot.py` passes.
4. A timing line: seconds per attempted episode per worker, so the volume run can be sized.

## Report

Write `briefs/WP2-REPORT.md`: what was built (file list), the measured acceptance output,
the audit findings with line numbers and the fix for each, the state/action ordering
constant, the lerobot version situation, anything left undone and why. Keep it factual.
