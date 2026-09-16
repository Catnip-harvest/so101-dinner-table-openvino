# WP2 report — collector, LeRobot conversion, private push, Kaggle audit

Date: 2026-09-15 (Asia/Saigon)

## Built

- `sim/collect.py`: Windows-spawn parallel collector, instructions A/B, 50 Hz labels and images, clean-action DART recording, failure filtering, MP4 output, and JSON accounting.
- `sim/to_lerobot.py`: raw-to-LeRobot v3 conversion, decoded load-back, private push, and post-push privacy verification.
- `k1_kaggle_autopilot.py`: unattended Kaggle reliability fixes and offline `--selftest`.
- `briefs/WP2-REPORT.md`: measured acceptance record.

No protected simulator file was edited. No `.env*` file was accessed. Nothing was made public and no GitHub push was performed.

## State/action contract

`STATE_ACTION_ORDER` is at `sim/collect.py:51`. State and clean commanded action are both 12-D in this order:

```text
right.shoulder_pan, right.shoulder_lift, right.elbow_flex,
right.wrist_flex, right.wrist_roll, right.gripper,
left.shoulder_pan, left.shoulder_lift, left.elbow_flex,
left.wrist_flex, left.wrist_roll, left.gripper
```

`RecordingEnv` starts at line 62. It derives physics substeps from the model timestep to record at 50 Hz (line 146). DART selection is 30% and executed-command noise is sigma 0.02 rad (lines 48–49); recorded actions remain clean.

## Measured acceptance output

The resumed collector manifest records 5 attempts, 4 kept, 2 workers, 27.968 s total:

```text
wall_time_s_per_attempted_episode: 5.594
seconds_per_attempted_episode_per_worker: 11.187
per_instruction_counts: A attempts=3 kept=3; B attempts=2 kept=1
per_fault_counts: "right arm 83mrad from home"=1
```

Array and decoded-video verification of kept episode 0:

```text
episode 0 state (912, 12) action (912, 12) T 912
top frames 912
left_wrist frames 912
right_wrist frames 912
```

Raw storage is one NPZ plus three H.264 MP4 files per kept episode.

Installed writer version:

```text
lerobot 0.5.1
```

The resumed local dataset was already finalized. The non-push converter safely refused to overwrite it:

```text
lerobot package=0.5.1 codebase=v3.0
dataset root already exists: out\probe\lerobot\VietHwang__dinner-table-probe
```

Load-back and private push reused that finalized dataset and completed:

```text
lerobot package=0.5.1 codebase=v3.0
reusing finalized dataset root=out\probe\lerobot\VietHwang__dinner-table-probe
loadback episodes=4 frames=3213
sample keys=["action", "episode_index", "frame_index", "index", "observation.images.left_wrist", "observation.images.right_wrist", "observation.images.top", "observation.state", "task", "task_index", "timestamp"]
sample shapes={"action": [12], "episode_index": [], "frame_index": [], "index": [], "observation.images.left_wrist": [3, 256, 256], "observation.images.right_wrist": [3, 256, 256], "observation.images.top": [3, 256, 256], "observation.state": [12], "task": [], "task_index": [], "timestamp": []}
pushed private dataset: https://huggingface.co/datasets/VietHwang/dinner-table-probe
```

`push_to_hub(private=True)` was followed by `HfApi.dataset_info()` and fails unless `info.private` is true. A separate `hf.exe` check was blocked by Windows Application Control; it did not affect the successful Python API privacy assertion:

```text
repo_id=VietHwang/dinner-table-probe
private=True
Program 'hf.exe' failed to run: An Application Control policy has blocked this file
```

Autopilot verification:

```text
SELFTEST PASS: step budgeting, checkpoint pruning, status writing
py_compile: PASS
```

## LeRobot alignment

The writer has LeRobot 0.5.1 and dataset codebase `v3.0`. Kaggle is pinned at 0.6.1 (`k1_kaggle_autopilot.py:101`) instead of floating. Both use the v3 dataset schema. Writer keys exactly match `RENAME` at lines 31–33:

```text
observation.images.top
observation.images.left_wrist
observation.images.right_wrist
```

## Kaggle audit findings and fixes

1. Floating LeRobot dependency: pinned 0.6.1 and fail on pip error (101–105).
2. Dataset wait too short: `WAIT_H = 8.0` (18).
3. Step floor could overrun 12 h: safe budgeting (37–43) and explicit stop (347–351).
4. AMP resume used the wrong launcher: restore saved AMP choice (225–227).
5. Resume could exit zero without upload: require/upload a complete checkpoint and propagate failure (217–243).
6. Upload could race an incomplete checkpoint: require config, weights, and training state (57–62).
7. Durability interval exceeded 20 minutes: derive `save_freq` from measured speed (414–416).
8. Timing checkpoints threatened 20 GB: remove scratch (364–370) and prune under low space (384–410).
9. Existing policy repo privacy unverified: force private and confirm through Hub (148–151).
10. Wrong data/schema could train: validate cameras and 12-D arrays (321–331), and never train on probe (357–362).
11. Final failures could look successful: require trainer, checkpoint, upload, and load proof (419–445).

The dependency-free self-test is lines 64–77 and runs before GPU, network, or LeRobot imports. The requested bottom `__main__` guard is also present.

## Left undone

Nothing required for WP2 probe acceptance remains undone. The private probe repo contains the four successful episodes. Full-volume `dinner-table-v1` collection/push is outside this five-episode probe run.
