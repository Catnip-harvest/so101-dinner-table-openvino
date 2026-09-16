# Architecture

The system turns a physics-scripted bimanual cup relay into a private LeRobot dataset, then provides a Kaggle training path and an OpenVINO evaluation path.

## Problem and standard

The supported demonstration scope is a simulated fixed **130 mm cup** moved by two SO-101 arms through a table relay. Instruction A completes the relay to the plate; instruction B stops at the middle relay point. [Scope and task source](PLAN.md), [episode implementation](sim/episode.py)

The production training source is the private `VietHwang/dinner-table-v2`: **400 physics episodes / 323,442 frames at 50 fps**, collected from **749 attempts**. Private v1 is separate historical kinematic data and must not be merged implicitly with v2. [Volume ground truth](out/volume_v2.log), [v1 history](briefs/ORCHESTRATOR-LOG.md)

## Data path

```text
physics scripted controller
  -> success-filtering collector + DART
  -> raw NPZ + top/left-wrist/right-wrist video + manifest
  -> append conversion and private Hub verification
  -> VietHwang/dinner-table-v2
  -> SmolVLA Kaggle trainer
  -> checkpoint
  -> OpenVINO FP16/INT8 export
  -> closed-loop and instruction-swap evaluation
```

`sim/collect.py` records clean commanded actions, observations, task text, and the three camera streams. DART perturbs executed commands while labels remain clean. Only successful physics episodes enter v2; failed trajectories contribute fault accounting but not training episodes. [Collector parity and DART contract](briefs/WP2c-REPORT.md), [volume ground truth](out/volume_v2.log)

State and action are each **12-D**, ordered as the six right-arm joints followed by the six left-arm joints. The converted cameras are `observation.images.top`, `observation.images.left_wrist`, and `observation.images.right_wrist`, each **256x256**. The dataset contains exactly **two task strings**. [Converted Hub QA](briefs/ORCHESTRATOR-LOG.md)

`sim/to_lerobot.py` appends each chunk into the same local/remote dataset, reloads decoded data, pushes privately, and verifies the resulting counts. Four chunks contributed **100 kept episodes each**, totaling **400 episodes**. [Volume ground truth](out/volume_v2.log)

The conversion does not expose `is_perturbed` in frame features or episode metadata, even though DART was active during collection. Downstream consumers must not infer or claim a perturbation share from v2. [DART evidence](briefs/WP2c-REPORT.md), [converted QA](briefs/ORCHESTRATOR-LOG.md)

## Training procedure

`k1_kaggle_autopilot.py` targets `VietHwang/dinner-table-v2`, validates the dataset, budgets training, retains resumable checkpoints, and writes status. Its self-test passed, but Kaggle training was not launched because the credential file remained absent. The owner must **Save & Run All on the notebook with the current `k1_kaggle_autopilot.py` cell, GPU T4, internet on**. No checkpoint should be treated as trained until the remote status and artifact exist. [Trainer status](briefs/ORCHESTRATOR-LOG.md)

## Evaluation procedure

After a trained checkpoint exists, `intel_export.py` produces FP16 and INT8 OpenVINO models. `intel_eval.py` consumes the same state, image, normalization, tokenization, and task contracts, then evaluates normal and instruction-swapped runs in closed loop. The existing rehearsal is untrained and ran on non-required AMD/NVIDIA hardware; real-hardware and required-silicon evidence is still absent. [Export/evaluation contract and hardware status](briefs/WP3-REPORT.md)

## Design rationale and limits

Physics is the current standard because it passed scripted gates of **7/10** and **14/20** without predicate relaxation; the historical kinematic fallback measured **8/10**. [Physics gates](briefs/WP1b-REPORT.md), [kinematic gate](briefs/WP1-REPORT.md)

This architecture does not yet establish general object manipulation, variable cup geometry, mid-air handoff, real-hardware transfer, Intel target performance, or trained-policy success. Its validated dataset is simulated, fixed-cup, table-relay, and success-only. [Scope](PLAN.md), [volume evidence](out/volume_v2.log), [hardware status](briefs/WP3-REPORT.md)

Credits: MuJoCo Menagerie supplies the robot/gripper asset; PegBitStudio informed the holding check; peacestate informed the instruction-swap test. [Attribution notes](PLAN.md)
