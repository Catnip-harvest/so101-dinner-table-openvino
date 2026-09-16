# WP4b round-4 documentation reconciliation

Updated README.md and RESULTS.md from the round-4 entries in [ORCHESTRATOR-LOG.md](ORCHESTRATOR-LOG.md), with carry-mode gates checked against [WP1-REPORT.md](WP1-REPORT.md) and [WP1b-REPORT.md](WP1b-REPORT.md).

- Physics volume: 0 episodes, 0 frames, 0 attempts; first-fault histogram `{}` means no trials. Collection stopped on the raw-cleanup policy blocker before starting. Private v2 is planned, not verified.
- Prior verified private v1: 24 kinematic episodes / 16,236 frames, 24/24 completed attempts, A12/12 and B12/12, histogram `{}`. The 100% recorded yield is not the scripted gate rate. Both 400 and 600 episode thresholds remain unmet.
- Gate evidence: original physics 0/10; patched physics 7/10 and 14/20 (70% each), integrated reproduction 7/10; historical kinematic 8/10 (80%). No new test was run.
- Script DATASET remains v1. Kaggle owner action follows first verified physics chunk, v2 target update and selftest; exact settings and current-cell requirement are documented.
- Coordinator measured rebuilt `C:/pai/tiber_bundle` at 19:18:57 Hanoi: 90 files / 1,278,811,605 bytes and 6.6782112121582 GiB free afterward. Previous out staging was removed by owner. Target hardware remains untested.

Validation: read the cited local evidence and reviewed the edits for dataset identity, counts, gate/yield distinction, training prerequisite and bundle path consistency. Git diff was unavailable because this directory is not a Git working tree. No tests were needed for this documentation update. No .env access, uploads, deletions, collection, conversion or simulator edits.
