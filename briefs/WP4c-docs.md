# WP4c — final physics dataset documentation refresh

Update `README.md`, `RESULTS.md`, and `ARCHITECTURE.md` only. This is a documentation task; do not change simulator, dataset, trainer, evaluator, bundle, or orchestration files. Do not access any `.env*` file and do not publish anything.

Read these evidence sources before editing:

- `briefs/ORCHESTRATOR-LOG.md`, especially the round-5 20:45 and 21:40 entries
- `out/volume_v2.log` (collection/conversion ground truth)
- `briefs/WP2c-REPORT.md` (collector fix and 10/18 probe)
- `briefs/WP1-REPORT.md` and `briefs/WP1b-REPORT.md` (kinematic 8/10; physics 7/10 and 14/20 gates)
- existing `README.md`, `RESULTS.md`, `ARCHITECTURE.md`

Publish only final standards and observed results. Preserve useful existing content and credits. Incorporate:

- private physics dataset `VietHwang/dinner-table-v2`: 400 episodes, 323,442 frames, 50 fps, 749 attempts, 53.4% kept/attempt yield
- per chunk: chunk1 192/100/81,944 frames; chunk2 176/100/79,984; chunk3 204/100/80,044; chunk4 177/100/81,470
- kept instruction split: A 203, B 197; attempted split A 375, B 374
- QA PASS from `briefs/ORCHESTRATOR-LOG.md`: two task strings; three 256x256 cameras; 12-D state/action; visual wrist/jaw/cup check passed; `is_perturbed` is absent from converted frame and episode metadata (do not invent a share)
- carry-mode gates: kinematic 8/10; physics 7/10 and 14/20, each cited to its report/log
- collector probe: 10/18, 7,005 frames, cited to WP2c; make clear it is a small probe, not the volume yield
- volume fault histogram summary from `out/volume_v2.log`: 831 recorded fault events over 194 exact strings (faults include downstream checks, so not one per failed attempt). Categories: place_jaws_closed 343; pick_jaws_closed 172; pick_axis_or_body 130; relay_point 51; resting_height_release 56; target_release 30; left_grasp_set 41; plate_centre 6; left_placing_set 2. Top exact strings: place jaws closed 343; pick jaws closed 172.
- honest limitations: simulated cup-only scope, fixed 130 mm cup, table relay, no real-hardware/required-silicon result yet, kinematic v1 is separate historical data, v2 physics success-only collection, converted v2 omits perturbation metadata even though DART was used in collection
- credits already required: MuJoCo Menagerie, PegBitStudio holding check, peacestate instruction-swap test
- trainer now targets v2 and passed selftest, but Kaggle training was not launched because token remained absent; do not claim a trained checkpoint

Every numeric claim must cite its log/report path inline. Keep the docs concise and operator-friendly: problem, standard, reason, procedure. Write `briefs/WP4c-REPORT.md` summarizing files changed and evidence used. Do not commit or push.
