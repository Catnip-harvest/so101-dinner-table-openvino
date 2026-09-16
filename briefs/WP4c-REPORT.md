# WP4c — final physics dataset documentation refresh

Completed the bounded documentation update without changing simulator, dataset, trainer, evaluator, bundle, or orchestration files. Nothing was committed, pushed, or published, and no `.env*` file was accessed.

## Files changed

- `README.md`: replaced stale pre-volume status with the final private v2 dataset, Hub QA, carry gates, training status, limitations, and required credits.
- `RESULTS.md`: recorded total and per-chunk accounting, instruction splits, QA, scripted gates, collector probe distinction, fault histogram, training status, and pending evaluation.
- `ARCHITECTURE.md`: documented the final physics data contract and flow from collection through conversion, training, export, and evaluation.

## Evidence used

- `out/volume_v2.log`: final volume totals, per-chunk attempts/kept/frames, instruction splits, yield, completion, and fault histogram.
- `briefs/ORCHESTRATOR-LOG.md`: private Hub QA, camera/state/action/task contract, visual wrist/jaw/cup check, missing converted perturbation metadata, v1 history, trainer target/self-test, and absent Kaggle token.
- `briefs/WP2c-REPORT.md`: collector repair, DART contract, and the small **10/18 kept / 7,005-frame** probe.
- `briefs/WP1-REPORT.md`: historical kinematic **8/10** gate.
- `briefs/WP1b-REPORT.md`: physics **7/10** and **14/20** gates.
- Existing `README.md`, `RESULTS.md`, and `ARCHITECTURE.md`: retained useful scope, workflow, evaluation context, and credits while removing stale claims.

## Published standard

The docs now consistently identify private `VietHwang/dinner-table-v2` as the physics training dataset with **400 episodes, 323,442 frames, 50 fps, 749 attempts, and 53.4% yield**. They explicitly keep v1 historical and kinematic, distinguish the small collector probe from the volume yield, avoid inventing an `is_perturbed` share, and do not claim a Kaggle launch, trained checkpoint, real-hardware result, or required-silicon result.
