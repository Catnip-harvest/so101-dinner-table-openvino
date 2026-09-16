# WP5 — export and evaluate the trained checkpoint

You are a fresh worker. A real private checkpoint now exists at `VietHwang/smolvla-dinner-table` under `checkpoints/last/pretrained_model/`. STATUS observed 2026-09-15 23:36 Hanoi: training on Tesla T4, dataset `VietHwang/dinner-table-v2`, 100 episodes / 81,944 frames, step 10,300, last_upload 002100. Do not claim it is final training; evaluate the checkpoint that actually exists.

Rules:

- Do not access any `.env*` file. Use cached Hugging Face authentication.
- Do not publish or push anything.
- Do not use shell deletion commands. If an exact stale output must be replaced, inspect and validate the absolute path first, then use program code; otherwise choose a fresh output path.
- Do not change simulator, collector, trainer, dataset, bundle, or orchestration files.
- You may update `RESULTS.md` and must write `briefs/WP5-REPORT.md`.
- Every result must come from observed command output/artifacts. Preserve `NOT_THE_REQUIRED_SILICON: true` locally.

Procedure:

1. With `C:\pai\ev\Scripts\python.exe` / `huggingface_hub`, download the private repo snapshot or the `checkpoints/last/pretrained_model/**` files into a local checkpoint directory, preserving the checkpoint directory layout. Record revision, STATUS, training step, and exact local checkpoint path.
2. Inspect `intel_export.py` and run it with the actual local `pretrained_model` checkpoint, outputting both FP16 and INT8 under `C:\pai\policy_ov_trained\{fp16,int8}`. Use `--keep-checkpoint-shape` only if required by the real 12-D checkpoint; do not alter dimensions or tolerances. Verify both IRs load and record file sizes. If an output path already exists, inspect it and use a fresh sibling path unless you can safely prove it is the requested trained export.
3. Run the trained FP16 IR locally with `C:\pai\ev\Scripts\python.exe intel_eval.py --ir C:\pai\policy_ov_trained\fp16 --device CPU --seeds 20 --seed0 1000 --instruction A --out out\trained_eval_20 --swap`. This must exercise 20 seeds and both given instructions via `--swap`. Do not reduce seeds or `max_steps` to save time.
4. Record success counts, instruction-swap table, inference p50/p95, finite-output status, faults, and the required-silicon disclosure from `eval_results.json`. Update `RESULTS.md` with the trained checkpoint provenance and actual numbers, clearly separated from the old untrained rehearsal.
5. If export or evaluation fails, investigate the root cause without changing success predicates/tolerances. Record the exact blocker and partial artifacts in `briefs/WP5-REPORT.md`; do not invent results.

This worker may run for hours. Finish the requested export/evaluation unless blocked by a concrete error.
