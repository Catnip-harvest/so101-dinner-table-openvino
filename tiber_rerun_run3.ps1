# Re-evaluate the run-3 policy on the Tiber Core Ultra box. Reuses the venv and the render
# context that run_intel_demo.ps1 already proved; skips pip, the render gate and the latency
# benchmark (the graph is identical to run 1, so FP16/INT8 latency does not change).
#
#   1. copy policy_ov_run3\fp16\  ->  C:\pai\policy_ov_run3\fp16\   (764 MB)
#   2. copy this file             ->  C:\pai\tiber_rerun_run3.ps1
#   3. Set-ExecutionPolicy -Scope Process Bypass; C:\pai\tiber_rerun_run3.ps1
#
# Same protocol as the run-1 Core Ultra eval: GPU, FP16, 10 seeds from 1000, instruction A
# with the swap check (B), 1500 steps, videos. ~70 min. Ends with RUN3 EVAL COMPLETE and a zip.
$ErrorActionPreference = "Stop"
$PaiRoot = "C:\pai"
$Py = Join-Path $PaiRoot "dv\Scripts\python.exe"
$Ir = Join-Path $PaiRoot "policy_ov_run3\fp16"
$Out = Join-Path $PaiRoot "out_run3\eval_fp16_gpu"

foreach ($p in @($Py, (Join-Path $Ir "smolvla.xml"), (Join-Path $Ir "tokenizer.xml"), (Join-Path $PaiRoot "intel_eval.py"))) {
    if (-not (Test-Path -LiteralPath $p)) { throw "missing: $p" }
}
New-Item -ItemType Directory -Force -Path $Out | Out-Null
Set-Location -LiteralPath $PaiRoot

Write-Host "=== run-3 FP16 on GPU: 10 seeds x (A + swap) ==="
& $Py (Join-Path $PaiRoot "intel_eval.py") --ir $Ir --device GPU --seeds 10 --seed0 1000 `
    --instruction A --out $Out --swap --video --max-steps 1500
if ($LASTEXITCODE -ne 0) { throw "intel_eval exited $LASTEXITCODE" }

$Res = Join-Path $Out "eval_results.json"
if (-not (Test-Path -LiteralPath $Res)) { throw "no eval_results.json written" }
$J = Get-Content -LiteralPath $Res -Raw | ConvertFrom-Json
Write-Host ("successes: {0} / {1}   inference p50 {2} ms   required silicon: {3}" -f `
    $J.successes, $J.episodes.Count, $J.inference.p50_ms, (-not $J.NOT_THE_REQUIRED_SILICON))

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Zip = Join-Path $PaiRoot ("tiber_results_run3_{0}.zip" -f $Stamp)
Compress-Archive -LiteralPath (Join-Path $PaiRoot "out_run3") -DestinationPath $Zip -CompressionLevel Optimal
Write-Host "wrote $Zip"
Write-Host "RUN3 EVAL COMPLETE"
