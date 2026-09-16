[CmdletBinding()]
param(
    [ValidateRange(1, 100)] [int] $Seeds = 10,
    [ValidateRange(1, 100)] [int] $Repeats = 3,
    [int] $Seed0 = 1000,
    [ValidateRange(1, 100000)] [int] $MaxSteps = 1500,
    [string] $VenvPath = "",
    [switch] $XpuSmoke,
    [string] $XpuDataset = "C:\pai\xpu_smoke_dataset"
)

$ErrorActionPreference = "Stop"
$PaiRoot = "C:\pai"
$OutRoot = Join-Path $PaiRoot "out"
$CpuName = (Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name).Trim()
$IsCoreUltra = $CpuName -match "Intel.*Core.*Ultra"
if (-not $VenvPath) {
    $VenvPath = if ($IsCoreUltra) { Join-Path $PaiRoot "dv" } else { Join-Path $PaiRoot "dv2" }
}

New-Item -ItemType Directory -Force -Path $PaiRoot, $OutRoot | Out-Null
Set-Location -LiteralPath $PaiRoot

function Test-RealPython {
    # A WindowsApps "python.exe" is a 0-byte Microsoft Store app-execution alias: it prints
    # "Python was not found" and exits 9009 instead of running anything, which produced an
    # empty venv on the Tiber box. Reject it by path, by size, and by actually running it.
    param([string]$Exe)
    if (-not $Exe) { return $false }
    if (-not (Test-Path -LiteralPath $Exe)) { return $false }
    if ($Exe -like "*\WindowsApps\*") { return $false }
    try { if ((Get-Item -LiteralPath $Exe).Length -lt 1024) { return $false } } catch { return $false }
    $Ver = & $Exe -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $Ver) { return $false }
    $Parts = $Ver.Trim().Split('.')
    if ([int]$Parts[0] -ne 3) { return $false }
    return ([int]$Parts[1] -ge 12 -and [int]$Parts[1] -le 14)
}

function Find-BasePython {
    $Candidates = @()
    foreach ($V in 314, 313, 312) {
        $Candidates += "$env:LOCALAPPDATA\Programs\Python\Python$V\python.exe"
        $Candidates += "$env:ProgramFiles\Python$V\python.exe"
        $Candidates += "C:\Python$V\python.exe"
    }
    $Candidates += "$env:ProgramFiles\Python\python.exe"
    $Candidates += "C:\ProgramData\miniforge3\python.exe"
    $Candidates += "C:\ProgramData\Anaconda3\python.exe"
    foreach ($C in $Candidates) { if (Test-RealPython $C) { return $C } }

    # py launcher: ask it for every interpreter it knows about
    $Py = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($Py) {
        foreach ($Tag in "-3.13", "-3.12", "-3.14", "-3") {
            $Found = & $Py.Source $Tag -c "import sys; print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and (Test-RealPython $Found.Trim())) { return $Found.Trim() }
        }
    }
    foreach ($Cmd in (Get-Command python.exe -All -ErrorAction SilentlyContinue)) {
        if (Test-RealPython $Cmd.Source) { return $Cmd.Source }
    }

    # Nothing usable. Try winget once, then rescan - the operator is usually on a remote
    # desktop with little time, and this is a one-line install.
    $Winget = Get-Command winget.exe -ErrorAction SilentlyContinue
    if ($Winget) {
        Write-Host "no usable Python found; installing Python 3.12 via winget (one time)"
        & $Winget.Source install --id Python.Python.3.12 -e --silent `
            --accept-package-agreements --accept-source-agreements
        foreach ($V in 312, 313, 314) {
            foreach ($Root in "$env:LOCALAPPDATA\Programs\Python", "$env:ProgramFiles") {
                $C = Join-Path $Root "Python$V\python.exe"
                if (Test-RealPython $C) { return $C }
            }
        }
    }
    throw "Python 3.12-3.14 was not found. Install it from python.org (tick 'Add python.exe to PATH'), open a NEW PowerShell, then rerun this command."
}

if (-not (Test-Path -LiteralPath (Join-Path $VenvPath "Scripts\python.exe"))) {
    $BasePython = Find-BasePython
    Write-Host "creating venv $VenvPath with $BasePython"
    & $BasePython -m venv $VenvPath
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed with exit $LASTEXITCODE using $BasePython" }
}
if (-not (Test-Path -LiteralPath (Join-Path $VenvPath "Scripts\python.exe"))) {
    throw "venv at $VenvPath has no Scripts\python.exe - delete it and rerun."
}
$PythonExe = Join-Path $VenvPath "Scripts\python.exe"
Write-Host "using venv $VenvPath"
& $PythonExe -m pip install --disable-pip-version-check --no-cache-dir openvino openvino-tokenizers numpy mujoco mink imageio imageio-ffmpeg
if ($LASTEXITCODE -ne 0) { throw "dependency installation failed with exit $LASTEXITCODE" }
& $PythonExe -c "import sys,importlib.metadata as m; print(sys.version); [print(n,m.version(n)) for n in ('openvino','openvino-tokenizers','numpy','mujoco','mink','imageio','imageio-ffmpeg')]"
if ($LASTEXITCODE -ne 0) { throw "version probe failed with exit $LASTEXITCODE" }

# The render gate is deliberately the first workload after venv setup.
$RenderPath = Join-Path $OutRoot "render_test.png"
try {
    & $PythonExe (Join-Path $PaiRoot "intel_render_test.py") --out $RenderPath
    if ($LASTEXITCODE -ne 0) { throw "render test exited $LASTEXITCODE" }
} catch {
    Write-Error ("render error: " + $_.Exception.Message)
    Write-Host ("RENDER FAILED " + [char]0x2014 + " see fallbacks in PLAN.md " + [char]0x00A7 + "6")
    exit 2
}

$DevicesPath = Join-Path $OutRoot "devices.json"
& $PythonExe (Join-Path $PaiRoot "intel_bench.py") --out $DevicesPath
if ($LASTEXITCODE -ne 0) { throw "device probe failed with exit $LASTEXITCODE" }

$Fp16Xml = Join-Path $PaiRoot "policy_ov\fp16\smolvla.xml"
$Int8Xml = Join-Path $PaiRoot "policy_ov\int8\smolvla.xml"
foreach ($Spec in @(@("fp16", $Fp16Xml), @("int8", $Int8Xml))) {
    $Precision = $Spec[0]
    $Ir = $Spec[1]
    if (-not (Test-Path -LiteralPath $Ir)) { throw "missing $Precision IR: $Ir" }
    $BenchOut = Join-Path $OutRoot ("bench_{0}.json" -f $Precision)
    & $PythonExe (Join-Path $PaiRoot "intel_bench.py") --ir $Ir --runs $Repeats --out $BenchOut
    if ($LASTEXITCODE -ne 0) { throw "$Precision benchmark failed with exit $LASTEXITCODE" }
}

$Fp16Bench = Get-Content -LiteralPath (Join-Path $OutRoot "bench_fp16.json") -Raw | ConvertFrom-Json
$Candidates = @($Fp16Bench.benchmarks | Where-Object { $null -eq $_.error -and $_.output_finite -eq $true })
if ($Candidates.Count -eq 0) { throw "no device completed the FP16 benchmark with finite output" }
$Best = $Candidates | Sort-Object {[double]$_.mean_ms} | Select-Object -First 1
Write-Host ("best FP16 device: {0} ({1} ms mean)" -f $Best.device, $Best.mean_ms)

$EvalOut = Join-Path $OutRoot "eval_fp16_best"
& $PythonExe (Join-Path $PaiRoot "intel_eval.py") --ir (Split-Path -Parent $Fp16Xml) --device $Best.device --seeds $Seeds --seed0 $Seed0 --instruction A --out $EvalOut --swap --video --max-steps $MaxSteps
if ($LASTEXITCODE -ne 0) { throw "closed-loop evaluation failed with exit $LASTEXITCODE" }

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$ZipPath = Join-Path $PaiRoot ("tiber_results_{0}.zip" -f $Stamp)
Compress-Archive -LiteralPath $OutRoot -DestinationPath $ZipPath -CompressionLevel Optimal
$Zip = Get-Item -LiteralPath $ZipPath
Write-Host ("RESULT ZIP {0} bytes={1} size_mb={2:N1}" -f $Zip.FullName, $Zip.Length, ($Zip.Length / 1MB))

if ($XpuSmoke) {
    Write-Host "XPU smoke is optional and cannot change the completed OpenVINO result above."
    Write-Host "UNTESTED HERE (no Intel GPU on this laptop): python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu"
    try {
        $Xv = Join-Path $PaiRoot "xv"
        if (-not (Test-Path -LiteralPath (Join-Path $Xv "Scripts\python.exe"))) {
            $BasePython = Find-BasePython
            & $BasePython -m venv $Xv
        }
        $Xpy = Join-Path $Xv "Scripts\python.exe"
        & $Xpy -m pip install --disable-pip-version-check --no-cache-dir --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
        $XpuAvailable = & $Xpy -c "import torch; print(str(torch.xpu.is_available()).lower())"
        Write-Host ("torch.xpu.is_available() " + $XpuAvailable)
        if ($XpuAvailable -ne "true") { Write-Host "XPU unavailable; optional smoke stops here."; return }
        & $Xpy -m pip install --disable-pip-version-check --no-cache-dir "physicalai-train[smolvla]" "lerobot[dataset]==0.5.1"
        & $Xpy (Join-Path $PaiRoot "intel_xpu_smoke.py") --dataset $XpuDataset --out (Join-Path $OutRoot "xpu_smoke.json")
    } catch {
        Write-Host ("XPU SMOKE FAILED (non-blocking): " + $_.Exception.Message)
    }
}

Write-Host "INTEL DEMO COMPLETE"
exit 0
