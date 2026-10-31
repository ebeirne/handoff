<#
Two-party acceptance demo. The vendor checks a delivery in the offline container, the buyer checks the same files on
its own machine, and both get the same fingerprint. Then a robot dataset with desynced episodes is repaired from
rdq's findings.

    powershell -ExecutionPolicy Bypass -File tools\demo_two_party.ps1              # 4 clips
    powershell -ExecutionPolicy Bypass -File tools\demo_two_party.ps1 -Clips 20    # the full hour

Needs: the .venv install with the hands extra, the hand model, Docker with the handoff-check image
(`docker build -t handoff-check:0.1.0 .`), and for the repair part .venv-rdq (rdq) plus data\so100_planted.
#>
param(
  [int]$Clips = 4,
  # A clip the full-hour audit flags (hands in view in too few frames), so the report shows a reason and a time.
  [string]$Flagged = "factory051_worker008_00051.mp4",
  [string]$Source = "data\egocentric10k_sample",
  [string]$HandModel = "models\hand_landmarker.task",
  [string]$Image = "handoff-check:0.1.0",
  [string]$Out = "reports\two_party_demo",
  [int]$Workers = 4
)
$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path "$PSScriptRoot\..")

function Step($text) { Write-Host ""; Write-Host "== $text" -ForegroundColor Cyan }
function Fresh($path) { if (Test-Path $path) { Remove-Item -Recurse -Force $path }; New-Item -ItemType Directory -Force $path | Out-Null }

# 1. The delivery: the flagged clip, then the first clips up to N, plus the agreed spec. Hard links, so nothing is
#    copied.
$delivery = Join-Path $Out "delivery"
Fresh $Out
New-Item -ItemType Directory -Force $delivery | Out-Null
$all = Get-ChildItem $Source -Filter *.mp4 | Sort-Object Name
$picked = @($all | Where-Object { $_.Name -eq $Flagged }) + @($all | Where-Object { $_.Name -ne $Flagged })
$picked | Select-Object -First $Clips | ForEach-Object {
  New-Item -ItemType HardLink -Path (Join-Path $delivery $_.Name) -Target $_.FullName | Out-Null
}
Copy-Item handoff\example_spec.yaml (Join-Path $delivery "spec.yaml")
Step "Delivery: $Clips clips, spec egocentric-manipulation-v1 (agreed by both sides)"

# 2. Vendor (offline container) and buyer (native) check the same files at the same time.
$abs = (Resolve-Path $Out).Path
New-Item -ItemType Directory -Force (Join-Path $Out "vendor") | Out-Null
$model = (Resolve-Path $HandModel).Path
$dockerArgs = @("run", "--rm", "--network", "none",
  "-v", "$abs\delivery:/data:ro", "-v", "$(Split-Path $model):/models:ro", "-v", "$abs\vendor:/out",
  $Image, "check", "/data", "--spec", "/data/spec.yaml", "--hand-model", "/models/$(Split-Path $model -Leaf)",
  "--out", "/out", "--workers", "$Workers", "--quiet")
Step "Vendor: docker run --network none ... check   |   Buyer: handoff check (native)"
$buyerArgs = @("check", $delivery, "--spec", (Join-Path $delivery "spec.yaml"), "--hand-model", $HandModel,
  "--out", (Join-Path $Out "buyer"), "--workers", "$Workers", "--quiet")
$start = Get-Date
# Both sides' logs go to files (MediaPipe is chatty). Reading .Handle keeps the exit code readable in PowerShell 5.1.
$vendor = Start-Process docker -ArgumentList $dockerArgs -NoNewWindow -PassThru `
  -RedirectStandardOutput (Join-Path $Out "vendor.log") -RedirectStandardError (Join-Path $Out "vendor.err")
$buyer = Start-Process .venv\Scripts\handoff.exe -ArgumentList $buyerArgs -NoNewWindow -PassThru `
  -RedirectStandardOutput (Join-Path $Out "buyer.log") -RedirectStandardError (Join-Path $Out "buyer.err")
$null = $vendor.Handle, $buyer.Handle
$vendor.WaitForExit(); $buyer.WaitForExit()
$seconds = [int]((Get-Date) - $start).TotalSeconds

# 3. Compare what each side got.
$v = Get-Content (Join-Path $Out "vendor\results.json") -Raw | ConvertFrom-Json
$b = Get-Content (Join-Path $Out "buyer\results.json") -Raw | ConvertFrom-Json
Step "Results after $seconds s"
foreach ($side in @(@("vendor (container, no network)", $v), @("buyer  (native)               ", $b))) {
  $r = $side[1]
  $counts = ($r.summary.PSObject.Properties | Where-Object { $_.Name -ne "clips" } | ForEach-Object { "$($_.Name) $($_.Value)" }) -join ", "
  Write-Host ("  {0}  fingerprint {1}  complete {2}  ({3})" -f $side[0], $r.fingerprint.Substring(0, 16), $r.complete, $counts)
}
$match = ($v.fingerprint -eq $b.fingerprint) -and $v.complete -and $b.complete
if ($match) {
  Write-Host "  MATCH: same spec, same data, same verdicts. accepted.txt is the same list on both sides." -ForegroundColor Green
} else {
  Write-Host "  MISMATCH: the two sides disagree. See vendor.err, buyer.err and both results.json files." -ForegroundColor Red
}
Write-Host "  Reports: $Out\vendor\report.html and $Out\buyer\report.html"

# 4. Repair instead of reject: rdq finds the desynced episodes, Handoff writes the fixed copy.
if ((Test-Path ".venv-rdq\Scripts\rdq.exe") -and (Test-Path "data\so100_planted")) {
  Step "Repair: rdq check data\so100_planted, then value.repair --from-rdq"
  $env:PYTHONIOENCODING = "utf-8"
  & .venv-rdq\Scripts\rdq check data\so100_planted --json (Join-Path $Out "rdq.json") --no-color | Select-String "shifted|usable episodes"
  & .venv\Scripts\python -m value.repair --root data\so100_planted --out (Join-Path $Out "repaired") --from-rdq (Join-Path $Out "rdq.json") | Select-Object -Last 1
} else {
  Write-Host ""; Write-Host "(skipping repair: needs .venv-rdq and data\so100_planted)"
}

if (-not $match -or $buyer.ExitCode -ne 0 -or $vendor.ExitCode -ne 0) { exit 1 }
