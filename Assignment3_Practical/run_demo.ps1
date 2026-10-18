$ErrorActionPreference = 'Stop'
$taskRoot = $PSScriptRoot
$taskPython = Join-Path $taskRoot '.venv\Scripts\python.exe'
$taskRequirements = Join-Path $taskRoot 'requirements.txt'
$taskStamp = Join-Path $taskRoot '.venv\rmt_requirements.sha256'

try {
    Write-Host 'STEP 1/3  Checking the Python environment' -ForegroundColor Cyan
    if (-not (Test-Path -LiteralPath $taskPython)) {
        $taskBundledPython = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
        if (Test-Path -LiteralPath $taskBundledPython) {
            & $taskBundledPython -m venv (Join-Path $taskRoot '.venv')
        } elseif (Get-Command py -ErrorAction SilentlyContinue) {
            & py -3.12 -m venv (Join-Path $taskRoot '.venv')
        } else {
            throw 'Python 3.12 was not found. Install Python 3.12 and retry.'
        }
        if ($LASTEXITCODE -ne 0) { throw 'Could not create the Python environment.' }
    }
    $taskHash = (Get-FileHash -LiteralPath $taskRequirements -Algorithm SHA256).Hash
    $taskInstalledHash = if (Test-Path -LiteralPath $taskStamp) { (Get-Content -LiteralPath $taskStamp -Raw).Trim() } else { '' }
    if ($taskHash -ne $taskInstalledHash) {
        Write-Host 'First launch: installing pinned libraries. Internet access is required.'
        & $taskPython -I -m pip install -r $taskRequirements --disable-pip-version-check
        if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed. Check the messages above.' }
        Set-Content -LiteralPath $taskStamp -Value $taskHash -Encoding ASCII
    }
    Write-Host 'STEP 2/3  Training and evaluating on the real datasets' -ForegroundColor Cyan
    & $taskPython -I (Join-Path $taskRoot 'run_pipeline.py') --output outputs_demo
    if ($LASTEXITCODE -ne 0) { throw 'Pipeline failed. Results are not confirmed.' }
    Write-Host 'STEP 3/3  Test results' -ForegroundColor Cyan
    Import-Csv -LiteralPath (Join-Path $taskRoot 'outputs_demo\metrics.csv') |
        Where-Object { $_.split -eq 'test' } |
        Select-Object model, @{Name='Precision';Expression={ '{0:F3}' -f [double]::Parse($_.precision,[Globalization.CultureInfo]::InvariantCulture) }},
                            @{Name='Recall';Expression={ '{0:F3}' -f [double]::Parse($_.recall,[Globalization.CultureInfo]::InvariantCulture) }},
                            @{Name='F1';Expression={ '{0:F3}' -f [double]::Parse($_.f1,[Globalization.CultureInfo]::InvariantCulture) }},
                            @{Name='ROC_AUC';Expression={ '{0:F3}' -f [double]::Parse($_.roc_auc,[Globalization.CultureInfo]::InvariantCulture) }} |
        Format-Table -AutoSize | Out-Host
    Write-Host 'SUCCESS. New results are in outputs_demo.' -ForegroundColor Green
    Write-Host 'Open metrics.csv, test_predictions.csv and checks.txt in that folder.'
    exit 0
} catch {
    Write-Host ('ERROR: ' + $_.Exception.Message) -ForegroundColor Red
    exit 1
}
