param([ValidateSet('All','Holdout','Benchmark')][string]$Mode = 'Holdout')
$ErrorActionPreference = 'Stop'
$taskRoot = $PSScriptRoot
$taskPython = Join-Path $taskRoot '.venv\Scripts\python.exe'
try {
    if (-not (Test-Path -LiteralPath $taskPython)) {
        $taskBundled = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
        if (Test-Path -LiteralPath $taskBundled) { & $taskBundled -m venv (Join-Path $taskRoot '.venv') }
        elseif (Get-Command py -ErrorAction SilentlyContinue) { & py -3.12 -m venv (Join-Path $taskRoot '.venv') }
        else { throw 'Python 3.12 is required. Install it and retry.' }
        if ($LASTEXITCODE -ne 0) { throw 'Creating the Python environment failed.' }
    }
    $taskRequirements = Join-Path $taskRoot 'requirements.txt'
    $taskStamp = Join-Path $taskRoot '.venv\rmt_requirements.sha256'
    $taskHash = (Get-FileHash -LiteralPath $taskRequirements -Algorithm SHA256).Hash
    $taskStored = if (Test-Path -LiteralPath $taskStamp) { (Get-Content -LiteralPath $taskStamp -Raw).Trim() } else { '' }
    if ($taskStored -ne $taskHash) {
        & $taskPython -I -m pip install -r $taskRequirements --disable-pip-version-check
        if ($LASTEXITCODE -ne 0) { throw 'Installing dependencies failed.' }
        Set-Content -LiteralPath $taskStamp -Value $taskHash -Encoding ASCII
    }
    $taskTimestamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
    if ($Mode -eq 'All') {
        $taskOutput = 'outputs_assignment4_' + $taskTimestamp
        $taskBenchmarkOutput = Join-Path $taskOutput 'benchmark'
        $taskHoldoutOutput = Join-Path $taskOutput 'holdout'
        Write-Host 'Assignment 4: running the complete research reproduction.' -ForegroundColor Cyan
        Write-Host 'Step 1/4: five projects, three seeds and 1,080 inner candidate fits.'
        & $taskPython -I (Join-Path $taskRoot 'run_benchmark.py') --reproduce --output $taskBenchmarkOutput
        if ($LASTEXITCODE -ne 0) { throw 'Development benchmark did not complete.' }
        Write-Host 'Step 2/4: reproducing Synapse with the original frozen configuration.'
        & $taskPython -I (Join-Path $taskRoot 'evaluate_holdout.py') --output $taskHoldoutOutput
        if ($LASTEXITCODE -ne 0) { throw 'Synapse evaluation did not complete.' }
        $taskOriginalChoice = Get-Content -LiteralPath (Join-Path $taskRoot 'outputs_benchmark\final_selection.json') -Raw | ConvertFrom-Json
        $taskRepeatedChoice = Get-Content -LiteralPath (Join-Path $taskRoot (Join-Path $taskBenchmarkOutput 'final_selection.json')) -Raw | ConvertFrom-Json
        foreach ($taskField in @('condition','parameter','secondary_threshold')) {
            if ($taskOriginalChoice.$taskField -ne $taskRepeatedChoice.$taskField) {
                throw ('The repeated development choice differs in ' + $taskField + '. Results are retained; inspect the environment before combining summaries.')
            }
        }
        Write-Host 'Step 3/4: descriptive tables, threshold analysis and sensitivity summaries.'
        & $taskPython -I (Join-Path $taskRoot 'summarise_results.py') --benchmark-output $taskBenchmarkOutput --holdout-output $taskHoldoutOutput
        if ($LASTEXITCODE -ne 0) { throw 'Creating the result summaries failed.' }
        Write-Host 'Step 4/4: auditing the archived submission evidence.'
        & $taskPython -I (Join-Path $taskRoot 'verify_submission.py')
        if ($LASTEXITCODE -ne 0) { throw 'Submission evidence verification failed.' }
        $taskRun = [ordered]@{
            completed_utc = [DateTime]::UtcNow.ToString('o')
            benchmark_directory = $taskBenchmarkOutput
            holdout_directory = $taskHoldoutOutput
            holdout_selection_source = 'outputs_benchmark/final_selection.json'
            development_choice_matches_original = $true
            interpretation = 'Reproduction of the released study. The Synapse run uses the original frozen configuration; it is not a new untouched test.'
        }
        $taskRun | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskRoot (Join-Path $taskOutput 'run_manifest.json')) -Encoding UTF8
    } elseif ($Mode -eq 'Benchmark') {
        $taskOutput = 'outputs_benchmark_reproduced_' + $taskTimestamp
        Write-Host 'Reproducing the five-project nested benchmark and three seeds.' -ForegroundColor Cyan
        Write-Host 'This run fits more than 1,000 candidates and can take several minutes.'
        & $taskPython -I (Join-Path $taskRoot 'run_benchmark.py') --reproduce --output $taskOutput
        if ($LASTEXITCODE -ne 0) { throw 'Benchmark did not complete.' }
    } else {
        $taskOutput = 'outputs_holdout_demo_' + $taskTimestamp
        Write-Host 'Reproducing the frozen Synapse evaluation.' -ForegroundColor Cyan
        Write-Host 'This is a repeat of the recorded holdout, not a new independent test.'
        & $taskPython -I (Join-Path $taskRoot 'evaluate_holdout.py') --output $taskOutput
        if ($LASTEXITCODE -ne 0) { throw 'Holdout evaluation did not complete.' }
    }
    Write-Host ('SUCCESS. New results: ' + (Join-Path $taskRoot $taskOutput)) -ForegroundColor Green
    Write-Host 'Complete Assignment 4: START_ASSIGNMENT4.cmd. Quick Synapse demo: START_DEMO.cmd.'
    Write-Host 'Original pilot: START_PILOT.cmd. Development only: START_BENCHMARK.cmd.'
    exit 0
} catch {
    Write-Host ('ERROR: ' + $_.Exception.Message) -ForegroundColor Red
    exit 1
}
