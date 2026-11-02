# Windows equivalent of the Makefile. Usage: .\run.ps1 setup|data|train|train-custom|serve|test|lint|all|docker-build|docker-run [-Data path\to\my.csv]
param(
    [Parameter(Mandatory = $true)][ValidateSet("setup", "data", "train", "train-custom", "serve", "test", "lint", "all", "docker-build", "docker-run")]
    [string]$Target,
    [string]$Data
)

$py = "venv\Scripts\python.exe"

function Run($exe, $arguments) {
    & $exe @arguments
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

switch ($Target) {
    "setup"        { Run "python" @("-m", "venv", "venv"); Run $py @("-m", "pip", "install", "-r", "requirements.txt") }
    "data"         { Run $py @("scripts/download_data.py") }
    "train"        { Run $py @("-m", "src.train", "--config", "config.yaml") }
    "train-custom" { if (-not $Data) { throw "Pass -Data path\to\my.csv" }; Run $py @("-m", "src.train", "--config", "config.yaml", "--data", $Data) }
    "serve"        { Run $py @("-m", "uvicorn", "src.serve:create_app", "--factory", "--host", "127.0.0.1", "--port", "8000") }
    "test"         { Run $py @("-m", "pytest", "tests/") }
    "lint"         { Run $py @("-m", "ruff", "check", ".") }
    "all"          { & $PSCommandPath setup; & $PSCommandPath data; & $PSCommandPath train; & $PSCommandPath serve }
    "docker-build" { Run "docker" @("build", "-t", "recommendation-engine", ".") }
    "docker-run"   { Run "docker" @("run", "--rm", "-p", "8000:8000", "-v", "${PWD}\models:/app/models:ro", "-v", "${PWD}\data:/app/data:ro", "recommendation-engine") }
}
