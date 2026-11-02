# Publishes the free static demo to a Hugging Face static Space. Run from anywhere:
#   .\deploy\huggingface\push-static.ps1
# Needs your Hugging Face token in the HF_TOKEN environment variable. Set it in your own
# PowerShell window first, never in a file:   $env:HF_TOKEN = "hf_..."
# Create the empty Space on huggingface.co first (SDK: Static, Blank, Public).
# Needs a trained model in models/ (run .\run.ps1 train once).
param(
    [string]$User = "Onkar-Gupta-HF",
    [string]$Space = "recommendation-engine"
)

if (-not $env:HF_TOKEN) { throw 'Set the token first:  $env:HF_TOKEN = "hf_..."' }

$repo = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$python = Join-Path $repo "venv\Scripts\python.exe"
$site = Join-Path $env:TEMP "hf-site-$Space"
$work = Join-Path $env:TEMP "hf-space-$Space"
foreach ($d in @($site, $work)) { if (Test-Path $d) { Remove-Item $d -Recurse -Force } }

# Build the static copy of the demo from your trained model.
Push-Location $repo
& $python "scripts\export_static.py" --out $site
$exportCode = $LASTEXITCODE
Pop-Location
if ($exportCode -ne 0) { throw "Export failed. Is the model trained? Run .\run.ps1 train first." }

$auth = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes("${User}:$($env:HF_TOKEN)"))
$header = "http.extraHeader=Authorization: Basic $auth"
$url = "https://huggingface.co/spaces/$User/$Space"

git -c $header clone $url $work
if ($LASTEXITCODE -ne 0) { throw "Clone failed. Does the Space exist, and is the token a Write token?" }

# Replace the Space's files with the demo. A normal commit and push, never a force push.
Get-ChildItem $work -Force | Where-Object { $_.Name -ne ".git" } | Remove-Item -Recurse -Force
Copy-Item (Join-Path $site "*") $work -Recurse
Copy-Item (Join-Path $PSScriptRoot "static-README.md") (Join-Path $work "README.md")

Set-Location $work
git add -A
git -c user.name="Onkar Gupta" -c user.email="onkar.gupta.ai@gmail.com" commit -m "Deploy static demo"
git -c $header push
Write-Host "Done. Your demo: $url"
