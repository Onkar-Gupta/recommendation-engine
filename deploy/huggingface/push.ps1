# Publishes the Space. Run from anywhere:  .\deploy\huggingface\push.ps1
# Needs your Hugging Face token in the HF_TOKEN environment variable (set it in your own
# PowerShell window first, never in a file):   $env:HF_TOKEN = "hf_..."
# Create the empty Space on huggingface.co first (SDK: Docker, Blank, Public).
param(
    [string]$User = "Onkar-Gupta",
    [string]$Space = "recommendation-engine"
)

if (-not $env:HF_TOKEN) { throw 'Set the token first:  $env:HF_TOKEN = "hf_..."' }

$repo = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$work = Join-Path $env:TEMP "hf-space-$Space"
if (Test-Path $work) { Remove-Item $work -Recurse -Force }

$auth = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes("${User}:$($env:HF_TOKEN)"))
$header = "http.extraHeader=Authorization: Basic $auth"
$url = "https://huggingface.co/spaces/$User/$Space"

git -c $header clone $url $work
if ($LASTEXITCODE -ne 0) { throw "Clone failed. Does the Space exist, and is the token a Write token?" }

# Replace the Space's files with the ones it needs. A normal commit and push, never a force push.
Get-ChildItem $work -Force | Where-Object { $_.Name -ne ".git" } | Remove-Item -Recurse -Force
Copy-Item (Join-Path $repo "src") (Join-Path $work "src") -Recurse
Get-ChildItem (Join-Path $work "src") -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
New-Item -ItemType Directory (Join-Path $work "scripts") | Out-Null
Copy-Item (Join-Path $repo "scripts\download_data.py") (Join-Path $work "scripts")
Copy-Item (Join-Path $repo "config.yaml"), (Join-Path $repo "requirements.txt") $work
Copy-Item (Join-Path $PSScriptRoot "Dockerfile"), (Join-Path $PSScriptRoot "README.md") $work

Set-Location $work
git add -A
git -c user.name="Onkar Gupta" -c user.email="onkar.gupta.ai@gmail.com" commit -m "Deploy recommendation engine"
git -c $header push
Write-Host "Done. Watch the build at $url"
