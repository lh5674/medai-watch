$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$temporaryRoot = Join-Path $projectRoot ".runtime\tmp"
$pipCacheRoot = Join-Path $projectRoot ".pip-cache"
New-Item -ItemType Directory -Force -Path $temporaryRoot, $pipCacheRoot | Out-Null

$env:TEMP = $temporaryRoot
$env:TMP = $temporaryRoot
$env:PIP_CACHE_DIR = $pipCacheRoot

$pythonExecutable = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonExecutable -PathType Leaf)) {
    throw "未找到项目虚拟环境。请先运行 .\scripts\setup_e_drive.ps1"
}

Set-Location $projectRoot
& $pythonExecutable -m streamlit run app.py @args
