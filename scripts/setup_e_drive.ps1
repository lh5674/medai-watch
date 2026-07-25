$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $projectRoot.StartsWith("E:\", [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "项目必须位于 E 盘，当前路径：$projectRoot"
}

$runtimeRoot = Join-Path $projectRoot ".runtime"
$temporaryRoot = Join-Path $runtimeRoot "tmp"
$pipCacheRoot = Join-Path $projectRoot ".pip-cache"
New-Item -ItemType Directory -Force -Path $temporaryRoot, $pipCacheRoot | Out-Null

$env:TEMP = $temporaryRoot
$env:TMP = $temporaryRoot
$env:PIP_CACHE_DIR = $pipCacheRoot

$basePython = Join-Path $projectRoot ".python314\python.exe"
if (-not (Test-Path -LiteralPath $basePython -PathType Leaf)) {
    throw "未找到 E 盘基础 Python：$basePython"
}

$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    & $basePython -m venv (Join-Path $projectRoot ".venv")
}

& $venvPython -m pip config --site set global.cache-dir $pipCacheRoot
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r (Join-Path $projectRoot "requirements.txt")
& $venvPython -m pip check

Write-Host "MedAI Watch 已安装到：$projectRoot"
Write-Host "Python：$venvPython"
Write-Host "pip 缓存：$pipCacheRoot"
Write-Host "临时目录：$temporaryRoot"
