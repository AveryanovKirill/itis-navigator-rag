param([Parameter(ValueFromRemainingArguments=$true)][string[]]$CliArgs)
$ErrorActionPreference = 'Stop'
$venvPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $venvPython)) { throw 'Нет .venv. Сначала выполните setup.ps1 или команды установки из README.' }
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONUTF8='1'
Push-Location $PSScriptRoot
try {
    if (-not $CliArgs) { & $venvPython -m src.cli --help }
    else { & $venvPython -m src.cli @CliArgs }
    $runCode=$LASTEXITCODE
} finally { Pop-Location }
exit $runCode
