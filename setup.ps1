param(
    [string]$Python = '',
    [switch]$FetchOfficial,
    [switch]$SkipDependencies
)
$ErrorActionPreference = 'Stop'
$taskRoot = $PSScriptRoot
Push-Location $taskRoot
try {
    $env:PYTHONIOENCODING = 'utf-8'
    $env:PYTHONUTF8 = '1'
    $venvPython = Join-Path $taskRoot '.venv/Scripts/python.exe'
    if (-not (Test-Path -LiteralPath $venvPython)) {
        if ($Python) {
            & $Python -m venv (Join-Path $taskRoot '.venv')
        } elseif (Get-Command py -ErrorAction SilentlyContinue) {
            $selected = $null
            foreach ($version in @('-3.12', '-3.13', '-3.11')) {
                & py $version -c 'import sys; assert sys.version_info >= (3,11)' 2>$null
                if ($LASTEXITCODE -eq 0) { $selected=$version; break }
            }
            if (-not $selected) { throw 'Установите Python 3.12 или 3.13 с python.org, затем повторите setup.ps1.' }
            & py $selected -m venv (Join-Path $taskRoot '.venv')
        } else {
            throw 'Не найден Python Launcher py. Установите Python либо передайте -Python путь-к-python.exe.'
        }
        if ($LASTEXITCODE -ne 0) { throw 'Не удалось создать локальное виртуальное окружение.' }
    }
    if (-not $SkipDependencies) {
        & $venvPython -m pip install --cache-dir (Join-Path $taskRoot '.cache/pip') -r requirements-lock.txt
        if ($LASTEXITCODE -ne 0) { throw 'Не удалось установить зависимости. Проверьте интернет и версию Python; системные пакеты не изменялись.' }
        & $venvPython -m pip check
        if ($LASTEXITCODE -ne 0) { throw 'Обнаружена несовместимость Python зависимостей.' }
    }
    if (-not (Test-Path -LiteralPath '.env')) { Copy-Item -LiteralPath '.env.example' -Destination '.env' }
    if ($FetchOfficial) { & $venvPython -m src.bootstrap --fetch-official }
    else { & $venvPython -m src.bootstrap }
    if ($LASTEXITCODE -ne 0) { throw 'Подготовка данных/индекса не завершена. Следуйте сообщению выше; при отсутствии сети сохраните разрешённые UTF-8 документы вручную.' }
    Write-Host 'Готово. Установите/запустите Ollama, выполните ollama pull qwen3:4b и .\run.ps1 doctor.'
} finally { Pop-Location }
