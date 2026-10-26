# Дополнительная интеграция. Все записи — внутри текущего проекта.
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
Push-Location $taskRoot
try {
    if (-not (Get-Command node -ErrorAction SilentlyContinue)) { throw 'Нужен Node.js >=22.13. Основной Python RAG не требует Node.' }
    if (-not (Test-Path 'reference/MCP-Demo/package.json')) {
        git clone https://github.com/GreenSLA/MCP-Demo.git reference/MCP-Demo
        if ($LASTEXITCODE -ne 0) { throw 'Не удалось клонировать MCP-Demo' }
        git -C reference/MCP-Demo checkout --detach b45fbf29884eb6b5e88225440dc36110546c5958
        if ($LASTEXITCODE -ne 0) { throw 'Не удалось выбрать проверенную версию MCP-Demo' }
    }
    Push-Location reference/MCP-Demo
    try {
        npm.cmd ci --ignore-scripts --no-audit --no-fund --cache "$taskRoot/.cache/npm"
        if ($LASTEXITCODE -ne 0) { throw 'Не удалось установить локальные зависимости MCP-Demo' }
        node --test tests/*.test.mjs
        if ($LASTEXITCODE -ne 0) { throw 'Тесты MCP-Demo завершились ошибкой' }
    } finally { Pop-Location }
} finally { Pop-Location }
