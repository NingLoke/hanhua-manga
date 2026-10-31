param(
    [ValidateSet('cpu', 'dml', 'cuda', 'all')]
    [string]$Variant = 'cpu',
    [string]$Python = 'python'
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$variants = if ($Variant -eq 'all') { @('cpu', 'dml', 'cuda') } else { @($Variant) }

& $Python -c "import sys; assert sys.version_info >= (3,12), 'Python 3.12 or newer required'"
if ($LASTEXITCODE -ne 0) { throw '请用 -Python 指定 Python 3.12 或更新版本。' }

foreach ($name in $variants) {
    $envPath = Join-Path $projectRoot "envs\$name"
    $exe = Join-Path $envPath 'Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $exe)) {
        & $Python -m venv $envPath
        if ($LASTEXITCODE -ne 0) { throw "创建 $name 环境失败" }
    }
    & $exe -X utf8 -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "更新 $name 的 pip 失败" }
    & $exe -X utf8 -m pip install -e "${projectRoot}[$name,dev,build]"
    if ($LASTEXITCODE -ne 0) { throw "安装 $name 依赖失败" }
    & $exe -X utf8 -c "import onnxruntime as ort; print(ort.__version__, ort.get_available_providers())"
    if ($LASTEXITCODE -ne 0) { throw "验证 $name 环境失败" }
}
