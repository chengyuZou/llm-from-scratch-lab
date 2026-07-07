$repoRoot = Split-Path -Parent $PSScriptRoot

Set-Location $repoRoot
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = Join-Path $repoRoot "src"

if ($env:LLM_LAB_CONDA_ENV) {
    conda run -n $env:LLM_LAB_CONDA_ENV python -m llm_lab.day03_byte_bpe.run_demo
} else {
    python -m llm_lab.day03_byte_bpe.run_demo
}
