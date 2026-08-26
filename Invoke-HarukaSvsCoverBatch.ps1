param(
    [Parameter(Mandatory = $true)][string]$InputRoot,
    [ValidateSet('balanced', 'quality')][string]$Preset = 'balanced',
    [ValidateSet('prep', 'status', 'resume')][string]$Through = 'prep',
    [string]$JobId,
    [string]$OutputRoot = 'D:\语音模型\Haruka-SVS-Covers',
    [switch]$UseTta
)

# 主脚本只编排 C 盘源码与 D 盘资源路径，不覆盖旧版 cover-prep。
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $projectRoot 'coverprep_env\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { $python = (Get-Command python -ErrorAction Stop).Source }

$command = if ($Through -eq 'status') { 'status' } elseif ($Through -eq 'resume') { 'resume' } else { 'batch' }
$arguments = @('-m', 'coverprep.v3_cli', $command)
if ($command -eq 'status') { $arguments += @('--output-root', $OutputRoot) } else { $arguments += @('--input-root', $InputRoot, '--output-root', $OutputRoot, '--preset', $Preset) }
if ($JobId) { $arguments += @('--job-id', $JobId) }
if ($UseTta -and $command -eq 'batch') { $arguments += '--use-tta' }
& $python @arguments
exit $LASTEXITCODE
