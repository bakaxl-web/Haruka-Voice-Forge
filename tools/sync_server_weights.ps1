[CmdletBinding()]
param(
    [Alias("Host")]
    [Parameter(Mandatory = $true)]
    [string]$ServerHost,

    [Parameter(Mandatory = $true)]
    [string]$RemotePath,

    [Parameter(Mandatory = $true)]
    [string]$RunId,

    [Parameter(Mandatory = $true)]
    [string]$RegistryRoot,

    [string]$ScpExecutable = "scp.exe"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-FileSha256 {
    param([Parameter(Mandatory = $true)][string]$Path)

    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Copy-VerifiedTree {
    param(
        [Parameter(Mandatory = $true)][string]$SourceRoot,
        [Parameter(Mandatory = $true)][string]$DestinationRoot
    )

    $files = @(Get-ChildItem -LiteralPath $SourceRoot -File -Recurse)
    if ($files.Count -eq 0) {
        throw "服务器路径没有下载到文件: $SourceRoot"
    }

    $copied = 0
    foreach ($file in $files) {
        $relative = [IO.Path]::GetRelativePath($SourceRoot, $file.FullName)
        $destination = Join-Path $DestinationRoot $relative
        $parent = Split-Path -Parent $destination
        New-Item -ItemType Directory -Path $parent -Force | Out-Null

        if (Test-Path -LiteralPath $destination -PathType Leaf) {
            $existing = Get-Item -LiteralPath $destination
            if ($existing.Length -ne $file.Length -or (Get-FileSha256 $destination) -ne (Get-FileSha256 $file.FullName)) {
                throw "拒绝覆盖内容不同的已有权重: $destination"
            }
            continue
        }

        Copy-Item -LiteralPath $file.FullName -Destination $destination
        $copied++
    }
    return $copied
}

# RunId 只允许出现在 incoming 的单层目录中，防止远程输入形成路径穿越。
if ($RunId -notmatch "^[A-Za-z0-9][A-Za-z0-9._-]*$") {
    throw "RunId contains unsafe characters: $RunId"
}

$registryRootFull = [IO.Path]::GetFullPath($RegistryRoot)
$incoming = Join-Path $registryRootFull (Join-Path "incoming" $RunId)
$temporary = Join-Path $registryRootFull (".incoming-{0}-{1}" -f $RunId, [Guid]::NewGuid().ToString("N"))
$scpIsPath = Test-Path -LiteralPath $ScpExecutable -PathType Leaf
$scpCommand = Get-Command $ScpExecutable -ErrorAction SilentlyContinue
if (-not $scpIsPath -and $null -eq $scpCommand) {
    throw "找不到 scp 程序: $ScpExecutable"
}

New-Item -ItemType Directory -Path $incoming -Force | Out-Null
New-Item -ItemType Directory -Path $temporary -Force | Out-Null

try {
    $remoteSpec = "{0}:{1}" -f $ServerHost, $RemotePath
    & $ScpExecutable -r $remoteSpec $temporary
    if ($LASTEXITCODE -ne 0) {
        throw "scp 下载失败，退出码: $LASTEXITCODE"
    }

    $downloadRoot = $temporary
    $childFiles = @(Get-ChildItem -LiteralPath $temporary -File)
    $childDirectories = @(Get-ChildItem -LiteralPath $temporary -Directory)
    if ($childFiles.Count -eq 0 -and $childDirectories.Count -eq 1) {
        $downloadRoot = $childDirectories[0].FullName
    }

    $copied = Copy-VerifiedTree -SourceRoot $downloadRoot -DestinationRoot $incoming
    [pscustomobject]@{
        run_id = $RunId
        incoming = $incoming
        remote_path = $RemotePath
        files_copied = $copied
    } | ConvertTo-Json -Compress
}
finally {
    if (Test-Path -LiteralPath $temporary) {
        Remove-Item -LiteralPath $temporary -Recurse -Force
    }
}
