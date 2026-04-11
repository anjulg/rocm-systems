<#
.SYNOPSIS
    Deploy the HRR proxy DLL alongside a target application.

.DESCRIPTION
    1. Renames amdhip64_7.dll -> amdhip64_7_orig.dll in the ROCm bin directory
       (or a custom target directory).
    2. Copies the built proxy amdhip64_7.dll into that same directory.

    After deployment every process that loads amdhip64_7.dll from that
    directory will pick up the proxy.  The proxy forwards all un-intercepted
    HIP symbols to amdhip64_7_orig.dll via the forwarder entries in the .def
    file, so normal HIP workloads are unaffected.

    To undo, run with -Restore.

.PARAMETER ProxyDll
    Path to the built proxy amdhip64_7.dll
    (default: .\build\amdhip64_7.dll relative to this script).

.PARAMETER TargetDir
    Directory that contains the real amdhip64_7.dll to replace.
    Defaults to %ROCM_PATH%\bin.

.PARAMETER Restore
    Rename amdhip64_7_orig.dll back to amdhip64_7.dll (undo deployment).

.EXAMPLE
    # Deploy using ROCm from ROCM_PATH
    .\deploy_proxy.ps1 -ProxyDll .\build\amdhip64_7.dll

    # Deploy into an app-local directory
    .\deploy_proxy.ps1 -ProxyDll .\build\amdhip64_7.dll `
                       -TargetDir C:\MyApp

    # Undo
    .\deploy_proxy.ps1 -Restore -TargetDir C:\MyApp
#>
param(
    [string]$ProxyDll,

    [string]$TargetDir,

    [switch]$Restore
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# Resolve TargetDir
# ---------------------------------------------------------------------------
if (-not $TargetDir) {
    if ($env:ROCM_PATH) {
        $TargetDir = Join-Path $env:ROCM_PATH "bin"
    } else {
        Write-Error "TargetDir not specified and ROCM_PATH is not set.`nPass -TargetDir <path> explicitly."
        exit 1
    }
}

if (-not (Test-Path $TargetDir)) {
    Write-Error "TargetDir does not exist: $TargetDir"
    exit 1
}

$realDll  = Join-Path $TargetDir "amdhip64_7.dll"
$origDll  = Join-Path $TargetDir "amdhip64_7_orig.dll"

# ---------------------------------------------------------------------------
# Restore mode: put the original back
# ---------------------------------------------------------------------------
if ($Restore) {
    if (-not (Test-Path $origDll)) {
        Write-Error "amdhip64_7_orig.dll not found in $TargetDir — nothing to restore."
        exit 1
    }
    if (Test-Path $realDll) {
        Write-Host "Removing proxy:   $realDll"
        Remove-Item $realDll -Force
    }
    Write-Host "Restoring original: $origDll -> $realDll"
    Rename-Item $origDll $realDll
    Write-Host "Restored."
    exit 0
}

# ---------------------------------------------------------------------------
# Deploy mode
# ---------------------------------------------------------------------------
if (-not $ProxyDll) {
    $scriptDir = Split-Path $MyInvocation.MyCommand.Path
    $ProxyDll  = Join-Path $scriptDir "build\amdhip64_7.dll"
}
if (-not (Test-Path $ProxyDll)) {
    Write-Error "Proxy DLL not found: $ProxyDll`nBuild it first with: cmake --build build"
    exit 1
}

# Step 1: rename the real DLL
if (Test-Path $origDll) {
    Write-Host "amdhip64_7_orig.dll already exists — skipping rename (proxy was previously deployed)."
} else {
    if (-not (Test-Path $realDll)) {
        Write-Error "amdhip64_7.dll not found in $TargetDir"
        exit 1
    }
    Write-Host "Renaming: $realDll -> $origDll"
    Rename-Item $realDll $origDll
}

# Step 2: copy the proxy into place
Write-Host "Installing proxy: $ProxyDll -> $realDll"
Copy-Item $ProxyDll $realDll -Force

Write-Host ""
Write-Host "Deployed.  Recording is OFF by default."
Write-Host "To record a session:"
Write-Host '  $env:HRR_RECORD = "1"'
Write-Host '  $env:HRR_OUTPUT = ".\capture.hrr"'
Write-Host "  .\YourApp.exe"
Write-Host ""
Write-Host "To undo: .\deploy_proxy.ps1 -Restore -TargetDir `"$TargetDir`""
