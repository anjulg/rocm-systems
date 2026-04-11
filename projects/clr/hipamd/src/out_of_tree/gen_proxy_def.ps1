<#
.SYNOPSIS
    Generate amdhip64_7.def for the HRR proxy DLL.

.DESCRIPTION
    Reads all exports from the real amdhip64_7.dll using dumpbin and writes a
    .def file where:
      - Functions implemented directly in hrr_proxy_win.c are listed as-is
        (the linker picks up the __declspec(dllexport) definitions).
      - Every other export is forwarded to amdhip64_7_orig.dll by the Windows
        loader — no GetProcAddress needed for the forwarded symbols.

.PARAMETER RealDll
    Full path to the real amdhip64_7.dll (before renaming to _orig).

.PARAMETER OutDef
    Output .def file path.  Defaults to amdhip64_7.def in the current directory.

.PARAMETER ForwardTarget
    Name of the renamed real DLL (without .dll).  Defaults to amdhip64_7_orig.

.EXAMPLE
    .\gen_proxy_def.ps1 -RealDll "C:\Program Files\ROCm\6.3\bin\amdhip64_7.dll"
    .\gen_proxy_def.ps1 -RealDll "C:\rocm\bin\amdhip64_7.dll" -OutDef build\amdhip64_7.def
#>
param(
    [Parameter(Mandatory = $true)]
    [string]$RealDll,

    [string]$OutDef = "amdhip64_7.def",

    [string]$ForwardTarget = "amdhip64_7_orig"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# Validate inputs
# ---------------------------------------------------------------------------
if (-not (Test-Path $RealDll)) {
    Write-Error "Real DLL not found: $RealDll"
    exit 1
}

# Locate dumpbin (ships with Visual Studio)
$dumpbin = Get-Command dumpbin -ErrorAction SilentlyContinue
if (-not $dumpbin) {
    # Try common VS locations
    $vsPaths = @(
        "${env:ProgramFiles}\Microsoft Visual Studio\2022\*\VC\Tools\MSVC\*\bin\Hostx64\x64\dumpbin.exe",
        "${env:ProgramFiles}\Microsoft Visual Studio\2019\*\VC\Tools\MSVC\*\bin\Hostx64\x64\dumpbin.exe",
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\*\VC\Tools\MSVC\*\bin\Hostx64\x64\dumpbin.exe",
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2019\*\VC\Tools\MSVC\*\bin\Hostx64\x64\dumpbin.exe"
    )
    foreach ($p in $vsPaths) {
        $found = Resolve-Path $p -ErrorAction SilentlyContinue | Select-Object -Last 1
        if ($found) { $dumpbin = $found.Path; break }
    }
}
if (-not $dumpbin) {
    Write-Error "dumpbin not found.  Run this script from a Visual Studio Developer PowerShell or add VC tools to PATH."
    exit 1
}

# ---------------------------------------------------------------------------
# Functions that hrr_proxy_win.c implements directly — do NOT forward these.
# Keep in sync with the __declspec(dllexport) list in hrr_proxy_win.c.
# ---------------------------------------------------------------------------
$proxied = [System.Collections.Generic.HashSet[string]]@(
    'hipInit',
    'hipMalloc',
    'hipFree',
    'hipMemcpy',
    'hipMemset',
    'hipModuleLoad',
    'hipModuleLoadData',
    'hipModuleUnload',
    'hipModuleGetFunction',
    'hipModuleLaunchKernel',
    'hipExtModuleLaunchKernel',
    '__hipRegisterFatBinary',
    '__hipRegisterFunction',
    'hipLaunchKernel',
    'hipDeviceSynchronize',
    'hipStreamSynchronize'
)

# ---------------------------------------------------------------------------
# Run dumpbin /exports on the real DLL
# ---------------------------------------------------------------------------
Write-Host "Reading exports from: $RealDll"
$raw = & $dumpbin /exports $RealDll 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Error "dumpbin failed (exit $LASTEXITCODE)"
    exit 1
}

# dumpbin output lines look like:
#       1    0 00001000 hipInit
# Some lines are ordinal-only (no name column) — those are skipped.
$namePattern = [regex]'^\s+\d+\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]+\s+(\S+)'
$exports = $raw |
    ForEach-Object {
        $m = $namePattern.Match($_)
        if ($m.Success) { $m.Groups[1].Value }
    } |
    Where-Object { $_ } |
    Sort-Object

if ($exports.Count -eq 0) {
    Write-Error "No named exports found — check that $RealDll is a valid PE DLL."
    exit 1
}

# ---------------------------------------------------------------------------
# Write the .def file
# ---------------------------------------------------------------------------
$outDir = Split-Path $OutDef -Parent
if ($outDir -and -not (Test-Path $outDir)) {
    New-Item -ItemType Directory -Path $outDir | Out-Null
}

$lines = [System.Collections.Generic.List[string]]@()
$lines.Add("LIBRARY amdhip64_7")
$lines.Add("EXPORTS")

$forwardCount = 0
foreach ($fn in $exports) {
    if ($proxied.Contains($fn)) {
        $lines.Add("  $fn")
    } else {
        $lines.Add("  $fn = ${ForwardTarget}.$fn")
        $forwardCount++
    }
}

[System.IO.File]::WriteAllLines($OutDef, $lines, [System.Text.UTF8Encoding]::new($false))

Write-Host "Written: $OutDef"
Write-Host "  Total exports : $($exports.Count)"
Write-Host "  Proxied (impl): $($proxied.Count)"
Write-Host "  Forwarded      : $forwardCount  ->  $ForwardTarget"
