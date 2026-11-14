<#
.SYNOPSIS
    AbstractCore standalone installer for Windows 10/11 (PowerShell 5.1 and 7).

.DESCRIPTION
    One line, no admin rights, no system Python needed:

        powershell -ExecutionPolicy ByPass -c "irm https://abstractframework.ai/core/install.ps1 | iex"

    With options (a script block keeps the parameters):

        & ([scriptblock]::Create((irm https://abstractframework.ai/core/install.ps1))) -Profile gpu

    The Windows twin of scripts/install.sh (its header has the reasoning): AbstractCore alone,
    the abstractcore, abstractcore-config, abstractcore-chat and abstractcore-endpoint commands
    and the apps (summarizer, extractor, judge, intent, deepsearch), no gateway, no services.
    Each step prints its command; -Print shows them and changes nothing. Steps: preflight; uv
    into %USERPROFILE%\.local\bin when missing; Python 3.12 through uv;
    uv tool install --python 3.12 "abstractcore[<profile>]==<version>" (prebuilt wheels only, no
    MSVC Build Tools); `uv tool update-shell` for the user PATH.

    Profiles (-Profile auto|light|gpu; auto takes the first that fits):
      gpu    nvidia-smi answers: abstractcore[gpu] with PyTorch's CUDA build for the driver
             (580+ and compute capability 7.5+: --torch-backend cu130; 525+: cu126; else PyPI's
             CPU build). On Windows abstractcore[gpu] carries neither vLLM (Linux only) nor
             llama.cpp (a source build on PyPI), so llama.cpp comes from upstream's prebuilt
             builds, as in the framework installer: the CUDA build for the driver first (cu130
             then cu132 for CUDA 13, cu125 for CUDA 12; its first choice goes into the install
             itself, so the uv receipt keeps it), kept only when it loads the way AbstractCore
             loads it and llama_supports_gpu_offload() is True; else the CPU build, with the
             reason. The order is cu130 -> cu132 -> cpu (or cu125 -> cpu) on purpose, with no
             Vulkan step, unlike the framework installer: a GPU build that does not offload falls
             back to the CPU build. Asked for without a card: the same stack, on the processor.
      light  abstractcore with no extra: every remote provider, the built-in tools, media inputs
             and the HTTP server
      apple  Apple Silicon Macs only: refused here
    Windows on ARM is refused: tiktoken, lxml and pandas publish no Windows ARM64 wheels for the
    versions AbstractCore needs, and this installer never starts a compiler.

    Upgrade: run the same line again ("AbstractCore <old> found: upgrading to <new>", or "already
    up to date"); the profile is kept unless -Profile says otherwise. Remove: uv tool uninstall abstractcore

    Options (environment twins AC_PROFILE, AC_PIN, AC_FROM): -Profile auto|light|gpu;
    -Pin VERSION|latest (default: the release this script ships with; latest = the newest on PyPI);
    -From PATH (a checkout, sdist or wheel, for testing); -NoModifyPath; -Print (alias -DryRun);
    -Verbose (every command's full output). The body runs only from Main, called on the last lines.

.EXAMPLE
    .\install.ps1 -Print
#>
[CmdletBinding()]
param(
    # No [ValidateSet]: `irm | iex` evaluates this block as statements; Main validates the value.
    [Alias('Profile')]
    [string]$InstallProfile = '',
    [string]$Pin = '',
    [string]$From = '',
    [switch]$NoModifyPath,
    [Alias('DryRun')]
    [switch]$Print
)

# The release this script ships with = abstractcore/utils/version.py (tests/test_install_scripts.py).
$AcPinDefault = '2.35.2'
$AcPython = '3.12'
$AcScriptUrl = 'https://abstractframework.ai/core/install.ps1'
$AcDocs = 'https://github.com/lpalbou/AbstractCore/blob/main/docs/installation.md#one-line-install'
# Every name of [project.scripts] in pyproject.toml (the test compares them): uv refuses the
# whole install when one of them already exists.
$AcCommands = @('abstractcore', 'abstractcore-config', 'abstractcore-chat', 'abstractcore-endpoint', 'summarizer', 'abstractcore-summarizer', 'extractor', 'abstractcore-extractor', 'judge', 'abstractcore-judge', 'intent', 'abstractcore-intent', 'deepsearch', 'abstractcore-deepsearch')
# llama.cpp (gpu profile): upstream's prebuilt Windows builds, the same pin as install.sh's CPU wheel.
$AcLlamaIndex = 'https://abetlen.github.io/llama-cpp-python/whl'
$AcLlamaPin = '0.3.36'
# The framework installer's llama.cpp check, in a fresh process of the tool's environment (no torch
# import, which would mask missing DLLs): AbstractCore's DLL preparation, import, backend init, GPU
# offload, and the CPU instructions the build needs. One JSON line "ACSMOKE {...}".
$AcLlamaSmoke = @'
import json
r = {"import": False, "offload": False, "error": ""}
try:
    from abstractcore.utils.windows_dll import prepare_llama_cpp_import
    prepare_llama_cpp_import()
    import llama_cpp
    r["import"] = True
    r["version"] = getattr(llama_cpp, "__version__", "")
    init = getattr(llama_cpp, "llama_backend_init", None)
    if callable(init):
        init()
    r["offload"] = bool(llama_cpp.llama_supports_gpu_offload())
    # A build made with AVX-512 imports here but ends the process with an illegal instruction at
    # its first model load on a CPU without it (most desktop Intel CPUs).
    info = llama_cpp.llama_print_system_info()
    info = info.decode("utf-8", "replace") if isinstance(info, bytes) else str(info or "")
    built = [k.strip() for k, _, v in (p.partition("=") for p in info.split("|")) if v.strip() == "1"]
    import ctypes
    pf = {"AVX": 39, "AVX2": 40, "AVX512": 41}
    have = ctypes.windll.kernel32.IsProcessorFeaturePresent
    r["cpu_missing"] = [k for k in built if k in pf and not have(pf[k])]
except BaseException as e:
    r["error"] = "%s: %s" % (type(e).__name__, e)
print("ACSMOKE " + json.dumps(r))
'@

$script:RanAsFile = [bool]$PSCommandPath; $script:DryRun = [bool]$Print; $script:LogFile = ''; $script:WorkDir = ''; $script:StepNo = 0
$script:ShowAll = ($VerbosePreference -eq 'Continue')   # -Verbose (CmdletBinding): show every command's output

function Write-Step([string]$Text) { $script:StepNo++; Write-Host ''; Write-Host "[$($script:StepNo)] $Text" -ForegroundColor White }
function Write-Ok([string]$Text) { Write-Host '  + ' -ForegroundColor Green -NoNewline; Write-Host $Text }
function Write-Info([string]$Text) { Write-Host '  . ' -ForegroundColor Cyan -NoNewline; Write-Host $Text }
function Write-Warn2([string]$Text) { Write-Host '  ! ' -ForegroundColor Yellow -NoNewline; Write-Host $Text }
function Stop-Install([string]$Text) { throw "ACBOOT: $Text" }
function Test-Command([string]$Name) { return [bool](Get-Command $Name -ErrorAction SilentlyContinue) }
function Format-Arg([string]$Value) { if ($Value -match '^[A-Za-z0-9_./:=@,+%\\-]+$') { return $Value }; return "'" + ($Value -replace "'", "''") + "'" }
function Format-Cmd([string[]]$Argv) { return (($Argv | ForEach-Object { Format-Arg $_ }) -join ' ') }
function Get-Http([string]$Url) { try { return [string](Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 20 -ErrorAction Stop -Verbose:$false).Content } catch { return $null } }

# Run a native command: print it, log its output (and show it with -Live), return $true on exit 0.
# A failure stops the install unless -Soft, which warns and returns $false.
function Invoke-Native {
    param([string]$Description, [string[]]$Argv, [switch]$Soft, [switch]$Live, [string]$Shown = '')
    if (-not $Shown) { $Shown = Format-Cmd $Argv }
    Write-Host "  `$ $Shown" -ForegroundColor DarkGray
    if ($script:DryRun) { return $true }
    $exe = $Argv[0]; $rest = @(); if ($Argv.Count -gt 1) { $rest = $Argv[1..($Argv.Count - 1)] }
    $old = $ErrorActionPreference; $ErrorActionPreference = 'Continue'   # PS 5.1 turns native stderr into errors
    try {
        Add-Content -Path $script:LogFile -Value "`r`n`$ $Shown" -Encoding UTF8
        & $exe @rest 2>&1 | ForEach-Object {
            $line = "$_"; Add-Content -Path $script:LogFile -Value $line -Encoding UTF8
            if ($Live -or $script:ShowAll) { Write-Host "    $line" -ForegroundColor DarkGray }
        }
        $code = $LASTEXITCODE
    } catch { Add-Content -Path $script:LogFile -Value "$_" -Encoding UTF8; $code = 1 } finally { $ErrorActionPreference = $old }
    if ($code -eq 0) { return $true }
    if ($Soft) { Write-Warn2 "$Description did not succeed (exit $code; log: $($script:LogFile))"; return $false }
    Get-Content -Path $script:LogFile -Tail 15 | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkGray }
    Stop-Install "$Description failed (exit $code). Log: $($script:LogFile)`nWhat to do: run the installer again (it repairs a half-finished install); if it stops at the same step, report it with that log ($AcDocs)."
}

# The framework installer's Get-NvidiaInfo: Ok, Driver, DriverMajor, Cc (the lowest compute
# capability, $null when the driver cannot report it), Name, Error (why an installed nvidia-smi
# gives no GPU; empty when there is no nvidia-smi at all).
function Get-NvidiaInfo {
    $info = @{ Ok = $false; Driver = ''; DriverMajor = 0; Cc = $null; Name = ''; Error = '' }
    $noAnswer = 'nvidia-smi is installed but does not answer (driver not loaded?)'
    if (-not (Test-Command 'nvidia-smi')) { return $info }
    $old = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    try {
        $withCc = $true
        $lines = @(& nvidia-smi '--query-gpu=driver_version,compute_cap,name' '--format=csv,noheader' 2>$null | ForEach-Object { "$_" })
        if ($LASTEXITCODE -ne 0) {   # drivers too old to report compute_cap: the driver alone
            $withCc = $false; $lines = @(& nvidia-smi '--query-gpu=driver_version,name' '--format=csv,noheader' 2>$null | ForEach-Object { "$_" }) }
        if ($LASTEXITCODE -ne 0) { $info.Error = $noAnswer; return $info }
    } catch { $info.Error = $noAnswer; return $info } finally { $ErrorActionPreference = $old }
    $inv = [Globalization.CultureInfo]::InvariantCulture; $nameAt = if ($withCc) { 2 } else { 1 }
    foreach ($l in ($lines | Where-Object { $_.Trim() })) {
        $parts = $l.Split(',', $nameAt + 1); $major = 0; $cc = 0.0
        if (-not [int]::TryParse(($parts[0].Trim().Split('.')[0]), [ref]$major)) { continue }
        if (-not $info.Driver) { $info.Driver = $parts[0].Trim(); $info.DriverMajor = $major }
        if ($withCc -and $parts.Count -ge 2 -and [double]::TryParse($parts[1].Trim(), [Globalization.NumberStyles]::Float, $inv, [ref]$cc)) {
            if ($null -eq $info.Cc -or $cc -lt $info.Cc) { $info.Cc = $cc }
        }
        if (-not $info.Name -and $parts.Count -gt $nameAt) { $info.Name = $parts[$nameAt].Trim() }
    }
    $info.Ok = [bool]$info.Driver; if (-not $info.Ok) { $info.Error = 'nvidia-smi gave no driver version' }; return $info
}

function Find-Uv([string]$HomeDir) {
    $cmd = Get-Command uv -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    foreach ($d in @($env:UV_INSTALL_DIR, $env:XDG_BIN_HOME, [IO.Path]::Combine($HomeDir, '.local', 'bin'), [IO.Path]::Combine($HomeDir, '.cargo', 'bin'))) {
        if ($d -and (Test-Path -LiteralPath (Join-Path $d 'uv.exe'))) { return (Join-Path $d 'uv.exe') }
    }
    return ''
}
function Get-ToolList([string]$Uv) {  # `uv tool list` lines (none without uv)
    if (-not $Uv) { return @() }
    $old = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    try { return @(& $Uv tool list 2>$null | ForEach-Object { "$_" }) } catch { return @() } finally { $ErrorActionPreference = $old }
}
function Get-ToolVersion([string]$Uv) { foreach ($l in (Get-ToolList $Uv)) { if ($l -match '^abstractcore v(\S+)') { return $Matches[1] } }; return '' }
# Run one smoke script with the tool environment's Python: the parsed JSON, or $null. Always from a
# .py file in the installer's folder, run by path, never `python -c`: Windows PowerShell 5.1 strips
# double quotes inside native arguments (pwsh 7 does not), so inline code holding a `"` would run
# altered and misreport there. tests/test_install_scripts.py rejects any -c argument with a `"`.
function Invoke-Smoke([string]$Python, [string]$Code) {
    if (-not $Python -or -not (Test-Path -LiteralPath $Python)) { return $null }
    $dir = if ($script:WorkDir) { $script:WorkDir } else { [System.IO.Path]::GetTempPath() }
    $file = Join-Path $dir ("smoke-{0}-{1}.py" -f $PID, [guid]::NewGuid().ToString('N'))
    [System.IO.File]::WriteAllText($file, $Code, (New-Object System.Text.UTF8Encoding($false)))
    $old = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    try { $out = @(& $Python $file 2>&1 | ForEach-Object { "$_" }) } catch { $out = @("$_") } finally {
        $ErrorActionPreference = $old
        Remove-Item -LiteralPath $file -Force -ErrorAction SilentlyContinue
    }
    if ($script:LogFile) { Add-Content -Path $script:LogFile -Value (@("`r`n`$ $Python <smoke check>") + $out) -Encoding UTF8 }
    $line = $out | Where-Object { $_ -like 'ACSMOKE *' } | Select-Object -Last 1
    if (-not $line) { return $null }
    try { return ($line.Substring(8) | ConvertFrom-Json) } catch { return $null }
}

function Main {
    $onWindows = ($PSVersionTable.PSEdition -eq 'Desktop') -or ($IsWindows -eq $true)
    if (-not $onWindows -and -not $script:DryRun) {
        Stop-Install 'this installer is for Windows; on macOS and Linux run: curl -LsSf https://abstractframework.ai/core/install.sh | sh'
    }
    $homeDir = if ($env:USERPROFILE) { $env:USERPROFILE } else { $HOME }
    $profileName = $InstallProfile
    if (-not $profileName) { $profileName = if ($env:AC_PROFILE) { $env:AC_PROFILE } else { 'auto' } }
    if (-not $Pin -and $env:AC_PIN) { $Pin = $env:AC_PIN }
    if (-not $From -and $env:AC_FROM) { $From = $env:AC_FROM }
    $arch = if ($env:PROCESSOR_ARCHITEW6432) { $env:PROCESSOR_ARCHITEW6432 } else { $env:PROCESSOR_ARCHITECTURE }

    Write-Host "AbstractCore installer  $(if ($script:DryRun) { '(-Print: preflight only, nothing is installed)' } else { $AcScriptUrl })" -ForegroundColor White
    Write-Step 'Preflight'
    Write-Ok "system: Windows $([Environment]::OSVersion.Version) $arch"
    if ($arch -eq 'ARM64') {
        Stop-Install 'Windows on ARM: tiktoken, lxml and pandas publish no Windows ARM64 wheels for the versions AbstractCore needs, and this installer never starts a compiler. Install an x64 Python 3.12 (Windows 11 runs it under emulation) and use: pip install abstractcore. Nothing was changed.'
    }

    $uv = Find-Uv $homeDir; $toolBin = ''; $toolDir = ''
    if ($uv) { $toolBin = (& $uv tool dir --bin 2>$null | Select-Object -First 1); $toolDir = (& $uv tool dir 2>$null | Select-Object -First 1) }
    if (-not $toolBin) { $toolBin = if ($env:UV_TOOL_BIN_DIR) { $env:UV_TOOL_BIN_DIR } else { [IO.Path]::Combine($homeDir, '.local', 'bin') } }
    if (-not $toolDir) { $toolDir = if ($env:UV_TOOL_DIR) { $env:UV_TOOL_DIR } else { [IO.Path]::Combine($(if ($env:APPDATA) { $env:APPDATA } else { $homeDir }), 'uv', 'tools') } }
    $before = Get-ToolVersion $uv; $prevProfile = ''   # the existing install: its version, and its profile from the uv receipt
    $receipt = [IO.Path]::Combine($toolDir, 'abstractcore', 'uv-receipt.toml')
    if ($before -and (Test-Path -LiteralPath $receipt)) {
        $rc = Get-Content -LiteralPath $receipt -Raw
        if ($rc -match 'name = "abstractcore", extras = \[[^\]]*"(gpu|all-gpu|vllm)"') { $prevProfile = 'gpu' }
        elseif ($rc -match 'name = "abstractcore"' -and $rc -notmatch 'name = "abstractcore", extras = ') { $prevProfile = 'light' } }

    $gpu = Get-NvidiaInfo
    switch ($profileName) {
        'auto' {
            if ($prevProfile) { $profileName = $prevProfile; $why = 'kept from the previous install' }
            elseif ($gpu.Ok) { $profileName = 'gpu'; $why = "nvidia-smi found $(if ($gpu.Name) { $gpu.Name } else { 'a GPU' })" }
            elseif ($gpu.Error) { $profileName = 'light'; $why = "$($gpu.Error), so remote providers and servers only; once the driver works, run the installer again" }
            else { $profileName = 'light'; $why = 'no GPU found, so remote providers and servers only' }
            Write-Ok "profile: $profileName ($why; override with -Profile)"
        }
        'light' { Write-Ok 'profile: light (every remote provider, the tools and the HTTP server; no local engine)' }
        'gpu' {
            if (-not $gpu.Ok) { Write-Warn2 "gpu profile requested, but $(if ($gpu.Error) { $gpu.Error } else { 'there is no nvidia-smi' }): its engines will run on the processor" }
            Write-Ok 'profile: gpu (CUDA engines when a card works, else the processor)'
        }
        'apple' { Stop-Install 'the apple profile is for Apple Silicon Macs; use -Profile light or gpu' }
        default { Stop-Install "unknown profile '$profileName' (expected auto, light or gpu)" }
    }
    $extras = if ($profileName -eq 'gpu') { 'gpu' } else { '' }
    $ggufResult = "not applicable ($profileName profile)"

    # PyTorch's build (gpu profile; the framework installer's Select-GpuStack, without llama.cpp).
    $torchArgs = @()
    if ($profileName -eq 'gpu') {
        $torch = ''
        if ($gpu.Ok -and $gpu.DriverMajor -ge 580 -and $null -ne $gpu.Cc -and $gpu.Cc -ge 7.5) { $torch = 'cu130' }
        elseif ($gpu.Ok -and $gpu.DriverMajor -ge 525) { $torch = 'cu126' }
        if ($torch) {
            $help = if ($uv) { (& $uv tool install --help 2>$null) -join "`n" } else { '--torch-backend' }
            if ($help -match '--torch-backend') { $torchArgs = @('--torch-backend', $torch); Write-Ok "PyTorch: CUDA build $torch (driver $($gpu.Driver))" }
            else { Write-Warn2 "this uv has no --torch-backend: PyTorch's CPU build is installed; update uv (uv self update) and run the installer again" }
        } else { Write-Info "PyTorch: PyPI's CPU build ($(if ($gpu.Ok) { "driver $($gpu.Driver) is older than 525" } else { 'no NVIDIA GPU' }))" }
    }
    # llama.cpp's builds to try, in order (the framework installer's Select-GpuStack): CUDA 13 ->
    # cu130 then cu132 (cu130 is built with AVX-512, cu132 is not); CUDA 12 -> cu125; then cpu.
    # The CUDA builds load cuBLAS/cudart from PyTorch's CUDA build, so without one only cpu.
    $llamaVariants = @(); $llamaCpuWhy = ''
    if ($profileName -eq 'gpu') {
        if ($torchArgs -contains 'cu130') { $llamaVariants = @('cu130', 'cu132', 'cpu') }
        elseif ($torchArgs -contains 'cu126') { $llamaVariants = @('cu125', 'cpu') }
        else {
            $llamaVariants = @('cpu')
            $llamaCpuWhy = if (-not $gpu.Ok) { $(if ($gpu.Error) { $gpu.Error } else { 'no NVIDIA GPU answers nvidia-smi' }) } elseif ($torch) { "this uv has no --torch-backend, so PyTorch is a CPU build" } else { "driver $($gpu.Driver) is older than 525 (CUDA 12)" }
        }
        if ($llamaCpuWhy) { Write-Info "llama.cpp (GGUF): its CPU build ($llamaCpuWhy)" }
        else { Write-Ok "llama.cpp (GGUF): llama-cpp-python $AcLlamaPin, builds tried in order: $($llamaVariants -join ', then ')" }
    }

    if ($From) {
        if (-not (Test-Path -LiteralPath $From)) { Stop-Install "-From: no such file or folder: $From" }
        $From = (Resolve-Path -LiteralPath $From).Path; $target = "(the build from $From)"; $pinWhy = '-From'
    } elseif ($Pin -eq 'latest') {
        $latest = ''; $j = Get-Http 'https://pypi.org/pypi/abstractcore/json'
        if ($j -and $j -match '"version":\s*"([^"]+)"') { $latest = $Matches[1] }
        $target = if ($latest) { $latest } else { 'the newest release on PyPI' }; $pinWhy = '-Pin latest'
    } else {
        if ($Pin) { $pinWhy = '-Pin' } else { $Pin = $AcPinDefault; $pinWhy = 'the release this installer ships with' }; $target = $Pin
    }
    function Get-Spec([string]$Extras) {  # the requirement given to uv
        $x = if ($Extras) { "[$Extras]" } else { '' }
        if ($From) { return "abstractcore$x @ " + [Uri]::new($From, [UriKind]::Absolute).AbsoluteUri }
        if ($Pin -eq 'latest') { return "abstractcore$x" }; return "abstractcore$x==$Pin"
    }
    Write-Ok "package: $(Get-Spec $extras)  (version: $pinWhy)"
    if (-not $before) { Write-Ok "No AbstractCore install found: installing AbstractCore $target" }
    elseif ($From) { Write-Ok "AbstractCore $before found: replacing it with the build from $From (-From)" }
    elseif ($before -eq $target) { Write-Ok "AbstractCore $before found: already up to date (checked, repaired if needed)" }
    elseif ($Pin -ne 'latest' -and $(try { [version]($before -replace '[^0-9.].*$', '') -gt [version]($Pin -replace '[^0-9.].*$', '') } catch { $false })) {
        Write-Ok "AbstractCore $before found: going back to AbstractCore $target (-Pin)"
    } else { Write-Ok "AbstractCore $before found: upgrading to AbstractCore $target" }

    # A command of the same name from another program (often AbstractFramework's gateway tool):
    # uv would refuse the install, so stop and say why.
    if (-not $before) {
        foreach ($n in $AcCommands) {
            $exe = Join-Path $toolBin "$n.exe"
            if (-not (Test-Path -LiteralPath $exe)) { continue }
            $owner = ''; $t = ''
            foreach ($l in (Get-ToolList $uv)) { if ($l -match '^(\S+) v') { $t = $Matches[1] } elseif ($l -eq "- $n") { $owner = $t; break } }
            $msg = "$exe already exists ($(if ($owner) { "from the uv tool '$owner'" } else { "another program's" }))"
            if ($owner -eq 'abstractgateway') { $msg += ': AbstractFramework is installed here and already includes AbstractCore; upgrade it with: powershell -ExecutionPolicy ByPass -c "irm https://abstractframework.ai/install.ps1 | iex". A standalone AbstractCore beside the framework''s gateway tool is not supported (there is no -Force: both would own the same commands)' }
            if ($script:DryRun) { Write-Warn2 "$msg (the real run stops here; nothing is changed)"; break }
            Stop-Install "$msg`nNothing was changed."
        }
    }

    $needMb = if ($profileName -eq 'gpu') { 14000 } else { 1500 }
    try {
        $root = [System.IO.Path]::GetPathRoot($homeDir)
        $freeMb = [int]((New-Object System.IO.DriveInfo($root)).AvailableFreeSpace / 1MB)
        if ($freeMb -lt 500) { Stop-Install "only $freeMb MB free on $root (about $needMb MB needed)" }
        elseif ($freeMb -lt $needMb) { Write-Warn2 "only $freeMb MB free on $root; the $profileName profile needs about $needMb MB (models need more)" }
        else { Write-Ok "disk: $freeMb MB free (about $needMb MB needed, models extra)" }
    } catch { if ("$_" -like 'ACBOOT:*') { throw } }
    if ($null -ne (Get-Http 'https://pypi.org/simple/pip/')) { Write-Ok 'internet: pypi.org reachable' }
    elseif ($script:DryRun) { Write-Warn2 'pypi.org is not reachable (a real run stops here)' }
    else { Stop-Install 'this computer cannot reach pypi.org, where AbstractCore is downloaded (check the connection or the proxy). Nothing was changed.' }
    if (-not $From -and $Pin -ne 'latest' -and $null -eq (Get-Http "https://pypi.org/pypi/abstractcore/$Pin/json")) {
        $m = "abstractcore $Pin is not on PyPI (yet): use -Pin latest for the newest published version"
        if ($script:DryRun) { Write-Warn2 $m } else { Stop-Install "$m. Nothing was changed." }
    }
    $work = [IO.Path]::Combine($(if ($env:LOCALAPPDATA) { $env:LOCALAPPDATA } else { $homeDir }), 'abstractcore', 'installer')
    if (-not $script:DryRun) { New-Item -ItemType Directory -Force -Path $work | Out-Null
        $script:WorkDir = $work; $script:LogFile = Join-Path $work 'install.log'; Set-Content -Path $script:LogFile -Value '' -Encoding UTF8 }

    # --- 2. uv and Python ------------------------------------------------------------------------
    Write-Step 'uv (Python toolchain manager)'
    if ($uv) { Write-Ok "uv found: $uv" }
    else {
        Write-Info 'installing uv from astral.sh into %USERPROFILE%\.local\bin (no admin)'
        # A child PowerShell with a per-process ByPass policy, so the vendor script's `exit` cannot end this one.
        $command = "`$env:UV_NO_MODIFY_PATH='1'; irm https://astral.sh/uv/install.ps1 | iex"
        Invoke-Native -Description 'install uv' -Argv @((Get-Process -Id $PID).Path, '-NoProfile', '-ExecutionPolicy', 'ByPass', '-Command', $command) `
            -Shown "powershell -ExecutionPolicy ByPass -c `"$command`"" | Out-Null
        if ($script:DryRun) { $uv = 'uv' }
        else { $uv = Find-Uv $homeDir; if (-not $uv) { Stop-Install 'uv was installed but cannot be found in %USERPROFILE%\.local\bin' }
            Write-Ok "uv installed: $uv"
            $toolBin = (& $uv tool dir --bin 2>$null | Select-Object -First 1); $toolDir = (& $uv tool dir 2>$null | Select-Object -First 1) }
    }
    Write-Step "Python $AcPython (managed by uv, isolated from any system Python)"
    Invoke-Native -Description "install Python $AcPython" -Argv @($uv, 'python', 'install', $AcPython) | Out-Null

    # --- 3. AbstractCore -------------------------------------------------------------------------
    # Prebuilt wheels only: on Windows x64 every package of these profiles has a wheel for Python
    # 3.12 (aec-audio-processing too); vLLM, llama-cpp-python and stable-diffusion-cpp-python are
    # marked out of abstractcore[gpu] there, so no override is needed (install.sh needs three).
    # llama.cpp is added with --with from upstream's page for one build (--find-links, recorded in
    # the uv receipt), never built (--no-build-package).
    Write-Step 'AbstractCore'
    function Install-Tool([string]$Extras, [string]$Llama = '', [switch]$Soft, [string[]]$More = @()) {
        $argv = @($uv, 'tool', 'install', '--python', $AcPython) + $More
        if ($Extras -eq 'gpu') { $argv += $torchArgs }
        if ($Llama) { $argv += @('--with', "llama-cpp-python==$AcLlamaPin", '--find-links', "$AcLlamaIndex/$Llama/llama-cpp-python/", '--no-build-package', 'llama-cpp-python') }
        if ($From) { $argv += @('--reinstall-package', 'abstractcore') }        # same version number, new code
        elseif ($Pin -eq 'latest') { $argv += '--upgrade' }                     # else uv keeps a satisfying install
        $argv += (Get-Spec $Extras)
        $desc = "install $(Get-Spec $Extras)$(if ($Llama) { " with llama.cpp's $Llama build" })"
        return (Invoke-Native -Description $desc -Argv $argv -Live -Soft:$Soft)
    }
    $llamaInstalled = ''   # the llama.cpp build the install itself included
    if ($llamaVariants.Count) {
        if (Install-Tool $extras -Llama $llamaVariants[0] -Soft) { $llamaInstalled = $llamaVariants[0] }
        else { Write-Warn2 "the install with llama.cpp's $($llamaVariants[0]) build did not succeed: retrying without llama.cpp (it is tried again afterwards)"; Install-Tool $extras | Out-Null }
    } else { Install-Tool $extras | Out-Null }
    $acExe = Join-Path $toolBin 'abstractcore.exe'
    $after = $before
    if (-not $script:DryRun) {
        # Repair: uv can report the tool installed while its command cannot start; reinstall in place.
        & $acExe --help *> $null
        if ($LASTEXITCODE -ne 0) {
            Write-Warn2 'the installed abstractcore does not start (an earlier install was interrupted or damaged): reinstalling it'
            Install-Tool $extras -Llama $llamaInstalled -More @('--reinstall') | Out-Null
            & $acExe --help *> $null
            if ($LASTEXITCODE -ne 0) { Stop-Install "abstractcore still does not start after reinstalling it (log: $($script:LogFile))" }
        }
        $after = Get-ToolVersion $uv
        if (-not $before) { Write-Ok "installed abstractcore $after" }
        elseif ($before -ne $after) { Write-Ok "abstractcore $before -> $after" }
        elseif ($From) { Write-Ok "abstractcore $after reinstalled from $From" }
        else { Write-Ok "abstractcore $after already installed" }
    }

    # --- 3b. llama.cpp: keep the first build that loads ------------------------------------------
    # Swapped with uv pip inside the tool's environment (--no-index: only that build's page);
    # a CUDA build is kept when llama_supports_gpu_offload() is True, the cpu build when it loads;
    # none left behind when no build loads (never a broken import).
    if ($llamaVariants.Count) {
        Write-Step "llama.cpp's build (GGUF)"
        $toolPy = [IO.Path]::Combine($toolDir, 'abstractcore', 'Scripts', 'python.exe')
        $twin = 'from abstractcore.utils.windows_dll import prepare_llama_cpp_import; prepare_llama_cpp_import(); import llama_cpp; print(llama_cpp.llama_supports_gpu_offload())'
        function Get-SwapArgv([string]$V) { return @($uv, 'pip', 'install', '--python', $toolPy, '--no-index', '--find-links', "$AcLlamaIndex/$V/llama-cpp-python/", '--no-deps', '--reinstall-package', 'llama-cpp-python', '--refresh-package', 'llama-cpp-python', "llama-cpp-python==$AcLlamaPin") }
        if ($script:DryRun) {
            Write-Info "builds tried in this order: $($llamaVariants -join ', then '); one is kept when it loads the way AbstractCore loads it (CUDA builds: when llama_supports_gpu_offload() is True), else none"
            if ($llamaVariants.Count -gt 1) { Write-Host "  `$ $(Format-Cmd (Get-SwapArgv $llamaVariants[1]))   # only when the build before it does not work" -ForegroundColor DarkGray }
            Write-Host "  `$ $(Format-Cmd @($toolPy, '-c', $twin))" -ForegroundColor DarkGray
            $ggufResult = "planned: llama-cpp-python $AcLlamaPin, $($llamaVariants -join ' then ') build$(if ($llamaCpuWhy) { " ($llamaCpuWhy)" })"
        } else {
            $current = $llamaInstalled; $notKept = @(); $kept = ''; $where = ''
            foreach ($v in $llamaVariants) {
                if ($current -ne $v) {
                    Write-Info "llama.cpp: installing its $v build"
                    if (-not (Invoke-Native -Description "install llama.cpp's $v build" -Argv (Get-SwapArgv $v) -Soft -Live)) { $notKept += "${v}: did not install"; $current = ''; continue }
                    $current = $v
                }
                Write-Host "  `$ $(Format-Cmd @($toolPy, '-c', $twin))" -ForegroundColor DarkGray
                $r = Invoke-Smoke $toolPy $AcLlamaSmoke
                $cpuMissing = @(if ($r -and $r.cpu_missing) { $r.cpu_missing })
                if ($r -and $r.import -and -not $cpuMissing.Count -and ($v -eq 'cpu' -or $r.offload)) {
                    $kept = $v; $where = if ($r.offload) { 'GPU offload' } else { 'CPU' }
                    Write-Ok "llama.cpp $($r.version) ($v build): loads, $(if ($r.offload) { 'llama_supports_gpu_offload() = True' } else { 'CPU' })"
                    break
                }
                $reason = if (-not $r) { 'no answer from the check' } elseif (-not $r.import) { "does not load: $($r.error)" } elseif ($cpuMissing.Count) { "is built for CPU instructions this processor lacks ($($cpuMissing -join ', '))" } else { 'loads, but reports no GPU offload' }
                Write-Warn2 "llama.cpp $v build $reason$(if ($v -ne $llamaVariants[-1]) { '; trying the next build' })"
                $notKept += "${v}: $reason"
            }
            if ($kept) {
                $ggufResult = "llama-cpp-python $AcLlamaPin ($kept build, $where)$(if ($llamaCpuWhy) { "; CPU because $llamaCpuWhy" })$(if ($notKept.Count) { "; not kept: $($notKept -join '; ')" })"
            } else {
                if ($current) { Invoke-Native -Description 'remove llama.cpp (no build loads here)' -Argv @($uv, 'pip', 'uninstall', '--python', $toolPy, 'llama-cpp-python') -Soft | Out-Null }
                $ggufResult = "skipped: no llama.cpp build loads on this machine ($($notKept -join '; ')); GGUF models run in Ollama or LM Studio"
                Write-Warn2 "llama.cpp: $ggufResult"
            }
        }
    }

    # --- 4. PATH and summary -----------------------------------------------------------------------
    Write-Step 'PATH'
    if (($env:PATH -split [IO.Path]::PathSeparator) -contains $toolBin) { Write-Ok "$toolBin is on PATH" }
    elseif ($NoModifyPath) { Write-Warn2 "$toolBin is not on PATH (-NoModifyPath); add it to your user PATH or use $acExe" }
    elseif (Invoke-Native -Description "add $toolBin to your user PATH" -Argv @($uv, 'tool', 'update-shell') -Soft) {
        Write-Info "open a new terminal for the 'abstractcore' command to be on PATH"
    } else { Write-Info "add $toolBin to your user PATH yourself (Settings > System > About > Advanced system settings > Environment Variables)" }

    Write-Host ''
    Write-Host $(if ($script:DryRun) { 'Plan (-Print: nothing was changed)' } else { 'AbstractCore is installed' }) -ForegroundColor White
    $ver = if ($script:DryRun -or -not $after) { $target } else { $after }
    Write-Host "  version:   $ver$(if (-not $script:DryRun -and $before -and $before -ne $after) { " (was $before)" })"
    Write-Host "  profile:   $profileName ($(Get-Spec $extras))"
    Write-Host "  llama.cpp: $ggufResult"
    Write-Host "  commands:  $acExe, abstractcore-config, abstractcore-chat, abstractcore-endpoint"
    Write-Host '             and the apps summarizer, extractor, judge, intent, deepsearch (each also abstractcore-<app>)'
    if (-not $script:DryRun) { Write-Host "  log:       $($script:LogFile)" }
    Write-Host ''
    Write-Host 'Try it:'
    Write-Host '  abstractcore --help'
    Write-Host '  abstractcore config defaults'
    Write-Host "Docs: $AcDocs"
}

# The caller's $ErrorActionPreference is not ours (`irm | iex` shares the caller's scope; under
# 'Stop', PowerShell 5.1 turns any native stderr line into a terminating error), so Main runs under
# 'Continue', as with -File, and the caller's value comes back after.
$AcCallerErrorAction = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
try {
    Main
} catch {
    Write-Host ''
    Write-Host "ERROR: $("$_" -replace '^ACBOOT: ', '')" -ForegroundColor Red
    if ("$_" -notlike 'ACBOOT:*') { Write-Host $_.ScriptStackTrace -ForegroundColor DarkGray }
    # `exit` would close the window of someone who pasted `irm | iex`; only exit when run as a file.
    if ($script:RanAsFile) { exit 1 }
} finally {
    $ErrorActionPreference = $AcCallerErrorAction
}
