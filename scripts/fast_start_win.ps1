[CmdletBinding()]
param(
    [switch]$CheckOnly,
    [switch]$SkipBuild,
    [switch]$Rebuild,
    [switch]$SkipTest,
    [switch]$SmokeMlip,
    [string]$Arch = "",
    [string]$CudaIndex = "",
    [switch]$NoPrompt,
    [string]$VenvDir = ".venv-mlip"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot

function Write-Step([string]$Msg) {
    Write-Host "`n==> $Msg" -ForegroundColor Cyan
}

function Write-Ok([string]$Msg) {
    Write-Host "  [OK] $Msg" -ForegroundColor Green
}

function Write-WarnLine([string]$Msg) {
    Write-Host "  [WARN] $Msg" -ForegroundColor Yellow
}

function Write-Miss([string]$Msg) {
    Write-Host "  [MISSING] $Msg" -ForegroundColor Red
}

function Ensure-DirInPath([string]$DirPath) {
    if ([string]::IsNullOrWhiteSpace($DirPath) -or -not (Test-Path $DirPath)) {
        return
    }
    $parts = $env:PATH -split ";"
    if ($parts -notcontains $DirPath) {
        $env:PATH = "$DirPath;$env:PATH"
    }
}

function Refresh-ProcessPath {
    $machine = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $user = [Environment]::GetEnvironmentVariable("Path", "User")
    $combined = @($machine, $user) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    $env:PATH = ($combined -join ";")
}

function Confirm-Or-Abort([string]$PromptText) {
    if ($NoPrompt) { return $true }
    $ans = Read-Host "$PromptText [Y/n]"
    return ([string]::IsNullOrWhiteSpace($ans) -or $ans.Trim().ToLower().StartsWith("y"))
}

function Install-ViaWinget([string]$WingetId, [string]$ExtraOverride = "") {
    $wg = Get-Command winget.exe -ErrorAction SilentlyContinue
    if (-not $wg) {
        throw "winget.exe is not available to install '$WingetId'. Please install it manually and re-run fast_start.bat."
    }
    $args = @("install", "-e", "--id", $WingetId, "--accept-package-agreements", "--accept-source-agreements")
    if (-not [string]::IsNullOrWhiteSpace($ExtraOverride)) {
        $args += @("--override", $ExtraOverride)
    }
    Write-Host "  Running: winget $($args -join ' ')"
    & winget.exe @args
    if ($LASTEXITCODE -ne 0) {
        throw "winget install '$WingetId' exited with code $LASTEXITCODE."
    }
    Refresh-ProcessPath
}

function Find-Nvcc {
    $cmd = Get-Command nvcc.exe -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $cudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if (Test-Path $cudaRoot) {
        $dirs = Get-ChildItem $cudaRoot -Directory | Sort-Object Name -Descending
        foreach ($d in $dirs) {
            $cand = Join-Path $d.FullName "bin\nvcc.exe"
            if (Test-Path $cand) {
                Ensure-DirInPath (Split-Path $cand -Parent)
                return $cand
            }
        }
    }
    return $null
}

function Find-VcVars64 {
    if (-not [string]::IsNullOrWhiteSpace($env:VCVARS64) -and (Test-Path $env:VCVARS64)) {
        return $env:VCVARS64
    }
    $candidates = @(
        "${env:ProgramFiles}\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
        "${env:ProgramFiles}\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
        "${env:ProgramFiles}\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat",
        "${env:ProgramFiles}\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat",
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat",
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2019\Professional\VC\Auxiliary\Build\vcvars64.bat",
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2019\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) { return $c }
    }
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $p = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
        if ($p) {
            $vc = Join-Path ($p | Select-Object -First 1) "VC\Auxiliary\Build\vcvars64.bat"
            if (Test-Path $vc) { return $vc }
        }
    }
    return $null
}

function Find-SystemPython310Plus {
    $candidates = New-Object System.Collections.Generic.List[string]
    $wherePy = Get-Command python.exe -All -ErrorAction SilentlyContinue
    foreach ($item in $wherePy) {
        if ($item.Source -notlike "*\WindowsApps\*") {
            $candidates.Add($item.Source)
        }
    }
    foreach ($ver in @("313", "312", "311", "310")) {
        foreach ($base in @("C:\Python$ver\python.exe", "$env:LocalAppData\Programs\Python\Python$ver\python.exe", "$env:ProgramFiles\Python$ver\python.exe")) {
            if (Test-Path $base) { $candidates.Add($base) }
        }
    }
    foreach ($py in ($candidates | Select-Object -Unique)) {
        try {
            $verStr = & $py -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')" 2>$null
            if ($LASTEXITCODE -eq 0 -and $verStr) {
                $parts = $verStr.Trim().Split(".")
                if ([int]$parts[0] -eq 3 -and [int]$parts[1] -ge 10) {
                    return @{ Path = $py; Version = $verStr.Trim() }
                }
            }
        } catch {}
    }
    return $null
}

Write-Host "=== cuda-mof-gcmc Windows Fast-Start ===" -ForegroundColor White
Write-Host "Workspace: $RepoRoot"

$missingItems = New-Object System.Collections.Generic.List[string]

# 1. GPU Check
Write-Step "1/6 Checking NVIDIA GPU & Driver"
$smiCmd = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
if (-not $smiCmd) {
    $fallbackSmi = "C:\Windows\System32\nvidia-smi.exe"
    if (Test-Path $fallbackSmi) { $smiCmd = Get-Command $fallbackSmi }
}
if (-not $smiCmd) {
    Write-Miss "nvidia-smi.exe not found. Please install NVIDIA GPU drivers first."
    exit 1
}
$gpuRaw = (& $smiCmd.Source --query-gpu=name,compute_cap,memory.total --format=csv,noheader 2>$null | Select-Object -First 1)
if (-not $gpuRaw) {
    Write-Miss "No active NVIDIA GPU reported by nvidia-smi."
    exit 1
}
$gpuParts = $gpuRaw.Split(",") | ForEach-Object { $_.Trim() }
$gpuName = $gpuParts[0]
$ccRaw = $gpuParts[1]
$gpuMem = if ($gpuParts.Count -ge 3) { $gpuParts[2] } else { "unknown" }
$detectedArch = "sm_" + ($ccRaw -replace "\.", "")
if (-not [string]::IsNullOrWhiteSpace($Arch)) {
    $env:CUDA_ARCH = $Arch
} elseif (-not [string]::IsNullOrWhiteSpace($env:CUDA_ARCH)) {
    # preserve caller override
} else {
    $env:CUDA_ARCH = $detectedArch
}
Write-Ok "$gpuName | compute $ccRaw -> CUDA_ARCH=$($env:CUDA_ARCH) | VRAM $gpuMem"

# 2. CUDA Toolkit Check
Write-Step "2/6 Checking CUDA Toolkit (nvcc)"
$nvccPath = Find-Nvcc
$cudaVerText = ""
if ($nvccPath) {
    $nvccOut = (& $nvccPath --version 2>$null) -join " "
    if ($nvccOut -match "release\s+([0-9]+\.[0-9]+)") { $cudaVerText = $Matches[1] }
    Write-Ok "nvcc ($cudaVerText) at $nvccPath"
} else {
    Write-Miss "CUDA Toolkit (nvcc.exe) not found."
    if ($CheckOnly) {
        $missingItems.Add("CUDA Toolkit (Nvidia.CUDA)")
    } elseif (Confirm-Or-Abort "Install NVIDIA CUDA Toolkit via winget (Nvidia.CUDA)?") {
        Install-ViaWinget "Nvidia.CUDA"
        $nvccPath = Find-Nvcc
        if (-not $nvccPath) { throw "CUDA Toolkit installed, but nvcc.exe was not found on PATH yet. Open a new terminal and re-run fast_start.bat." }
        Write-Ok "Installed nvcc at $nvccPath"
    } else {
        throw "CUDA Toolkit is required."
    }
}

# 3. MSVC Build Tools Check
Write-Step "3/6 Checking MSVC C++ Build Tools"
$vcVars = Find-VcVars64
if ($vcVars) {
    $env:VCVARS64 = $vcVars
    Write-Ok "Found vcvars64.bat at $vcVars"
} else {
    Write-Miss "Visual Studio 2022 or 2019 C++ Build Tools (vcvars64.bat) not found."
    if ($CheckOnly) {
        $missingItems.Add("MSVC C++ Build Tools (Microsoft.VisualStudio.2022.BuildTools)")
    } elseif (Confirm-Or-Abort "Install VS 2022 C++ Build Tools via winget?") {
        Install-ViaWinget "Microsoft.VisualStudio.2022.BuildTools" "--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
        $vcVars = Find-VcVars64
        if (-not $vcVars) { throw "MSVC Build Tools installation finished, but vcvars64.bat was not found." }
        $env:VCVARS64 = $vcVars
        Write-Ok "Installed vcvars64.bat at $vcVars"
    } else {
        throw "MSVC C++ Build Tools are required."
    }
}

# 4. Python & .venv-mlip Check
Write-Step "4/6 Checking Python & MLIP Environment ($VenvDir)"
$venvPath = if ([System.IO.Path]::IsPathRooted($VenvDir)) { $VenvDir } else { Join-Path $RepoRoot $VenvDir }
$venvPy = Join-Path $venvPath "Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    $sysPy = Find-SystemPython310Plus
    if (-not $sysPy) {
        Write-Miss "Python >= 3.10 not found."
        if ($CheckOnly) {
            $missingItems.Add("Python >= 3.10 + $VenvDir")
        } elseif (Confirm-Or-Abort "Install Python 3.12 via winget?") {
            Install-ViaWinget "Python.Python.3.12"
            $sysPy = Find-SystemPython310Plus
            if (-not $sysPy) { throw "Python installed, please re-open terminal and re-run fast_start.bat." }
        } else {
            throw "Python >= 3.10 is required."
        }
    }
    if ($sysPy -and -not $CheckOnly) {
        Write-Host "  Creating venv at $venvPath using $($sysPy.Path) (Python $($sysPy.Version))..."
        & $sysPy.Path -m venv $venvPath
        if ($LASTEXITCODE -ne 0) { throw "Failed to create venv at $venvPath." }
        & $venvPy -m pip install --upgrade pip
    }
}

if (Test-Path $venvPy) {
    $probeCode = @'
import importlib, json
out = {"torch_ok": False, "torch_cuda": False, "torch_ver": "", "missing": []}
try:
    import torch
    out["torch_ok"] = True
    out["torch_ver"] = getattr(torch, "__version__", "")
    out["torch_cuda"] = bool(torch.cuda.is_available())
except Exception:
    pass
reqs = [
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("ase", "ase"),
    ("mace", "mace-torch"),
    ("torch_dftd", "torch-dftd"),
    ("e3nn", "e3nn"),
    ("matscipy", "matscipy"),
    ("matplotlib", "matplotlib"),
    ("pytest", "pytest>=8.0"),
]
for mod_name, pkg_spec in reqs:
    try:
        importlib.import_module(mod_name)
    except Exception:
        out["missing"].append(pkg_spec)
print(json.dumps(out))
'@
    $probeRaw = $probeCode | & $venvPy -
    if ($LASTEXITCODE -ne 0) { throw "Failed to probe Python packages inside $venvPy." }
    $probe = $probeRaw | ConvertFrom-Json

    if ($probe.torch_ok -and $probe.torch_cuda) {
        Write-Ok "PyTorch $($probe.torch_ver) (CUDA enabled)"
    } else {
        Write-Miss "CUDA-enabled PyTorch missing in $VenvDir (torch_ok=$($probe.torch_ok), cuda=$($probe.torch_cuda))."
        if ($CheckOnly) {
            $missingItems.Add("CUDA PyTorch in $VenvDir")
        } else {
            $idx = $CudaIndex
            if ([string]::IsNullOrWhiteSpace($idx)) {
                $ccNum = [double]::Parse($ccRaw, [System.Globalization.CultureInfo]::InvariantCulture)
                if ($ccNum -ge 12.0 -or ($cudaVerText -and [double]::Parse($cudaVerText, [System.Globalization.CultureInfo]::InvariantCulture) -ge 12.8)) {
                    $idx = "cu128"
                } else {
                    $idx = "cu124"
                }
            }
            Write-Host "  Installing CUDA PyTorch from https://download.pytorch.org/whl/$idx ..."
            & $venvPy -m pip install torch --index-url "https://download.pytorch.org/whl/$idx"
            if ($LASTEXITCODE -ne 0) { throw "Failed to install CUDA PyTorch ($idx)." }
            Write-Ok "Installed CUDA PyTorch ($idx)"
        }
    }

    if ($probe.missing.Count -eq 0) {
        Write-Ok "All MLIP & pytest Python packages present."
    } else {
        $missList = @($probe.missing)
        Write-Miss "Missing Python packages: $($missList -join ', ')"
        if ($CheckOnly) {
            $missingItems.Add("Python packages: $($missList -join ', ')")
        } else {
            Write-Host "  Installing missing packages..."
            & $venvPy -m pip install @missList
            if ($LASTEXITCODE -ne 0) { throw "Failed to install missing Python packages." }
            Write-Ok "Installed: $($missList -join ', ')"
        }
    }
}

# 5. Build Core Binaries
Write-Step "5/6 Checking CUDA Binaries (build\mof_gcmc_cli.exe & build\mof_gcmc_proposer.dll)"
$cliExe = Join-Path $RepoRoot "build\mof_gcmc_cli.exe"
$propDll = Join-Path $RepoRoot "build\mof_gcmc_proposer.dll"
$needBuild = $Rebuild -or -not (Test-Path $cliExe) -or -not (Test-Path $propDll)
if (-not $needBuild) {
    $oldestBin = [Math]::Min((Get-Item $cliExe).LastWriteTimeUtc.Ticks, (Get-Item $propDll).LastWriteTimeUtc.Ticks)
    $srcItems = @(Get-ChildItem (Join-Path $RepoRoot "include") -File) + @(Get-ChildItem (Join-Path $RepoRoot "src") -File)
    $newestSrc = ($srcItems | Measure-Object -Property LastWriteTimeUtc -Maximum).Maximum
    if ($newestSrc -and $newestSrc.Ticks -gt $oldestBin) {
        $needBuild = $true
        Write-Host "  Source files are newer than existing binaries; rebuild required."
    }
}

if (-not $needBuild) {
    Write-Ok "Core binaries are up-to-date."
} elseif ($CheckOnly -or $SkipBuild) {
    if ($CheckOnly) {
        Write-Miss "Core binaries missing or outdated (run 'cmd /c fast_start.bat' to build)."
        $missingItems.Add("Core CUDA build")
    } else {
        Write-WarnLine "Skipping build (-SkipBuild)."
    }
} else {
    Write-Host "  Building core binaries via build.bat core (CUDA_ARCH=$($env:CUDA_ARCH))..."
    & cmd.exe /c "build.bat core"
    if ($LASTEXITCODE -ne 0) { throw "build.bat core failed with exit code $LASTEXITCODE." }
    Write-Ok "Built mof_gcmc_cli.exe and mof_gcmc_proposer.dll"
}

if ($CheckOnly) {
    Write-Step "CheckOnly Summary"
    if ($missingItems.Count -eq 0) {
        Write-Ok "All Windows single-GPU prerequisites and builds are ready!"
        exit 0
    } else {
        Write-Miss "Missing items: $($missingItems -join '; ')"
        exit 2
    }
}

# 6. Verification & Quickstart Commands
if (-not $SkipTest) {
    Write-Step "6/6 Running Self-Checks"
    Write-Host "  [a] Running Pol-SCF unit verification (mof_gcmc_cli.exe --test-scf)..."
    & $cliExe --test-scf | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "mof_gcmc_cli.exe --test-scf failed." }
    Write-Ok "mof_gcmc_cli.exe --test-scf passed."

    Write-Host "  [b] Running pytest classical regression suite (-m 'not mlip')..."
    & $venvPy -m pytest -v -m "not mlip" | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "pytest failed." }
    Write-Ok "All non-MLIP pytest tests passed."

    if ($SmokeMlip) {
        Write-Host "  [c] Running live MLIP smoke test (IRMOF-1 + CH4, MACE small)..."
        $tmpSmoke = Join-Path $RepoRoot "build\_tmp_faststart_smoke.json"
        if (Test-Path $tmpSmoke) { Remove-Item -Force $tmpSmoke }
        & $venvPy -u scripts\run_mlip_gcmc.py `
            --cif data\IRMOF-1.cif --supercell 1 1 1 --temp 298 --pressures 5.0 `
            --guest ch4 --proposer uff --charge-mode auto `
            --model small --dispersion off --boxes 2 --k-steps 15 --pilot-cycles 0 --eq-cycles 2 --prod-cycles 4 `
            --out-json $tmpSmoke | Out-Host
        $rc = $LASTEXITCODE
        if (Test-Path $tmpSmoke) { Remove-Item -Force $tmpSmoke }
        if ($rc -ne 0) { throw "Live MLIP smoke test failed with code $rc." }
        Write-Ok "Live MLIP smoke test passed."
    }
}

Write-Host "`n=== Fast-Start Complete! Ready for Single-GPU Runs ===" -ForegroundColor Green
Write-Host @"

Copy-pasteable single-GPU examples:

1) Classical / Pol-SCF GPU GCMC (seconds):
   .\build\mof_gcmc_cli.exe --config configs\mg_mof74_co2_298K_uff.json

2) Any rigid MOF CIF + built-in guest MLIP-GCMC (auto spacegroup + QEq):
   .\.venv-mlip\Scripts\python.exe -u scripts\run_mlip_gcmc.py ``
     --cif data\IRMOF-1.cif --supercell 1 1 1 --temp 298 --pressures 5.0 ``
     --guest ch4 --proposer uff --charge-mode auto --model medium --dispersion on --xc pbe ``
     --boxes 4 --k-steps 25 --eq-cycles 120 --prod-cycles 320 ``
     --out-json build\irmof1_ch4.json --out-csv build\irmof1_ch4.csv

3) Resume an interrupted isotherm without recomputing finished pressures:
   (add --resume to the same run_mlip_gcmc.py command)
"@
