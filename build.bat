@echo off
setlocal enabledelayedexpansion

REM Auto-detect vcvars64.bat (respect VCVARS64 override if provided)
set "VCVARS_FOUND="
if defined VCVARS64 (
    if exist "%VCVARS64%" set "VCVARS_FOUND=%VCVARS64%"
)
if not defined VCVARS_FOUND (
    for %%V in (
        "%ProgramFiles%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
        "%ProgramFiles%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
        "%ProgramFiles%\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
        "%ProgramFiles%\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
        "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
        "%ProgramFiles(x86)%\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
        "%ProgramFiles(x86)%\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
        "%ProgramFiles(x86)%\Microsoft Visual Studio\2019\Professional\VC\Auxiliary\Build\vcvars64.bat"
        "%ProgramFiles(x86)%\Microsoft Visual Studio\2019\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
    ) do (
        if not defined VCVARS_FOUND (
            if exist %%~V set "VCVARS_FOUND=%%~V"
        )
    )
)
if not defined VCVARS_FOUND (
    set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
    if exist "!VSWHERE!" (
        for /f "usebackq delims=" %%P in (`"!VSWHERE!" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
            if not defined VCVARS_FOUND (
                if exist "%%P\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS_FOUND=%%P\VC\Auxiliary\Build\vcvars64.bat"
            )
        )
    )
)
if not defined VCVARS_FOUND (
    echo Error: Could not locate Visual Studio 2022 or 2019 C++ Build Tools ^(vcvars64.bat^).
    echo Run "cmd /c fast_start.bat" to detect/install prerequisites, or set VCVARS64.
    exit /b 1
)

call "!VCVARS_FOUND!" >nul
if errorlevel 1 (
    echo Failed to initialize x64 MSVC build environment from "!VCVARS_FOUND!".
    exit /b 1
)

REM Auto-detect CUDA compute capability (respect CUDA_ARCH override, e.g. set CUDA_ARCH=sm_89)
if not defined CUDA_ARCH (
    for /f "usebackq delims=" %%G in (`nvidia-smi --query-gpu^=compute_cap --format^=csv^,noheader 2^>nul`) do (
        if not defined CUDA_ARCH (
            set "CC_RAW=%%G"
            set "CC_RAW=!CC_RAW: =!"
            set "CC_NODOT=!CC_RAW:.=!"
            if not "!CC_NODOT!"=="" set "CUDA_ARCH=sm_!CC_NODOT!"
        )
    )
)
if not defined CUDA_ARCH set "CUDA_ARCH=sm_120"

if not exist build mkdir build

set TARGET=%~1
if "%TARGET%"=="" set TARGET=core

if /I "%TARGET%"=="core" goto build_core
if /I "%TARGET%"=="cli" goto build_cli
if /I "%TARGET%"=="proposer" goto build_proposer
if /I "%TARGET%"=="legacy" goto build_legacy
if /I "%TARGET%"=="single" goto build_single
if /I "%TARGET%"=="mix" goto build_mix
if /I "%TARGET%"=="v2" goto build_v2
if /I "%TARGET%"=="all" goto build_all

echo Unknown target "%TARGET%". Usage: build.bat [core^|cli^|proposer^|legacy^|single^|mix^|v2^|all]
exit /b 1

:build_all
:build_core
:build_cli
echo Building build\mof_gcmc_cli.exe for %CUDA_ARCH%...
nvcc -O3 -std=c++17 -arch=%CUDA_ARCH% -Iinclude ^
    -Xcompiler "/O2 /arch:AVX2 /utf-8 /FS" ^
    src\cif_loader.cpp src\unified_grid_builder.cu src\unified_gcmc_kernel.cu src\main_cli.cu ^
    -o build\mof_gcmc_cli.exe
if errorlevel 1 (
    echo Build mof_gcmc_cli.exe failed!
    exit /b 1
)
echo Build succeeded: build\mof_gcmc_cli.exe
if /I "%TARGET%"=="cli" goto done

:build_proposer
echo Building build\mof_gcmc_proposer.dll for %CUDA_ARCH%...
nvcc -O3 -std=c++17 -arch=%CUDA_ARCH% -Iinclude --shared ^
    -Xcompiler "/O2 /arch:AVX2 /utf-8 /FS /MD" ^
    src\cif_loader.cpp src\unified_grid_builder.cu src\unified_gcmc_kernel.cu src\proposer_dll.cu ^
    -o build\mof_gcmc_proposer.dll
if errorlevel 1 (
    echo Build mof_gcmc_proposer.dll failed!
    exit /b 1
)
echo Build succeeded: build\mof_gcmc_proposer.dll
if /I "%TARGET%"=="core" goto done
if /I "%TARGET%"=="proposer" goto done

:build_legacy
:build_single
echo Building build\mof_gcmc.exe for %CUDA_ARCH%...
nvcc -O3 -std=c++17 -arch=%CUDA_ARCH% -Iinclude ^
    -Xcompiler "/O2 /arch:AVX2 /utf-8 /FS" ^
    src\cif_loader.cpp src\legacy\grid_builder.cu src\legacy\gcmc_kernel.cu src\legacy\main.cu ^
    -o build\mof_gcmc.exe
if errorlevel 1 (
    echo Build mof_gcmc.exe failed!
    exit /b 1
)
echo Build succeeded: build\mof_gcmc.exe
if /I "%TARGET%"=="single" goto done

:build_mix
echo Building build\mof_gcmc_mix.exe for %CUDA_ARCH%...
nvcc -O3 -std=c++17 -arch=%CUDA_ARCH% -Iinclude ^
    -Xcompiler "/O2 /arch:AVX2 /utf-8 /FS" ^
    src\cif_loader.cpp src\legacy\mix_grid_builder.cu src\legacy\mix_gcmc_kernel.cu src\legacy\main_mix.cu ^
    -o build\mof_gcmc_mix.exe
if errorlevel 1 (
    echo Build mof_gcmc_mix.exe failed!
    exit /b 1
)
echo Build succeeded: build\mof_gcmc_mix.exe
if /I "%TARGET%"=="mix" goto done

:build_v2
echo Building build\mof_gcmc_v2.exe for %CUDA_ARCH%...
nvcc -O3 -std=c++17 -arch=%CUDA_ARCH% -Iinclude ^
    -Xcompiler "/O2 /arch:AVX2 /utf-8 /FS" ^
    src\cif_loader.cpp src\legacy\grid_builder.cu src\legacy\grid_builder_v2.cu src\legacy\gcmc_kernel_v2.cu src\legacy\main_v2.cu ^
    -o build\mof_gcmc_v2.exe
if errorlevel 1 (
    echo Build mof_gcmc_v2.exe failed!
    exit /b 1
)
echo Build succeeded: build\mof_gcmc_v2.exe
if /I "%TARGET%"=="v2" goto done

:done
exit /b 0
