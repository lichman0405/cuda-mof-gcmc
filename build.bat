@echo off
setlocal enabledelayedexpansion

call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 (
    echo Failed to initialize VS2019 x64 build environment.
    exit /b 1
)

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
echo Building build\mof_gcmc_cli.exe for sm_120 (RTX 5070)...
nvcc -O3 -std=c++17 -arch=sm_120 -Iinclude ^
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
echo Building build\mof_gcmc_proposer.dll for sm_120 (RTX 5070)...
nvcc -O3 -std=c++17 -arch=sm_120 -Iinclude --shared ^
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
echo Building build\mof_gcmc.exe for sm_120 (RTX 5070)...
nvcc -O3 -std=c++17 -arch=sm_120 -Iinclude ^
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
echo Building build\mof_gcmc_mix.exe for sm_120 (RTX 5070)...
nvcc -O3 -std=c++17 -arch=sm_120 -Iinclude ^
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
echo Building build\mof_gcmc_v2.exe for sm_120 (RTX 5070)...
nvcc -O3 -std=c++17 -arch=sm_120 -Iinclude ^
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
