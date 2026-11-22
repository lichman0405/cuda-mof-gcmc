@echo off
setlocal enabledelayedexpansion

call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 (
    echo Failed to initialize VS2019 x64 build environment.
    exit /b 1
)

if not exist build mkdir build

set TARGET=%~1
if "%TARGET%"=="" set TARGET=all

if /I "%TARGET%"=="single" goto build_single
if /I "%TARGET%"=="mix" goto build_mix
if /I "%TARGET%"=="v2" goto build_v2
if /I "%TARGET%"=="all" goto build_single

echo Unknown target "%TARGET%". Usage: build.bat [all^|single^|mix^|v2]
exit /b 1

:build_single
echo Building build\mof_gcmc.exe for sm_120 (RTX 5070)...
nvcc -O3 -std=c++17 -arch=sm_120 -Iinclude ^
    -Xcompiler "/O2 /arch:AVX2 /utf-8 /FS" ^
    src\cif_loader.cpp src\grid_builder.cu src\gcmc_kernel.cu src\main.cu ^
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
    src\cif_loader.cpp src\mix_grid_builder.cu src\mix_gcmc_kernel.cu src\main_mix.cu ^
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
    src\cif_loader.cpp src\grid_builder.cu src\grid_builder_v2.cu src\gcmc_kernel_v2.cu src\main_v2.cu ^
    -o build\mof_gcmc_v2.exe
if errorlevel 1 (
    echo Build mof_gcmc_v2.exe failed!
    exit /b 1
)
echo Build succeeded: build\mof_gcmc_v2.exe

:done
exit /b 0
