---
layout: default
title: Build Scripts
parent: HDK
nav_order: 5
description: HDK build automation scripts
permalink: /hdk/build-scripts/
---

# Build Scripts
{: .fs-9 }

Automating HDK plugin builds with batch scripts.
{: .fs-6 .fw-300 }

---

## Basic build.bat

Minimal build script for Windows:

```batch
@echo off
setlocal

:: Find Houdini installation
for /d %%d in ("C:\Program Files\Side Effects Software\Houdini 21.*") do (
    set "HFS=%%d"
)

if not defined HFS (
    echo ERROR: Houdini not found
    exit /b 1
)

echo Using Houdini: %HFS%

:: Configure and build
cmake -G "Visual Studio 17 2022" -A x64 -B build -S .
cmake --build build --config Release

endlocal
```

---

## Full-Featured build.bat

Complete build script with error handling and options:

```batch
@echo off
setlocal enabledelayedexpansion

:: ============================================================================
:: Build Script for Houdini CUDA Plugin
:: ============================================================================

echo ============================================
echo  Building Houdini CUDA Plugin
echo ============================================
echo.

:: Configuration
set "BUILD_DIR=build"
set "BUILD_CONFIG=Release"
set "PACKAGE_DSO_DIR=%~dp0package\dso"

:: ============================================================================
:: Find Visual Studio
:: ============================================================================

echo [1/5] Finding Visual Studio...

:: Try VS 2022 first, then 2019
set "VS_VERSION="
set "VS_GENERATOR="

if exist "C:\Program Files\Microsoft Visual Studio\2022\Professional" (
    set "VS_VERSION=2022 Professional"
    set "VS_GENERATOR=Visual Studio 17 2022"
) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Community" (
    set "VS_VERSION=2022 Community"
    set "VS_GENERATOR=Visual Studio 17 2022"
) else if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional" (
    set "VS_VERSION=2019 Professional"
    set "VS_GENERATOR=Visual Studio 16 2019"
) else if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community" (
    set "VS_VERSION=2019 Community"
    set "VS_GENERATOR=Visual Studio 16 2019"
)

if not defined VS_VERSION (
    echo ERROR: Visual Studio 2019 or 2022 not found
    exit /b 1
)

echo       Found: %VS_VERSION%

:: ============================================================================
:: Find Houdini
:: ============================================================================

echo [2/5] Finding Houdini...

set "HFS="
for /d %%d in ("C:\Program Files\Side Effects Software\Houdini 21.*") do (
    set "HFS=%%d"
)

if not defined HFS (
    echo ERROR: Houdini 21.x not found in Program Files
    echo        Install Houdini or set HFS environment variable
    exit /b 1
)

echo       Found: %HFS%

:: ============================================================================
:: Verify CUDA
:: ============================================================================

echo [3/5] Verifying CUDA Toolkit...

where nvcc >nul 2>&1
if errorlevel 1 (
    echo ERROR: CUDA Toolkit not found
    echo        Install CUDA Toolkit and ensure nvcc is in PATH
    exit /b 1
)

for /f "tokens=*" %%i in ('nvcc --version ^| findstr /C:"release"') do (
    echo       %%i
)

:: ============================================================================
:: Handle Locked DSOs
:: ============================================================================

echo [4/5] Preparing build...

:: Rename old DSOs if they exist (in case Houdini has them locked)
if exist "%PACKAGE_DSO_DIR%\*.dll" (
    for %%f in ("%PACKAGE_DSO_DIR%\*.dll") do (
        if exist "%%f.old" del "%%f.old" 2>nul
        ren "%%f" "%%~nxf.old" 2>nul
    )
)

:: ============================================================================
:: CMake Configure and Build
:: ============================================================================

echo [5/5] Building...

:: Configure (only if needed)
if not exist "%BUILD_DIR%\CMakeCache.txt" (
    echo       Configuring CMake...
    cmake -G "%VS_GENERATOR%" -A x64 -B "%BUILD_DIR%" -S . -DHFS="%HFS%"
    if errorlevel 1 (
        echo ERROR: CMake configuration failed
        exit /b 1
    )
)

:: Build
echo       Compiling %BUILD_CONFIG%...
cmake --build "%BUILD_DIR%" --config %BUILD_CONFIG% --parallel
if errorlevel 1 (
    echo.
    echo ERROR: Build failed
    exit /b 1
)

:: ============================================================================
:: Success
:: ============================================================================

echo.
echo ============================================
echo  Build Successful!
echo ============================================
echo.
echo DSO files in: %PACKAGE_DSO_DIR%
echo.
dir /b "%PACKAGE_DSO_DIR%\*.dll" 2>nul
echo.
echo Restart Houdini to load the new plugins.

endlocal
exit /b 0
```

---

## Clean Build Script

Script to perform a clean rebuild:

```batch
@echo off
setlocal

echo Cleaning build directory...

:: Remove build folder
if exist build (
    rmdir /s /q build
    echo       Removed: build/
)

:: Remove old DSOs
if exist package\dso\*.dll (
    del /q package\dso\*.dll
    echo       Removed: package/dso/*.dll
)

if exist package\dso\*.dll.old (
    del /q package\dso\*.dll.old
    echo       Removed: package/dso/*.dll.old
)

echo.
echo Clean complete. Run build.bat to rebuild.

endlocal
```

---

## Build Options

Add command-line options to your build script:

```batch
@echo off
setlocal enabledelayedexpansion

:: Parse arguments
set "BUILD_CONFIG=Release"
set "CLEAN_BUILD=0"
set "REBUILD=0"

:parse_args
if "%~1"=="" goto done_parsing
if /i "%~1"=="debug" set "BUILD_CONFIG=Debug"
if /i "%~1"=="release" set "BUILD_CONFIG=Release"
if /i "%~1"=="clean" set "CLEAN_BUILD=1"
if /i "%~1"=="rebuild" set "REBUILD=1"
shift
goto parse_args
:done_parsing

:: Handle clean
if "%CLEAN_BUILD%"=="1" (
    echo Cleaning...
    if exist build rmdir /s /q build
)

:: Handle rebuild
if "%REBUILD%"=="1" (
    echo Rebuilding...
    cmake --build build --config %BUILD_CONFIG% --clean-first --parallel
) else (
    cmake --build build --config %BUILD_CONFIG% --parallel
)

endlocal
```

Usage:
```batch
build.bat              :: Release build
build.bat debug        :: Debug build
build.bat clean        :: Clean build directory
build.bat rebuild      :: Clean and rebuild
```

---

## Test Script

Script to build and run tests:

```batch
@echo off
setlocal

:: Build first
call build.bat
if errorlevel 1 exit /b 1

:: Set environment
set "HFS=C:\Program Files\Side Effects Software\Houdini 21.0.559"
set "HOUDINI_DSO_PATH=%~dp0package\dso;&"

:: Run test
echo.
echo Running test...
"%HFS%\bin\hython.exe" scripts\run_test.py

endlocal
```

---

## Multi-Configuration Build

Build both Debug and Release:

```batch
@echo off
setlocal

echo Building Debug configuration...
cmake --build build --config Debug --parallel
if errorlevel 1 exit /b 1

echo Building Release configuration...
cmake --build build --config Release --parallel
if errorlevel 1 exit /b 1

echo.
echo Both configurations built successfully.

endlocal
```

---

## Linux Build Script

Equivalent bash script for Linux:

```bash
#!/bin/bash

# Find Houdini
HFS=$(ls -d /opt/hfs21.* 2>/dev/null | tail -1)
if [ -z "$HFS" ]; then
    echo "ERROR: Houdini not found"
    exit 1
fi

echo "Using Houdini: $HFS"

# Source Houdini environment
source "$HFS/houdini_setup"

# Configure and build
mkdir -p build
cd build

cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --parallel $(nproc)
```

---

## CMake Presets (Modern Approach)

For CMake 3.20+, use `CMakePresets.json`:

```json
{
    "version": 3,
    "configurePresets": [
        {
            "name": "default",
            "displayName": "Default Config",
            "generator": "Visual Studio 17 2022",
            "architecture": "x64",
            "binaryDir": "${sourceDir}/build",
            "cacheVariables": {
                "CMAKE_BUILD_TYPE": "Release"
            }
        }
    ],
    "buildPresets": [
        {
            "name": "release",
            "configurePreset": "default",
            "configuration": "Release"
        },
        {
            "name": "debug",
            "configurePreset": "default",
            "configuration": "Debug"
        }
    ]
}
```

Usage:
```batch
cmake --preset default
cmake --build --preset release
```

---

## Integration with IDEs

### Visual Studio

After CMake configuration, open `build/YourProject.sln`.

### VS Code

Create `.vscode/tasks.json`:

```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Build Release",
            "type": "shell",
            "command": "${workspaceFolder}/build.bat",
            "group": {
                "kind": "build",
                "isDefault": true
            }
        },
        {
            "label": "Clean Build",
            "type": "shell",
            "command": "${workspaceFolder}/clean.bat"
        }
    ]
}
```

---

## Troubleshooting

### CMake Not Found

```batch
:: Add CMake to PATH
set PATH=%PATH%;C:\Program Files\CMake\bin
```

### CUDA Architecture Mismatch

Check your GPU architecture:
```batch
nvidia-smi --query-gpu=compute_cap --format=csv
```

Update `CMAKE_CUDA_ARCHITECTURES` in CMakeLists.txt accordingly.

### Houdini Version Mismatch

Ensure you're building against the same Houdini version you're running:
```batch
echo %HFS%
"%HFS%\bin\houdini.exe" -version
```
