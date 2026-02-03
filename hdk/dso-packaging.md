---
layout: default
title: DSO Packaging
parent: HDK
nav_order: 4
description: Packaging DSO plugins for distribution
permalink: /hdk/dso-packaging/
---

# DSO Packaging
{: .fs-9 }

Organizing and distributing Houdini plugin files.
{: .fs-6 .fw-300 }

---

## What is a DSO?

DSO (Dynamic Shared Object) is Houdini's term for plugin libraries:
- **Windows**: `.dll` files
- **Linux**: `.so` files
- **macOS**: `.dylib` files

These contain compiled HDK code that Houdini loads at startup.

---

## Package Structure

Recommended folder structure for a plugin package:

```
my_cuda_plugin/
├── CMakeLists.txt
├── build.bat
├── src/
│   ├── SOP_MyNode.cpp
│   ├── SOP_MyNode.h
│   ├── CudaBridge.cpp
│   ├── CudaBridge.h
│   └── cuda/
│       ├── MyKernels.cu
│       └── MyKernels.cuh
├── package/
│   ├── CudaPlugins.json     ← Package descriptor
│   ├── dso/
│   │   └── SOP_MyNode.dll   ← Output DSO
│   ├── apexdso/
│   │   └── apex_nodes.dll   ← APEX-specific nodes
│   └── otls/
│       └── my_tools.hda     ← Digital assets
└── build/                   ← CMake build directory
```

---

## CMake Output Configuration

Direct DSO output to the package folder:

```cmake
# Define output directory
set(PACKAGE_DSO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/package/dso")

# Configure output for all configurations (Debug, Release)
set_target_properties(SOP_MyNode PROPERTIES
    # Runtime output (Windows .dll)
    RUNTIME_OUTPUT_DIRECTORY "${PACKAGE_DSO_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${PACKAGE_DSO_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${PACKAGE_DSO_DIR}"

    # Library output (Linux/Mac .so/.dylib)
    LIBRARY_OUTPUT_DIRECTORY "${PACKAGE_DSO_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_DEBUG "${PACKAGE_DSO_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${PACKAGE_DSO_DIR}"
)
```

---

## Package Descriptor (JSON)

Create a package file for Houdini to discover your plugins:

```json
{
    "env": [
        {
            "HOUDINI_DSO_PATH": {
                "value": "$PACKAGE_PATH/dso",
                "method": "prepend"
            }
        },
        {
            "HOUDINI_APEX_DSO_PATH": {
                "value": "$PACKAGE_PATH/apexdso",
                "method": "prepend"
            }
        },
        {
            "HOUDINI_OTLSCAN_PATH": {
                "value": "$PACKAGE_PATH/otls",
                "method": "prepend"
            }
        }
    ],
    "path": "$PACKAGE_PATH"
}
```

Save this as `package/CudaPlugins.json`.

---

## Installing the Package

### Option 1: User Packages Folder (Development)

Copy or symlink your package folder to:

```
Windows: C:\Users\<user>\Documents\houdini21.0\packages\
Linux:   ~/houdini21.0/packages/
macOS:   ~/Library/Preferences/houdini/21.0/packages/
```

Create a JSON file pointing to your development folder:

```json
{
    "package_path": "C:/Development/my_cuda_plugin/package"
}
```

### Option 2: HOUDINI_PACKAGE_DIR (Production)

Set environment variable before launching Houdini:

```batch
set HOUDINI_PACKAGE_DIR=C:\Studio\houdini_packages
houdini
```

Place your package JSON in that directory.

### Option 3: Direct DSO Path (Quick Testing)

Set the DSO path directly:

```batch
set HOUDINI_DSO_PATH=C:\Development\my_cuda_plugin\package\dso;&
houdini
```

The `&` at the end appends to the default path instead of replacing it.

---

## DSO Types and Paths

| DSO Type | Environment Variable | Description |
|:---------|:--------------------|:------------|
| SOP, OBJ, etc. | `HOUDINI_DSO_PATH` | Standard operator plugins |
| APEX nodes | `HOUDINI_APEX_DSO_PATH` | APEX graph nodes |
| VEX functions | `HOUDINI_VEX_PATH` | Custom VEX operators |

---

## Multiple Plugin Build

For projects with multiple DSOs:

```cmake
# Common output directory
set(PACKAGE_DSO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/package/dso")
set(PACKAGE_APEX_DIR "${CMAKE_CURRENT_SOURCE_DIR}/package/apexdso")

# Helper function
function(configure_dso_output target output_dir)
    set_target_properties(${target} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${output_dir}"
        RUNTIME_OUTPUT_DIRECTORY_DEBUG "${output_dir}"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${output_dir}"
        LIBRARY_OUTPUT_DIRECTORY "${output_dir}"
        LIBRARY_OUTPUT_DIRECTORY_DEBUG "${output_dir}"
        LIBRARY_OUTPUT_DIRECTORY_RELEASE "${output_dir}"
    )
endfunction()

# Standard SOP nodes
add_library(SOP_CudaDeltaMush SHARED ...)
configure_dso_output(SOP_CudaDeltaMush ${PACKAGE_DSO_DIR})

add_library(SOP_CudaCloth SHARED ...)
configure_dso_output(SOP_CudaCloth ${PACKAGE_DSO_DIR})

# APEX nodes (different output directory)
add_library(apex_cuda_nodes SHARED ...)
configure_dso_output(apex_cuda_nodes ${PACKAGE_APEX_DIR})
```

---

## Handling Build Conflicts

When rebuilding while Houdini has the DSO loaded:

### Windows: Rename Old DSO

```batch
@echo off
:: Rename old DSO before building
if exist "package\dso\SOP_MyNode.dll" (
    if exist "package\dso\SOP_MyNode.dll.old" (
        del "package\dso\SOP_MyNode.dll.old"
    )
    ren "package\dso\SOP_MyNode.dll" "SOP_MyNode.dll.old"
)

:: Build
cmake --build build --config Release

echo Build complete. Restart Houdini to load new DSO.
```

### Linux/Mac: Similar Approach

```bash
#!/bin/bash
if [ -f "package/dso/SOP_MyNode.so" ]; then
    mv "package/dso/SOP_MyNode.so" "package/dso/SOP_MyNode.so.old"
fi

cmake --build build --config Release
```

---

## Debug vs Release

### Development Workflow

1. Build **Release** for normal testing (faster execution)
2. Build **Debug** only when debugging crashes

```batch
:: Release build (default)
cmake --build build --config Release

:: Debug build (when needed)
cmake --build build --config Debug
```

### CMake Configuration

```cmake
# Different flags for Debug
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # CUDA debugging
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")

    # Extra debug info
    add_definitions(-DDEBUG_MODE)
endif()
```

---

## Verifying DSO Loading

Check if Houdini loaded your DSO:

### Using dsoinfo

```batch
:: Windows
"%HFS%\bin\dsoinfo.exe" package\dso\SOP_MyNode.dll
```

### In Houdini Python Shell

```python
import hou

# List all SOP operators
for op in hou.sopNodeTypeCategory().nodeTypes().values():
    if 'cuda' in op.name().lower():
        print(op.name())
```

### Check DSO Path

```python
import os
print(os.environ.get('HOUDINI_DSO_PATH', 'Not set'))
```

---

## Common Issues

### DSO Not Loading

1. Check `HOUDINI_DSO_PATH` includes your directory
2. Verify DSO was built for correct Houdini version
3. Check for missing dependencies with `dsoinfo` or `dumpbin`

### Wrong Houdini Version

DSOs must match the exact Houdini version:
- Built with HDK 21.0.559 → Only works with Houdini 21.0.559
- Rebuild when upgrading Houdini

### Missing CUDA DLLs

If using CUDA, ensure the CUDA runtime is available:

```batch
:: Check CUDA path
where cudart64_*.dll

:: Or include CUDA bin in PATH
set PATH=%PATH%;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin
```
