---
layout: default
title: CMake Setup
parent: HDK
nav_order: 1
description: CMake configuration for HDK projects
permalink: /hdk/cmake-setup/
---

# CMake Setup for HDK
{: .fs-9 }

Configuring CMakeLists.txt to build Houdini plugins.
{: .fs-6 .fw-300 }

---

## Basic CMakeLists.txt

Minimal configuration for a HDK plugin:

```cmake
cmake_minimum_required(VERSION 3.18)
project(MyHDKPlugin LANGUAGES CXX)

# === Houdini Configuration ===

# HFS can be set via cmake variable or environment variable
if(DEFINED HFS)
    # HFS passed as cmake variable - use it directly
elseif(DEFINED ENV{HFS})
    set(HFS $ENV{HFS})
else()
    message(FATAL_ERROR "HFS not set. Set via -DHFS=... or environment variable")
endif()

message(STATUS "Using Houdini at: ${HFS}")

# Add Houdini's CMake modules to the path
list(APPEND CMAKE_PREFIX_PATH "${HFS}/toolkit/cmake")

# Find Houdini package
find_package(Houdini REQUIRED)

# === Build Plugin ===

add_library(SOP_MyNode SHARED
    src/SOP_MyNode.cpp
)

# Configure for Houdini
houdini_configure_target(SOP_MyNode)

target_link_libraries(SOP_MyNode PRIVATE
    Houdini
)
```

## HFS Environment Variable

The `HFS` variable points to your Houdini installation:

```
Windows: C:\Program Files\Side Effects Software\Houdini 21.0.559
Linux:   /opt/hfs21.0.559
macOS:   /Applications/Houdini/Houdini21.0.559
```

### Setting HFS

**Option 1: Environment variable**
```batch
set HFS=C:\Program Files\Side Effects Software\Houdini 21.0.559
cmake -B build
```

**Option 2: CMake parameter**
```batch
cmake -B build -DHFS="C:\Program Files\Side Effects Software\Houdini 21.0.559"
```

**Option 3: Auto-detection in build script** (recommended)
```batch
for /d %%d in ("C:\Program Files\Side Effects Software\Houdini 21.*") do (
    set "HFS=%%d"
)
```

---

## Adding CUDA Support

For GPU-accelerated plugins:

```cmake
cmake_minimum_required(VERSION 3.18)
project(MyCUDAPlugin LANGUAGES CXX CUDA)

# === CUDA Configuration ===
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86 89 100)  # RTX 30/40/50 series

find_package(CUDAToolkit REQUIRED)

# === Houdini Configuration ===
# (same as above)
if(DEFINED HFS)
elseif(DEFINED ENV{HFS})
    set(HFS $ENV{HFS})
else()
    message(FATAL_ERROR "HFS not set")
endif()

list(APPEND CMAKE_PREFIX_PATH "${HFS}/toolkit/cmake")
find_package(Houdini REQUIRED)

# === CUDA Static Library ===
add_library(MyCUDA STATIC
    src/cuda/MyKernels.cu
)

set_target_properties(MyCUDA PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_link_libraries(MyCUDA PRIVATE
    CUDA::cudart_static
)

# === HDK Plugin (Shared Library) ===
add_library(SOP_MyCudaNode SHARED
    src/SOP_MyCudaNode.cpp
    src/CudaBridge.cpp
)

houdini_configure_target(SOP_MyCudaNode)

target_link_libraries(SOP_MyCudaNode PRIVATE
    Houdini
    MyCUDA
    CUDA::cudart_static
)
```

### Why Static CUDA Library?

The pattern `CUDA Static Library → HDK Shared Library` is recommended because:

1. **CUDA device code** needs special linking (`CUDA_RESOLVE_DEVICE_SYMBOLS`)
2. **Houdini expects shared libraries** (.dll/.so) for plugins
3. **Separation** keeps CUDA code isolated from HDK code

---

## Output Directory Configuration

Place DSO files in a specific location:

```cmake
# Define output directory
set(PACKAGE_DSO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/package/dso")

# Configure output for all build types
set_target_properties(SOP_MyNode PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PACKAGE_DSO_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${PACKAGE_DSO_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${PACKAGE_DSO_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${PACKAGE_DSO_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_DEBUG "${PACKAGE_DSO_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${PACKAGE_DSO_DIR}"
)
```

---

## Multi-Plugin Build

Build multiple plugins in one CMakeLists.txt:

```cmake
cmake_minimum_required(VERSION 3.18)
project(HoudiniPlugins LANGUAGES CXX CUDA)

# Shared configuration
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86 89 100)

# Houdini setup
if(DEFINED ENV{HFS})
    set(HFS $ENV{HFS})
endif()
list(APPEND CMAKE_PREFIX_PATH "${HFS}/toolkit/cmake")
find_package(Houdini REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Output directory
set(PACKAGE_DSO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/package/dso")

# === Plugin 1: Delta Mush ===
add_library(DeltaMushCUDA STATIC src/cuda/DeltaMush.cu)
set_target_properties(DeltaMushCUDA PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_link_libraries(DeltaMushCUDA PRIVATE CUDA::cudart_static)

add_library(SOP_CudaDeltaMush SHARED
    src/SOP_CudaDeltaMush.cpp
    src/CudaBridge.cpp
)
houdini_configure_target(SOP_CudaDeltaMush)
target_link_libraries(SOP_CudaDeltaMush PRIVATE Houdini DeltaMushCUDA CUDA::cudart_static)
set_target_properties(SOP_CudaDeltaMush PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PACKAGE_DSO_DIR}")

# === Plugin 2: Cloth Sim ===
add_library(ClothCUDA STATIC
    src/cuda/ClothSolver.cu
    src/cuda/Constraints.cu
)
set_target_properties(ClothCUDA PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_link_libraries(ClothCUDA PRIVATE
    CUDA::cudart_static
    CUDA::cublas
    CUDA::cusparse
)

add_library(SOP_CudaCloth SHARED
    src/SOP_CudaCloth.cpp
    src/ClothBridge.cpp
)
houdini_configure_target(SOP_CudaCloth)
target_link_libraries(SOP_CudaCloth PRIVATE Houdini ClothCUDA CUDA::cudart_static)
set_target_properties(SOP_CudaCloth PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PACKAGE_DSO_DIR}")
```

---

## Shared CMake Module

Create reusable configuration in `cmake/HoudiniCudaCommon.cmake`:

```cmake
# cmake/HoudiniCudaCommon.cmake

# CUDA setup
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86 89 100)

# Houdini setup
if(DEFINED HFS)
    # Already set
elseif(DEFINED ENV{HFS})
    set(HFS $ENV{HFS})
else()
    message(FATAL_ERROR "HFS not set")
endif()

message(STATUS "Houdini: ${HFS}")

list(APPEND CMAKE_PREFIX_PATH "${HFS}/toolkit/cmake")
find_package(Houdini REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Output directory
set(PACKAGE_DSO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../package/dso"
    CACHE PATH "DSO output directory")
```

Use in plugin CMakeLists.txt:

```cmake
cmake_minimum_required(VERSION 3.18)
project(MyPlugin LANGUAGES CXX CUDA)

# Include shared configuration
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/HoudiniCudaCommon.cmake)

# Build plugin...
```

---

## CUDA Libraries

Link additional CUDA libraries as needed:

```cmake
target_link_libraries(MyCUDALib PRIVATE
    CUDA::cudart_static    # CUDA runtime (always needed)
    CUDA::cublas           # Linear algebra (dot, norm, etc.)
    CUDA::cusparse         # Sparse matrix operations
    CUDA::cusolver         # Linear system solvers
)
```

---

## Compiler Flags

### CUDA Flags

```cmake
# Line info for profiling
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

# Debug: device-side debugging
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")
endif()
```

### C++ Flags (Windows)

```cmake
if(MSVC)
    # Disable warnings about dll-interface
    target_compile_options(SOP_MyNode PRIVATE /wd4251)
endif()
```

---

## CMake Configuration Commands

```batch
:: Configure (first time)
cmake -G "Visual Studio 17 2022" -A x64 -B build -S .

:: Build Release
cmake --build build --config Release

:: Build Debug
cmake --build build --config Debug

:: Clean rebuild
cmake --build build --config Release --clean-first
```

---

## Troubleshooting

### "HFS not set"
Set the HFS variable via environment or CMake parameter.

### "Could not find Houdini"
Ensure `${HFS}/toolkit/cmake/HoudiniConfig.cmake` exists.

### "CUDA not found"
Install CUDA Toolkit and ensure `nvcc` is in PATH.

### "Link errors"
Check that CUDA libraries match (static vs shared).
