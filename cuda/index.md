---
layout: default
title: CUDA
nav_order: 4
has_children: true
description: CUDA development for Houdini HDK
permalink: /cuda/
---

# CUDA for Houdini
{: .fs-9 }

A guide to GPU-accelerated computing with CUDA in Houdini.
{: .fs-6 .fw-300 }

---

## What is CUDA?

CUDA (Compute Unified Device Architecture) is NVIDIA's parallel computing platform that allows you to run computations on the GPU. For Houdini, this means massive speedups for operations like:

- **Deformers** - Delta Mush, smoothing operations
- **Simulations** - Cloth, soft body physics
- **UV Operations** - Unwrapping, seam detection
- **Any parallel computation** - Anything that processes many vertices/points

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Houdini SOP Node                      │
│  ┌─────────────────────────────────────────────────┐    │
│  │              Bridge Class (C++)                  │    │
│  │  ┌─────────────┐       ┌─────────────┐         │    │
│  │  │   Upload    │       │  Download   │         │    │
│  │  │ GU_Detail → │       │ → GU_Detail │         │    │
│  │  │    GPU      │       │    GPU      │         │    │
│  │  └─────────────┘       └─────────────┘         │    │
│  └─────────────────────────────────────────────────┘    │
│                          ↓ ↑                             │
│  ┌─────────────────────────────────────────────────┐    │
│  │           CUDA Static Library (.lib)             │    │
│  │  ┌─────────────┐  ┌─────────────┐               │    │
│  │  │   Kernels   │  │   Launch    │               │    │
│  │  │  __global__ │  │  Wrappers   │               │    │
│  │  └─────────────┘  └─────────────┘               │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

## Project Structure

A typical CUDA Houdini plugin has this structure:

```
my_cuda_plugin/
├── CMakeLists.txt           # Build configuration
├── build.bat                # Windows build script
├── src/
│   ├── SOP_MyNode.cpp       # Houdini SOP node
│   ├── SOP_MyNode.h
│   ├── CudaBridge.cpp       # CPU↔GPU data transfer
│   ├── CudaBridge.h
│   └── cuda/
│       ├── MyKernels.cu     # CUDA kernel implementations
│       └── MyKernels.cuh    # Kernel declarations
└── package/
    └── dso/                 # Output DSO files
```

## Key Components

| Component | Purpose |
|:----------|:--------|
| **CUDA Kernels** (`.cu`) | GPU code that runs in parallel |
| **Header Files** (`.cuh`) | Kernel declarations and shared types |
| **Bridge Class** | Manages GPU memory and data transfer |
| **SOP Node** | Houdini interface and parameter handling |

## Quick Start Example

Here's a minimal CUDA kernel that doubles all point positions:

```cpp
// MyKernels.cu
__global__ void doublePositions(float3* positions, int numPoints) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPoints) return;

    positions[idx].x *= 2.0f;
    positions[idx].y *= 2.0f;
    positions[idx].z *= 2.0f;
}

void launchDoublePositions(float3* d_positions, int numPoints) {
    int blockSize = 256;
    int numBlocks = (numPoints + blockSize - 1) / blockSize;
    doublePositions<<<numBlocks, blockSize>>>(d_positions, numPoints);
}
```

## Documentation Sections

| Section | Description |
|:--------|:------------|
| [Kernel Patterns](kernel-patterns.html) | Standard conventions for writing CUDA kernels |
| [Bridge Pattern](bridge-pattern.html) | CPU↔GPU data transfer architecture |
| [Memory Management](memory-management.html) | GPU memory allocation and data layouts |
| [Debugging](debugging.html) | Error handling and debugging techniques |

## Prerequisites

- **CUDA Toolkit** - NVIDIA's development tools (nvcc compiler)
- **Compatible GPU** - NVIDIA GPU with compute capability 8.6+ (RTX 30/40/50 series)
- **Houdini HDK** - Houdini Development Kit
- **Visual Studio 2022** - Windows compiler (or GCC on Linux)

## Environment

This documentation is based on:
- **Houdini 21.0.559**
- **CUDA Toolkit 12.x**
- **GPU Architectures**: 86 (Ampere), 89 (Ada), 100 (Blackwell)
