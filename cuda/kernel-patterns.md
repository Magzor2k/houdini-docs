---
layout: default
title: Kernel Patterns
parent: CUDA
nav_order: 1
description: CUDA kernel patterns for Houdini
permalink: /cuda/kernel-patterns/
---

# CUDA Kernel Patterns
{: .fs-9 }

Standard conventions for writing CUDA kernels in Houdini plugins.
{: .fs-6 .fw-300 }

---

## Basic Kernel Structure

Every CUDA kernel follows this standard pattern:

```cpp
__global__ void myKernel(
    const float3* __restrict__ input,    // Read-only input with restrict
    float3* __restrict__ output,          // Output buffer with restrict
    const int* __restrict__ indices,      // Optional: index arrays
    int count                             // Element count
) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Bounds check - CRITICAL!
    if (idx >= count) return;

    // Perform computation on idx-th element
    output[idx] = input[idx] * 2.0f;
}
```

## Key Conventions

### Block Size: 256 Threads

Use 256 threads per block as the standard:

```cpp
void launchMyKernel(float3* d_input, float3* d_output, int count) {
    int blockSize = 256;  // Standard block size
    int numBlocks = (count + blockSize - 1) / blockSize;  // Round up

    myKernel<<<numBlocks, blockSize>>>(d_input, d_output, count);
}
```

Why 256?
- Divisible by warp size (32)
- Good occupancy on most GPUs
- Consistent across the codebase

### Grid Dimension Calculation

Always round up to cover all elements:

```cpp
int numBlocks = (count + blockSize - 1) / blockSize;
```

This formula ensures:
- `count = 1000, blockSize = 256` → `numBlocks = 4` (covers 1024 threads)
- Extra threads are filtered by bounds check

### The `__restrict__` Keyword

Use `__restrict__` on pointer parameters to tell the compiler that pointers don't alias:

```cpp
__global__ void addKernel(
    const float* __restrict__ a,    // Compiler knows a, b, c don't overlap
    const float* __restrict__ b,
    float* __restrict__ c,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    c[idx] = a[idx] + b[idx];  // Compiler can optimize better
}
```

### Bounds Checking

**Always** check bounds at the start of every kernel:

```cpp
if (idx >= count) return;
```

Without this check, threads beyond your data will access invalid memory.

---

## Common Kernel Patterns

### Per-Point Operation

Process each point independently:

```cpp
__global__ void smoothPositions(
    const float3* __restrict__ positions,
    const int* __restrict__ neighborOffsets,
    const int* __restrict__ neighborIndices,
    float3* __restrict__ smoothed,
    int numPoints
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPoints) return;

    // Get neighbor range for this point
    int start = neighborOffsets[idx];
    int end = neighborOffsets[idx + 1];

    // Average neighbor positions
    float3 sum = make_float3(0, 0, 0);
    for (int i = start; i < end; i++) {
        int neighborIdx = neighborIndices[i];
        sum.x += positions[neighborIdx].x;
        sum.y += positions[neighborIdx].y;
        sum.z += positions[neighborIdx].z;
    }

    int count = end - start;
    if (count > 0) {
        smoothed[idx].x = sum.x / count;
        smoothed[idx].y = sum.y / count;
        smoothed[idx].z = sum.z / count;
    } else {
        smoothed[idx] = positions[idx];
    }
}
```

### Per-Edge Operation

Process each edge (two-point operation):

```cpp
__global__ void computeEdgeLengths(
    const float3* __restrict__ positions,
    const int2* __restrict__ edges,       // (v0, v1) pairs
    float* __restrict__ lengths,
    int numEdges
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numEdges) return;

    int v0 = edges[idx].x;
    int v1 = edges[idx].y;

    float3 p0 = positions[v0];
    float3 p1 = positions[v1];

    float dx = p1.x - p0.x;
    float dy = p1.y - p0.y;
    float dz = p1.z - p0.z;

    lengths[idx] = sqrtf(dx*dx + dy*dy + dz*dz);
}
```

### Per-Triangle Operation

Process each triangle (three-point operation):

```cpp
__global__ void computeTriangleNormals(
    const float3* __restrict__ positions,
    const int3* __restrict__ triangles,   // (v0, v1, v2) triplets
    float3* __restrict__ normals,
    int numTriangles
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numTriangles) return;

    int3 tri = triangles[idx];
    float3 p0 = positions[tri.x];
    float3 p1 = positions[tri.y];
    float3 p2 = positions[tri.z];

    // Edge vectors
    float3 e1 = make_float3(p1.x - p0.x, p1.y - p0.y, p1.z - p0.z);
    float3 e2 = make_float3(p2.x - p0.x, p2.y - p0.y, p2.z - p0.z);

    // Cross product
    float3 n;
    n.x = e1.y * e2.z - e1.z * e2.y;
    n.y = e1.z * e2.x - e1.x * e2.z;
    n.z = e1.x * e2.y - e1.y * e2.x;

    // Normalize
    float len = sqrtf(n.x*n.x + n.y*n.y + n.z*n.z);
    if (len > 1e-8f) {
        normals[idx] = make_float3(n.x/len, n.y/len, n.z/len);
    } else {
        normals[idx] = make_float3(0, 1, 0);
    }
}
```

### Reduction with Atomics

Accumulate results across threads (use sparingly - atomics are slow):

```cpp
__global__ void computeBoundingBox(
    const float3* __restrict__ positions,
    float3* __restrict__ minBounds,
    float3* __restrict__ maxBounds,
    int numPoints
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPoints) return;

    float3 p = positions[idx];

    // Atomic min/max for each component
    atomicMin((int*)&minBounds->x, __float_as_int(p.x));
    atomicMin((int*)&minBounds->y, __float_as_int(p.y));
    atomicMin((int*)&minBounds->z, __float_as_int(p.z));

    atomicMax((int*)&maxBounds->x, __float_as_int(p.x));
    atomicMax((int*)&maxBounds->y, __float_as_int(p.y));
    atomicMax((int*)&maxBounds->z, __float_as_int(p.z));
}
```

---

## Launch Wrapper Pattern

Always wrap kernel launches in a C++ function:

```cpp
// In .cuh header
void launchSmoothPositions(
    const float3* d_positions,
    const int* d_neighborOffsets,
    const int* d_neighborIndices,
    float3* d_smoothed,
    int numPoints
);

// In .cu implementation
void launchSmoothPositions(
    const float3* d_positions,
    const int* d_neighborOffsets,
    const int* d_neighborIndices,
    float3* d_smoothed,
    int numPoints
) {
    if (numPoints == 0) return;  // Early exit for empty data

    int blockSize = 256;
    int numBlocks = (numPoints + blockSize - 1) / blockSize;

    smoothPositions<<<numBlocks, blockSize>>>(
        d_positions,
        d_neighborOffsets,
        d_neighborIndices,
        d_smoothed,
        numPoints
    );

    // Optional: check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "smoothPositions kernel failed: %s\n",
                cudaGetErrorString(err));
    }
}
```

Benefits:
- Hides CUDA syntax from callers
- Centralizes grid/block calculation
- Adds error checking
- Can be called from non-CUDA code

---

## Header File Organization

```cpp
// MyKernels.cuh
#pragma once

#include <cuda_runtime.h>

// Kernel launch wrappers (callable from C++)
void launchDoublePositions(float3* d_positions, int numPoints);
void launchSmoothPositions(const float3* d_positions,
                           const int* d_neighborOffsets,
                           const int* d_neighborIndices,
                           float3* d_smoothed, int numPoints);
void launchComputeNormals(const float3* d_positions,
                          const int3* d_triangles,
                          float3* d_normals, int numTriangles);

// Don't expose __global__ kernels in headers - keep them in .cu files
```

---

## Performance Tips

1. **Minimize divergence** - Avoid different threads in a warp taking different branches
2. **Coalesced memory access** - Adjacent threads should access adjacent memory
3. **Use shared memory** - For data reused within a block
4. **Avoid atomic operations** - Use reduction patterns instead when possible
5. **Profile first** - Use NVIDIA Nsight to identify bottlenecks
