---
layout: default
title: Bridge Pattern
parent: CUDA
nav_order: 2
description: CUDA-Houdini bridge pattern for HDK
permalink: /cuda/bridge-pattern/
---

# The Bridge Pattern
{: .fs-9 }

Isolating CUDA code from Houdini through a data transfer layer.
{: .fs-6 .fw-300 }

---

## Why Use a Bridge?

The Bridge pattern separates concerns:

- **Houdini code** knows about `GU_Detail`, `GA_Attribute`, etc.
- **CUDA code** knows about `float3*`, `cudaMemcpy`, etc.
- **Bridge class** translates between them

This makes:
- CUDA code reusable outside Houdini
- Houdini code testable without GPU
- Memory management centralized

## Bridge Class Structure

```cpp
// CudaBridge.h
#pragma once

#include <GU/GU_Detail.h>
#include <cuda_runtime.h>
#include <vector>

class CudaBridge {
public:
    CudaBridge();
    ~CudaBridge();

    // === Data Transfer ===
    void uploadPositions(const GU_Detail* gdp);
    void downloadPositions(GU_Detail* gdp);
    void uploadTopology(const GU_Detail* gdp);

    // === GPU Operations ===
    void smooth(int iterations, float weight);
    void computeNormals();

    // === Accessors ===
    int numPoints() const { return m_numPoints; }
    float3* devicePositions() { return m_d_positions; }

private:
    // Host buffers (CPU)
    std::vector<float3> m_hostPositions;
    std::vector<int> m_hostNeighborOffsets;
    std::vector<int> m_hostNeighborIndices;

    // Device buffers (GPU)
    float3* m_d_positions = nullptr;
    float3* m_d_smoothed = nullptr;
    int* m_d_neighborOffsets = nullptr;
    int* m_d_neighborIndices = nullptr;

    // Counts
    int m_numPoints = 0;
    int m_totalNeighbors = 0;

    // Caching
    GA_DataId m_cachedTopologyId = GA_INVALID_DATAID;
};
```

---

## Upload Pattern: Houdini → GPU

### Step 1: Extract from GU_Detail

Use Houdini's GA (Geometry Attribute) handles:

```cpp
void CudaBridge::uploadPositions(const GU_Detail* gdp) {
    // Get point count
    m_numPoints = gdp->getNumPoints();
    if (m_numPoints == 0) return;

    // Resize host buffer
    m_hostPositions.resize(m_numPoints);

    // Get position attribute handle (read-only)
    GA_ROHandleV3 posHandle(gdp->getP());

    // Extract positions to host buffer
    GA_Offset ptoff;
    GA_FOR_ALL_PTOFF(gdp, ptoff) {
        GA_Index idx = gdp->pointIndex(ptoff);
        UT_Vector3 pos = posHandle.get(ptoff);

        m_hostPositions[idx].x = pos.x();
        m_hostPositions[idx].y = pos.y();
        m_hostPositions[idx].z = pos.z();
    }

    // Step 2: Upload to GPU...
}
```

### Step 2: Allocate GPU Memory

```cpp
void CudaBridge::uploadPositions(const GU_Detail* gdp) {
    // ... (extraction code above) ...

    // Allocate GPU memory if needed
    if (m_d_positions == nullptr) {
        cudaMalloc(&m_d_positions, m_numPoints * sizeof(float3));
    }

    // Step 3: Transfer to GPU...
}
```

### Step 3: cudaMemcpy to GPU

```cpp
void CudaBridge::uploadPositions(const GU_Detail* gdp) {
    // ... (extraction and allocation above) ...

    // Transfer to GPU
    cudaMemcpy(
        m_d_positions,                          // Destination (GPU)
        m_hostPositions.data(),                 // Source (CPU)
        m_numPoints * sizeof(float3),           // Size in bytes
        cudaMemcpyHostToDevice                  // Direction
    );
}
```

### Complete Upload Function

```cpp
void CudaBridge::uploadPositions(const GU_Detail* gdp) {
    m_numPoints = gdp->getNumPoints();
    if (m_numPoints == 0) return;

    // Resize host buffer
    m_hostPositions.resize(m_numPoints);

    // Extract from Houdini
    GA_ROHandleV3 posHandle(gdp->getP());
    GA_Offset ptoff;
    GA_FOR_ALL_PTOFF(gdp, ptoff) {
        GA_Index idx = gdp->pointIndex(ptoff);
        UT_Vector3 pos = posHandle.get(ptoff);
        m_hostPositions[idx] = make_float3(pos.x(), pos.y(), pos.z());
    }

    // Allocate GPU if needed
    if (m_d_positions == nullptr) {
        cudaMalloc(&m_d_positions, m_numPoints * sizeof(float3));
    }

    // Upload to GPU
    cudaMemcpy(m_d_positions, m_hostPositions.data(),
               m_numPoints * sizeof(float3), cudaMemcpyHostToDevice);
}
```

---

## Download Pattern: GPU → Houdini

### Step 1: Sync GPU

Ensure all kernels are complete:

```cpp
void CudaBridge::downloadPositions(GU_Detail* gdp) {
    if (m_numPoints == 0) return;

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Step 2: Download...
}
```

### Step 2: cudaMemcpy from GPU

```cpp
void CudaBridge::downloadPositions(GU_Detail* gdp) {
    // ... sync ...

    // Download from GPU
    cudaMemcpy(
        m_hostPositions.data(),                 // Destination (CPU)
        m_d_positions,                          // Source (GPU)
        m_numPoints * sizeof(float3),           // Size in bytes
        cudaMemcpyDeviceToHost                  // Direction
    );

    // Step 3: Write to Houdini...
}
```

### Step 3: Write to GU_Detail

```cpp
void CudaBridge::downloadPositions(GU_Detail* gdp) {
    // ... sync and download ...

    // Get position attribute handle (read-write)
    GA_RWHandleV3 posHandle(gdp->getP());

    // Write back to Houdini
    GA_Offset ptoff;
    GA_FOR_ALL_PTOFF(gdp, ptoff) {
        GA_Index idx = gdp->pointIndex(ptoff);
        float3 pos = m_hostPositions[idx];
        posHandle.set(ptoff, UT_Vector3(pos.x, pos.y, pos.z));
    }
}
```

### Complete Download Function

```cpp
void CudaBridge::downloadPositions(GU_Detail* gdp) {
    if (m_numPoints == 0) return;

    // Sync GPU
    cudaDeviceSynchronize();

    // Download from GPU
    cudaMemcpy(m_hostPositions.data(), m_d_positions,
               m_numPoints * sizeof(float3), cudaMemcpyDeviceToHost);

    // Write to Houdini
    GA_RWHandleV3 posHandle(gdp->getP());
    GA_Offset ptoff;
    GA_FOR_ALL_PTOFF(gdp, ptoff) {
        GA_Index idx = gdp->pointIndex(ptoff);
        float3 pos = m_hostPositions[idx];
        posHandle.set(ptoff, UT_Vector3(pos.x, pos.y, pos.z));
    }
}
```

---

## Topology Caching

Topology (edges, neighbors) doesn't change every frame. Cache it:

```cpp
void CudaBridge::uploadTopology(const GU_Detail* gdp) {
    // Check if topology changed
    GA_DataId currentTopologyId = gdp->getTopology().getDataId();
    if (currentTopologyId == m_cachedTopologyId) {
        return;  // Topology unchanged, skip re-upload
    }
    m_cachedTopologyId = currentTopologyId;

    // ... extract and upload topology ...
}
```

Use `GA_DataId` for:
- **Position attribute**: `gdp->getP()->getDataId()`
- **Topology**: `gdp->getTopology().getDataId()`
- **Any attribute**: `attrib->getDataId()`

---

## Uploading Topology (Neighbor Lists)

Build CSR (Compressed Sparse Row) format for neighbor connectivity:

```cpp
void CudaBridge::uploadTopology(const GU_Detail* gdp) {
    // Check cache...

    m_numPoints = gdp->getNumPoints();

    // Build neighbor lists
    m_hostNeighborOffsets.resize(m_numPoints + 1);
    m_hostNeighborIndices.clear();

    // First pass: count neighbors
    GA_Offset ptoff;
    int totalNeighbors = 0;
    GA_FOR_ALL_PTOFF(gdp, ptoff) {
        GA_Index idx = gdp->pointIndex(ptoff);
        m_hostNeighborOffsets[idx] = totalNeighbors;

        // Get connected points
        GA_OffsetArray neighbors;
        gdp->getPointNeighbours(ptoff, neighbors);
        totalNeighbors += neighbors.size();
    }
    m_hostNeighborOffsets[m_numPoints] = totalNeighbors;

    // Second pass: fill neighbor indices
    m_hostNeighborIndices.resize(totalNeighbors);
    int writeIdx = 0;
    GA_FOR_ALL_PTOFF(gdp, ptoff) {
        GA_OffsetArray neighbors;
        gdp->getPointNeighbours(ptoff, neighbors);

        for (GA_Offset neighborOff : neighbors) {
            GA_Index neighborIdx = gdp->pointIndex(neighborOff);
            m_hostNeighborIndices[writeIdx++] = neighborIdx;
        }
    }

    // Upload to GPU
    cudaMalloc(&m_d_neighborOffsets, (m_numPoints + 1) * sizeof(int));
    cudaMalloc(&m_d_neighborIndices, totalNeighbors * sizeof(int));

    cudaMemcpy(m_d_neighborOffsets, m_hostNeighborOffsets.data(),
               (m_numPoints + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(m_d_neighborIndices, m_hostNeighborIndices.data(),
               totalNeighbors * sizeof(int), cudaMemcpyHostToDevice);
}
```

---

## Using the Bridge in a SOP Node

```cpp
// In SOP_MyNode.cpp

OP_ERROR SOP_MyNode::cookMySop(OP_Context& context) {
    // Lock inputs
    if (lockInputs(context) >= UT_ERROR_ABORT)
        return error();

    // Get input geometry
    const GU_Detail* restGdp = inputGeo(0);
    const GU_Detail* deformedGdp = inputGeo(1);

    // Duplicate input to output
    gdp->clearAndDestroy();
    gdp->copy(*deformedGdp);

    // Get parameters
    int iterations = evalInt("iterations", 0, context.getTime());
    float weight = evalFloat("weight", 0, context.getTime());

    // Upload to GPU
    m_bridge->uploadTopology(restGdp);      // Only if changed
    m_bridge->uploadPositions(gdp);

    // Run GPU computation
    m_bridge->smooth(iterations, weight);

    // Download results
    m_bridge->downloadPositions(gdp);

    unlockInputs();
    return error();
}
```

---

## Memory Cleanup

Always free GPU memory in destructor:

```cpp
CudaBridge::~CudaBridge() {
    if (m_d_positions) cudaFree(m_d_positions);
    if (m_d_smoothed) cudaFree(m_d_smoothed);
    if (m_d_neighborOffsets) cudaFree(m_d_neighborOffsets);
    if (m_d_neighborIndices) cudaFree(m_d_neighborIndices);
}
```

---

## Best Practices

1. **Cache topology** - Use `GA_DataId` to skip redundant uploads
2. **Reuse GPU buffers** - Don't reallocate every frame if size unchanged
3. **Batch operations** - Upload once, run multiple kernels, download once
4. **Sync only when needed** - `cudaDeviceSynchronize()` is expensive
5. **Use pinned memory** - For frequent transfers, use `cudaMallocHost` for faster copies
