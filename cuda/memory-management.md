---
layout: default
title: Memory Management
parent: CUDA
nav_order: 3
description: GPU memory management for Houdini CUDA
permalink: /cuda/memory-management/
---

# GPU Memory Management
{: .fs-9 }

Allocating, organizing, and managing GPU memory efficiently.
{: .fs-6 .fw-300 }

---

## GPU Memory Basics

GPU memory is separate from CPU memory. Data must be explicitly transferred:

```
┌─────────────────┐         ┌─────────────────┐
│   CPU Memory    │ ──────► │   GPU Memory    │
│  (Host/RAM)     │ cudaMemcpy │  (Device/VRAM) │
│                 │ ◄────── │                 │
└─────────────────┘         └─────────────────┘
```

## Memory Allocation

### Basic Allocation

```cpp
float3* d_positions = nullptr;  // Prefix d_ for device pointers
int numPoints = 10000;

// Allocate
cudaMalloc(&d_positions, numPoints * sizeof(float3));

// Use in kernels...

// Free when done
cudaFree(d_positions);
```

### Safe Allocation Helper

Template function to handle allocation safely:

```cpp
template<typename T>
void allocateGPU(T** ptr, size_t count) {
    // Free existing memory
    if (*ptr) {
        cudaFree(*ptr);
        *ptr = nullptr;
    }

    // Allocate new memory if count > 0
    if (count > 0) {
        cudaError_t err = cudaMalloc(ptr, count * sizeof(T));
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed: %s\n",
                    cudaGetErrorString(err));
            *ptr = nullptr;
        }
    }
}

// Usage
float3* d_positions = nullptr;
allocateGPU(&d_positions, numPoints);  // Safe allocation
allocateGPU(&d_positions, newSize);    // Safe reallocation (frees old)
allocateGPU(&d_positions, 0);          // Safe free
```

### Conditional Reallocation

Only reallocate if size changes:

```cpp
void CudaBridge::ensureCapacity(int newNumPoints) {
    if (newNumPoints != m_numPoints) {
        // Size changed - reallocate
        cudaDeviceSynchronize();  // Wait for pending operations

        allocateGPU(&m_d_positions, newNumPoints);
        allocateGPU(&m_d_velocities, newNumPoints);
        allocateGPU(&m_d_normals, newNumPoints);

        m_numPoints = newNumPoints;
    }
}
```

---

## Data Layouts

### AoS (Array of Structures)

Each element is a complete structure:

```cpp
// float3 is AoS - x, y, z are contiguous per point
float3* positions;  // [x0,y0,z0, x1,y1,z1, x2,y2,z2, ...]

// Access
positions[idx].x = value;
positions[idx].y = value;
positions[idx].z = value;
```

**Pros:** Natural for per-element operations
**Cons:** Not optimal for component-wise operations

### SoA (Structure of Arrays)

Each component is a separate array:

```cpp
// Separate arrays for each component
float* posX;  // [x0, x1, x2, ...]
float* posY;  // [y0, y1, y2, ...]
float* posZ;  // [z0, z1, z2, ...]

// Access
posX[idx] = value;
posY[idx] = value;
posZ[idx] = value;
```

**Pros:** Better memory coalescing for component-wise operations
**Cons:** More arrays to manage

### Recommendation

Use `float3` (AoS) for:
- Positions, velocities, normals
- When operations process all components together

Use SoA for:
- When processing components independently
- Performance-critical code (profile first!)

---

## Common Data Structures

### Edge List

Flat array of vertex pairs:

```cpp
// Two vertices per edge
int* d_edgeIndices;     // [v0, v1, v0, v1, v0, v1, ...]
                        //  edge0   edge1   edge2

// Or as int2 pairs
int2* d_edges;          // [(v0,v1), (v0,v1), (v0,v1), ...]

// With per-edge data
float* d_restLengths;   // [len0, len1, len2, ...]
```

**Uploading edges:**
```cpp
void uploadEdges(const GU_Detail* gdp) {
    std::vector<int2> edges;
    std::vector<float> restLengths;

    // Iterate over primitives (edges)
    GA_Offset primoff;
    GA_FOR_ALL_PRIMOFF(gdp, primoff) {
        const GA_Primitive* prim = gdp->getPrimitive(primoff);
        if (prim->getVertexCount() == 2) {
            GA_Offset v0 = prim->getPointOffset(0);
            GA_Offset v1 = prim->getPointOffset(1);

            int idx0 = gdp->pointIndex(v0);
            int idx1 = gdp->pointIndex(v1);

            edges.push_back(make_int2(idx0, idx1));

            // Compute rest length
            UT_Vector3 p0 = gdp->getPos3(v0);
            UT_Vector3 p1 = gdp->getPos3(v1);
            restLengths.push_back((p1 - p0).length());
        }
    }

    // Upload
    allocateGPU(&m_d_edges, edges.size());
    allocateGPU(&m_d_restLengths, edges.size());
    cudaMemcpy(m_d_edges, edges.data(),
               edges.size() * sizeof(int2), cudaMemcpyHostToDevice);
    cudaMemcpy(m_d_restLengths, restLengths.data(),
               restLengths.size() * sizeof(float), cudaMemcpyHostToDevice);
}
```

### CSR Format (Compressed Sparse Row)

Efficient storage for variable-length neighbor lists:

```
Point 0 has neighbors: [1, 2, 3]
Point 1 has neighbors: [0, 2]
Point 2 has neighbors: [0, 1, 3, 4]

Offsets:  [0, 3, 5, 9]     // Start index for each point, + total at end
Indices:  [1, 2, 3, 0, 2, 0, 1, 3, 4]  // Flattened neighbor lists
           ^^^^^^  ^^^^  ^^^^^^^^^^^
           pt 0    pt 1    pt 2
```

```cpp
int* d_neighborOffsets;   // [numPoints + 1] - start indices
int* d_neighborIndices;   // [totalNeighbors] - neighbor point indices

// In kernel: get neighbors of point idx
int start = neighborOffsets[idx];
int end = neighborOffsets[idx + 1];
for (int i = start; i < end; i++) {
    int neighborIdx = neighborIndices[i];
    // Process neighbor...
}
```

### Triangle List

Three vertices per triangle:

```cpp
int3* d_triangles;  // [(v0,v1,v2), (v0,v1,v2), ...]

// Or flat array
int* d_triangleIndices;  // [v0, v1, v2, v0, v1, v2, ...]
                         //   tri 0       tri 1
```

### Quad List (Bend Constraints)

Four vertices for dihedral angle calculations:

```cpp
int4* d_bendQuads;       // [(v0, v1, v2, v3), ...] - shared edge v0-v1
float* d_restDihedrals;  // Rest angles
```

---

## Memory Lifecycle

### Initialization Pattern

```cpp
class CudaBridge {
public:
    CudaBridge() {
        // Initialize all pointers to nullptr
        m_d_positions = nullptr;
        m_d_velocities = nullptr;
        m_d_normals = nullptr;
    }

    void initialize(int numPoints) {
        m_numPoints = numPoints;

        // Allocate all buffers
        allocateGPU(&m_d_positions, numPoints);
        allocateGPU(&m_d_velocities, numPoints);
        allocateGPU(&m_d_normals, numPoints);

        // Initialize velocity to zero
        cudaMemset(m_d_velocities, 0, numPoints * sizeof(float3));
    }
```

### Reset Pattern

Clear data without deallocating:

```cpp
void CudaBridge::reset() {
    if (m_d_positions) {
        cudaMemset(m_d_positions, 0, m_numPoints * sizeof(float3));
    }
    if (m_d_velocities) {
        cudaMemset(m_d_velocities, 0, m_numPoints * sizeof(float3));
    }
}
```

### Cleanup Pattern

```cpp
CudaBridge::~CudaBridge() {
    cleanup();
}

void CudaBridge::cleanup() {
    allocateGPU(&m_d_positions, 0);   // Free via helper
    allocateGPU(&m_d_velocities, 0);
    allocateGPU(&m_d_normals, 0);
    allocateGPU(&m_d_neighborOffsets, 0);
    allocateGPU(&m_d_neighborIndices, 0);
    m_numPoints = 0;
}
```

---

## Double Buffering

For operations that read and write the same data:

```cpp
// Two position buffers
float3* m_d_positions;    // Current positions
float3* m_d_positionsNew; // New positions

void smooth() {
    // Read from positions, write to positionsNew
    launchSmooth(m_d_positions, m_d_positionsNew, ...);

    // Swap pointers
    std::swap(m_d_positions, m_d_positionsNew);
}
```

---

## Pinned Memory (for frequent transfers)

Pinned (page-locked) memory enables faster CPU↔GPU transfers:

```cpp
// Allocate pinned host memory
float3* h_positions;
cudaMallocHost(&h_positions, numPoints * sizeof(float3));

// Use like normal memory
for (int i = 0; i < numPoints; i++) {
    h_positions[i] = make_float3(...);
}

// Transfer is faster than regular memory
cudaMemcpy(d_positions, h_positions, size, cudaMemcpyHostToDevice);

// Free with cudaFreeHost
cudaFreeHost(h_positions);
```

**When to use:**
- Frequent uploads/downloads every frame
- Large data transfers
- Async transfers with streams

---

## Memory Size Estimation

```cpp
size_t estimateGPUMemory(int numPoints, int numEdges, int numTriangles) {
    size_t total = 0;

    // Positions, velocities, normals
    total += numPoints * sizeof(float3) * 3;

    // Neighbor CSR (estimate 6 neighbors per point)
    total += (numPoints + 1) * sizeof(int);  // Offsets
    total += numPoints * 6 * sizeof(int);    // Indices

    // Edges
    total += numEdges * sizeof(int2);
    total += numEdges * sizeof(float);  // Rest lengths

    // Triangles
    total += numTriangles * sizeof(int3);

    return total;
}
```

---

## Best Practices

1. **Initialize pointers to nullptr** - Prevents accidental double-free
2. **Use allocation helpers** - Consistent, safe memory management
3. **Sync before realloc** - `cudaDeviceSynchronize()` before freeing active buffers
4. **Check allocation errors** - GPU can run out of memory
5. **Minimize transfers** - Keep data on GPU as long as possible
6. **Use CSR for sparse data** - More efficient than dense arrays
7. **Profile memory usage** - Use `nvidia-smi` or Nsight to monitor
