---
layout: default
title: Debugging
parent: CUDA
nav_order: 4
description: CUDA debugging techniques for Houdini
permalink: /cuda/debugging/
---

# CUDA Debugging
{: .fs-9 }

Error handling, diagnostics, and debugging techniques.
{: .fs-6 .fw-300 }

---

## Error Checking Macros

### CUDA_CHECK Macro

Wrap every CUDA call with error checking:

```cpp
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(error)); \
        } \
    } while(0)

// Usage
CUDA_CHECK(cudaMalloc(&d_positions, size));
CUDA_CHECK(cudaMemcpy(d_positions, h_positions, size, cudaMemcpyHostToDevice));
```

### cuBLAS Error Checking

```cpp
#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", \
                    __FILE__, __LINE__, (int)status); \
        } \
    } while(0)

// Usage
CUBLAS_CHECK(cublasSdot(handle, n, x, 1, y, 1, &result));
```

### cuSPARSE Error Checking

```cpp
#define CUSPARSE_CHECK(call) \
    do { \
        cusparseStatus_t status = call; \
        if (status != CUSPARSE_STATUS_SUCCESS) { \
            fprintf(stderr, "cuSPARSE error at %s:%d: %d\n", \
                    __FILE__, __LINE__, (int)status); \
        } \
    } while(0)
```

### Kernel Launch Error Checking

Check for errors after kernel launches:

```cpp
void launchMyKernel(float3* d_data, int count) {
    int blockSize = 256;
    int numBlocks = (count + blockSize - 1) / blockSize;

    myKernel<<<numBlocks, blockSize>>>(d_data, count);

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n",
                cudaGetErrorString(err));
    }

    // Optional: check for execution errors (slower - forces sync)
    #ifdef DEBUG
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel execution failed: %s\n",
                cudaGetErrorString(err));
    }
    #endif
}
```

---

## GPU-Side Validity Checking

### Explosion Detection Kernel

Detect invalid values during simulation:

```cpp
// Flags for different error types (bit field)
#define BAD_POSITION    (1 << 0)  // NaN or out of bounds
#define BAD_VELOCITY    (1 << 1)  // NaN or excessive speed
#define CLOTH_SCRAMBLED (1 << 2)  // Extent too large
#define PINNED_DRIFT    (1 << 3)  // Pinned point moved

__global__ void checkExplosion(
    const float3* __restrict__ positions,
    const float3* __restrict__ velocities,
    const int* __restrict__ pinnedMask,
    const float3* __restrict__ restPositions,
    int* __restrict__ errorFlags,
    float positionThreshold,
    float velocityThreshold,
    float pinnedTolerance,
    int numPoints
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPoints) return;

    float3 pos = positions[idx];
    float3 vel = velocities[idx];

    int flags = 0;

    // Check for NaN
    if (isnan(pos.x) || isnan(pos.y) || isnan(pos.z)) {
        flags |= BAD_POSITION;
    }

    // Check for extreme values
    if (fabsf(pos.x) > positionThreshold ||
        fabsf(pos.y) > positionThreshold ||
        fabsf(pos.z) > positionThreshold) {
        flags |= BAD_POSITION;
    }

    // Check velocity
    if (isnan(vel.x) || isnan(vel.y) || isnan(vel.z)) {
        flags |= BAD_VELOCITY;
    }

    float speed = sqrtf(vel.x*vel.x + vel.y*vel.y + vel.z*vel.z);
    if (speed > velocityThreshold) {
        flags |= BAD_VELOCITY;
    }

    // Check pinned point drift
    if (pinnedMask[idx]) {
        float3 rest = restPositions[idx];
        float drift = sqrtf(
            (pos.x - rest.x) * (pos.x - rest.x) +
            (pos.y - rest.y) * (pos.y - rest.y) +
            (pos.z - rest.z) * (pos.z - rest.z)
        );
        if (drift > pinnedTolerance) {
            flags |= PINNED_DRIFT;
        }
    }

    // Write flags using atomic OR
    if (flags != 0) {
        atomicOr(errorFlags, flags);
    }
}

// Host function to check results
bool checkSimulationHealth(CudaBridge* bridge) {
    int h_flags = 0;
    int* d_flags;
    cudaMalloc(&d_flags, sizeof(int));
    cudaMemset(d_flags, 0, sizeof(int));

    launchCheckExplosion(
        bridge->positions(), bridge->velocities(),
        bridge->pinnedMask(), bridge->restPositions(),
        d_flags, 1000.0f, 100.0f, 0.01f, bridge->numPoints()
    );

    cudaMemcpy(&h_flags, d_flags, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_flags);

    if (h_flags != 0) {
        if (h_flags & BAD_POSITION) fprintf(stderr, "Bad position detected!\n");
        if (h_flags & BAD_VELOCITY) fprintf(stderr, "Bad velocity detected!\n");
        if (h_flags & CLOTH_SCRAMBLED) fprintf(stderr, "Cloth scrambled!\n");
        if (h_flags & PINNED_DRIFT) fprintf(stderr, "Pinned point drifted!\n");
        return false;
    }
    return true;
}
```

### NaN/Inf Validation

Simple inline checks for kernels:

```cpp
__device__ bool isValidFloat3(float3 v) {
    return !isnan(v.x) && !isnan(v.y) && !isnan(v.z) &&
           !isinf(v.x) && !isinf(v.y) && !isinf(v.z);
}

__global__ void safeComputation(float3* positions, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float3 pos = positions[idx];

    // Validate input
    if (!isValidFloat3(pos)) {
        positions[idx] = make_float3(0, 0, 0);  // Reset to safe value
        return;
    }

    // Computation...
    float3 result = ...;

    // Validate output
    if (!isValidFloat3(result)) {
        result = pos;  // Keep original if computation failed
    }

    positions[idx] = result;
}
```

---

## Debug Logging

### File-Based Debug Output

Write debug info from the SOP node:

```cpp
// In SOP node header
class SOP_MyNode : public SOP_Node {
    std::ofstream m_debugFile;
    bool m_debugEnabled = false;
};

// In cookMySop
void SOP_MyNode::writeDebugInfo(int frame, const char* phase) {
    if (!m_debugEnabled) return;

    if (!m_debugFile.is_open()) {
        m_debugFile.open("C:/temp/my_debug.txt");
    }

    m_debugFile << "Frame " << frame << " - " << phase << std::endl;

    // Download sample positions
    std::vector<float3> positions(m_bridge->numPoints());
    cudaMemcpy(positions.data(), m_bridge->positions(),
               positions.size() * sizeof(float3), cudaMemcpyDeviceToHost);

    // Write sample data
    for (int i = 0; i < std::min(10, (int)positions.size()); i++) {
        m_debugFile << "  Point " << i << ": ("
                    << positions[i].x << ", "
                    << positions[i].y << ", "
                    << positions[i].z << ")" << std::endl;
    }

    m_debugFile.flush();
}
```

### CSV Position Dumps

Export positions for frame-by-frame analysis:

```cpp
void CudaBridge::dumpPositionsToCSV(const char* filename) {
    // Download positions
    std::vector<float3> positions(m_numPoints);
    cudaMemcpy(positions.data(), m_d_positions,
               m_numPoints * sizeof(float3), cudaMemcpyDeviceToHost);

    // Write CSV
    std::ofstream file(filename);
    file << "point,x,y,z" << std::endl;
    for (int i = 0; i < m_numPoints; i++) {
        file << i << ","
             << positions[i].x << ","
             << positions[i].y << ","
             << positions[i].z << std::endl;
    }
}

// Usage
char filename[256];
sprintf(filename, "C:/temp/positions_frame_%04d.csv", frame);
m_bridge->dumpPositionsToCSV(filename);
```

### Per-Substep Logging

For simulations with substeps:

```cpp
void SOP_ClothSim::simulate(float dt, int substeps) {
    float subDt = dt / substeps;

    for (int sub = 0; sub < substeps; sub++) {
        // Pre-step diagnostics
        if (m_debugEnabled) {
            logSubstepStart(sub, subDt);
        }

        // Integration
        m_bridge->integrate(subDt);

        // Constraint solving
        for (int iter = 0; iter < m_iterations; iter++) {
            m_bridge->solveConstraints();

            // Per-iteration logging (expensive!)
            if (m_verboseDebug) {
                float residual = m_bridge->computeResidual();
                fprintf(stderr, "  Substep %d, Iter %d: residual = %f\n",
                        sub, iter, residual);
            }
        }

        // Post-step diagnostics
        if (m_debugEnabled) {
            if (!checkSimulationHealth(m_bridge)) {
                fprintf(stderr, "Explosion at substep %d!\n", sub);
                break;
            }
        }
    }
}
```

---

## Debug Parameters

Add debug toggles to your SOP node:

```cpp
// Parameter definitions
static PRM_Name debugName("debug", "Debug Mode");
static PRM_Name dumpPositionsName("dumppositions", "Dump Positions");
static PRM_Name debugFilePath("debugpath", "Debug File Path");

PRM_Template SOP_MyNode::myTemplateList[] = {
    // ... other parameters ...

    PRM_Template(PRM_SEPARATOR, 1, &sep2Name),
    PRM_Template(PRM_TOGGLE, 1, &debugName, PRMzeroDefaults),
    PRM_Template(PRM_TOGGLE, 1, &dumpPositionsName, PRMzeroDefaults),
    PRM_Template(PRM_FILE, 1, &debugFilePath,
                 new PRM_Default(0, "C:/temp/debug.txt")),

    PRM_Template()
};
```

---

## Common Issues and Solutions

### Issue: Kernel produces zeros

**Cause:** Data not uploaded, wrong pointer
```cpp
// Check
if (m_d_positions == nullptr) {
    fprintf(stderr, "Positions not allocated!\n");
}
```

### Issue: Kernel produces NaN

**Cause:** Division by zero, uninitialized memory
```cpp
// Add safety
float denom = ...;
if (fabsf(denom) < 1e-8f) denom = 1e-8f;
float result = value / denom;
```

### Issue: Results don't change

**Cause:** Writing to wrong buffer, not downloading
```cpp
// Verify download
cudaDeviceSynchronize();
cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
// Check h_data values
```

### Issue: Crash on second cook

**Cause:** Double-free, stale pointers
```cpp
// Always null after free
cudaFree(d_ptr);
d_ptr = nullptr;
```

### Issue: Different results each run

**Cause:** Race conditions, uninitialized memory
```cpp
// Initialize memory
cudaMemset(d_buffer, 0, size);
```

---

## NVIDIA Tools

### nvidia-smi

Monitor GPU usage:
```bash
nvidia-smi
# Shows memory usage, GPU utilization, temperature
```

### Nsight Systems

Profile your application:
1. Install NVIDIA Nsight Systems
2. Launch: `nsys profile your_app.exe`
3. View timeline of CPU/GPU activity

### Nsight Compute

Detailed kernel analysis:
1. Install NVIDIA Nsight Compute
2. Profile specific kernels
3. View occupancy, memory throughput, bottlenecks

### cuda-memcheck (older GPUs)

Check for memory errors:
```bash
cuda-memcheck your_app.exe
```

---

## Debug Build Configuration

In CMakeLists.txt:

```cmake
# Debug configuration
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(MyTarget PRIVATE DEBUG=1)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -lineinfo")
endif()
```

In code:
```cpp
#ifdef DEBUG
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());
#endif
```
