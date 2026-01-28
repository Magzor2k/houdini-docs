---
layout: default
title: CUDA Checklist
parent: Code Review
nav_order: 3
description: CUDA-specific code review checklist for GPU kernels and host code
permalink: /code-review/checklists/cuda/
---

# CUDA Code Review Checklist

Apply these checks to CUDA kernels, device code, and host-side GPU management.

## Memory Management

| Check | Description |
|:------|:------------|
| Allocation/Deallocation Paired | Every `cudaMalloc` has matching `cudaFree` |
| Host/Device Memory Correct | Using right memory type for context |
| No Double Free | Memory not freed multiple times |
| Unified Memory Appropriate | `cudaMallocManaged` used correctly |
| Memory Limits Checked | Large allocations validate available memory |

### Memory Patterns
```cpp
// Good - RAII wrapper
class CudaBuffer {
    void* ptr = nullptr;
public:
    CudaBuffer(size_t size) {
        cudaMalloc(&ptr, size);
    }
    ~CudaBuffer() {
        if (ptr) cudaFree(ptr);
    }
};

// Bad - manual management without cleanup
void* ptr;
cudaMalloc(&ptr, size);
// ... code that might throw ...
// cudaFree never called on error path
```

### Memory Transfer
```cpp
// Explicit direction
cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
cudaMemcpy(h_result, d_result, size, cudaMemcpyDeviceToHost);

// Async with streams
cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, stream);
```

## Thread Safety

| Check | Description |
|:------|:------------|
| No Race Conditions | Shared data properly synchronized |
| Atomic Operations | Use atomics for concurrent updates |
| Proper Barriers | `__syncthreads()` where needed |
| No Deadlocks | All threads in block reach barriers |
| Warp Divergence Minimized | Branches affect whole warps |

### Synchronization Patterns
```cpp
// Good - all threads reach barrier
__global__ void kernel(float* data, int n) {
    __shared__ float shared[256];

    int tid = threadIdx.x;
    shared[tid] = data[blockIdx.x * blockDim.x + tid];

    __syncthreads();  // All threads must reach this

    // Now safe to read any shared[i]
}

// Bad - conditional barrier (deadlock risk)
if (threadIdx.x < 128) {
    __syncthreads();  // WRONG: not all threads reach this
}
```

### Atomic Operations
```cpp
// Thread-safe increment
atomicAdd(&counter, 1);

// Thread-safe min/max
atomicMin(&min_val, local_val);
atomicMax(&max_val, local_val);
```

## Kernel Bounds

| Check | Description |
|:------|:------------|
| Grid/Block Size Valid | Dimensions within device limits |
| Array Bounds Checked | Threads check they're in valid range |
| Shared Memory Fits | Shared memory within block limit |
| Register Pressure | Not exceeding register limits |

### Bounds Checking Pattern
```cpp
__global__ void kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // CRITICAL: bounds check before access
    if (idx >= n) return;

    data[idx] = process(data[idx]);
}

// Launch with ceiling division
int blockSize = 256;
int numBlocks = (n + blockSize - 1) / blockSize;
kernel<<<numBlocks, blockSize>>>(d_data, n);
```

### Device Limits
```cpp
// Query and respect device limits
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);

// Max threads per block
int maxThreads = prop.maxThreadsPerBlock;  // Usually 1024

// Max shared memory per block
size_t maxShared = prop.sharedMemPerBlock;  // Usually 48KB

// Max grid dimensions
int maxGridX = prop.maxGridSize[0];  // Usually 2^31-1
```

## Error Checking

| Check | Description |
|:------|:------------|
| API Calls Checked | All CUDA API returns checked |
| Kernel Launches Checked | `cudaGetLastError` after kernels |
| Async Errors Caught | `cudaDeviceSynchronize` for debugging |
| Meaningful Error Messages | Errors include context |

### Error Checking Macro
```cpp
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Usage
CUDA_CHECK(cudaMalloc(&d_ptr, size));
CUDA_CHECK(cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice));

kernel<<<blocks, threads>>>(d_ptr, n);
CUDA_CHECK(cudaGetLastError());  // Check kernel launch
CUDA_CHECK(cudaDeviceSynchronize());  // Check kernel execution
```

## Memory Coalescing

| Check | Description |
|:------|:------------|
| Sequential Access | Adjacent threads access adjacent memory |
| Aligned Access | Memory accesses aligned to 32/128 bytes |
| Stride-Free Reads | No strided access patterns |
| Structure of Arrays | SoA preferred over AoS |

### Access Patterns
```cpp
// Good - coalesced access
// Thread 0 reads data[0], thread 1 reads data[1], etc.
__global__ void coalesced(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = data[idx];  // Coalesced
    }
}

// Bad - strided access
// Thread 0 reads data[0], thread 1 reads data[stride], etc.
__global__ void strided(float* data, int stride, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = data[idx * stride];  // Poor memory efficiency
    }
}
```

### Structure of Arrays (SoA)
```cpp
// Bad - Array of Structures (AoS)
struct Particle { float x, y, z, w; };
Particle* particles;
// Access: particles[i].x - strided in memory

// Good - Structure of Arrays (SoA)
struct ParticleArrays {
    float* x;
    float* y;
    float* z;
    float* w;
};
// Access: particles.x[i] - coalesced
```

## Shared Memory

| Check | Description |
|:------|:------------|
| Bank Conflicts Minimized | Access patterns avoid same bank |
| Proper Padding | Padding added to avoid conflicts |
| Size Within Limits | Shared memory fits block limit |
| Lifetime Understood | Shared memory scope is block |

### Bank Conflict Avoidance
```cpp
// Bad - bank conflicts (32 threads hit same bank)
__shared__ float shared[32][32];
float val = shared[threadIdx.x][0];  // All access bank 0

// Good - padded to avoid conflicts
__shared__ float shared[32][33];  // +1 padding
float val = shared[threadIdx.x][0];  // Spread across banks
```

### Shared Memory Patterns
```cpp
__global__ void tiled_matmul(float* A, float* B, float* C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;
    for (int t = 0; t < N / TILE_SIZE; ++t) {
        // Load tiles into shared memory
        As[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];

        __syncthreads();

        // Compute using shared memory
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    C[row * N + col] = sum;
}
```

## Occupancy

| Check | Description |
|:------|:------------|
| Block Size Optimal | Block size maximizes occupancy |
| Register Usage Known | Compile with `-Xptxas -v` to check |
| Shared Memory Considered | Balance shared memory vs occupancy |
| Launch Bounds Set | `__launch_bounds__` for critical kernels |

### Occupancy Calculation
```cpp
// Query occupancy
int minGridSize, blockSize;
cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel, 0, 0);

// Launch with optimal config
kernel<<<minGridSize, blockSize>>>(args);
```

## Summary Checklist

Quick reference for CUDA review:

- [ ] **Memory Management**: Proper allocation/deallocation
- [ ] **Thread Safety**: No race conditions, proper synchronization
- [ ] **Kernel Bounds**: Grid/block size validation, array bounds
- [ ] **Error Checking**: CUDA error codes checked
- [ ] **Memory Coalescing**: Efficient memory access patterns
- [ ] **Shared Memory**: Proper bank conflict avoidance
- [ ] **Occupancy**: Block sizes optimized for hardware
