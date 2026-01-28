---
layout: default
title: Binding Guide
parent: OpenCL
nav_order: 1
description: OpenCL binding setup in Houdini
permalink: /opencl/binding-guide/
---

# Houdini OpenCL Binding Guide

This guide explains how to properly set up bindings in Houdini's OpenCL nodes.

---

## Understanding Bindings

Bindings tell Houdini how to pass data between the CPU and GPU. There are two syntaxes:
1. **Manual bindings** - Set in the Bindings tab
2. **@-bindings** - Automatic setup using `#bind` directives

---

## @-Bindings Syntax (Recommended)

Enable @-bindings in the OpenCL node options. Use `#bind` directives to declare data.

### Basic Syntax
```c
#bind <class> [modifiers]<name> <type> [options]
```

### Classes
| Class | Description |
|-------|-------------|
| `point` | Point attributes |
| `prim` | Primitive attributes |
| `vertex` | Vertex attributes |
| `detail` | Detail (global) attributes |
| `parm` | Node parameters |
| `layer` | COP layers (Copernicus) |
| `ramp` | Ramp parameters |

### Modifiers
| Modifier | Meaning |
|----------|---------|
| `&` | Writeable |
| `?` | Optional (won't error if missing) |
| `!` | No read (write only) |

### Examples
```c
#bind point &P float3        // Writeable point position
#bind point N float3         // Read-only normals
#bind point &Cd? float3      // Optional writeable color
#bind parm scale float val=1 // Parameter with default
#bind detail &sum float      // Writeable detail attribute
```

### Data Types
| Type | Description |
|------|-------------|
| `int` | Integer |
| `float` | Float |
| `float2` | Vector2 |
| `float3` | Vector |
| `float4` | Vector4/Quaternion |
| `float9` | Matrix3 |
| `float16` | Matrix4 |
| `int[]` | Integer array |
| `float[]` | Float array |
| `fpreal` | Variable precision float |
| `fpreal3` | Variable precision vector |
| `exint` | Variable precision integer |

---

## Manual Bindings (Bindings Tab)

When @-bindings are disabled, configure bindings in the Bindings tab.

### Critical Rule
**Binding order in the tab MUST match kernel parameter order exactly!**

Example bindings:
```
1. P (float, size 3, readable, writeable)
2. N (float, size 3, readable)
3. noise (float, size 1, readable)
```

Must match:
```c
kernel void kernelName(
    int P_length,           // 1st binding
    global float* P,
    int N_length,           // 2nd binding
    global float* N,
    int noise_length,       // 3rd binding
    global float* noise
)
```

### Binding Types

#### Float/Integer Attributes
Adds 2 kernel arguments:
```c
int attr_length,        // Number of entries
global float* attr      // Data array
```

#### Vector Attributes
Same as float, but data is interleaved:
```c
// Point 0: P[0], P[1], P[2]
// Point 1: P[3], P[4], P[5]
// ...
```
Use `vload3(idx, P)` and `vstore3(val, idx, P)`.

#### Array Attributes
Adds 3 kernel arguments:
```c
int attr_length,        // Number of entries
global int* attr_index, // Start index of each subarray
global float* attr      // Flattened array data
```

#### Volume Attributes
Adds many kernel arguments for strides, resolution, transforms:
```c
int density_stride_x,
int density_stride_y,
int density_stride_z,
int density_stride_offset,
int density_res_x,
int density_res_y,
int density_res_z,
float density_voxelsize_x,
float density_voxelsize_y,
float density_voxelsize_z,
float16 density_xformtoworld,
float16 density_xformtovoxel,
global float* density
```

---

## Reading and Writing Data

### @-Bindings Syntax

```c
// Read
float3 pos = @P;
float val = @noise;

// Write
@P.set(pos + offset);
@Cd.set((float3)(1, 0, 0));

// Access specific element
float3 other = @P.getAt(5);  // Get point 5's position
@P.setAt(10, newPos);        // Set point 10's position

// Array length
int count = @P.len;
```

### Plain OpenCL Syntax

```c
// Read float
float val = noise[idx];

// Write float
noise[idx] = val + 1.0f;

// Read vector
float3 pos = vload3(idx, P);

// Write vector
vstore3(pos, idx, P);

// Read vector4 (quaternion)
float4 orient = vload4(idx, orient);

// Write vector4
vstore4(orient, idx, orient);
```

---

## Matrix Operations

Include the matrix header:
```c
#include <matrix.h>
```

### Creating Matrices
```c
mat3 m;
mat3zero(m);      // Fill with zeros
mat3identity(m);  // Identity matrix
```

### Reading/Writing Matrix Attributes
```c
// Read matrix3
mat3 m;
mat3load(idx, matrix_attr, m);

// Write matrix3
mat3store(m, idx, matrix_attr);
```

### Matrix Operations
```c
// Transform vector by matrix
float3 result = mat3vecmul(m, vec);

// Matrix multiplication
mat3 c;
mat3mul(a, b, c);

// Transpose
mat3 t;
transpose3(m, t);
```

---

## Volume Bindings

Volumes require special handling for voxel access.

### Calculate Voxel Index
```c
int gidx = get_global_id(0);
int gidy = get_global_id(1);
int gidz = get_global_id(2);

int idx = density_stride_offset
        + density_stride_x * gidx
        + density_stride_y * gidy
        + density_stride_z * gidz;
```

### Get World Position
```c
float4 worldPos = gidx * density_xformtoworld.lo.lo +
                  gidy * density_xformtoworld.lo.hi +
                  gidz * density_xformtoworld.hi.lo +
                  1 * density_xformtoworld.hi.hi;
```

### Read/Write Voxel
```c
float value = density[idx];
density[idx] = newValue;
```

---

## COP/Copernicus Bindings

### Layer Bindings
```c
#bind layer src? val=0   // Input layer (optional)
#bind layer !&dst        // Output layer (write-only)
```

### Accessing Pixels
```c
// Read current pixel
float4 color = @src;

// Write current pixel
@dst.set(color);

// Sample at UV coordinates
float4 sampled = @src.textureSample(uv);

// Sample at image coordinates
float4 sampled = @src.imageSample(pos);
```

### Built-in Variables
```c
@ix, @iy        // Integer pixel coordinates
@ixy            // int2 pixel coordinates
@xres, @yres    // Image resolution
@res            // int2 resolution
@P              // Normalized position (0-1)
@P.texture      // float2 UV coordinates
@Time           // Current time
@Frame          // Current frame
```

---

## Parameter Bindings

### Node Parameters
```c
#bind parm scale float val=1.0
#bind parm offset float3 val=0
#bind parm iterations int val=10
```

### Ramp Parameters
```c
#bind ramp myRamp float val=0

// Usage
float result = @myRamp(0.5);  // Sample at 0.5
```

---

## Common Binding Patterns

### Point Displacement
```c
#bind point &P float3
#bind point N float3
#bind parm amount float val=1

@KERNEL
{
    @P.set(@P + @N * @amount);
}
```

### Global Accumulation (Detail Attribute)
```c
#bind point values float
#bind detail &sum float

@KERNEL
{
    // Use atomics for parallel safety
    atomic_add_f(@sum.data, @values);
}
```

### Two-Pass with Scratch Attribute
```c
#bind point &P float3
#bind point &__scratch float3

// First kernel writes to scratch
// Second kernel reads scratch, writes to P
```

---

## Troubleshooting

### "Parameter count mismatch"
- Check binding order matches kernel parameter order
- Verify all attributes exist

### "CL_OUT_OF_RESOURCES"
- Too much data bound
- Reduce bound attributes
- Use compiled blocks

### Memory Leaks/Crashes
- Missing bounds check: `if (idx >= length) return;`
- Writing out of bounds
- Wrong vload/vstore size

### Wrong Results
- Using `1.0` instead of `1.0f` (double vs float)
- Missing `&` for writeable attributes
- Incorrect binding type (size mismatch)
