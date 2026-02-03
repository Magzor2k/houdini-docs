---
layout: default
title: Learning Resources
parent: OpenCL
nav_order: 2
description: OpenCL learning resources for Houdini
permalink: /opencl/learning-resources/
---

# Houdini OpenCL Learning Resources

A curated list of tutorials, documentation, and learning materials for OpenCL in Houdini.

---

## Official SideFX Documentation

### Node Documentation

| Resource | Description |
|:---------|:------------|
| [OpenCL SOP](https://www.sidefx.com/docs/houdini/nodes/sop/opencl.html) | Main SOP documentation |
| [Gas OpenCL DOP](https://www.sidefx.com/docs/houdini/nodes/dop/gasopencl.html) | DOPs documentation |
| [OpenCL COP](https://www.sidefx.com/docs/houdini/nodes/cop/opencl.html) | Copernicus documentation |
| [OpenCL for VEX Users](https://www.sidefx.com/docs/houdini/vex/ocl.html) | VEX-to-OpenCL guide |

### Official Example Files

| Resource | Description |
|:---------|:------------|
| [SOP Snippets](https://www.sidefx.com/docs/houdini/examples/nodes/sop/opencl/SimpleOpenCLSOPSnippets.html) | SOP examples |
| [COP Snippets](https://www.sidefx.com/docs/houdini/examples/nodes/cop/opencl/SimpleOpenCLCOPSnippets.html) | COP examples |
| [OpenCL Smoke](https://www.sidefx.com/docs/houdini/examples/nodes/dop/smokeobject/OpenCL.html) | Pyro example |

---

## Video Tutorials

### SideFX Official

| Title | Presenter | Level |
|:------|:----------|:------|
| [H16.5 Masterclass: OpenCL](https://www.sidefx.com/tutorials/houdini-165-masterclass-opencl/) | Jeff Lait | Intermediate |
| [OpenCL COP for VEX Users](https://www.sidefx.com/tutorials/opencl-cop-for-vex-users/) | Luca Pataracchia | Beginner |
| [Minimal OpenCL Solver (Explosion)](https://www.sidefx.com/tutorials/minimal-opencl-solver-part-01-explosion/) | SideFX | Intermediate |
| [Speed Up Fluid with OpenCL](https://www.sidefx.com/tutorials/speed-up-fluid-simulation-using-opencl/) | SideFX | Beginner |
| [Create Wetmaps with COPS](https://www.sidefx.com/tutorials/create-wetmaps-with-cops/) | SideFX | Intermediate |

### Community Tutorials

| Title | Creator | Level |
|:------|:--------|:------|
| [OpenCL for Copernicus Part 1](https://youtu.be/UX5y14cTZpk) | Konstantin Magnus | Beginner |
| [OpenCL for Copernicus Part 2](https://youtu.be/OGhumSrYzTw) | Konstantin Magnus | Intermediate |
| [OpenCL Copernicus Feedback Loops](https://github.com/jhorikawa/HoudiniHowtos/tree/master/Live-0140) | Junichiro Horikawa | Intermediate |
| [NileRed Logo with OpenCL](https://entagma.com/simulating-the-nilered-logo-with-copernicus-and-opencl/) | Entagma | Advanced |

---

## GitHub Repositories

### Comprehensive Guides

| Repository | Description |
|:-----------|:------------|
| [MysteryPancake/Houdini-OpenCL](https://github.com/MysteryPancake/Houdini-OpenCL) | Best overall guide with examples |
| [JoseZalez/Houdini-scripts](https://github.com/JoseZalez/Houdini-scripts) | OpenCL kernels with documentation |
| [toby5001/Houdini-Snippets](https://github.com/toby5001/Houdini-Snippets) | VEX and OpenCL snippets |

### Specialized Examples

| Repository | Description |
|:-----------|:------------|
| [Amir-Ashkezari/Houdini-TaubinSmoothingCL](https://github.com/Amir-Ashkezari/Houdini-TaubinSmoothingCL) | GPU polygon smoothing |
| [melMass/cops-cl](https://github.com/melMass/cops-cl) | Copernicus experiments |
| [thi-ng/vexed-generation](https://github.com/thi-ng/vexed-generation) | Polymorphic VEX/OpenCL helpers |
| [MysteryPancake/Houdini-VBD](https://github.com/MysteryPancake/Houdini-VBD) | Vertex Block Descent solver |

### Resource Collections

| Repository | Description |
|:-----------|:------------|
| [wyhinton/AwesomeHoudini](https://github.com/wyhinton/AwesomeHoudini) | Curated Houdini resources |
| [Aeoll/Aelib](https://github.com/Aeoll/Aelib) | ~100 HDAs and tools |

---

## Blogs & Websites

| Site | Description |
|:-----|:------------|
| [Procegen](https://procegen.konstantinmagnus.de/opencl-resources) | OpenCL resources for COPs/SOPs |
| [cgwiki](https://www.tokeru.com/cgwiki/HoudiniHDA.html) | Matt Estela's Houdini tips |
| [Houdini Gubbins](https://houdinigubbins.wordpress.com/tag/opencl/) | OpenCL tutorials |
| [ikrima Gamedev Guide](https://ikrima.dev/houdini/dops/dop-opencl/) | DOP OpenCL guide |
| [VFX Mentor](https://thevfxmentor.com/quicktips/COP) | COP tips and tricks |

---

## Forums & Community

### SideFX Forums

| Topic | Description |
|:------|:------------|
| [OpenCL Snippets Search](https://www.sidefx.com/forum/topic/92954/) | Community snippet collection |
| [OpenCL Libraries Discussion](https://www.sidefx.com/forum/topic/97120/) | Library discussion |
| [OpenCL Wrangle SOP](https://www.sidefx.com/forum/topic/44718/) | HDA 144x faster than VEX |

### od|forum

| Topic | Description |
|:------|:------------|
| [Gas OpenCL](https://forums.odforce.net/topic/26391-gas-opencl/) | DOPs discussion |
| [OpenCL Wrangle](https://forums.odforce.net/topic/26679-opencl-wrangle-from-animatrix/) | Animatrix's wrangle |
| [OpenCL and VDB](https://forums.odforce.net/topic/50616-opencl-and-vdb-examples/) | VDB examples |

---

## Built-in Houdini Resources

### OCL Library Location

```
$HFS/houdini/ocl/
```

On Windows typically:

```
C:/Program Files/Side Effects Software/Houdini XX.X.XXX/houdini/ocl/
```

### Key Headers

| Header | Location | Purpose |
|:-------|:---------|:--------|
| matrix.h | `$HH/ocl/include/matrix.h` | Matrix operations |
| interpolate.h | `$HH/ocl/include/interpolate.h` | Interpolation |
| xnoise.h | `$HH/ocl/include/xnoise.h` | Simplex/curl noise |
| reduce.h | `$HH/ocl/include/reduce.h` | Workgroup reduction |
| color.h | `$HH/ocl/include/color.h` | Color space conversions |

### Embedded OpenCL Examples

Look inside these nodes for real-world OpenCL usage:
- Ripple Solver
- Vellum solver (XPBD)
- Copernicus nodes (Dilate/Erode, etc.)

---

## Learning Path

### Beginner

1. Read "OpenCL for VEX Users" documentation
2. Watch Jeff Lait's H16.5 Masterclass
3. Study the SimpleOpenCLSOPSnippets example
4. Try converting simple VEX to OpenCL

### Intermediate

1. Learn @-bindings syntax
2. Understand workgroups and parallel execution
3. Study volume operations
4. Practice with Copernicus

### Advanced

1. Implement custom solvers
2. Learn atomic operations and workgroup reduction
3. Study Vellum/VBD solver patterns
4. Optimize with memory management

---

## Key Technical Notes

### Performance Tips

- Chain OpenCL nodes in compiled blocks to reduce GPU<->CPU transfers
- OpenCL is only faster when data stays on GPU
- Avoid binding unnecessary data
- Use workgroup reduction for global accumulation

### Common Gotchas

- Binding order must match kernel parameter order
- Always check bounds: `if (idx >= length) return;`
- No VEX functions available (no `intersect()`, `xyzdist()`)
- Floats require `f` suffix: `1.0f` not `1.0`
- Limited matrix support (use `matrix.h`)

### Houdini 20.5+ Features

- `#import` directive for prequel code
- Improved Copernicus integration
- Better error messages
