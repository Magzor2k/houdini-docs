---
layout: home
title: Home
nav_order: 1
description: Central documentation hub for Houdini development tools and techniques
permalink: /
---

# Houdini R&D Documentation

Central documentation for Houdini development tools, HDK extensions, and Python scripting.

---

## Quick Topic Lookup

| Looking for... | Go to |
|:---------------|:------|
| **Viewer States** | |
| Python viewer states overview | [viewer-states/](viewer-states/index.md) |
| Getting started tutorial | [viewer-states/getting-started.md](viewer-states/getting-started.md) |
| State class structure | [viewer-states/reference/state-class.md](viewer-states/reference/state-class.md) |
| Mouse/keyboard events | [viewer-states/reference/events.md](viewer-states/reference/events.md) |
| Drawables (points, lines, HUD) | [viewer-states/reference/drawables.md](viewer-states/reference/drawables.md) |
| Registration & file naming (`_state.py`) | [viewer-states/reference/registration.md](viewer-states/reference/registration.md) |
| HDA integration (DefaultState) | [viewer-states/guides/hda-integration.md](viewer-states/guides/hda-integration.md) |
| Testing with `-waitforui` | [viewer-states/guides/testing.md](viewer-states/guides/testing.md) |
| Troubleshooting | [viewer-states/troubleshooting.md](viewer-states/troubleshooting.md) |
| **APEX** | |
| APEX scripting overview | [apex/](apex/index.md) |
| APEX functions reference | [apex/reference/functions.md](apex/reference/functions.md) |
| APEX types | [apex/reference/types.md](apex/reference/types.md) |
| Common patterns | [apex/reference/patterns.md](apex/reference/patterns.md) |
| Animate state tool events | [apex/reference/tool-events.md](apex/reference/tool-events.md) |
| Animate state controls API | [apex/reference/controls.md](apex/reference/controls.md) |
| Troubleshooting | [apex/reference/troubleshooting.md](apex/reference/troubleshooting.md) |
| Creating reusable subgraphs | [apex/guides/subgraph-guide.md](apex/guides/subgraph-guide.md) |
| External resources | [apex/reference/resources.md](apex/reference/resources.md) |
| **APEX Tools (HDAs)** | |
| APEX Tools overview | [projects/apex-tools/](projects/apex-tools/index.md) |
| Animate state tools guide | [projects/apex-tools/animate-state-tools.md](projects/apex-tools/animate-state-tools.md) |
| Skeleton Placer | See APEX Tools - Interactive joint placement for sceneanimate |
| **CUDA** | |
| CUDA HDK development | [cuda/](cuda/index.md) |
| Kernel patterns | [cuda/kernel-patterns.md](cuda/kernel-patterns.md) |
| Bridge pattern (CPU↔GPU) | [cuda/bridge-pattern.md](cuda/bridge-pattern.md) |
| Memory management | [cuda/memory-management.md](cuda/memory-management.md) |
| Debugging CUDA | [cuda/debugging.md](cuda/debugging.md) |
| **HDK** | |
| HDK development overview | [hdk/](hdk/index.md) |
| CMake setup | [hdk/cmake-setup.md](hdk/cmake-setup.md) |
| SOP node development | [hdk/sop-node.md](hdk/sop-node.md) |
| Parameters | [hdk/parameters.md](hdk/parameters.md) |
| DSO packaging | [hdk/dso-packaging.md](hdk/dso-packaging.md) |
| Build scripts | [hdk/build-scripts.md](hdk/build-scripts.md) |
| USD Procedurals (HoudiniProceduralAPI, BRAY) | [hdk/usd-procedurals.md](hdk/usd-procedurals.md) |
| GitHub Pages setup | [hdk/github-pages.md](hdk/github-pages.md) |
| **OpenCL** | |
| OpenCL in Houdini | [opencl/](opencl/index.md) |
| Learning resources | [opencl/learning_resources.md](opencl/learning_resources.md) |
| Binding guide | [opencl/binding_guide.md](opencl/binding_guide.md) |
| **MCP/RPC** | |
| Houdini MCP overview | [mcp/](mcp/index.md) |
| RPC gotchas & pitfalls | [mcp/rpc-gotchas.md](mcp/rpc-gotchas.md) |
| **Projects** | |
| Project documentation hub | [projects/](projects/index.md) |
| TumbleheadRig | [projects/TumbleheadRig/](projects/TumbleheadRig/index.md) |

---

## Documentation Sections

### [Viewer States](viewer-states/index.md)
Python-based interactive tools for Houdini's viewport. Handle mouse/keyboard input, draw custom geometry, integrate with HDAs.

### [APEX](apex/index.md)
APEX scripting language for procedural rigging and animation graphs.

### [CUDA](cuda/index.md)
GPU-accelerated HDK development using CUDA for high-performance geometry processing.

### [HDK](hdk/index.md)
Houdini Development Kit - creating custom C++ nodes, operators, and extensions.

### [OpenCL](opencl/index.md)
OpenCL integration for GPU-accelerated SOPs and geometry processing.

### [MCP/RPC](mcp/index.md)
Remote Python execution and scene control via Model Context Protocol. Connect Claude Code to a running Houdini session.

### [Projects](projects/index.md)
Documentation for individual projects in this workspace, including rigging tools, solvers, and plugins.

---

## Environment

- **Houdini**: 21.0.559
- **Python**: 3.11
- **Platform**: Windows

---

## Contributing

Each documentation section follows a consistent structure:
- `index.md` - Overview and navigation
- `reference/` - API documentation (one topic per file)
- `guides/` - Task-based tutorials
- `troubleshooting.md` - Common issues and solutions
