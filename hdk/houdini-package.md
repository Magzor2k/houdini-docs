---
layout: default
title: Houdini Package
parent: HDK
nav_order: 8
description: Package folder structure for deploying HDAs, DSOs, and Python scripts
permalink: /hdk/houdini-package/
---

# Houdini Package

This repository uses a centralized `package/` folder for all Houdini assets. All builds and HDAs deploy here.

---

## Directory Structure

```
package/
├── dso/              # Compiled C++/CUDA plugins (.dll/.so)
├── apexdso/          # APEX graph node plugins
├── otls/             # HDAs and OTLs (.hda, .otl)
│   └── backup/       # Auto-saved HDA backups
├── python3.11libs/   # Python modules for HDAs
├── viewer_states/    # Python viewer state scripts
└── scripts/          # Utility and test scripts
```

## Asset Deployment

| Asset Type | Target Directory | Extension |
|:-----------|:-----------------|:----------|
| C++/CUDA SOP nodes | `package/dso/` | `.dll` (Windows) |
| APEX graph nodes | `package/apexdso/` | `.dll` (Windows) |
| HDAs | `package/otls/` | `.hda` |
| Python libraries | `package/python3.11libs/` | `.py` |
| Viewer states | `package/viewer_states/` | `.py` |

## Build Output

CMake builds in `houdini-cuda-deformers/` automatically install to `package/dso/`:

```bash
# From houdini-cuda-deformers directory
cmake --build build --config Release

# Output goes to: ../package/dso/
```

## Environment Setup

To load plugins from the package folder, set `HOUDINI_DSO_PATH`:

```bash
# Windows (Git Bash)
export HOUDINI_DSO_PATH="c:/path/to/repo/package/dso;&"

# The ;&' suffix is required on Windows to append to default paths
```

## HDA Development

When creating or modifying HDAs:

1. Save HDA to `package/otls/`
2. Associated Python modules go in `package/python3.11libs/`
3. Viewer states go in `package/viewer_states/`

Example structure for a Skeleton Builder HDA:

```
package/
├── otls/sop_th.skeleton_builder.1.0.hda
├── python3.11libs/skeleton_builder/
│   ├── __init__.py
│   ├── skeleton_graph_builder.py
│   └── skeleton_drawable.py
└── viewer_states/skeleton_builder_state.py
```

## Naming Conventions

| Type | Convention | Example |
|:-----|:-----------|:--------|
| HDAs | `[context]_[namespace].[name].[version].hda` | `sop_th.skeleton_builder.1.0.hda` |
| DSOs | `SOP_[Name].dll` | `SOP_VBDSolver.dll` |
| Python modules | `snake_case/` | `skeleton_builder/` |
| Viewer states | `[name]_state.py` | `skeleton_builder_state.py` |

## See Also

- [DSO Packaging](dso-packaging.md) - How to package compiled plugins
- [CMake Setup](cmake-setup.md) - Build configuration for HDK projects
- [Build Scripts](build-scripts.md) - Automation for building and deploying
