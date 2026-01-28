---
layout: default
title: APEX Tools
parent: Projects
nav_order: 5
has_children: true
description: Utility HDAs and tools for APEX rigging workflow
permalink: /projects/apex-tools/
---

# APEX Tools
{: .fs-9 }

Collection of HDAs and Python tools for APEX rigging workflow. Includes tools for the `apex:sceneanimate` viewer state.
{: .fs-6 .fw-300 }

**Keywords**: APEX HDA, APEX tool, skeleton placer, sceneanimate, animate state, interactive rigging

---

## Overview

APEX Tools provides utility HDAs and Python libraries for building APEX rigs interactively. This is different from APEX scripting (see [APEX Documentation](../../apex/)) - these are ready-to-use tools built ON TOP of APEX.

## Quick Start

| Task | Command |
|:-----|:--------|
| Open examples | `/launch-houdini apex-tools` |
| Default scene | `examples/skeleton_builder.hip` |

## Available Tools

| Tool | Description |
|:-----|:------------|
| **Skeleton Placer** | Interactive joint placement for sceneanimate (H key) |
| Lattice Deformer | Lattice-based deformation HDA |
| Cloth Analysis | Analyze cloth simulation quality HDA |
| Rig Controller Placer | Interactive control placement |

See [Animate State Tools Guide](animate-state-tools.md) for building custom tools.

## Python Libraries

Located in `package/python3.11libs/`:

| Module | Description |
|:-------|:------------|
| `canvas_tools/` | Animate state tools (Skeleton Placer, etc.) |
| `cloth_analysis/` | Simulation metrics and analysis |

## Animate State Tools

Tools that work within the `apex:sceneanimate` viewer state:

| Tool | Location | Activation |
|:-----|:---------|:-----------|
| Skeleton Placer | `canvas_tools/skeleton_placer/` | H key or shelf button |

## Documentation

| Document | Description |
|:---------|:------------|
| [Animate State Tools Guide](animate-state-tools.md) | Building custom tools for sceneanimate |

## Documentation To Add

- [ ] HDA reference pages
- [ ] Cloth analysis metrics
