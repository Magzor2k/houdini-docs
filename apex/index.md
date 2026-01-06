---
layout: default
title: APEX
nav_order: 3
has_children: true
description: APEX scripting and tools for Houdini
permalink: /apex/
---

# APEX Tools
{: .fs-9 }

Python viewer state tools for building APEX rigs interactively in Houdini's viewport.
{: .fs-6 .fw-300 }

---

## Overview

APEX Tools provides interactive viewer state tools that generate APEX Script code. Instead of manually writing node connections, you can:

- **Click to place** joints in the viewport
- **Set parent-child** relationships visually
- **Generate APEX Script** code automatically

## Available Tools

| Tool | Description |
|:-----|:------------|
| [Skeleton Builder](skeleton-builder.html) | Click-to-place skeleton joints with parent-child hierarchy |
| [Spline Rig Setup](spline-rig-setup.html) | Building spline-based rigs with CV controls, pins, and stretch |

## Quick Start

```python
# The Skeleton Builder generates code like this:
graph = ApexGraphHandle()
parms = graph.addNode('parms', '__parms__')
output = graph.addNode('output', '__output__')

# Hip joint
hip = graph.addNode('hip', 'TransformObject')
hip.t_in.set(Vector3(0.0, 1.0, 0.0))
hip.xform_out.promoteOutput('hip_xform')

# Spine (child of hip)
spine = graph.addNode('spine', 'TransformObject')
spine.t_in.set(Vector3(0.0, 1.5, 0.0))
spine.parentxform_in.wire(hip.xform_out)
spine.xform_out.promoteOutput('spine_xform')

graph.sort(True)
geo = graph.saveToGeometry()
BindOutput(geo)
```

## Installation

### Launcher Batch File Setup

Add apex_tools to your Houdini launcher batch file:

```batch
set APEX_TOOLS=C:\path\to\apex_tools
set HOUDINI_PATH=%APEX_TOOLS%;%HOUDINI_PATH%
```

This adds the package to Houdini's search path, enabling:
- Auto-registration of viewer states from `viewer_states/`
- Python modules from `python3.11libs/`

### Package Structure

```
apex_tools/
├── viewer_states/
│   └── skeleton_builder.py    # Viewer state (auto-registered)
└── python3.11libs/
    └── skeleton_builder/      # Support modules
        ├── skeleton_drawable.py
        └── apex_script_generator.py
```

### Usage

1. Start Houdini via your launcher batch file
2. Go inside a Geometry node (SOP context)
3. Press **Enter** in viewport, type "skeleton_builder"
4. Click to place joints, press **G** to generate APEX Script

## Environment

This documentation is for:
- **Houdini 21.0.559**
- **APEX Script SOP** (`apex::script`)

---

## Related Projects

- [APEX Script Docs](https://Magzor2k.github.io/apex-script-docs) - APEX Script language reference
