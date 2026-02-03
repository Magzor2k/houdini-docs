---
layout: default
title: Karma Procedural
parent: Projects
nav_order: 1
has_children: true
description: Custom Karma procedural geometry rendering using BRAY API
permalink: /projects/karma-procedural/
---

# Karma Procedural
{: .fs-9 }

Custom procedural geometry for Karma CPU rendering using the BRAY API.
{: .fs-6 .fw-300 }

---

## Overview

The Karma Procedural project creates geometry at render-time using Karma's BRAY (Bifrost Ray) API. This is different from Hydra procedurals - BRAY procedurals run within Karma's CPU rendering pipeline.

## Quick Start

| Task | Command |
|:-----|:--------|
| Build | `/build karma-procedural` |
| Test | Open `test_cube_grid.hip` and render with Karma CPU |

## What is a BRAY Procedural?

BRAY procedurals are C++ plugins that generate geometry during Karma CPU rendering. They:

- Run at render-time, not at scene evaluation time
- Can generate geometry that doesn't exist in the scene graph
- Are registered as USD procedural prims
- Only work with Karma CPU (not Karma XPU)

## Current Implementation

**BRAY_CubeGrid**: A simple example that generates a grid of cubes at render time.

| File | Description |
|:-----|:------------|
| `src/BRAY_CubeGrid.cpp` | Main procedural implementation |
| `src/BRAY_CubeGrid.h` | Header file |
| `CMakeLists.txt` | Build configuration |

## Build Output

```
package/dso/karma/BRAY_CubeGrid.dll
```

## Documentation To Add

- [ ] BRAY API reference
- [ ] Creating new procedurals
- [ ] Parameter binding
- [ ] Performance considerations
