---
layout: default
title: Hydra Procedural
parent: Projects
nav_order: 3
has_children: true
description: C++ Hydra Generative Procedural for Solaris/Karma
permalink: /projects/hydra-procedural/
---

# Hydra Procedural Example
{: .fs-9 }

C++ Hydra 2.0 Generative Procedural for render-time geometry generation in Solaris and Karma.
{: .fs-6 .fw-300 }

---

## Overview

This project demonstrates how to create a Hydra Generative Procedural - a C++ plugin that generates geometry at render/viewport time within the Hydra 2.0 rendering framework.

**Key difference from BRAY procedurals**: Hydra procedurals work with any Hydra-compatible renderer, not just Karma.

## Quick Start

| Task | Command |
|:-----|:--------|
| Build | `/build hydra-procedural` |
| Test | Open `examples/test_procedural.hip` |

## What is a Hydra Generative Procedural?

Hydra Generative Procedurals are USD plugins that:

- Generate geometry at render/viewport time
- Integrate with any Hydra 2.0 renderer
- Are registered via `plugInfo.json`
- Inherit from `HdGpGenerativeProcedural`

## Project Structure

| File | Description |
|:-----|:------------|
| `src/ExampleProcedural.cpp` | Main procedural implementation |
| `src/ExampleProcedural.h` | Header file |
| `resources/plugInfo.json` | USD plugin registration |
| `CMakeLists.txt` | Build configuration |

## Build Output

```
package/dso/ExampleProcedural.dll
package/usd/ExampleProcedural/resources/plugInfo.json
```

## Plugin Discovery

The procedural is registered via `plugInfo.json`. Houdini finds it through:
- `PXR_PLUGINPATH_NAME` environment variable
- Or the package's `usd/` directory

## Documentation To Add

- [ ] HdGpGenerativeProcedural API
- [ ] Creating new procedurals
- [ ] plugInfo.json schema
- [ ] Comparison with BRAY and HoudiniProceduralAPI
