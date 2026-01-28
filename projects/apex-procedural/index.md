---
layout: default
title: APEX Procedural
parent: Projects
nav_order: 7
has_children: true
description: Python USD procedural using APEX graphs for render-time geometry
permalink: /projects/apex-procedural/
---

# APEX USD Procedural
{: .fs-9 }

Python-based USD procedural that invokes APEX graphs at render time.
{: .fs-6 .fw-300 }

---

## Overview

The APEX Procedural allows you to define geometry in SOPs, compile it to an APEX graph, and have that graph execute at render time to generate procedural geometry.

**Key difference from Hydra procedurals**: This uses Python and the HoudiniProceduralAPI, not C++.

## Quick Start

| Task | Command |
|:-----|:--------|
| Open examples | `/launch-houdini apex-procedural` |
| Default scene | `examples/apex_usd_procedural_test.hip` |

## How It Works

1. **Author time**: SOP network defines geometry creation logic
2. **Compile**: SOP network is compiled to APEX graph (`.bgeo`)
3. **Export**: LOP HDA creates USD prim with procedural reference
4. **Render time**: `apex_invokegraph.py` executes the graph

## LOP HDA Parameters

| Parameter | Description |
|:----------|:------------|
| Graph File | Path to compiled `.bgeo` APEX graph |
| Prim Path | USD prim path for the procedural |
| Parameters | Graph input parameters |

## Build Output

```
package/otls/lop_th.apex_procedural.0.3.hda
package/python3.11libs/apex_invokegraph.py
```

## Documentation To Add

- [ ] Compiling APEX graphs
- [ ] HDA parameter reference
- [ ] Debugging render-time execution
