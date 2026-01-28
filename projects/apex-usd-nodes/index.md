---
layout: default
title: APEX USD Nodes
parent: Projects
nav_order: 6
has_children: true
description: Custom APEX callback nodes for USD procedural generation
permalink: /projects/apex-usd-nodes/
---

# APEX USD Nodes
{: .fs-9 }

Custom APEX nodes for procedural USD generation and manipulation.
{: .fs-6 .fw-300 }

---

## Overview

APEX USD Nodes provides custom APEX callback nodes that can create, modify, and export USD content from within APEX graphs. This enables procedural USD generation controlled by APEX Canvas.

## Quick Start

| Task | Command |
|:-----|:--------|
| Build | `/build apex-usd-nodes` |
| Test | Open `examples/apex_canvas_test.hip` |

## Available Nodes

| Node | Description |
|:-----|:------------|
| `APEX_UsdCreate` | Create USD layer |
| `APEX_UsdLight` | Create USD light prim |
| `APEX_UsdCamera` | Create USD camera prim (planned) |
| `APEX_UsdExport` | Export layer to file |
| `APEX_UsdLayerClear` | Clear layer contents |

## Architecture

The nodes use a **LayerRegistry** pattern:

1. Each APEX graph evaluation gets a unique layer
2. Nodes write to the layer via callback execution
3. The layer is exported or consumed by LOPs
4. Registry tracks layers by graph ID for cleanup

## Build Output

```
package/apexdso/apex_usd_nodes.dll
```

## Documentation To Add

- [ ] Node reference pages
- [ ] LayerRegistry pattern
- [ ] Integration with APEX Canvas
- [ ] Data flow diagrams
