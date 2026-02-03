---
layout: default
title: APEX Script
parent: Projects
nav_order: 4
has_children: true
description: APEX scripting examples and tutorials for procedural rigging
permalink: /projects/apex-script/
---

# APEX Script Examples
{: .fs-9 }

Learning resource for APEX Script - Houdini's graph-building language for procedural rigging.
{: .fs-6 .fw-300 }

---

## Overview

This project contains examples and tutorials for learning APEX Script, the Python API for building APEX graphs programmatically.

## Quick Start

| Task | Command |
|:-----|:--------|
| Open examples | `/launch-houdini apex-script` |
| Default scene | `examples/spline_rig_test.hip` |

## Example Scenes

| Scene | Description |
|:------|:------------|
| `01_hello_world/` | Basic "add two floats" example |
| `02_geometry_basics/` | Geometry transformation |
| `03_fk_rig/` | Forward Kinematics skeleton rigging |
| `spline_rig_test.hip` | Spline-based character rig |

## Code Snippets

The `snippets/` folder contains reusable code patterns:

| File | Description |
|:-----|:------------|
| `bind_parameters.txt` | Parameter binding patterns |
| `iterate_geometry.txt` | Geometry iteration |
| `graph_building.txt` | Graph construction patterns |

## Related Documentation

For APEX Script language reference, see:
- [APEX Reference](../../apex/reference/) - Language reference
- [APEX Guides](../../apex/guides/) - Tutorials

## Documentation To Add

- [ ] Example walkthroughs
- [ ] Common patterns guide
- [ ] SOP configuration reference
