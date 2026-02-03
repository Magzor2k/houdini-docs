---
layout: default
title: VBD Solver
parent: Projects
nav_order: 2
has_children: true
description: CUDA cloth/soft body simulation with Vertex Block Descent
permalink: /projects/vbd/
---

# VBD Solver
{: .fs-9 }

GPU-accelerated cloth and soft body simulation using the Vertex Block Descent (XPBD) algorithm.
{: .fs-6 .fw-300 }

---

## Overview

The VBD Solver is a CUDA-based physics solver for cloth and soft body simulation. It uses Extended Position-Based Dynamics (XPBD) with the Vertex Block Descent algorithm for stable, fast simulation.

## Quick Start

| Task | Command |
|:-----|:--------|
| Build | `/build vbd` |
| Test | Open `examples/vbd_drape_test_v001.hip` |
| Launch | `/launch-houdini vbd` |

## Features

- **Constraint Types**: Distance, Shear, Bend, Volume, Pin
- **Self-Collision**: GPU-accelerated with spatial hashing
- **Sphere Colliders**: External collision objects
- **Soft/Hard Pins**: Animated pin constraints with per-pin stiffness

## Solver Inputs

| Input | Description |
|:------|:------------|
| Input 1 | Rest geometry with topology |
| Input 2 | Constraints (from VBD Constraints SOP) |
| Input 3 | Pin geometry (optional) |
| Input 4 | Sphere colliders (optional) |

## Key Parameters

| Parameter | Description | Default |
|:----------|:------------|:--------|
| Substeps | Simulation substeps per frame | 10 |
| Iterations | Constraint iterations per substep | 10 |
| Gravity | Gravity vector | (0, -9.81, 0) |
| Self Collision | Enable self-collision detection | Off |

## Build Output

```
package/dso/SOP_VBDSolver.dll
```

## Documentation To Add

- [ ] Constraint types guide
- [ ] Pin constraint tutorial
- [ ] Self-collision setup
- [ ] Troubleshooting
- [ ] Parameter reference
