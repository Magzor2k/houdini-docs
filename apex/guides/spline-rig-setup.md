---
layout: default
title: Spline Rig Setup
parent: Guides
grand_parent: APEX
nav_order: 2
description: Building spline-based rigs with CV controls, pins, and stretch
permalink: /apex/guides/spline-rig-setup/
---

# Spline Rig Setup

Guide to building spline-based rigs using APEX in Houdini 21+, based on analysis of the SideFX earthworm example rig.

---

## Overview

A spline rig uses a curve to drive joint transforms, providing smooth, flexible deformation ideal for:
- Spines and tails
- Tentacles and worms
- Tongues and cables
- Any chain requiring smooth bending

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           SPLINE RIG FLOW                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  TransformObject     ┌──────────────┐                                   │
│    (C_main)  ───────►│  CV Controls │ (C_cv_0..N)                       │
│       │              └──────┬───────┘                                   │
│       │                     │                                           │
│       │              ┌──────▼───────┐                                   │
│       │              │ cv_enablers  │ Collects transforms into array    │
│       │              └──────┬───────┘                                   │
│       │                     │ .xforms                                   │
│       │              ┌──────▼───────────────────────┐                   │
│       │              │ rig::ControlSplineFromArray  │                   │
│       │              │  - cvs (Matrix4Array)        │                   │
│       │              │  - splinetype (0=Bezier,1=NURBS)                 │
│       │              │  - order (3-4 typical)       │                   │
│       │              └──────┬───────────────────────┘                   │
│       │                     │ .geo (spline geometry)                    │
│       │                     │                                           │
│       │              ┌──────▼────────────────────────────┐              │
│       │              │ rig::SampleSplineTransformsToArray│              │
│       │              │  - geo (spline)                   │              │
│       │              │  - numsamples (joint count)       │              │
│       │              │  - restlengths                    │              │
│       │              │  - pins, twists                   │              │
│       │              │  - stretch, stretchscale          │              │
│       │              └──────┬────────────────────────────┘              │
│       │                     │ .xforms (Matrix4Array)                    │
│       │              ┌──────▼───────┐                                   │
│       │              │ joint_xforms │ Distributes to individual joints  │
│       │              └──────┬───────┘                                   │
│       │                     │                                           │
│       │              ┌──────▼───────┐                                   │
│       └─────────────►│  transforms  │ Final joint outputs               │
│                      └──────────────┘                                   │
│                                                                         │
│  Pin Controls ───────► pin_enablers ───► samplespline.pins/twists      │
│  (C_pin_0..N)          (arrays)                                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Core APEX Nodes

### rig::ControlSplineFromArray::3.0

Builds a spline curve from an array of CV transforms.

**Inputs:**

| Port | Type | Description |
|:-----|:-----|:------------|
| `cvs` | Matrix4Array | Array of CV transforms (position + orientation) |
| `splinetype` | Int | 0 = Bezier, 1 = NURBS |
| `order` | Int | Curve order (3-4 typical for smooth curves) |

**Outputs:**

| Port | Type | Description |
|:-----|:-----|:------------|
| `geo` | Geometry | Spline geometry with `transform` point attribute |

**Example Config:**
```python
parms = {
    'order': 3,
    'splinetype': 1  # NURBS
}
```

### rig::SampleSplineTransformsToArray::3.0

Samples transforms along a spline to drive a joint chain.

**Inputs:**

| Port | Type | Description |
|:-----|:-----|:------------|
| `geo` | Geometry | Spline geometry from ControlSpline |
| `numsamples` | Int | Number of joints to generate |
| `restlengths` | FloatArray | Distance between consecutive joints |
| `stretch` | Float | 0 = fixed length, 1 = fully stretchy |
| `stretchscale` | Float | Max stretch multiplier |
| `squashscale` | Float | Max squash multiplier |
| `pins` | Vector2Array | (offset, jointindex) pairs |
| `twists` | Vector2Array | (angle, offset) pairs |
| `pininterps` | IntArray | Interpretation mode per pin |
| `twistinterps` | IntArray | Interpretation mode per twist |
| `overshoot` | Bool | Allow joints past spline ends |
| `stretchsquashtype` | Int | How scaling is applied |

**Outputs:**

| Port | Type | Description |
|:-----|:-----|:------------|
| `xforms` | Matrix4Array | Sampled joint transforms |
| `outgeo` | Geometry | Resampled spline as skeleton |
| `arclength` | Float | Total spline length |

**Example Config:**
```python
parms = {
    'numsamples': 100,      # 100 joints along the chain
    'stretchsquashtype': 1
}
```

### TransformObject

Creates interactive controls. When t/r/s ports are promoted, becomes pickable in animate state.

**Key Properties:**
```python
properties = {
    'shape': 'circle_wires',      # Control shape
    'shapescale': [1, 1, 1],      # Shape size
    'shaperotate': [90, 0, 0],    # Shape orientation
    'promote': 't r s',           # Exposed channels
    'override': True              # Allow parameter override
}
```

## Control Hierarchy

### Main Control

The root of the rig hierarchy:

```
C_main (TransformObject)
├── Properties:
│   ├── shape: circle_wires
│   ├── promote: t r s
│   └── tags: ('main',)
├── Outputs:
│   ├── xform → CV controls .parent
│   └── localxform → CV controls .parentlocal
```

### CV Controls (Spline Shape)

Control vertices that define the spline shape:

```
C_cv_N_grp (__subnet__)
├── Contains: TransformObject for the CV
├── Inputs:
│   ├── parent ← C_main.xform
│   ├── parentlocal ← C_main.localxform
│   ├── restlocal ← C_rest_cv_N_grp.restlocal
│   ├── enable ← parms.cv_N_enable
│   ├── t ← parms.cv_N_t
│   ├── r ← parms.cv_N_r
│   └── s ← parms.cv_N_s
├── Outputs:
│   ├── xform → cv_enablers.xform[N]
│   └── enable → cv_enablers.enable[N]
└── Properties:
    └── control: {controlgroup: 1, primary: 'C_cv_N_ctrl'}
```

### Rest CV Groups

Store the rest/default positions:

```
C_rest_cv_N_grp (__subnet__)
├── Parms:
│   ├── enable: True
│   └── restlocal_t: Vector3  # Rest position along spline
├── Outputs:
│   ├── restlocal → C_cv_N_grp.restlocal
│   └── enable → C_cv_N_grp.enable
```

### Pin Controls (Joint Constraints)

Pins constrain specific joints to positions along the spline:

```
C_pin_N_grp (__subnet__)
├── Inputs:
│   ├── spline ← spline.geo
│   ├── jointoffset ← C_rest_pin_N_grp.jointoffset
│   ├── enable ← parms.enable_N
│   ├── splineoffset ← parms.splineoffset_N
│   ├── twist ← parms.twist_N
│   └── twistoffset ← parms.twistoffset_N
├── Outputs:
│   ├── pin → pin_enablers.pin[N]
│   ├── twist → pin_enablers.twist[N]
│   └── enable → pin_enablers.enable[N]
├── Parms:
│   ├── splineoffset: 0.0  # Position along spline (0-1)
│   └── twistoffset: 0.0
└── Properties:
    └── control: {controlgroup: 1, primary: 'C_pin_N_ctrl'}
```

## Promoted Parameters

These are exposed to the animate state for interactive manipulation:

### Spline Configuration

```python
C_spline_numcvs: 4           # Active CV count
C_spline_numpins: 3          # Active pin count
C_spline_stretch: 0.0        # 0=fixed, 1=stretchy
C_spline_stretchscale: 2.0   # Max stretch factor
C_spline_squashscale: 2.0    # Max squash factor
C_spline_overshoot: False    # Joints past spline ends
```

### Per-CV Parameters

```python
cv_N_t: Vector3      # Translation
cv_N_r: Vector3      # Rotation (Euler)
cv_N_s: Vector3      # Scale
cv_N_enable: Bool    # Toggle CV
cv_N_restlocal_t: Vector3  # Rest position
```

### Per-Pin Parameters

```python
splineoffset_N: Float    # Position along spline (0-1)
twist_N: Float           # Rotation about tangent
twistoffset_N: Float     # Additional twist offset
enable_N: Bool           # Toggle pin
jointoffset_N: Int       # Which joint this pin affects
```

## Data Flow

```
1. CV Controls
   C_cv_0..N.xform → cv_enablers → .xforms (Matrix4Array)

2. Build Spline
   cv_enablers.xforms → spline.cvs
   spline.geo → (curve geometry with transform attrib)

3. Pin Controls
   C_pin_0..N → pin_enablers → .pins, .twists (Vector2Arrays)

4. Sample Spline
   spline.geo → samplespline.geo
   pin_enablers.pins → samplespline.pins
   pin_enablers.twists → samplespline.twists
   C_spline.stretch/scale → samplespline.stretch/scale

5. Output Joints
   samplespline.xforms → joint_xforms → transforms → output
```

## Key Connections

| Source | Target | Purpose |
|:-------|:-------|:--------|
| `cv_enablers.xforms` | `spline.cvs` | CV transforms to spline builder |
| `spline.geo` | `samplespline.geo` | Spline geometry to sampler |
| `pin_enablers.pins` | `samplespline.pins` | Pin constraints |
| `pin_enablers.twists` | `samplespline.twists` | Twist values |
| `C_spline.stretch` | `samplespline.stretch` | Stretch blend |
| `samplespline.xforms` | `joint_xforms.xforms` | Sampled transforms to joints |

## Stretch vs Fixed Length

The `stretch` parameter blends between two modes:

| stretch | Behavior |
|:--------|:---------|
| 0.0 | Fixed length - joints slide along spline, chain length constant |
| 1.0 | Stretchy - joints distributed evenly, chain stretches with spline |
| 0.5 | Blend of both behaviors |

When stretchy:
- `stretchscale` limits maximum stretch (e.g., 2.0 = 200% max)
- `squashscale` limits maximum compression

## Example Stats (Earthworm)

| Component | Count |
|:----------|------:|
| CV Controls | 10 |
| Pin Controls | 10 |
| Output Joints | 100 |
| Active CVs (default) | 4 |
| Active Pins (default) | 3 |

## See Also

- [APEX Quick Reference](../reference/quick-reference.md) - APEX Script syntax
- [APEX Patterns](../reference/patterns.md) - Common APEX patterns
- [rig::SampleSplineTransforms](https://www.sidefx.com/docs/houdini/nodes/apex/rig--SampleSplineTransforms.html) - SideFX docs
- [Spline Rig Component](https://www.sidefx.com/docs/houdini/character/kinefx/rig_components/spline.html) - SideFX docs
- [TransformObject](https://www.sidefx.com/docs/houdini/nodes/apex/TransformObject.html) - SideFX docs
