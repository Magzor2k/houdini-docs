---
layout: default
title: APEX Rigging
parent: Projects
nav_order: 1
description: Tumblehead APEX rigging tools and HDAs
permalink: /projects/tumbleheadrig/
---
# APEX Rigging

Tumblehead APEX rigging tools for procedural character rigging in Houdini.

---

## Overview

This package provides a comprehensive set of HDAs and Python tools for building APEX-based character rigs. The tools support the full rigging workflow from skeleton creation to skinning and deformation.

---

## Package Contents

### HDAs (47 tools)

| Category | Tools |
|:---------|:------|
| **Skeleton** | fit_skeleton, skelPrepper, skeleton_doctor, skeleton_validator |
| **Joints** | apex_joints, addFitJoint, apex_inbetween_joints |
| **IK/FK** | apex_ik_chains, apex_bendy_system |
| **Deformation** | apex_bonedeform, apex_deltamush, apex_blendshapes |
| **Skinning** | apex_skinning, apex_capture_weights, capture_blur, capture_with_cage |
| **Systems** | apex_eye_system, apex_mouth_system, sticky_system |
| **Utilities** | apex_rig_builder, apex_rigger, ctrl_shapes, apex_lattice |

### Python Modules

- `TumbleheadRig.th_apex_wrapper` - Core APEX graph manipulation
- `TumbleheadRig.th_biped` - Biped rigging automation
- `TumbleheadRig.th_rig_utils` - General rigging utilities
- `TumbleheadRig.apex_utils` - APEX helper functions

---

## Quick Start

1. **Install the package** by adding it to your `HOUDINI_PATH`
2. **Create a new SOP network** for your rig
3. **Use the Skeleton Builder** tools to define your joint hierarchy
4. **Apply IK/FK systems** using the apex_ik_chains HDA
5. **Skin your geometry** with apex_skinning

---

## Key Concepts

### APEX Graphs
All rigging is built on APEX (Animated Procedural EXpressions) graphs. These define the evaluation order and connections between rig components.

### Fit Skeleton
The fit skeleton is your reference skeleton used during rig construction. It defines rest poses and joint orientations.

### Control Shapes
Custom controller shapes can be created with the ctrl_shapes HDA and assigned to rig controls.

---

## See Also

- [APEX Documentation](../../apex/) - Core APEX scripting reference
- [HDK APEX Nodes](../../hdk/) - Building custom APEX nodes
