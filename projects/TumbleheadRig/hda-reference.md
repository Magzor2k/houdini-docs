---
layout: default
title: HDA Reference
parent: APEX Rigging
grand_parent: Projects
nav_order: 2
description: Auto-generated documentation of all TumbleheadRig HDAs
permalink: /projects/tumbleheadrig/hda-reference/
---

# TumbleheadRig HDA Reference

> Auto-generated documentation of all TumbleheadRig HDAs

## Table of Contents

- [Skeleton Creation](#skeleton-creation)
- [APEX Rig Systems](#apex-rig-systems)
- [APEX Rig Assembly](#apex-rig-assembly)
- [Skinning & Capture](#skinning--capture)
- [Post-Skin Deformers](#post-skin-deformers)
- [Blendshape Tools](#blendshape-tools)
- [Utility & Helpers](#utility--helpers)

## Quick Reference

| HDA | Category | APEX | VEX | VS | Description |
|:----|:---------|:----:|:---:|:--:|:------------|
| apex_joints.6.0 | Skeleton Creation | 2 | 6 | Y | th Apex joints |
| fit_skeleton.2.0 | Skeleton Creation | - | 21 |  | th fit skeleton |
| skelPrepper.2 | Skeleton Creation | - | 6 |  | th skel prepper |
| addFitJoint.1 | Skeleton Creation | - | - |  | th skel addFitJoint |
| skeleton_doctor.2.0 | Skeleton Creation | - | 2 |  | th Skeleton Doctor |
| skeleton_validator.1.0 | Skeleton Creation | - | 2 |  | th Skeleton Validator |
| apex_ik_chains.2.0 | APEX Rig Systems | 3 | 9 |  | th apex ik chains |
| apex_bendy_system.1 | APEX Rig Systems | - | - |  | th apex bendy system |
| apex_eye_system.1 | APEX Rig Systems | - | - |  | th apex eye system |
| apex_mouth_system.1 | APEX Rig Systems | - | - |  | th apex mouth system |
| apex_inbetween_joints.1.0 | APEX Rig Systems | 1 | - |  | th Apex Inbetween Joints |
| sticky_system.1 | APEX Rig Systems | - | - |  | th sticky system |
| apex_rigger.1 | APEX Rig Assembly | - | - |  | th apex Rigger |
| apex_rig_builder.1.0 | APEX Rig Assembly | - | 3 |  | th apex rig builder |
| apex_connectSystems.1 | APEX Rig Assembly | - | - |  | th apex connectSystems |
| apex_graphToSystem.1 | APEX Rig Assembly | - | - |  | th apex graph to system |
| apex_visibility.1.0 | APEX Rig Assembly | 1 | 1 |  | th apex visibility |
| apex_skinning.2.0 | Skinning & Capture | - | - |  | Apex Skinning |
| apex_capture_weights.1.0 | Skinning & Capture | - | - |  | th Apex capture weights |
| capture_with_cage.2.0 | Skinning & Capture | - | - |  | th capture with cage |
| skin_cage.2.0 | Skinning & Capture | - | 12 |  | th skin cage |
| skin_base.1 | Skinning & Capture | - | - |  | th skin base |
| skin_layer.1 | Skinning & Capture | - | - |  | th skin layer |
| skinPrepper.1 | Skinning & Capture | - | 1 |  | th skin Prepper |
| capture_blur.1.0 | Skinning & Capture | - | - |  | th capture blur |
| capture_weights_extract.1.0 | Skinning & Capture | - | 2 |  | th capture weights extract |
| apex_bonedeform.2.0 | Post-Skin Deformers | 1 | - |  | th apex bonedeform |
| apex_deltamush.1.0 | Post-Skin Deformers | 1 | - |  | th Apex deltamush |
| deltamushapex.1.0 | Post-Skin Deformers | - | - |  | th deltamushApex |
| apex_lattice.1.0 | Post-Skin Deformers | 2 | - |  | th apex lattice |
| apex_blendshapes.1.0 | Blendshape Tools | 1 | - |  | th apex blendshapes |
| blendshape_comboExtract.1 | Blendshape Tools | - | 1 |  | th blendshape combo extract |
| apex_add_proxy_shp.1.0 | Blendshape Tools | 1 | - |  | th apex add proxy shp |
| ctrl_shapes.1 | Utility & Helpers | - | 7 |  | th ctrl shapes |
| mirror_position.1.0 | Utility & Helpers | - | 1 |  | th mirror position |
| name_from_path.1.0 | Utility & Helpers | - | - |  | th name from path |
| set_path_from_ref.1 | Utility & Helpers | - | - |  | th set path from ref |

---

## Skeleton Creation

### apex_joints.6.0

**Label:** th Apex joints
**Type:** `th::apex_joints::6.0`

Creates APEX joint visualizers with FK system setup.

#### Parameters

| Parameter | Label | Type |
|:----------|:------|:-----|
| `joint_scale` | Joint Scale | float |
| `showpackedgeo1` | Show Joint Visualizers | toggle |
| `outputSkel` | Output Skeleton Shape | toggle |
| `skel_output_name` | Skeleton Visualizer Name | string |
| `system_id` | FKIK system ID | integer |
| `skel_geo_name` | Skeleton Output Name | string |

This HDA has an interactive viewer state.

---

### fit_skeleton.2.0

**Label:** th fit skeleton
**Type:** `th::fit_skeleton::2.0`

Generates a complete biped fit skeleton with customizable proportions.

---

### skelPrepper.2

**Label:** th skel prepper
**Type:** `th::skelPrepper::2`

Prepares a skeleton for rigging by normalizing transforms and setting up groups.

---

## APEX Rig Systems

### apex_ik_chains.2.0

**Label:** th apex ik chains
**Type:** `th::apex_ik_chains::2.0`

Creates IK/FK switching systems for skeleton chains.

---

### apex_bendy_system.1

**Label:** th apex bendy system
**Type:** `th::apex_bendy_system::1`

Creates bendy limb systems with ribbon-style control.

---

## Skinning & Capture

### apex_skinning.2.0

**Label:** Apex Skinning
**Type:** `th::apex_skinning::2.0`

Wrapper for biharmonic/proximity skinning methods.

---

### capture_with_cage.2.0

**Label:** th capture with cage
**Type:** `th::capture_with_cage::2.0`

Captures skinning weights using a simplified cage geometry.

---

## Post-Skin Deformers

### apex_bonedeform.2.0

**Label:** th apex bonedeform
**Type:** `th::apex_bonedeform::2.0`

Applies bone deformation within an APEX graph.

---

### apex_deltamush.1.0

**Label:** th Apex deltamush
**Type:** `th::apex_deltamush::1.0`

Applies delta mush smoothing as an APEX graph node.

---

## Utility & Helpers

### ctrl_shapes.1

**Label:** th ctrl shapes
**Type:** `th::ctrl_shapes::1`

Library of control shapes for rig visualizers.

---

For detailed parameter reference and APEX/VEX code snippets, see the full HDA documentation in the TumbleheadRig package.
