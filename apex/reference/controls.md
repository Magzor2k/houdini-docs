---
layout: default
title: Controls
parent: Reference
grand_parent: APEX
nav_order: 8
description: APEX control system architecture and API
permalink: /apex/reference/controls/
---

# APEX Controls Reference

> Houdini 21.0 | Source: SideFX Internal Documentation

This document describes the APEX control system architecture used in the Animate State. Controls define how users interact with rigs in the viewport.

**Related:** [Tool Events](tool-events.md) - Event callbacks for building animation tools

---

## Control Types

Controls in the APEX animate state come in two varieties: Transform Controls and Abstract Controls. Both are created by adding specific nodes to a rig.

### Transform Controls

Transform controls are authored by adding a `TransformObject` callback to a rig, then wiring the `t`, `r`, and/or `s` ports to the rig inputs. When interacted with in the viewport, these controls use an Xform Handle to alter the promoted `t`, `r`, and `s` parameters.

### Abstract Controls

Abstract controls are authored by adding an `AbstractControl` callback to a rig. The `x` and `y` parameters can be wired to rig inputs. When interacted with, the abstract control acts as a 2-dimensional slider, altering `x` in the horizontal direction and `y` in the vertical.

### Custom Controls (Toggle)

Abstract controls can also be used for custom control types. The **Toggle Control** has these properties:

| Property | Type | Description |
|:---------|:-----|:------------|
| `on_text` | String | Text displayed when toggle is ON |
| `off_text` | String | Text displayed when toggle is OFF |
| `radius` | Float | Radius of the toggle |
| `width` | Float | Overall size of the widget |
| `toggle` | Int | Toggle state (needed by widget) |
| `font_color` | Vector3 | Color of on/off text |
| `on_color` | Vector3 | Widget color when ON |
| `off_color` | Vector3 | Widget color when OFF |
| `circle_color` | Vector3 | Color of the draggable circle |
| `drag_threshold` | Float | Drag distance to change state |
| `scale_factor` | Float | Scaling factor as you zoom out |
| `font_size` | Float | Font size |

---

## Control Classes

The `Scene.TransformControl`, `Scene.AbstractControl`, and `Scene.Control` classes store metadata about controls.

### Key Properties

| Property | Type | Description |
|:---------|:-----|:------------|
| `internal` | Bool | If True, control is hidden in viewport and selection manager but accessible from state |
| `enable` | Bool | Whether control is enabled. Update via `ControlManager.updateEnabledControls(scene)` |
| `animatable` | Bool | If False, control parameters are not bound to channel primitives |

Access these classes by indexing into `ControlManager.controls` with the control path.

---

## Helper Functions

### Group Control Paths by Rig

```python
apex.control_2.groupControlPathsByRig(control_paths)
```

Takes a list of control paths and groups them into a 2-tiered dictionary: first keys are rig paths, second keys are control paths in those rigs.

### Split a Control Path

```python
apex.splitControlPath(ctrl_path)      # Returns (rig_path, control_name)
apex.controlName(ctrl_path)           # Returns control name
apex.controlRigPath(ctrl_path)        # Returns rig path
```

---

## ControlManager

The `ControlManager` class is the primary interface for retrieving control data. It must be initialized before use.

Control managers are designed for custom control sets based on current task demands. They're used throughout the state:

- **Viewer state** - Tracks currently selected controls
- **Tools** - Track task-specific controls (e.g., dynamic motion tracks controls to bake, mirror tool tracks mirrored controls)
- **Scene** - Single control manager with all controls as ground truth
- **Rigs** - Each rig has a control manager in scene data for easy access

### Initialization

Control managers are created without variables. Pass the scene when access is needed.

To add a control, use:

```python
apex.control_2.addControlToManager(manager, ctrl_path, scene)
```

This checks the scene's control manager to determine existence and type.

### Control Managers in the State

| Location | Purpose |
|:---------|:--------|
| `Scene.control_manager` | Created on state init, updated when rigs are added/removed. Only one that analyzes rig graphs for controls |
| `State.control_manager` | Tracks currently selected controls, sometimes extended for scoped controls |
| `<rig_path>/control_manager` | Per-rig control managers in scene data for quick evaluation |

---

## ControlTransformData

The `apex.control_2.ControlTransformData` class provides transform data for controls.

### Usage

1. Call `Scene.updateEvaluationParms()` to ensure latest rig parameters
2. Call `ControlManager.update(scene)` to retrieve current matrices
3. Call `ControlManager.getControlData(ctrl_path)` to get the data

### Properties

| Property | Type | Description |
|:---------|:-----|:------------|
| `xform` | Matrix4 | World transform of the control |
| `local` | Matrix4 | Local transform (TransformObjects only) |
| `restlocal` | Matrix4 | Rest local transform (TransformObjects only) |
| `parentxform` | Matrix4 | Parent world transform (TransformObjects only) |
| `parentlocal` | Matrix4 | Parent local transform (TransformObjects only) |
| `xord` | Int | Transform order (use `xordString()` for string) |
| `rord` | Int | Rotation order (use `rordString()` for string) |
| `scaleinheritance` | Int | Scale inheritance mode |

---

## ControlMapping

The `apex.control_2.ControlMapping` class maps promoted control inputs to parameter names in the scene.

### Helper Function

```python
apex.control_2.getParmsForControls(scene, control_paths, include_non_animatable=False)
```

Returns parameter mapping for a list of controls, filtering by component and animatable status.

---

## Control Template

When a rig is loaded into the scene, a **control template** geometry is created to manage control appearance. Access it at `<rig_path>/control_template`.

Each control maps to a point on this geometry. Point positions are updated to match control positions.

### Attributes

| Attribute | Type | Description |
|:----------|:-----|:------------|
| `name` | String | Control name |
| `ctrl_path` | String | Path to control in scene |
| `label` | String | Label shown when pre-selecting in viewport |
| `shapeoffset` | Matrix4 | Offset matrix for drawing the control shape |
| `shapetype` | String | Shape type (must match shape in shapelibrary). Set from `shapeoverride` property |
| `Cd` | Vector3 | Control color. Set from `color` property |
| `hidden` | Int | Visibility (0 = visible) |
| `disable` | Int | Disabled state. Queried by `ControlManager.updateEnabledControls()` |

### Helper Functions

**Set Attributes:**

- `setControlTemplateInternal()` - Set internal flag
- `setControlTemplateVisibility()` - Set visibility
- `setControlTemplateEnabled()` - Set enabled state
- `setControlTemplateAttrib()` - Set arbitrary attribute

**Set Graph Node Properties:**

- `setControlGraphEnabled()` - Update `enable` property on rig graph controls

**Get Lists:**

- `getEnabledControlsFromTemplate()` - Get enabled control paths/names (can invert)
- `getVisibleControlsFromTemplate()` - Get visible control paths/names (can invert)

**Check State:**

- `getControlEnabledFromTemplate()` - Check if specific control is enabled

---

## Examples

### Get Control Parameter Values

```python
from apex.control_2 import getParmsForControls, controlRigPath

def getControlParmValues(scene, ctrl_path):
    rig_path = controlRigPath(ctrl_path)
    rig_parms = getParmsForControls(scene, (ctrl_path,), include_non_animatable=True)
    if rig_path not in rig_parms:
        return {}

    rig = scene.getData(rig_path)
    if rig is None:
        return {}

    result = {}
    for parm in rig_parms[rig_path]:
        result[parm] = rig.graph_parms[parm]
    return result
```

### Set Control Translation

```python
from apex.control_2 import getParmsForControls, controlRigPath

def setControlTranslation(scene, ctrl_path, frame, value=hou.Vector3(0, 0, 0)):
    rig_path = controlRigPath(ctrl_path)
    rig_control_manager = scene.getData(f"{rig_path}/control_manager")
    ctrl_mapping = rig_control_manager.getControlMapping()
    if ctrl_mapping.t == "":
        return False

    rig = scene.getData(rig_path)
    if rig is None:
        return False

    rig.graph_parms[ctrl_mapping.t] = value

    binding = scene.getData(f"{rig_path}/animbinding")
    binding.setKeysFromDict(scene, frame, pattern=ctrl_mapping.t, force_key=True)

    return True
```

### Set Control Shape

```python
from apex.control_2 import controlRigPath

def setControlShape(scene, ctrl_path, shape_type: str):
    """Sets the shapetype attribute for a control.

    The shapetype string must match a shape in the shapelibrary
    for the control type or character.
    """
    rig_path = controlRigPath(ctrl_path)
    ctrl_template = scene.getData(f"{rig_path}/control_template")

    geo = ctrl_template  # The control template geometry
    pts = geo.globPoints(f"@ctrl_path={ctrl_path}")
    if len(pts) == 0:
        return False

    shape_type_attrib = geo.findPointAttrib("shapetype")
    if not shape_type_attrib:
        return False

    pts[0].setAttribValue(shape_type_attrib, shape_type)
    shape_type_attrib.incrementDataId()
    geo.incrementModificationCounter()

    return True
```
