---
parent: Tools
grand_parent: APEX
layout: default
title: Skeleton Builder
nav_order: 2
---

# Skeleton Builder
{: .fs-9 }

Interactive viewer state for click-to-place skeleton joint creation.
{: .fs-6 .fw-300 }

---

## Overview

The Skeleton Builder is a Python viewer state that allows you to:

1. **Click in viewport** to place skeleton joints
2. **Set parent-child relationships** between joints
3. **Rename joints** interactively
4. **Mirror joints** for symmetric skeletons
5. **Undo/redo** any action
6. **Generate APEX Script code** for the skeleton hierarchy

## Controls

| Action | Input | Description |
|:-------|:------|:------------|
| Place joint | LMB click | Creates a new joint at the clicked position |
| Select joint | Shift + LMB | Selects an existing joint (for parenting) |
| Place with parent | Ctrl + LMB | Creates joint with selected joint as parent |
| Generate script | G | Outputs APEX Script code to console/clipboard |
| Rename selected | R | Opens dialog to rename the selected joint |
| Mirror joints | M | Mirrors selected joint subtree, or all joints on +X side |
| Delete selected | Delete | Removes the selected joint |
| Clear all | C | Clears all joints |
| Undo | Ctrl + Z | Undo the last action |
| Redo | Ctrl + Shift + Z | Redo the last undone action |

## Workflow

### Step 1: Place Root Joint
Click in the viewport to place your first joint (typically hip or root).

### Step 2: Create Child Joints
1. **Shift+Click** an existing joint to select it as parent
2. **Ctrl+Click** to place a new joint as a child

### Step 3: Rename Joints
1. **Shift+Click** to select a joint
2. Press **R** to open the rename dialog
3. Enter the new name (e.g., `L_shoulder`, `hip`, `spine`)

### Step 4: Mirror for Symmetry
For symmetric skeletons:
1. Build one side using `L_` prefix (e.g., `L_shoulder`, `L_elbow`)
2. Press **M** to mirror all joints on the positive X side
3. Mirrored joints get `R_` prefix automatically

**Mirror naming conventions:**
- `L_name` ↔ `R_name`
- `name_L` ↔ `name_R`
- `left_name` ↔ `right_name`

### Step 5: Generate Code
Press **G** to generate APEX Script code. The output is:
- Copied to clipboard
- Printed to console
- Ready to paste into an APEX Script SOP

## Generated Code Example

For a simple spine with hip → spine → chest:

```python
graph = ApexGraphHandle()
parms = graph.addNode('parms', '__parms__')
output = graph.addNode('output', '__output__')

# hip
hip = graph.addNode('hip', 'TransformObject', parms={'t': Vector3(0.0, 1.0, 0.0)})
hip.xform_out.promoteOutput('hip_xform')

# spine
spine = graph.addNode('spine', 'TransformObject', parms={'t': Vector3(0.0, 1.3, 0.0)})
graph.addWire(hip.xform_out, spine.parent_in)
spine.xform_out.promoteOutput('spine_xform')

# chest
chest = graph.addNode('chest', 'TransformObject', parms={'t': Vector3(0.0, 1.6, 0.0)})
graph.addWire(spine.xform_out, chest.parent_in)
chest.xform_out.promoteOutput('chest_xform')

graph.sort(True)
geo = graph.saveToGeometry()
BindOutput(geo)
```

## HUD Display

The viewer state shows a HUD with:
- **Joints**: Current joint count
- **Selected**: Name of selected joint (for parenting)
- **Controls**: Quick reference for hotkeys

## Technical Details

### Position Calculation
Click positions are calculated using ray intersection:
- Uses the construction plane if available
- Falls back to ground plane (Y=0)

### Joint Naming
Joints are auto-named sequentially:
- `joint_1`, `joint_2`, etc.
- Press **R** to rename to meaningful names

### Parent-Child Connections
Parent connections use the `parent_in` port:
```python
graph.addWire(parent.xform_out, child.parent_in)
```

### Undo/Redo
The viewer state maintains up to 50 undo states. Actions that support undo:
- Place joint
- Delete joint
- Clear all
- Rename joint
- Mirror joints

## API Reference

### SkeletonBuilderState

Main viewer state class.

```python
class SkeletonBuilderState:
    def __init__(self, scene_viewer, state_name):
        """Initialize the skeleton builder state."""

    def onMouseEvent(self, kwargs):
        """Handle mouse events for joint placement."""

    def onKeyEvent(self, kwargs):
        """Handle keyboard shortcuts."""

    def onDraw(self, kwargs):
        """Draw joints and bones in viewport."""

    def onMenuAction(self, kwargs):
        """Handle menu actions."""
```

### generate_skeleton_script

Generates APEX Script code from joint data.

```python
def generate_skeleton_script(joints):
    """
    Generate APEX Script code for skeleton.

    Args:
        joints: List of (name, Vector3 position, parent_idx)
                parent_idx is -1 for root joints

    Returns:
        str: APEX Script code
    """
```

### generate_skeleton_with_controls

Generates APEX Script with promoted control inputs.

```python
def generate_skeleton_with_controls(joints):
    """
    Generate APEX Script code with control inputs.

    Each joint gets promoted t_in and r_in for animation.
    """
```

## Installation

### Package Structure

The skeleton builder is part of the main package:

```
package/
├── viewer_states/
│   └── skeleton_builder.py    # Viewer state (auto-registered)
├── python3.11libs/
│   └── skeleton_builder/      # Support modules
│       ├── skeleton_drawable.py
│       └── apex_script_generator.py
└── scripts/
    └── test_skeleton_builder.py  # Test script
```

### Manual Registration (for development)

```python
import hou
hou.ui.registerViewerStateFile("/path/to/package/viewer_states/skeleton_builder.py")
```

## Testing

Run the test script to verify installation:

```batch
hython package/scripts/test_skeleton_builder.py
```

## Dependencies

- Houdini 21.0+
- APEX Script SOP (`apex::script`)
- Python 3.11+

---

## See Also

- [APEX Script Reference](https://Magzor2k.github.io/apex-script-docs/reference.html)
- [TransformObject Node](https://Magzor2k.github.io/apex-script-docs/patterns.html)
