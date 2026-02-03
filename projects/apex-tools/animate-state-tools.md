---
layout: default
title: Animate State Tools Guide
parent: APEX Tools
grand_parent: Projects
nav_order: 2
description: Building custom tools for the apex:sceneanimate viewer state
permalink: /projects/apex-tools/animate-state-tools/
---

# Animate State Tools Guide
{: .fs-9 }

Complete guide to building custom tools for the `apex:sceneanimate` viewer state in Houdini.
{: .fs-6 .fw-300 }

---

## Overview

The `apex:sceneanimate` viewer state allows custom tools to run inside it, providing access to the APEX scene, rig graphs, and control system. This is how tools like the Skeleton Placer work.

**Prerequisites:**
- A scene with an `apex::sceneanimate` SOP node
- Understanding of APEX rig graphs and controls
- Python knowledge

---

## 1. Tool Structure

Every animate state tool is a Python class with a `load()` entry point.

### Minimal Template

```python
"""My Custom Tool for apex:sceneanimate."""

import hou
import apex

class MyTool:
    """Custom tool for the animate state."""

    def __init__(self, state):
        self.state = state
        self.scene = state.scene

    def label(self):
        """Display name shown in the UI."""
        return "My Tool"

    def hudTemplate(self):
        """Define keyboard shortcuts for the HUD."""
        return [
            {"label": "Do Action", "key": "H"},
            {"label": "Cancel", "key": "Esc"},
        ]

    def onActivate(self, kwargs=None):
        """Called when the tool is activated."""
        self.state.scene_viewer.setPromptMessage("My Tool: H=action, Esc=cancel")

    def onDeactivate(self):
        """Called when the tool is deactivated."""
        self.state.scene_viewer.clearPromptMessage()

    def onMouseEvent(self, kwargs):
        """Handle mouse events."""
        return False  # Let default state handle

    def onKeyEvent(self, kwargs):
        """Handle keyboard events."""
        return False  # Let default state handle

    def onMouseWheelEvent(self, kwargs):
        """Handle scroll wheel events."""
        return False  # Let default state handle


def load(viewer_state):
    """Entry point - called by the animate state to load the tool."""
    return MyTool(viewer_state)
```

### Key Attributes on `self.state`

| Attribute | Type | Description |
|:----------|:-----|:------------|
| `state.scene` | `apex.scene_2.Scene` | The APEX scene |
| `state.scene_viewer` | `hou.SceneViewer` | The viewport |
| `state.control_manager` | `ControlManager` | Manages control selection |
| `state.primary_control` | `str` | Path of primary selected control |
| `state.hotkeys` | `object` | Predefined hotkey symbols |

---

## 2. State Integration

### Starting a Tool

Tools are started via `apex.ui.statecommandutils`:

```python
from apex.ui.statecommandutils import startTool, inAnimateState, exitCurrentTool

# Check if we're in the animate state
if inAnimateState():
    # Start your tool by module path
    startTool("canvas_tools.my_tool.tool", {})
```

### Exiting a Tool

```python
from apex.ui.statecommandutils import exitCurrentTool

# Returns to default animate state behavior
exitCurrentTool()
```

### Accessing the Rig

```python
# Get a rig from the scene by path
rig = self.state.scene.getData("/MyFolder/MyRig")
if rig:
    graph = rig.graph  # The APEX graph
```

---

## 3. Event Handling

### Mouse Events

```python
def onMouseEvent(self, kwargs):
    ui_event = kwargs.get("ui_event")
    device = ui_event.device()
    reason = ui_event.reason()

    # Check mouse buttons
    if device.isLeftButton():
        pass
    if device.isMiddleButton():
        pass
    if device.isRightButton():
        pass

    # Check modifiers
    if device.isShiftKey():
        pass
    if device.isCtrlKey():
        pass

    # Event reasons
    if reason == hou.uiEventReason.Active:
        # Button pressed and held
        pass
    elif reason == hou.uiEventReason.Changed:
        # Mouse moved while button held
        pass
    elif reason == hou.uiEventReason.Picked:
        # Button released (click completed)
        pass

    # Get ray for intersection
    origin, direction = ui_event.ray()

    return False  # False = let default handle, True = consumed
```

### Keyboard Events

```python
import viewerstate.utils as su

def onKeyEvent(self, kwargs):
    ui_event = kwargs["ui_event"]
    device = ui_event.device()

    # Skip auto-repeat events
    if device.isAutoRepeat():
        return False

    key = su.hotkeySymbolOrKeyString(kwargs)
    if not key:
        return False

    # Check specific keys
    if key == "Esc":
        # Cancel operation
        return True

    # Check hotkey symbols (defined in state.hotkeys)
    if hou.hotkeys.isKeyMatch(key, self.state.hotkeys.tool_key_h):
        # H key pressed
        return True

    return False
```

### Scroll Wheel Events

```python
def onMouseWheelEvent(self, kwargs):
    ui_event = kwargs.get("ui_event")
    device = ui_event.device()

    scroll = device.mouseWheel()  # Positive = up, negative = down
    if scroll == 0:
        return False

    # Modifier-based increments
    if device.isShiftKey():
        increment = 1.025  # Fine (2.5%)
    elif device.isCtrlKey():
        increment = 1.5    # Coarse (50%)
    else:
        increment = 1.15   # Normal (15%)

    factor = increment if scroll > 0 else 1.0 / increment

    # Apply factor to something (e.g., scale)
    # ...

    return True
```

---

## 4. Control Properties

Controls have properties that affect their appearance: color, scale, alpha, shape.

### Setting Properties via Graph

```python
import apex
import apex.control_2 as cc

def set_control_property(self, joint_name, prop_name, value):
    """Set a property on a control's graph node."""
    # Get the TransformObject node ID
    nodeid = cc.getTransformObject(self.scene, self.rig().graph, joint_name)
    if nodeid < 0:
        return

    # Get existing properties
    properties = self.rig().graph.getNodeProperties(nodeid)
    if 'control' not in properties.keys():
        prop = apex.ParmDict()
    else:
        prop = properties['control']

    # Set the property
    prop[prop_name] = value
    properties['control'] = prop
    self.rig().graph.setNodeProperties(nodeid, properties)
```

### Property Reference

| Property | Type | Description |
|:---------|:-----|:------------|
| `color` | `hou.Vector3` | RGB color (0-1 range) |
| `shapeoffset` | `hou.Matrix4` | Scale/offset transform |
| `alpha` | `float` | Opacity (0.0-1.0) |
| `shapeoverride` | `str` | Shape name override |

### Color Example

```python
# Set color to blue
color = hou.Vector3(0.2, 0.4, 0.8)
set_control_property("joint_0", "color", color)
```

### Scale Example

```python
# Set scale to 2x
scale = hou.Vector3(2.0, 2.0, 2.0)
shapeoffset = hou.hmath.buildScale(scale)
set_control_property("joint_0", "shapeoffset", shapeoffset)
```

### Push Changes to Scene

After modifying properties, push changes:

```python
self.state.scene.setData(f"{RIG_PATH}/graph", self.rig().graph.freeze())
self.state.scene.addControlGraphForRig(RIG_PATH)
self.state.scene.setGraphDirty(f"{RIG_PATH}/graph")
self.state.scene.setGraphDirty(f"{RIG_PATH}/controls/graph")
self.state.scene.updateEvaluationParms(hou.frame())
self.state.control_manager.update(self.state.scene)
self.state.runSceneCallbacks()
```

---

## 5. Drag Gestures

For operations like color cycling or smooth adjustments, use a drag gesture pattern.

### State Variables

```python
def __init__(self, state):
    # ...
    self.is_dragging = False
    self.drag_start_x = 0
    self.drag_accumulated = 0
```

### Drag Detection

```python
DRAG_THRESHOLD = 40  # Pixels per step

def onMouseEvent(self, kwargs):
    ui_event = kwargs.get("ui_event")
    device = ui_event.device()
    reason = ui_event.reason()

    # Handle ongoing drag
    if self.is_dragging:
        if reason == hou.uiEventReason.Picked:
            # Release - end drag
            self.is_dragging = False
            return True
        elif reason in (hou.uiEventReason.Changed, hou.uiEventReason.Active):
            # Update accumulated distance
            current_x = device.mouseX()
            delta = current_x - self.drag_start_x
            self.drag_accumulated += delta
            self.drag_start_x = current_x

            # Check threshold
            steps = int(self.drag_accumulated / DRAG_THRESHOLD)
            if steps != 0:
                self.drag_accumulated -= steps * DRAG_THRESHOLD
                self._do_step_action(steps)
            return True

    # Start drag on Ctrl+LMB
    if device.isCtrlKey() and device.isLeftButton() and reason == hou.uiEventReason.Active:
        self.is_dragging = True
        self.drag_start_x = device.mouseX()
        self.drag_accumulated = 0
        return True

    return False
```

---

## 6. Selection

### Get Selected Controls

```python
def _get_selected_controls(self):
    """Get list of selected controls."""
    sel_paths = self.state._getControlSelectionList()
    controls = []
    for path in sel_paths:
        ctrl = self.state.control_manager.controls.get(path, None)
        if ctrl:
            controls.append(ctrl)
    return controls

def _get_primary_control(self):
    """Get the primary (first) selected control."""
    if self.state.primary_control:
        return self.state.control_manager.controls.get(self.state.primary_control, None)
    return None
```

---

## 7. HUD Template

The `hudTemplate()` method defines keyboard shortcuts shown in the viewport HUD.

```python
def hudTemplate(self):
    return [
        {"label": "Place Joint", "key": "H"},
        {"label": "Confirm", "key": "LMB"},
        {"label": "Cancel", "key": "Esc"},
        {"label": "Scale", "key": "MMB Scroll"},
        {"label": "Fine Scale", "key": "Shift+Scroll"},
        {"label": "Cycle Color", "key": "Ctrl+LMB Drag"},
    ]
```

---

## 8. Shelf Integration

To launch your tool from a shelf button:

### Find and Enter Sceneanimate State

```python
def find_sceneanimate_node():
    """Find a sceneanimate SOP to enter."""
    # Check selected nodes first
    for node in hou.selectedNodes():
        if node.type().name() == "apex::sceneanimate":
            return node

    # Search in /obj children
    for obj_child in hou.node("/obj").children():
        if obj_child.type().name() == "geo":
            for child in obj_child.allSubChildren():
                if child.type().name() == "apex::sceneanimate":
                    if child.isDisplayFlagSet():
                        return child

    return None

def enter_sceneanimate_state(node):
    """Enter the sceneanimate viewer state."""
    desktop = hou.ui.curDesktop()
    scene_viewer = desktop.paneTabOfType(hou.paneTabType.SceneViewer)
    if not scene_viewer:
        raise RuntimeError("No Scene Viewer found")

    node.setDisplayFlag(True)
    node.setSelected(True, clear_all_selected=True)
    scene_viewer.setCurrentState("apex::sceneanimate")
```

### Start Tool with Delayed Callback

```python
def enter_my_tool():
    from apex.ui.statecommandutils import inAnimateState, startTool

    if inAnimateState():
        startTool("canvas_tools.my_tool.tool", {})
        return

    node = find_sceneanimate_node()
    if node is None:
        hou.ui.displayMessage("No sceneanimate node found.")
        return

    enter_sceneanimate_state(node)

    # Delay tool start until state is active
    def start_tool_callback():
        if inAnimateState():
            startTool("canvas_tools.my_tool.tool", {})

    hou.ui.postEventCallback(start_tool_callback)

enter_my_tool()
```

---

## 9. File Location

Place your tool module in:

```
apex_tools/package/python3.11libs/canvas_tools/my_tool/tool.py
```

The module path for `startTool()` would be: `"canvas_tools.my_tool.tool"`

---

## 10. Reference Implementation

For a complete working example, see the Skeleton Placer:

**Path:** `apex_tools/package/python3.11libs/canvas_tools/skeleton_placer/tool.py`

**Features demonstrated:**
- Preview mode with semi-transparent controls
- Keyboard shortcuts (H, Shift+H, Ctrl+H, Esc)
- Scroll wheel scaling with Shift/Ctrl modifiers
- Color cycling via Ctrl+LMB drag
- Selection handling
- Control property manipulation
- Processor graph integration

**Skeleton Placer Controls:**

| Action | Input |
|:-------|:------|
| Preview joint | H |
| Place joint | H then LMB |
| Place parented | H + Selection then LMB |
| Reparent/Unparent | Shift+H |
| Remove joint | Ctrl+H |
| Cancel preview | Esc |
| Scale control | MMB Scroll |
| Fine scale | Shift+Scroll |
| Coarse scale | Ctrl+Scroll |
| Cycle color | Ctrl+LMB Drag |
