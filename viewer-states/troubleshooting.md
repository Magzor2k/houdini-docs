---
layout: default
title: Troubleshooting
parent: Viewer States
nav_order: 5
description: Common issues and solutions for Houdini Python viewer states
permalink: /viewer-states/troubleshooting/
---

# Viewer States Troubleshooting
{: .fs-9 }

Common issues and solutions when working with Houdini Python viewer states.
{: .fs-6 .fw-300 }

---

## Registration Issues

### State Not Registered

**Check:**
```python
hou.ui.isRegisteredViewerState("my_state")  # Returns False
```

**Causes:**
1. File not in `viewer_states/` folder
2. Syntax error in Python file
3. Missing `createViewerStateTemplate()` function

**Fix:**
```python
# Manually register
hou.ui.registerViewerStateFile("/full/path/to/my_state.py")
```

### State File Not Auto-Loading

**Symptom:** State file exists in `viewer_states/` folder but `hou.ui.isRegisteredViewerState()` returns False.

**Cause:** Houdini only auto-loads viewer state files that end with `_state.py`.

**Check your filename:**
- `my_state.py` - Will auto-load
- `my_viewer_state.py` - Will auto-load
- `my_tool.py` - Will NOT auto-load (missing `_state.py` suffix)

**Fix:** Rename the file to end with `_state.py`:

```bash
# Wrong
viewer_states/my_tool.py

# Correct
viewer_states/my_tool_state.py
```

**Verification:**
```python
# After renaming, trigger re-scan
hou.ui.registerViewerStates()

# Check registration
print(hou.ui.isRegisteredViewerState("my_state_name"))
```

### State Registered But Won't Activate

**Error:**
```
hou.OperationFailed: The attempted operation failed.
Failed setting state named "my_state".
```

**Causes:**
1. Wrong context (SOP state in OBJ level)
2. Runtime error in state code
3. Cached old version with bugs

**Fix:**
```python
# Check context
viewer = hou.ui.paneTabOfType(hou.paneTabType.SceneViewer)
print(viewer.pwd())  # Must be inside geo for SOP states

# Reload state
hou.ui.unregisterViewerState("my_state")
hou.ui.registerViewerStateFile("/path/to/my_state.py")
```

---

## Runtime Errors

### HUD Initialization Error

**Error:**
```
AttributeError: 'NoneType' object has no attribute 'installEventFilter'
```

**Cause:** Bug in Houdini 21.0's HUD overlay Qt system.

**Fix:** Remove all `hudInfo()` calls:
```python
# REMOVE THIS:
# self.scene_viewer.hudInfo(template=self.HUD_TEMPLATE)
# self.scene_viewer.hudInfo(hud_values={"count": "5"})

# USE print() INSTEAD:
print(f"Count: {len(self.points)}")
```

### Drawable NoneType Error

**Error:**
```
AttributeError: 'NoneType' object has no attribute 'setGeometry'
```

**Cause:** Drawable not created in `onEnter()` or already destroyed.

**Fix:**
```python
def _updateDrawable(self):
    if not self.point_drawable:  # Check first!
        return
    geo = hou.Geometry()
    geo.createPoints(self.points)
    self.point_drawable.setGeometry(geo)
```

### Invalid setParams Parameter

**Error:**
```
hou.InvalidInput: Invalid input.
Unrecognized parameter: point_size
```

**Cause:** Using incorrect parameter names for `GeometryDrawable.setParams()`.

**Fix:** Use `radius` instead of `point_size`:
```python
# WRONG:
self.point_drawable.setParams({
    "point_size": 15.0,  # Invalid!
})

# CORRECT:
self.point_drawable.setParams({
    "radius": 0.1,  # Point radius in world units
    "color1": (1.0, 1.0, 0.0, 1.0),
})
```

See [Drawables](drawables.md) for complete parameter reference.

---

## HDA DefaultState Not Triggering

### Symptom

HDA has `DefaultState` section, but state doesn't activate when node is selected.

### Understanding the Behavior (This is Expected!)

The `DefaultState` field specifies which state to use when **entering Handle mode**. This is **intentional design** - Houdini does NOT auto-activate states on node selection alone.

**To enter the state:**
1. Select the HDA node
2. Press **Enter** in the viewport (enters Handle mode)
3. State activates via `onEnter` handler

This behavior is documented by SideFX: *"This method is called when the state is activated by the user creating a new node, or selecting an existing node and pressing Enter in the viewport."*

### Why No Auto-Activation?

1. **User Control**: Auto-activating states could interrupt normal viewport navigation
2. **Performance**: States may have expensive initialization
3. **Multi-Selection**: User may select multiple nodes without wanting to enter a state
4. **Tool Workflow**: Houdini's tool system (View, Handles, etc.) manages state transitions

### Workarounds for Auto-Activation

If you truly need auto-activation when selecting a node:

**Option 1: Python Callback on Selection**
```python
# Add to 456.py or shelf tool
def on_selection_changed(selection):
    for node in selection.nodes():
        if node.type().name() == "mycompany::my_hda::1.0":
            viewer = hou.ui.paneTabOfType(hou.paneTabType.SceneViewer)
            viewer.setCurrentState("my_hda_state")
            break

hou.ui.addSelectionCallback(on_selection_changed)
```

**Option 2: HDA OnCreated Script**
In HDA Type Properties > Scripts > OnCreated:
```python
# Auto-enter state when node is created
import hou
viewer = hou.ui.paneTabOfType(hou.paneTabType.SceneViewer)
if viewer:
    viewer.setCurrentState("my_hda_state")
```

**Option 3: Button Parameter**
Add a button parameter to the HDA that activates the state:
```python
# Callback script for button
viewer = hou.ui.paneTabOfType(hou.paneTabType.SceneViewer)
viewer.setCurrentState("my_hda_state")
```

### Verification

```python
# Check DefaultState section exists
node_type = hou.nodeType(hou.sopNodeTypeCategory(), "mycompany::my_hda::1.0")
defn = node_type.definition()

if "DefaultState" in defn.sections():
    print("DefaultState:", defn.sections()["DefaultState"].contents())
else:
    print("No DefaultState section!")

# Also check ExtraFileOptions
if "ExtraFileOptions" in defn.sections():
    print("ExtraFileOptions:", defn.sections()["ExtraFileOptions"].contents())
```

### Common Issues

1. **Missing DefaultState section** - Add it via hotl or Python
2. **State name mismatch** - DefaultState content must exactly match the state's template name
3. **State not registered** - External state file must be loaded first
4. **Expecting auto-activation** - Remember: Enter key required in viewport!

---

## HDA ViewerStateModule Conflicts

### Symptom

HDA has both `ViewerStateModule` section AND external state file. State doesn't activate or has unpredictable behavior.

### Cause

Embedding `ViewerStateModule` in HDA can conflict with external state file registration. The recommended approach is to use external state files only.

### Diagnosis

```bash
# Check HDA sections
hotl -l my_hda.hda

# Look for ViewerStateModule - if present, may conflict
```

### Fix

Remove `ViewerStateModule` from HDA, keep only `DefaultState`:

**Option 1: Via hotl**
1. Expand HDA: `hotl -X expanded_dir my_hda.hda`
2. Delete `ViewerStateModule` file from definition folder
3. Update `Sections.list` to remove ViewerStateModule entry
4. Rebuild: `hotl -C expanded_dir my_hda.hda`

**Option 2: When creating HDA via Python**
```python
# Only add DefaultState, NOT ViewerStateModule
definition.addSection("DefaultState", STATE_NAME)
definition.setExtraFileOption("DefaultState", STATE_NAME)
# Do NOT add: definition.addSection("ViewerStateModule", code)
```

### Reference

See [HDA Integration Guide](hda-viewer-state-setup.md#automated-hda-creation-hythonpython-script) for the complete automated HDA creation workflow.

---

## Debugging Tips

### Print Statements

Add prints to trace execution:
```python
def onEnter(self, kwargs):
    print(">>> onEnter called")
    # ...

def onMouseEvent(self, kwargs):
    print(f">>> Mouse event: {kwargs['ui_event'].reason()}")
    # ...
```

### Check Houdini Console

Errors appear in:
- Python Shell output
- Main Houdini console (Help > Show Console on Windows)

### Test State Manually

```python
# Step by step activation
viewer = hou.ui.paneTabOfType(hou.paneTabType.SceneViewer)
print("Current location:", viewer.pwd())
print("State registered:", hou.ui.isRegisteredViewerState("my_state"))

try:
    viewer.setCurrentState("my_state")
    print("State activated!")
except Exception as e:
    print(f"Error: {e}")
```

### Reload After Changes

```python
hou.ui.reloadViewerState("my_state")
# Or full unregister/register cycle
hou.ui.unregisterViewerState("my_state")
hou.ui.registerViewerStateFile("/path/to/state.py")
```

---

## Context Issues

### SOP State in Wrong Context

**Symptom:** State works in some scenes but not others.

**Cause:** SOP states require being inside a SOP network (Geometry node).

**Fix:** Navigate into the geo node first:
```python
viewer = hou.ui.paneTabOfType(hou.paneTabType.SceneViewer)
geo = hou.node("/obj/geo1")
viewer.setPwd(geo)
viewer.setCurrentState("my_sop_state")
```

### Multiple Viewer Panes

**Symptom:** State activates in wrong pane.

**Fix:** Get the specific viewer:
```python
# Get viewer containing specific node
node = hou.node("/obj/geo1/my_node")
# ... or ensure correct pane type
viewer = hou.ui.paneTabOfType(hou.paneTabType.SceneViewer)
```

---

## Performance Issues

### Slow Drawing

**Symptom:** Viewport lags when state is active.

**Causes:**
1. Creating geometry every frame in `onDraw()`
2. Too many points in drawable

**Fix:** Only update geometry when data changes:
```python
def onDraw(self, kwargs):
    # Just draw, don't recreate geometry
    if self.point_drawable:
        self.point_drawable.draw(kwargs["draw_handle"])

def _updateDrawable(self):
    # Only called when points change
    geo = hou.Geometry()
    geo.createPoints(self.points)
    self.point_drawable.setGeometry(geo)
```

---

## Version Compatibility

### Houdini 21.0 Specific

- HUD `hudInfo()` may crash - avoid using
- Use `hou.GeometryDrawable` for custom graphics

### Testing Across Versions

State code should work across versions if you:
1. Avoid version-specific APIs
2. Don't use HUD features (buggy)
3. Stick to documented drawable types
