---
layout: default
title: Testing
parent: Guides
grand_parent: Viewer States
nav_order: 2
description: Guide to testing Python viewer states in Houdini
permalink: /viewer-states/guides/testing/
---

# Testing Viewer States

Strategies for testing and debugging Python viewer states.

---

## In Houdini Python Shell

The quickest way to test changes during development:

```python
# Register/reload
hou.ui.unregisterViewerState("my_state")
hou.ui.registerViewerStateFile("/path/to/my_state.py")

# Activate
viewer = hou.ui.paneTabOfType(hou.paneTabType.SceneViewer)
viewer.setCurrentState("my_state")
```

---

## Via Viewport UI

1. Go inside a Geometry node (for SOP states)
2. Press **Enter** in viewport
3. Type state name to filter
4. Select state from list

---

## GUI vs Headless Mode

### hython (headless)

- No GUI window
- `hou.ui` is NOT available
- Cannot test interactive features (mouse events, keyboard, dialogs)
- Use for: batch processing, data validation, non-UI logic tests

### houdini (GUI)

- Full graphical interface
- `hou.ui` fully available
- Can test all viewer state features including drawables and interaction

---

## GUI Testing with -waitforui

Houdini has a hidden command-line option `-waitforui` that specifies a synchronization point in the startup sequence. Scripts listed **after** this flag will only run once the GUI is fully initialized, giving them full access to `hou.ui`.

### Usage

```bash
houdini -waitforui test_viewer_state.py my_scene.hip
```

In this example:
1. Houdini starts and initializes the GUI
2. Once the UI is ready, `test_viewer_state.py` executes with full `hou.ui` access
3. Then `my_scene.hip` loads

### What This Enables

- Full `hou.ui` access in startup scripts
- Register and reload viewer states programmatically
- Display dialogs (`hou.ui.displayMessage`, `hou.ui.readInput`)
- Access panes, desktops, and viewport elements
- Test state lifecycle (`onEnter`, `onExit`) with a real viewport

### Example Test Script

```python
"""
test_viewer_state.py - Run with: houdini -waitforui test_viewer_state.py
"""
import hou

# Reload the state to pick up changes
state_name = "my_state"
state_path = "C:/path/to/package/viewer_states/my_state.py"

if hou.ui.isRegisteredViewerState(state_name):
    hou.ui.unregisterViewerState(state_name)
hou.ui.registerViewerStateFile(state_path)
print(f"Registered: {state_name}")

# Create test geometry node
obj = hou.node("/obj")
geo = obj.createNode("geo", "test_geo")

# Get viewer, navigate inside geo, and activate state
viewer = hou.ui.paneTabOfType(hou.paneTabType.SceneViewer)
viewer.setPwd(geo)
viewer.setCurrentState(state_name)

print(f"State '{state_name}' activated - test interactively in the viewport")
```

### Important

The `-waitforui` flag only works when launching the full Houdini GUI application. It is NOT a way to enable `hou.ui` in hython - hython is fundamentally headless by design.

---

## Testing Non-UI Logic with hython

While you can't test interactive features in hython, you can test:

- Data structures and algorithms
- Geometry creation/manipulation
- Serialization (save/load from stash)
- Helper functions

```python
"""
test_state_logic.py - Run with: hython test_state_logic.py
"""
import hou

# Test helper functions without UI
from my_state_module import calculate_positions, validate_input

def test_position_calculation():
    points = [hou.Vector3(0, 0, 0), hou.Vector3(1, 0, 0)]
    result = calculate_positions(points)
    assert len(result) == 2

def test_validation():
    assert validate_input("valid_name") == True
    assert validate_input("") == False

if __name__ == "__main__":
    test_position_calculation()
    test_validation()
    print("All tests passed!")
```

---

## Quick Debugging

### Check Registration

```python
# Is state registered?
hou.ui.isRegisteredViewerState("my_state")

# List all registered states
# (No direct API, but you can try activating)
```

### Reload After Changes

```python
hou.ui.unregisterViewerState("my_state")
hou.ui.registerViewerStateFile("/path/to/my_state.py")
```

### Add Print Statements

```python
def onEnter(self, kwargs):
    print(">>> onEnter called")
    # ...

def onMouseEvent(self, kwargs):
    print(f">>> Mouse: {kwargs['ui_event'].reason()}")
    # ...
```

### Check Houdini Console

Errors appear in:
- Python Shell output
- Main Houdini console (Help > Show Console on Windows)

---

## References

- [SideFX: Starting Houdini from the Command Line](https://www.sidefx.com/docs/houdini/ref/commandline.html)
- [SideFX Docs: Python States](https://www.sidefx.com/docs/houdini/hom/python_states.html)
