---
parent: Viewer States
layout: default
title: Registration
nav_order: 2
---

# Viewer State Registration
{: .fs-9 }

How to register viewer states so they persist and are available in Houdini.
{: .fs-6 .fw-300 }

---

## Overview

Viewer states must be **registered** with Houdini before they can be used. Registration tells Houdini about the state's name, label, category, and factory function.

There are **four main methods** for registration, each suited to different use cases.

---

## Method 1: Package viewer_states Folder (Recommended)

The recommended method for distributing viewer states with your tools.

### Location

```
your_package/viewer_states/
```

For our projects:
```
apex_tools/viewer_states/
```

### File Structure

Each state is a single `.py` file with a `createViewerStateTemplate()` function:

```python
# my_custom_state.py

import hou

class MyCustomState:
    def __init__(self, state_name, scene_viewer):
        self.state_name = state_name
        self.scene_viewer = scene_viewer

    def onEnter(self, kwargs):
        print(f"Entered {self.state_name}")

    def onMouseEvent(self, kwargs):
        return False

def createViewerStateTemplate():
    """Required function - Houdini calls this on startup."""
    template = hou.ViewerStateTemplate(
        "my_custom_state",           # state_name (unique identifier)
        "My Custom State",           # state_label (display name)
        hou.sopNodeTypeCategory()    # category
    )
    template.bindFactory(MyCustomState)
    return template
```

### Package Structure

```
apex_tools/
├── viewer_states/
│   └── skeleton_builder.py # Viewer state file
└── python3.11libs/
    └── skeleton_builder/   # Support modules
```

### Launcher Batch File Setup

Add your package to `HOUDINI_PATH` in your launcher batch file:

```batch
set APEX_TOOLS=C:\path\to\apex_tools
set HOUDINI_PATH=%APEX_TOOLS%;%HOUDINI_PATH%
```

### How It Works

1. Your launcher sets `HOUDINI_PATH` to include your package
2. On Houdini startup, the package's `viewer_states/` folder is scanned
3. Each `.py` file's `createViewerStateTemplate()` is called
4. States are registered automatically and persist across sessions

### Triggering Re-registration

```python
# Rescan and register all states
hou.ui.registerViewerStates()

# Register a specific file
hou.ui.registerViewerStateFile("/path/to/my_state.py")

# Reload a modified state
hou.ui.reloadViewerState("my_custom_state")
```

### Example: APEX Package (Reference)

```
$HH/packages/apex/
├── apex.json
├── viewer_states/
│   ├── apexanimate.py
│   ├── apexdebugstate.py
│   └── apexmapcharacter.py
└── python3.11libs/
    └── apex/
```

---

## Method 2: User Preferences Folder

For personal states not distributed with a package.

### Location

```
$HOUDINI_USER_PREF_DIR/viewer_states/
```

On Windows:
```
C:/Users/<username>/Documents/houdini21.0/viewer_states/
```

Same file structure as package method - each `.py` file needs `createViewerStateTemplate()`.

---

## Method 3: Programmatic Registration

For dynamic or runtime-generated states. **Not recommended** - states don't persist.

### Basic Registration

```python
import hou

class DynamicState:
    def __init__(self, state_name, scene_viewer):
        self.state_name = state_name
        self.scene_viewer = scene_viewer

def register_my_state():
    template = hou.ViewerStateTemplate(
        "dynamic_state",
        "Dynamic State",
        hou.sopNodeTypeCategory()
    )
    template.bindFactory(DynamicState)

    # Register the template
    hou.ui.registerViewerState(template)

# Call this in Python Shell or 456.py
register_my_state()
```

### In 456.py (Session Startup)

```python
# $HOUDINI_USER_PREF_DIR/scripts/456.py

def register_custom_states():
    import sys
    sys.path.insert(0, "/path/to/my/scripts")

    from my_states import my_custom_state
    template = my_custom_state.createViewerStateTemplate()
    hou.ui.registerViewerState(template)

register_custom_states()
```

### Limitations

- State does not persist if registered only via `hou.ui.registerViewerState()`
- Must re-register each Houdini session
- Use `viewer_states/` folder for persistence

---

## Method 4: HDA Embedded States

States embedded in Houdini Digital Assets.

### Setting Up

1. Create HDA with **Type Properties**
2. Go to **Interactive > State Script**
3. Paste state code or link to external file
4. Set **Default State** to your state name

### State Script Template

```python
# Embedded in HDA Type Properties > State Script

class MyHDAState:
    def __init__(self, state_name, scene_viewer):
        self.state_name = state_name
        self.scene_viewer = scene_viewer

    def onEnter(self, kwargs):
        node = kwargs.get("node")
        if node:
            print(f"Editing node: {node.path()}")

    def onMouseEvent(self, kwargs):
        return False

def createViewerStateTemplate():
    template = hou.ViewerStateTemplate(
        "my_hda_state",
        "My HDA State",
        hou.sopNodeTypeCategory()
    )
    template.bindFactory(MyHDAState)
    return template
```

### Associating State with Node Type

In **Type Properties > Interactive**:
- **Default State**: `my_hda_state`
- State automatically activates when node is selected

---

## Summary: Choosing a Registration Method

| Method | Persistence | Best For |
|:-------|:------------|:---------|
| **Package folder** | Yes | Distributing with tools (recommended) |
| **User prefs folder** | Yes | Personal/development states |
| **Programmatic** | No | Testing, dynamic states |
| **HDA embedded** | Yes | Node-specific behavior |

---

## Registration API Reference

### hou.ui Functions

| Function | Description |
|:---------|:------------|
| `registerViewerState(template)` | Register a ViewerStateTemplate |
| `registerViewerStateFile(path)` | Register state from a .py file |
| `registerViewerStates()` | Rescan and register all states |
| `unregisterViewerState(name)` | Unregister a state |
| `reloadViewerState(name)` | Reload state code |
| `isRegisteredViewerState(name)` | Check if state is registered |
| `viewerStateInfo(name)` | Get state metadata |

### ViewerStateTemplate Constructor

```python
hou.ViewerStateTemplate(
    state_name,           # str: Unique identifier
    state_label,          # str: Display name
    node_type_category,   # hou.NodeTypeCategory
    contexts=None         # Optional: Additional categories
)
```

### Node Type Categories

```python
hou.sopNodeTypeCategory()    # SOP context
hou.objNodeTypeCategory()    # OBJ context
hou.lopNodeTypeCategory()    # LOP/USD context
hou.dopNodeTypeCategory()    # DOP context
hou.cop2NodeTypeCategory()   # COP2 context
hou.chopNodeTypeCategory()   # CHOP context
hou.topNodeTypeCategory()    # TOP context
```

---

## Entering a Registered State

### From Python

```python
# Get scene viewer
pane = hou.ui.curDesktop().paneTabOfType(hou.paneTabType.SceneViewer)

# Enter the state
pane.setCurrentState("my_custom_state")
```

### From Viewport

1. Press **Enter** in viewport to open state selector
2. Type state name or browse list
3. Click to enter state

### Via Shelf Tool

```python
# Shelf tool script
import toolutils

viewer = toolutils.sceneViewer()
viewer.setCurrentState("my_custom_state")
```

---

## Troubleshooting

### State Not Found

```python
# Check if registered
print(hou.ui.isRegisteredViewerState("my_state"))

# List all registered states
# (No direct API, but you can try entering states)
```

### State Not Loading

1. Check file is in `viewer_states/` folder
2. Verify `createViewerStateTemplate()` function exists
3. Check for Python syntax errors in Houdini console
4. Try manual registration: `hou.ui.registerViewerStateFile(path)`

### State Not Persisting

- Ensure file is in `viewer_states/` folder (not programmatic registration)
- Check file permissions
- Verify Houdini has read access to the folder

---

## Best Practices

1. **Use unique state names** - Prefix with project/company name
2. **Store in viewer_states folder** - For persistence
3. **Use packages** - For distribution
4. **Test with reload** - `hou.ui.reloadViewerState(name)`
5. **Handle errors gracefully** - Wrap in try/except in production
