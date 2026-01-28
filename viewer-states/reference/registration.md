---
layout: default
title: Registration
parent: Reference
grand_parent: Viewer States
nav_order: 4
description: Viewer state registration methods
permalink: /viewer-states/reference/registration/
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
your_package/
├── viewer_states/
│   └── my_tool_state.py            # MUST end with _state.py
└── python3.11libs/
    └── my_tool/                    # Support modules (optional)
```

**Important:** Viewer state files MUST end with `_state.py` suffix for Houdini to auto-discover them during startup. Files without this suffix will not be registered automatically.

### Launcher Batch File Setup

Add your package to `HOUDINI_PATH` in your launcher batch file:

```batch
set MY_PACKAGE=C:\path\to\your_package
set HOUDINI_PATH=%MY_PACKAGE%;%HOUDINI_PATH%
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

### Example: APEX Package (SideFX Reference)

SideFX's built-in APEX package follows the same structure:

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

## Method 4: HDA with External State (DefaultState Section)

Link an HDA to an external viewer state file. This is the recommended approach for HDAs.

### How It Works

1. Create viewer state file in `package/viewer_states/my_state.py`
2. Add `DefaultState` section to HDA containing the state name
3. State activates when node is selected and Enter is pressed

### Setup

See [HDA Viewer State Setup](hda-viewer-state-setup.md) for complete instructions.

**Key points:**
- State file MUST end with `_state.py` (e.g., `my_hda_state.py`)
- HDA needs only a `DefaultState` section (NOT `ViewerStateModule`)
- Do NOT embed state code in the HDA - use external files

---

## Summary: Choosing a Registration Method

| Method | Persistence | Best For |
|:-------|:------------|:---------|
| **Package folder** | Yes | Distributing with tools (recommended) |
| **User prefs folder** | Yes | Personal/development states |
| **Programmatic** | No | Testing, dynamic states |
| **HDA + External State** | Yes | Node-specific behavior (recommended for HDAs) |

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
    contexts=None         # Optional: List of additional hou.NodeTypeCategory for multi-context states
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

See [Troubleshooting](troubleshooting.md) for common issues and solutions.

---

## Best Practices

1. **Use unique state names** - Prefix with project/company name
2. **Store in viewer_states folder** - For persistence
3. **Use packages** - For distribution
4. **Test with reload** - `hou.ui.reloadViewerState(name)`
5. **Handle errors gracefully** - Wrap in try/except in production
