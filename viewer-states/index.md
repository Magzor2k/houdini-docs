---
layout: default
title: Viewer States
nav_order: 2
has_children: true
description: Python viewer states for Houdini viewport interaction
permalink: /viewer-states/
---

# Houdini Viewer States

A comprehensive guide to creating custom Python viewer states for Houdini's viewport interaction.

---

## What is a Viewer State?

A **Viewer State** is a Python-based interaction mode that controls how users interact with the viewport. When you use tools like Sculpt, Curve Draw, or APEX Animate, you're using viewer states.

Viewer states handle:
- **Mouse events** - clicks, drags, hover
- **Keyboard events** - hotkeys and shortcuts
- **Drawing** - custom geometry overlay in viewport
- **HUD display** - on-screen information panels
- **Handles** - interactive 3D widgets

---

## Quick Topic Lookup

| Looking for... | Go to |
|:---------------|:------|
| First-time tutorial | [getting-started.md](getting-started.md) |
| State class structure | [reference/state-class.md](reference/state-class.md) |
| Mouse/keyboard events | [reference/events.md](reference/events.md) |
| Drawables (points, lines, HUD) | [reference/drawables.md](reference/drawables.md) |
| Registration & file naming | [reference/registration.md](reference/registration.md) |
| HDA integration (DefaultState) | [guides/hda-integration.md](guides/hda-integration.md) |
| Animate State config, animation tools | [guides/animate-state-config.md](guides/animate-state-config.md) |
| Testing with `-waitforui` | [guides/testing.md](guides/testing.md) |
| Common problems | [troubleshooting.md](troubleshooting.md) |

---

## Architecture Overview

```
+-----------------------------------------------------------+
|                      Scene Viewer                         |
|  +-----------------------------------------------------+  |
|  |                   Viewer State                      |  |
|  |  +---------------+  +---------------+               |  |
|  |  |    Events     |  |   Drawables   |               |  |
|  |  | onMouseEvent  |  | GeometryDraw  |               |  |
|  |  | onKeyEvent    |  | TextDrawable  |               |  |
|  |  | onDraw        |  | HUD           |               |  |
|  |  +---------------+  +---------------+               |  |
|  |  +---------------+  +---------------+               |  |
|  |  |    Handles    |  |     Menu      |               |  |
|  |  | xformHandle   |  | RMB context   |               |  |
|  |  | customHandle  |  | actions       |               |  |
|  |  +---------------+  +---------------+               |  |
|  +-----------------------------------------------------+  |
+-----------------------------------------------------------+
```

---

## Quick Start

A minimal viewer state requires two components:

### 1. State Class

```python
class MyState:
    def __init__(self, state_name, scene_viewer):
        self.state_name = state_name
        self.scene_viewer = scene_viewer

    def onEnter(self, kwargs):
        """Called when entering the state."""
        pass

    def onExit(self, kwargs):
        """Called when exiting the state."""
        pass

    def onMouseEvent(self, kwargs):
        """Handle mouse interaction."""
        ui_event = kwargs["ui_event"]
        device = ui_event.device()

        if device.isLeftButton():
            # Handle left click
            return True
        return False
```

### 2. Template Function

```python
def createViewerStateTemplate():
    template = hou.ViewerStateTemplate(
        "my_state",                    # Unique state name
        "My Custom State",             # Display label
        hou.sopNodeTypeCategory()      # Node category
    )
    template.bindFactory(MyState)
    return template
```

---

## Documentation Sections

### Tutorials & Guides

| Section | Description |
|:--------|:------------|
| [Getting Started](getting-started.md) | Build your first viewer state |
| [HDA Integration](guides/hda-integration.md) | Connect states to HDAs |
| [Testing Guide](guides/testing.md) | Testing and debugging strategies |

### Reference

| Section | Description |
|:--------|:------------|
| [State Class](reference/state-class.md) | Class structure and lifecycle |
| [Events](reference/events.md) | Event handler reference |
| [Drawables](reference/drawables.md) | GeometryDrawable, TextDrawable, HUD |
| [Registration](reference/registration.md) | Registration methods |

### Support

| Section | Description |
|:--------|:------------|
| [Troubleshooting](troubleshooting.md) | Common issues and solutions |

---

## Registration Methods

| Method | Location | Use Case |
|:-------|:---------|:---------|
| **Package folder** | `your_package/viewer_states/` | Distributed with tools (recommended) |
| **User prefs folder** | `$HOUDINI_USER_PREF_DIR/viewer_states/` | Personal/development states |
| **Programmatic** | `hou.ui.registerViewerState()` | Dynamic/runtime states (not persistent) |
| **HDA embedded** | Inside HDA definition | Node-specific states |

**Recommended**: Use **package folder** method for production tools. See [Registration](reference/registration.md) for details.

---

## Key Locations

| Path | Description |
|:-----|:------------|
| `$HH/houdini/viewer_states/` | Built-in SideFX states |
| `$HOUDINI_USER_PREF_DIR/viewer_states/` | User custom states |
| `$HH/packages/apex/viewer_states/` | APEX viewer states |
| `$HH/houdini/python3.11libs/viewerstate/` | Utility modules |

---

## Environment

- **Houdini**: 21.0.559
- **Python**: 3.11