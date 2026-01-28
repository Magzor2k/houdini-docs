---
layout: default
title: Getting Started
parent: Viewer States
nav_order: 1
description: Quick start guide for creating Houdini Python viewer states
permalink: /viewer-states/getting-started/
---

# Getting Started with Viewer States

A quick tutorial to create your first Python viewer state.

---

## What You'll Build

A simple "Click Counter" state that:
- Places yellow points where you click
- Clears all points when you press 'C'
- Displays the count in the console

---

## Step 1: Create the State File

Create `package/viewer_states/click_counter_state.py`:

```python
"""
Click Counter State

A simple viewer state that places yellow points where you click.
LMB to add points, C to clear.
"""

import hou


class ClickCounterState:
    """Simple click counter viewer state."""

    def __init__(self, state_name, scene_viewer):
        self.state_name = state_name
        self.scene_viewer = scene_viewer
        self.points = []
        self.point_drawable = None

    def onEnter(self, kwargs):
        """Called when entering the state."""
        self.point_drawable = hou.GeometryDrawable(
            self.scene_viewer,
            hou.drawableGeometryType.Point,
            "click_points"
        )
        self.point_drawable.setParams({
            "radius": 0.1,  # Point radius in world units
            "color1": (1.0, 1.0, 0.0, 1.0),
        })
        self.point_drawable.show(True)

    def onExit(self, kwargs):
        """Called when exiting the state."""
        if self.point_drawable:
            self.point_drawable.show(False)
            self.point_drawable = None

    def onMouseEvent(self, kwargs):
        """Handle mouse events."""
        ui_event = kwargs["ui_event"]
        device = ui_event.device()
        reason = ui_event.reason()

        if reason == hou.uiEventReason.Picked and device.isLeftButton():
            origin, direction = ui_event.ray()
            pos = hou.hmath.intersectPlane(
                hou.Vector3(0, 0, 0),
                hou.Vector3(0, 1, 0),
                origin, direction
            )
            if pos:
                self.points.append(pos)
                self._updateDrawable()
                print(f"Added point {len(self.points)}: {pos}")
            return True
        return False

    def onKeyEvent(self, kwargs):
        """Handle key events."""
        key = kwargs["ui_event"].device().keyString()
        if key == "c":
            self.points = []
            self._updateDrawable()
            print("Cleared all points")
            return True
        return False

    def onDraw(self, kwargs):
        """Draw the points."""
        if self.point_drawable:
            self.point_drawable.draw(kwargs["draw_handle"])

    def _updateDrawable(self):
        """Update drawable geometry."""
        if not self.point_drawable:
            return
        geo = hou.Geometry()
        if self.points:
            geo.createPoints(self.points)
        self.point_drawable.setGeometry(geo)


def createViewerStateTemplate():
    """Create and return the viewer state template."""
    template = hou.ViewerStateTemplate(
        "click_counter_state",    # Unique state name
        "Click Counter",          # Display label
        hou.sopNodeTypeCategory()
    )
    template.bindFactory(ClickCounterState)
    return template
```

**Important:** The filename MUST end with `_state.py` for Houdini to auto-discover it.

---

## Step 2: Set Up Your Package

Create this folder structure:

```
your_package/
├── viewer_states/
│   └── click_counter_state.py
└── ...
```

---

## Step 3: Launch Houdini

Set `HOUDINI_PATH` to include your package:

```batch
set HOUDINI_PATH=C:\path\to\your_package;&
"C:\Program Files\Side Effects Software\Houdini 21.0.559\bin\houdini.exe"
```

---

## Step 4: Test the State

1. Create a Geometry node and dive inside
2. Press **Enter** in the viewport
3. Type "click" to filter states
4. Select "Click Counter"
5. Click in the viewport to add points
6. Press 'C' to clear

---

## Next Steps

- [State Class Reference](reference/state-class.md) - Understand the class structure
- [Events Reference](reference/events.md) - All event handlers
- [Drawables Reference](reference/drawables.md) - Drawing in the viewport
- [HDA Integration](guides/hda-integration.md) - Connect to an HDA
- [Testing Guide](guides/testing.md) - Testing and debugging

---

## File Structure Example

```
your_package/
├── otls/
│   └── mycompany.my_hda.1.0.hda
├── viewer_states/
│   └── my_tool_state.py
└── scripts/
    └── rebuild_my_hda.py
```

---

## Storing Geometry from Viewer State

To persist geometry created in a viewer state (so it saves with the hip file), use a **Stash SOP** inside an HDA.

### Method: Stash SOP with Promoted Parameter

1. Add a Stash SOP inside your HDA's network
2. Promote the `stash` parameter to the HDA level (or make the node editable)
3. Set geometry from your viewer state using `parm().set(geometry)`

```python
def _saveToStash(self):
    """Save points to the HDA's stash node."""
    if not self.node:
        return
    geo = hou.Geometry()
    if self.points:
        geo.createPoints(self.points)
    stash = self.node.node("stash1")
    if stash:
        stash.parm("stash").set(geo)

def _loadFromStash(self):
    """Load points from the HDA's stash node."""
    if not self.node:
        return
    stash = self.node.node("stash1")
    if stash:
        geo = stash.geometry()
        if geo:
            self.points = [pt.position() for pt in geo.points()]
            self._updateDrawable()
```

---

## References

- [SideFX Docs: Python States](https://www.sidefx.com/docs/houdini/hom/python_states.html)
- [SideFX Docs: hou.ViewerStateTemplate](https://www.sidefx.com/docs/houdini/hom/hou/ViewerStateTemplate.html)
- [SideFX Docs: hou.GeometryDrawable](https://www.sidefx.com/docs/houdini/hom/hou/GeometryDrawable.html)
