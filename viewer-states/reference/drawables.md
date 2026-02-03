---
layout: default
title: Drawables
parent: Reference
grand_parent: Viewer States
nav_order: 3
description: GeometryDrawable, TextDrawable, and HUD reference
permalink: /viewer-states/reference/drawables/
---

# Drawables and HUD
{: .fs-9 }

Drawing custom geometry, text, and HUD elements in the viewport.
{: .fs-6 .fw-300 }

---

## Overview

Viewer states can draw custom visuals in the viewport using:

- **GeometryDrawable** - Points, lines, polygons
- **TextDrawable** - Text labels in 2D/3D
- **HUD (Heads-Up Display)** - On-screen information panel (see Known Issues below)

---

## GeometryDrawable

### Creating Drawables

```python
def onEnter(self, kwargs):
    # Point drawable
    self.point_drawable = hou.GeometryDrawable(
        self.scene_viewer,
        hou.drawableGeometryType.Point,
        "my_points"
    )

    # Line drawable
    self.line_drawable = hou.GeometryDrawable(
        self.scene_viewer,
        hou.drawableGeometryType.Line,
        "my_lines"
    )

    # Face drawable
    self.face_drawable = hou.GeometryDrawable(
        self.scene_viewer,
        hou.drawableGeometryType.Face,
        "my_faces"
    )

    # Show them
    self.point_drawable.show(True)
    self.line_drawable.show(True)
```

### Drawable Geometry Types

| Type | Description |
|:-----|:------------|
| `hou.drawableGeometryType.Point` | Points/vertices |
| `hou.drawableGeometryType.Line` | Lines/curves |
| `hou.drawableGeometryType.Face` | Filled polygons |

### Drawable Parameters

> **Important:** Use `radius` for point size (in world units), NOT `point_size`. Using `point_size` will cause an `hou.InvalidInput` error.

```python
# Point parameters
self.point_drawable.setParams({
    "radius": 0.1,                # Point radius in world units (NOT point_size!)
    "color1": (1, 1, 1, 1),       # RGBA color
    "fade_factor": 1.0,           # Opacity (0-1)
    "use_cd_attrib": True,        # Use Cd attribute for color
})

# Line parameters
self.line_drawable.setParams({
    "line_width": 2.0,            # Line width in pixels
    "color1": (0.8, 0.8, 0.8, 1), # RGBA color
    "style": hou.drawableGeometryLineStyle.Plain,
    "fade_factor": 1.0,
})

# Face parameters
self.face_drawable.setParams({
    "color1": (0.2, 0.5, 1.0, 0.5),  # Semi-transparent blue
    "fade_factor": 0.5,
    "lit": False,                     # Unlit (flat shading)
})
```

### Setting Geometry

```python
def _updateDrawable(self):
    geo = hou.Geometry()
    if self.points:
        geo.createPoints(self.points)
    self.point_drawable.setGeometry(geo)
```

### Drawing in onDraw

```python
def onDraw(self, kwargs):
    if self.point_drawable:
        self.point_drawable.draw(kwargs["draw_handle"])
```

### Cleanup in onExit

```python
def onExit(self, kwargs):
    if self.point_drawable:
        self.point_drawable.show(False)
```

---

## TextDrawable

### Creating Text Drawables

```python
self.text_drawable = hou.TextDrawable(self.scene_viewer, "my_labels")
self.text_drawable.setParams({
    "text": "Hello World",
    "position": hou.Vector3(0, 1, 0),
    "color1": (1, 1, 1, 1),
    "font_size": 14,
})
self.text_drawable.show(True)
```

---

## HUD (Heads-Up Display)

> **Not Recommended (Houdini 21.0):** The `hudInfo()` method causes crashes due to a Qt overlay bug. Use **Prompt Messages** (below) instead. See [Troubleshooting](troubleshooting.md) for details.

If you need HUD functionality in future Houdini versions where the bug is fixed:

```python
HUD_TEMPLATE = {
    "title": "My State",
    "rows": [
        {"id": "count", "label": "Count", "value": "0"},
        {"type": "divider"},
        {"label": "LMB", "key": "Place point"},
    ]
}

# Setup (avoid in Houdini 21.0)
# self.scene_viewer.hudInfo(template=HUD_TEMPLATE)
```

---

## Prompt Messages (Recommended)

Display a message in the viewport status bar - the reliable alternative to HUD:

```python
def onEnter(self, kwargs):
    self.scene_viewer.setPromptMessage("LMB: Place point | C: Clear")

def onExit(self, kwargs):
    self.scene_viewer.clearPromptMessage()
```

---

## Best Practices

1. **Create drawables in onEnter** - Not in `__init__`
2. **Hide drawables in onExit** - Clean up properly
3. **Use `radius` for points** - Not `point_size` (causes error)
4. **Use prompt messages** - More reliable than HUD in Houdini 21.0
5. **Update geometry efficiently** - Only when data changes
