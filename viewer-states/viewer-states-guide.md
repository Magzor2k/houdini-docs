---
parent: Viewer States
layout: default
title: Complete Guide
nav_order: 6
---

# Houdini Python Viewer States Guide
{: .fs-9 }

A comprehensive guide to creating and deploying Python viewer states in Houdini.
{: .fs-6 .fw-300 }

---

## Overview

Python viewer states provide custom interactive tools in Houdini's viewport. They can:
- Handle mouse/keyboard input
- Draw custom geometry (points, lines, etc.)
- Integrate with HDAs for node-specific tools

---

## Installation Methods

### Method 1: Auto-Registration (Recommended)

Place `.py` files in a `viewer_states/` folder within your Houdini path:

```
your_package/
├── viewer_states/
│   └── my_state.py      # Auto-registered on startup
└── ...
```

**Launcher setup:**
```batch
set HOUDINI_PATH=C:\path\to\your_package;&
```

States are automatically registered when Houdini starts.

### Method 2: Manual Registration

```python
hou.ui.registerViewerStateFile("/path/to/my_state.py")
```

### Method 3: HDA-Embedded (Advanced)

Embed the state code in an HDA's `ViewerStateModule` section. See [HDA Integration](#hda-integration) below.

---

## Basic Viewer State Structure

```python
"""
My Custom State

Description of what this state does.
"""

import hou


class MyState:
    """Custom viewer state class."""

    def __init__(self, state_name, scene_viewer):
        self.state_name = state_name
        self.scene_viewer = scene_viewer
        # Initialize your data here
        self.my_drawable = None

    def onEnter(self, kwargs):
        """Called when entering the state."""
        # Create drawables, set up state
        pass

    def onExit(self, kwargs):
        """Called when exiting the state."""
        # Clean up drawables
        pass

    def onMouseEvent(self, kwargs):
        """Handle mouse events."""
        ui_event = kwargs["ui_event"]
        device = ui_event.device()
        reason = ui_event.reason()

        if reason == hou.uiEventReason.Picked and device.isLeftButton():
            # Handle left click
            return True
        return False

    def onKeyEvent(self, kwargs):
        """Handle keyboard events."""
        key = kwargs["ui_event"].device().keyString()
        if key == "c":
            # Handle 'c' key
            return True
        return False

    def onDraw(self, kwargs):
        """Draw custom geometry."""
        if self.my_drawable:
            self.my_drawable.draw(kwargs["draw_handle"])


def createViewerStateTemplate():
    """Required entry point - creates the state template."""
    template = hou.ViewerStateTemplate(
        "my_state_name",              # Unique state identifier
        "My State Label",             # Display name in UI
        hou.sopNodeTypeCategory()     # Context (SOP, OBJ, etc.)
    )
    template.bindFactory(MyState)
    return template
```

---

## Drawables

Drawables render custom geometry in the viewport.

### Point Drawable

```python
def onEnter(self, kwargs):
    self.point_drawable = hou.GeometryDrawable(
        self.scene_viewer,
        hou.drawableGeometryType.Point,
        "my_points"
    )
    self.point_drawable.setParams({
        "radius": 0.1,  # Point radius in world units (NOT point_size!)
        "color1": (1.0, 1.0, 0.0, 1.0),  # Yellow RGBA
    })
    self.point_drawable.show(True)

def onExit(self, kwargs):
    if self.point_drawable:
        self.point_drawable.show(False)
        self.point_drawable = None

def _updateDrawable(self):
    geo = hou.Geometry()
    if self.points:
        geo.createPoints(self.points)
    self.point_drawable.setGeometry(geo)
```

**Valid GeometryDrawable parameters for points:**
- `radius` (float): Point radius in world units. Default: 0.05
- `color1` (tuple): RGBA color
- `style` (hou.drawableGeometryPointStyle): Point shape (optional)

### Line Drawable

```python
self.line_drawable = hou.GeometryDrawable(
    self.scene_viewer,
    hou.drawableGeometryType.Line,
    "my_lines"
)
self.line_drawable.setParams({
    "line_width": 2.0,
    "color1": (0.0, 1.0, 0.0, 1.0),  # Green
})
```

---

## Mouse Event Handling

### Getting Click Position (Ground Plane)

```python
def onMouseEvent(self, kwargs):
    ui_event = kwargs["ui_event"]
    device = ui_event.device()
    reason = ui_event.reason()

    if reason == hou.uiEventReason.Picked and device.isLeftButton():
        # Ray from camera through mouse position
        origin, direction = ui_event.ray()

        # Intersect with ground plane (Y=0)
        pos = hou.hmath.intersectPlane(
            hou.Vector3(0, 0, 0),  # Plane origin
            hou.Vector3(0, 1, 0),  # Plane normal (up)
            origin, direction
        )

        if pos:
            # Use the position
            self.points.append(pos)
            self._updateDrawable()

        return True  # Event consumed

    return False  # Event not consumed
```

### Event Reasons

- `hou.uiEventReason.Picked` - Click completed
- `hou.uiEventReason.Start` - Drag started
- `hou.uiEventReason.Changed` - Dragging
- `hou.uiEventReason.Active` - Mouse moved (no button)

### Modifier Keys

```python
device = ui_event.device()
shift_held = device.isShiftKey()
ctrl_held = device.isCtrlKey()
alt_held = device.isAltKey()
```

---

## Keyboard Event Handling

```python
def onKeyEvent(self, kwargs):
    key = kwargs["ui_event"].device().keyString()

    if key == "c":
        self._clearAll()
        return True
    elif key == "g":
        self._generate()
        return True
    elif key == "Delete":
        self._deleteSelected()
        return True

    return False
```

---

## HDA Integration

### Understanding DefaultState Behavior

**Important:** The `DefaultState` section specifies which state to use when entering **Handle mode** - it does NOT auto-activate on node selection.

**To activate the state:**
1. Select the HDA node
2. Press **Enter** in the viewport (enters Handle mode)
3. State activates via `onEnter`

This is intentional Houdini design - auto-activating states on selection would interrupt normal viewport navigation.

### Option 1: External State File with DefaultState Section (Recommended)

1. **Create viewer state file** in `package/viewer_states/my_state.py`

2. **Add DefaultState section to HDA:**

   Using hotl (expand, edit, collapse):
   ```bash
   hotl -X expanded_dir my_hda.hda
   # Create file: expanded_dir/HDA_NAME/DefaultState containing just the state name
   echo "my_state_name" > expanded_dir/HDA_NAME/DefaultState
   # Update Sections.list to include DefaultState
   hotl -C expanded_dir my_hda.hda
   ```

   Or programmatically:
   ```python
   definition.addSection("DefaultState", "my_state_name")
   definition.save(hda_path)
   ```

### Option 2: Embedded ViewerStateModule (Complex)

The state code can be embedded in the HDA's `ViewerStateModule` section. However, this requires specific extra file options and can be tricky to set up correctly.

**Required sections:**
- `ViewerStateModule` - Contains the Python state code
- `DefaultState` - Contains the state name

**Required extra file options:**
- `ViewerStateModule/IsPython: True`
- `ViewerStateModule/IsScript: True`
- `ViewerStateModule/IsViewerStateModule: True`

---

## Known Issues & Workarounds

### HUD Initialization Error (Houdini 21.0)

**Error:**
```
AttributeError: 'NoneType' object has no attribute 'installEventFilter'
```

**Cause:** Bug in Houdini's Qt overlay system when calling `scene_viewer.hudInfo()`.

**Workaround:** Avoid using HUD templates. Use `print()` for debugging or create custom Qt overlays.

```python
# AVOID THIS (causes error in some Houdini versions):
# self.scene_viewer.hudInfo(template=self.HUD_TEMPLATE)

# USE THIS INSTEAD:
print(f"Point count: {len(self.points)}")
```

### State Not Activating

If `viewer.setCurrentState("my_state")` fails:

1. **Check context:** Must be in correct network type (SOP for sopNodeTypeCategory)
   ```python
   viewer = hou.ui.paneTabOfType(hou.paneTabType.SceneViewer)
   print(viewer.pwd())  # Should be inside a geo node for SOP states
   ```

2. **Reload the state:**
   ```python
   hou.ui.unregisterViewerState("my_state")
   hou.ui.registerViewerStateFile("/path/to/my_state.py")
   ```

3. **Restart Houdini** if states are cached incorrectly.

### Checking Registered States

```python
# Check if state is registered
hou.ui.isRegisteredViewerState("my_state")

# Get state info
hou.ui.viewerStateInfo("my_state")
```

---

## Testing Viewer States

### In Houdini Python Shell

```python
# Register/reload
hou.ui.unregisterViewerState("my_state")
hou.ui.registerViewerStateFile("/path/to/my_state.py")

# Activate
viewer = hou.ui.paneTabOfType(hou.paneTabType.SceneViewer)
viewer.setCurrentState("my_state")
```

### Via Viewport UI

1. Go inside a Geometry node (for SOP states)
2. Press **Enter** in viewport
3. Type state name to filter
4. Select state from list

### hython Limitations

`hou.ui` is not available in hython (headless mode). Viewer states cannot be tested in hython - use the Houdini GUI.

---

## File Structure Example

```
viewer_states/
├── docs/
│   └── viewer-states-guide.md
├── examples/
│   └── click_counter_test.hip
├── package/
│   ├── otls/
│   │   └── th.click_counter.1.0.hda
│   └── viewer_states/
│       └── click_counter_state.py
├── scripts/
│   ├── create_click_counter_hda.py
│   └── create_click_counter_example.py
└── hda_expanded/
    └── th_8_8Sop_1click__counter_8_81.0/
        ├── Contents.gz
        ├── CreateScript
        ├── DefaultState
        ├── DialogScript
        ├── ExtraFileOptions
        ├── InternalFileOptions
        ├── Sections.list
        └── TypePropertiesOptions
```

---

## Complete Working Example

### click_counter_state.py

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
        "th_click_counter",
        "Click Counter",
        hou.sopNodeTypeCategory()
    )
    template.bindFactory(ClickCounterState)
    return template
```

---

## Storing Geometry from Viewer State

To persist geometry created in a viewer state (so it saves with the hip file), use a **Stash SOP** inside the HDA.

### Method: Stash SOP with Promoted Parameter

1. **Add a Stash SOP** inside your HDA's network
2. **Promote the `stash` parameter** to the HDA level (or make the node editable)
3. **Set geometry from your viewer state** using `parm().set(geometry)`

### Example: Viewer State Setting Geometry on Stash

```python
def _updateGeometry(self):
    """Store points in the HDA's stash node."""
    if not self.node:
        return

    # Create geometry with clicked points
    geo = hou.Geometry()
    if self.points:
        geo.createPoints(self.points)

    # Find the stash node inside the HDA
    stash = self.node.node("stash1")
    if stash:
        # Set the geometry on the stash parameter
        stash.parm("stash").set(geo)
```

### HDA Setup for Stash

**Option 1: Editable Node**
- In HDA Type Properties → Node tab → Editable Nodes
- Add your stash node path (e.g., `stash1`)
- The stash can then modify itself inside the locked HDA

**Option 2: Promoted Parameter (Recommended)**
- Promote the stash's `stash` parameter to the HDA interface
- Reference `../stash` in the internal stash node
- Set via Python: `hda_node.parm("stash").set(geo)`

### Important Notes

- **File size**: Stashed geometry is saved in the .hip file, increasing file size
- **One geometry per stash**: Each stash parameter holds one geometry object
- **Freeze if needed**: Use `geo.freeze()` if the source geometry might change

### Complete Viewer State with Persistent Geometry

```python
class ClickCounterState:
    def __init__(self, state_name, scene_viewer):
        self.state_name = state_name
        self.scene_viewer = scene_viewer
        self.node = None
        self.points = []
        self.point_drawable = None

    def onEnter(self, kwargs):
        self.node = kwargs.get("node")
        # ... create drawable ...

        # Load existing points from stash if any
        self._loadFromStash()

    def onMouseEvent(self, kwargs):
        # ... handle click, add to self.points ...
        self._updateDrawable()
        self._saveToStash()
        return True

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
- [SideFX Forum: HDA Default State](https://forums.odforce.net/topic/25641-hda-default-state-field/)
