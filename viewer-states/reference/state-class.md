---
layout: default
title: State Class
parent: Reference
grand_parent: Viewer States
nav_order: 1
description: State class structure and lifecycle
permalink: /viewer-states/reference/state-class/
---

# State Class Structure
{: .fs-9 }

The anatomy of a viewer state class and its lifecycle.
{: .fs-6 .fw-300 }

---

## Class Requirements

A viewer state class must implement:

1. **Constructor** with specific signature
2. **Event handlers** (optional but usually needed)

### Constructor Signature

```python
class MyState:
    def __init__(self, state_name, scene_viewer):
        """
        Initialize the viewer state.

        Args:
            state_name: str - The registered state name
            scene_viewer: hou.SceneViewer - The viewport this state runs in
        """
        self.state_name = state_name
        self.scene_viewer = scene_viewer

        # Initialize your state data here
        self.my_data = []
        self.selected_index = -1
```

**Important**: The constructor signature must match exactly. Houdini passes these two arguments when creating the state instance.

---

## Lifecycle

```
┌────────────────────────────────────────────────────────────┐
│                    State Lifecycle                          │
├────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────┐                                               │
│   │ __init__│  State class instantiated                     │
│   └────┬────┘                                               │
│        │                                                    │
│        ▼                                                    │
│   ┌─────────┐                                               │
│   │ onEnter │  User enters the state                        │
│   └────┬────┘                                               │
│        │                                                    │
│        ▼                                                    │
│   ┌─────────────────────────────────────┐                   │
│   │        Active State Loop             │                  │
│   │  ┌──────────────┐ ┌──────────────┐  │                   │
│   │  │onMouseEvent  │ │ onKeyEvent   │  │                   │
│   │  └──────────────┘ └──────────────┘  │                   │
│   │  ┌──────────────┐ ┌──────────────┐  │                   │
│   │  │   onDraw     │ │onMenuAction  │  │                   │
│   │  └──────────────┘ └──────────────┘  │                   │
│   │  ┌────────────────┐ ┌────────────────┐  │                 │
│   │  │onParmChangeEvt │ │onNodeChangeEvt │  │                 │
│   │  └────────────────┘ └────────────────┘  │                 │
│   └─────────────────────────────────────┘                   │
│        │                                                    │
│        ▼                                                    │
│   ┌─────────┐                                               │
│   │ onExit  │  User leaves the state                        │
│   └─────────┘                                               │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

---

## Complete Class Template

```python
import hou
import viewerstate.utils as su

class MyCompleteState:
    """
    A complete viewer state template with all common methods.
    """

    # Class constants
    MSG = "My State: LMB to interact"

    def __init__(self, state_name, scene_viewer):
        self.state_name = state_name
        self.scene_viewer = scene_viewer

        # State data
        self.node = None
        self.geometry = None

        # Drawables (created in onEnter when we have kwargs)
        self.drawable = None

    # =========================================================================
    # Lifecycle Methods
    # =========================================================================

    def onEnter(self, kwargs):
        """
        Called when entering the state.

        kwargs contains:
            - node: The current node (if any)
            - state_parms: Dict of state parameter values
        """
        self.node = kwargs.get("node")

        # Create drawables
        self.drawable = hou.GeometryDrawable(
            self.scene_viewer,
            hou.drawableGeometryType.Point,
            "my_drawable"
        )
        self.drawable.show(True)

        # Show prompt
        self.scene_viewer.setPromptMessage(self.MSG)

    def onExit(self, kwargs):
        """
        Called when exiting the state.
        Clean up resources here.
        """
        if self.drawable:
            self.drawable.show(False)

        self.scene_viewer.clearPromptMessage()

    def onResume(self, kwargs):
        """
        Called when returning to this state after it was interrupted.
        For example, after using a handle.
        """
        self.scene_viewer.setPromptMessage(self.MSG)

    def onInterrupt(self, kwargs):
        """
        Called when state is temporarily interrupted.
        For example, when a handle takes over.
        """
        pass

    # =========================================================================
    # Input Event Methods
    # =========================================================================

    def onMouseEvent(self, kwargs):
        """
        Handle mouse events.

        Returns:
            bool: True if event was consumed, False to pass to next handler
        """
        ui_event = kwargs["ui_event"]
        device = ui_event.device()
        reason = ui_event.reason()

        # Mouse button states
        is_left = device.isLeftButton()
        is_middle = device.isMiddleButton()
        is_right = device.isRightButton()

        # Modifier keys
        shift = device.isShiftKey()
        ctrl = device.isCtrlKey()
        alt = device.isAltKey()

        # Event reasons
        if reason == hou.uiEventReason.Start:
            # Button pressed
            pass
        elif reason == hou.uiEventReason.Active:
            # Dragging
            pass
        elif reason == hou.uiEventReason.Changed:
            # Button released
            pass
        elif reason == hou.uiEventReason.Located:
            # Mouse moved (no button)
            pass
        elif reason == hou.uiEventReason.Picked:
            # Click completed
            pass

        return False

    def onKeyEvent(self, kwargs):
        """
        Handle keyboard events.

        Returns:
            bool: True if event was consumed
        """
        ui_event = kwargs["ui_event"]
        device = ui_event.device()
        key = device.keyString()

        if key == "g":
            self._doGenerate()
            return True
        elif key == "Escape":
            self._cancel()
            return True

        return False

    def onMouseWheelEvent(self, kwargs):
        """
        Handle mouse wheel events.
        """
        ui_event = kwargs["ui_event"]
        device = ui_event.device()
        scroll = device.mouseWheel()  # Positive = up, negative = down
        return False

    # =========================================================================
    # Drawing Methods
    # =========================================================================

    def onDraw(self, kwargs):
        """
        Draw custom geometry in the viewport.

        kwargs contains:
            - draw_handle: hou.DrawHandle for rendering
        """
        handle = kwargs["draw_handle"]

        if self.drawable:
            self.drawable.draw(handle)

    def onDrawSetup(self, kwargs):
        """
        Called before onDraw to set up drawing state.
        Useful for updating drawable geometry.
        """
        pass

    # =========================================================================
    # Node and Parameter Events
    # =========================================================================

    def onNodeChangeEvent(self, kwargs):
        """
        Called when the associated node changes.

        kwargs contains:
            - node: The node that changed
            - node_events: List of hou.nodeEventType values
        """
        node = kwargs.get("node")
        events = kwargs.get("node_events", [])

        if hou.nodeEventType.BeingDeleted in events:
            # Node is being deleted
            pass

    def onParmChangeEvent(self, kwargs):
        """
        Called when a state parameter changes.

        kwargs contains:
            - parm_name: Name of changed parameter
            - parm_value: New value
        """
        parm_name = kwargs.get("parm_name")
        parm_value = kwargs.get("parm_value")

    # =========================================================================
    # Menu Methods
    # =========================================================================

    def onMenuAction(self, kwargs):
        """
        Called when a menu item is selected.

        kwargs contains:
            - menu_item: The menu item ID that was selected
        """
        action = kwargs["menu_item"]

        if action == "do_action":
            self._doAction()
        elif action == "clear":
            self._clear()

    def onMenuPreOpen(self, kwargs):
        """
        Called before menu opens. Use to enable/disable items.

        kwargs contains:
            - menu: The menu about to open
        """
        menu = kwargs["menu"]
        # menu.setItemEnabled("item_id", True/False)

    # =========================================================================
    # Selection Methods
    # =========================================================================

    def onSelection(self, kwargs):
        """
        Called when geometry is selected.

        kwargs contains:
            - selection: hou.Selection object
            - node: The node
        """
        selection = kwargs.get("selection")
        if selection:
            # Process selection
            pass
        return False

    # =========================================================================
    # Handle Methods
    # =========================================================================

    def onHandleToState(self, kwargs):
        """
        Called when handle value changes.
        Transfer handle values to state/node.
        """
        handle = kwargs["handle"]
        parms = kwargs["parms"]
        # Update node parameters from handle

    def onStateToHandle(self, kwargs):
        """
        Called to update handle from state/node.
        Transfer state/node values to handle.
        """
        handle = kwargs["handle"]
        parms = kwargs["parms"]
        # Update handle from node parameters

    # =========================================================================
    # Private Helper Methods
    # =========================================================================

    def _doGenerate(self):
        """Generate output."""
        pass

    def _cancel(self):
        """Cancel current operation."""
        pass

    def _doAction(self):
        """Perform menu action."""
        pass

    def _clear(self):
        """Clear state data."""
        pass
```

---

## Instance Variables

Common instance variables to maintain:

```python
def __init__(self, state_name, scene_viewer):
    # Required
    self.state_name = state_name
    self.scene_viewer = scene_viewer

    # Node reference (set in onEnter)
    self.node = None

    # Drawables
    self.point_drawable = None
    self.line_drawable = None
    self.text_drawable = None

    # State data
    self.points = []
    self.selected_index = -1
    self.is_dragging = False
    self.drag_start = None

    # Cached geometry
    self.cached_geo = None
```

---

## Accessing Node and Geometry

```python
def onEnter(self, kwargs):
    self.node = kwargs.get("node")

def _getGeometry(self):
    """Get input geometry from node."""
    if self.node is None:
        return None

    # Get the node's input geometry
    inputs = self.node.inputs()
    if inputs and inputs[0]:
        return inputs[0].geometry()

    return None

def _getOutputGeometry(self):
    """Get node's output geometry."""
    if self.node is None:
        return None

    return self.node.geometry()
```

---

## Getting Click Position

```python
def onMouseEvent(self, kwargs):
    ui_event = kwargs["ui_event"]

    # Get ray from click
    origin, direction = ui_event.ray()

    # Intersect with construction plane
    cplane = self.scene_viewer.constructionPlane()
    if cplane.isVisible():
        plane_pt = cplane.origin()
        plane_n = cplane.normal()
    else:
        # Default ground plane
        plane_pt = hou.Vector3(0, 0, 0)
        plane_n = hou.Vector3(0, 1, 0)

    position = hou.hmath.intersectPlane(plane_pt, plane_n, origin, direction)
    return position
```

---

## State Parameters

State parameters allow user-configurable values:

```python
def createViewerStateTemplate():
    template = hou.ViewerStateTemplate(
        "my_state", "My State", hou.sopNodeTypeCategory()
    )
    template.bindFactory(MyState)

    # Add parameters
    template.bindParameter(
        hou.parmTemplateType.Float,
        name="radius",
        label="Radius",
        default_value=1.0
    )

    template.bindParameter(
        hou.parmTemplateType.Toggle,
        name="show_preview",
        label="Show Preview",
        default_value=True
    )

    return template
```

Access in state:

```python
def onEnter(self, kwargs):
    state_parms = kwargs.get("state_parms", {})
    self.radius = state_parms.get("radius", 1.0)
    self.show_preview = state_parms.get("show_preview", True)

def onParmChangeEvent(self, kwargs):
    parm_name = kwargs.get("parm_name")
    parm_value = kwargs.get("parm_value")

    if parm_name == "radius":
        self.radius = parm_value
    elif parm_name == "show_preview":
        self.show_preview = parm_value
```

---

## Best Practices

1. **Initialize drawables in onEnter** - Not in `__init__`
2. **Clean up in onExit** - Hide drawables, clear prompts
3. **Return True/False correctly** - From event handlers
4. **Store node reference** - From `kwargs["node"]` in onEnter
5. **Use viewerstate.utils** - For common operations
6. **Handle None cases** - Node may be None
