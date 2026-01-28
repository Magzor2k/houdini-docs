---
layout: default
title: Events
parent: Reference
grand_parent: Viewer States
nav_order: 2
description: Event handler reference for viewer states
permalink: /viewer-states/reference/events/
---

# Event Handlers Reference
{: .fs-9 }

Complete reference for viewer state event handlers.
{: .fs-6 .fw-300 }

---

## Event Handler Summary

| Handler | When Called |
|:--------|:------------|
| `onEnter` | State becomes active |
| `onExit` | State becomes inactive |
| `onResume` | Returning from interruption |
| `onInterrupt` | State temporarily interrupted |
| `onMouseEvent` | Mouse button/movement |
| `onMouseWheelEvent` | Mouse wheel scroll |
| `onKeyEvent` | Keyboard input |
| `onDraw` | Viewport redraw |
| `onDrawSetup` | Before onDraw |
| `onMenuAction` | Menu item selected |
| `onMenuPreOpen` | Before menu opens |
| `onSelection` | Geometry selected |
| `onParmChangeEvent` | State parameter changed |
| `onNodeChangeEvent` | Node event occurred |
| `onHandleToState` | Handle value changed |
| `onStateToHandle` | Update handle from state |

---

## Mouse Events

### onMouseEvent

```python
def onMouseEvent(self, kwargs):
    """
    Handle all mouse events.

    Args:
        kwargs: Dict containing:
            - ui_event: hou.UIEvent
            - node: Current node (may be None)

    Returns:
        bool: True if event was consumed
    """
    ui_event = kwargs["ui_event"]
    device = ui_event.device()
    reason = ui_event.reason()
```

### UIEvent Device Properties

```python
device = ui_event.device()

# Mouse buttons
device.isLeftButton()      # bool
device.isMiddleButton()    # bool
device.isRightButton()     # bool

# Modifier keys
device.isShiftKey()        # bool
device.isCtrlKey()         # bool
device.isAltKey()          # bool

# Position
device.mouseX()            # int - Screen X
device.mouseY()            # int - Screen Y

# Tablet pressure (0.0 - 1.0)
device.tabletPressure()    # float

# Tablet tilt
device.tabletTilt()        # (float, float)
```

### UIEvent Reason Values

| Reason | Description |
|:-------|:------------|
| `hou.uiEventReason.Start` | Mouse button pressed down |
| `hou.uiEventReason.Active` | Mouse dragging (button held) |
| `hou.uiEventReason.Changed` | Mouse button released |
| `hou.uiEventReason.Located` | Mouse moved (no button) |
| `hou.uiEventReason.Picked` | Click completed |

### Getting World Position from Click

```python
def onMouseEvent(self, kwargs):
    ui_event = kwargs["ui_event"]

    # Get ray from camera through click point
    origin, direction = ui_event.ray()

    # Option 1: Intersect with plane
    plane_point = hou.Vector3(0, 0, 0)
    plane_normal = hou.Vector3(0, 1, 0)
    position = hou.hmath.intersectPlane(
        plane_point, plane_normal, origin, direction
    )

    # Option 2: Use construction plane
    import viewerstate.utils as su
    position = su.cplaneIntersection(
        self.scene_viewer, origin, direction
    )

    # Option 3: Intersect with geometry
    node = kwargs.get("node")
    if node:
        geo = node.geometry()
        gi = su.GeometryIntersector(geo)
        gi.intersect(origin, direction)
        if gi.prim_num >= 0:
            position = gi.position
```

### Mouse Event Patterns

#### Click Detection

```python
def onMouseEvent(self, kwargs):
    ui_event = kwargs["ui_event"]
    device = ui_event.device()
    reason = ui_event.reason()

    if reason == hou.uiEventReason.Picked:
        if device.isLeftButton():
            self._handleLeftClick(ui_event)
            return True
    return False
```

#### Drag Detection

```python
def onMouseEvent(self, kwargs):
    ui_event = kwargs["ui_event"]
    device = ui_event.device()
    reason = ui_event.reason()

    if device.isLeftButton():
        if reason == hou.uiEventReason.Start:
            # Start drag
            self.drag_start = self._getPosition(ui_event)
            self.is_dragging = True
            return True

        elif reason == hou.uiEventReason.Active:
            # Continue drag
            if self.is_dragging:
                current = self._getPosition(ui_event)
                self._handleDrag(self.drag_start, current)
                return True

        elif reason == hou.uiEventReason.Changed:
            # End drag
            if self.is_dragging:
                self.is_dragging = False
                self._finishDrag()
                return True

    return False
```

#### Hover Detection

```python
def onMouseEvent(self, kwargs):
    ui_event = kwargs["ui_event"]
    reason = ui_event.reason()

    if reason == hou.uiEventReason.Located:
        # Mouse moved without button
        position = self._getPosition(ui_event)
        self._updatePreview(position)
        return True

    return False
```

---

## Keyboard Events

### onKeyEvent

```python
def onKeyEvent(self, kwargs):
    """
    Handle keyboard input.

    Args:
        kwargs: Dict containing:
            - ui_event: hou.UIEvent

    Returns:
        bool: True if event was consumed
    """
    ui_event = kwargs["ui_event"]
    device = ui_event.device()
    key = device.keyString()
```

### Key String Values

| Key | keyString() |
|:----|:------------|
| Letters | `"a"`, `"b"`, ..., `"z"` |
| Numbers | `"0"`, `"1"`, ..., `"9"` |
| Function keys | `"F1"`, `"F2"`, ..., `"F12"` |
| Special | `"Escape"`, `"Enter"`, `"Space"`, `"Tab"` |
| Delete | `"Delete"`, `"Backspace"` |
| Arrows | `"Left"`, `"Right"`, `"Up"`, `"Down"` |
| Modifiers | `"Shift"`, `"Ctrl"`, `"Alt"` |

### Keyboard Patterns

```python
def onKeyEvent(self, kwargs):
    ui_event = kwargs["ui_event"]
    device = ui_event.device()
    key = device.keyString()

    # Simple key
    if key == "g":
        self._generate()
        return True

    # With modifiers
    if key == "Delete":
        if device.isShiftKey():
            self._deleteAll()
        else:
            self._deleteSelected()
        return True

    # Escape to cancel
    if key == "Escape":
        self._cancel()
        return True

    return False
```

---

## Mouse Wheel Events

### onMouseWheelEvent

```python
def onMouseWheelEvent(self, kwargs):
    """
    Handle mouse wheel scrolling.

    Args:
        kwargs: Dict containing:
            - ui_event: hou.UIEvent

    Returns:
        bool: True if event was consumed
    """
    ui_event = kwargs["ui_event"]
    device = ui_event.device()

    # Scroll amount: positive = up, negative = down
    scroll = device.mouseWheel()

    if scroll > 0:
        self._zoomIn()
    else:
        self._zoomOut()

    return True
```

---

## Menu Events

### onMenuAction

```python
def onMenuAction(self, kwargs):
    """
    Handle menu item selection.

    Args:
        kwargs: Dict containing:
            - menu_item: str - ID of selected item
    """
    action = kwargs["menu_item"]

    if action == "generate":
        self._generate()
    elif action == "delete":
        self._delete()
    elif action == "clear":
        self._clear()
```

### Defining Menus in Template

```python
def createViewerStateTemplate():
    template = hou.ViewerStateTemplate(...)
    template.bindFactory(MyState)

    # Create menu
    menu = hou.ViewerStateMenu("my_menu", "My State Menu")

    # Add items
    menu.addActionItem("generate", "Generate Output")
    menu.addActionItem("delete", "Delete Selected")
    menu.addSeparator()
    menu.addActionItem("clear", "Clear All")

    # Add toggle item
    menu.addToggleItem("show_preview", "Show Preview", True)

    # Bind menu to template
    template.bindMenu(menu)

    return template
```

### onMenuPreOpen

```python
def onMenuPreOpen(self, kwargs):
    """
    Called before menu opens. Enable/disable items dynamically.
    """
    menu = kwargs["menu"]

    # Disable delete if nothing selected
    has_selection = self.selected_index >= 0
    menu.setItemEnabled("delete", has_selection)

    # Update toggle state
    menu.setItemChecked("show_preview", self.show_preview)
```

---

## Node Events

### onNodeChangeEvent

```python
def onNodeChangeEvent(self, kwargs):
    """
    Called when node events occur.

    Args:
        kwargs: Dict containing:
            - node: hou.Node
            - node_events: List of hou.nodeEventType
    """
    node = kwargs.get("node")
    events = kwargs.get("node_events", [])

    for event in events:
        if event == hou.nodeEventType.BeingDeleted:
            # Node is being deleted
            self._cleanup()

        elif event == hou.nodeEventType.NameChanged:
            # Node was renamed
            pass

        elif event == hou.nodeEventType.InputRewired:
            # Input connection changed
            self._refreshGeometry()

        elif event == hou.nodeEventType.ParmTupleChanged:
            # A parameter changed
            self._refreshDisplay()
```

### Node Event Types

| Event Type | Description |
|:-----------|:------------|
| `BeingDeleted` | Node is being deleted |
| `NameChanged` | Node was renamed |
| `FlagChanged` | Display/render flag changed |
| `InputRewired` | Input connection changed |
| `ParmTupleChanged` | Parameter value changed |
| `ChildCreated` | Child node created |
| `ChildDeleted` | Child node deleted |
| `SelectionChanged` | Selection changed in node |

---

## Parameter Events

### onParmChangeEvent

```python
def onParmChangeEvent(self, kwargs):
    """
    Called when a state parameter changes.

    Args:
        kwargs: Dict containing:
            - parm_name: str
            - parm_value: The new value
    """
    name = kwargs.get("parm_name")
    value = kwargs.get("parm_value")

    if name == "radius":
        self.radius = value
        self._updateDisplay()

    elif name == "color":
        self.color = value
        self._updateDrawableColor()
```

---

## Selection Events

### onSelection

```python
def onSelection(self, kwargs):
    """
    Called when geometry is selected via a selector.

    Args:
        kwargs: Dict containing:
            - selection: hou.Selection
            - name: Selector name
            - node: Current node

    Returns:
        bool: True if selection was accepted
    """
    selection = kwargs.get("selection")
    selector_name = kwargs.get("name")
    node = kwargs.get("node")

    if selection:
        # Get selected points/prims
        if selection.selectionType() == hou.geometryType.Points:
            points = selection.points(node.geometry())
            self._handlePointSelection(points)
            return True

    return False
```

---

## Handle Events

### onHandleToState

```python
def onHandleToState(self, kwargs):
    """
    Called when handle values change.
    Transfer handle values to node parameters.

    Args:
        kwargs: Dict containing:
            - handle: Handle name
            - parms: Dict of handle parameter values
            - node: Current node
    """
    handle = kwargs["handle"]
    parms = kwargs["parms"]
    node = kwargs["node"]

    if handle == "xform":
        # Get translation from handle
        tx = parms["tx"]
        ty = parms["ty"]
        tz = parms["tz"]

        # Apply to node
        node.parm("tx").set(tx)
        node.parm("ty").set(ty)
        node.parm("tz").set(tz)
```

### onStateToHandle

```python
def onStateToHandle(self, kwargs):
    """
    Called to update handle from node/state values.

    Args:
        kwargs: Dict containing:
            - handle: Handle name
            - parms: Dict of handle parameters (writable)
            - node: Current node
    """
    handle = kwargs["handle"]
    parms = kwargs["parms"]
    node = kwargs["node"]

    if handle == "xform":
        # Read from node
        parms["tx"] = node.parm("tx").eval()
        parms["ty"] = node.parm("ty").eval()
        parms["tz"] = node.parm("tz").eval()
```

---

## Best Practices

1. **Always return bool** from event handlers
2. **Return True** when you handle an event (consumes it)
3. **Return False** to let Houdini handle the event
4. **Check for None** - node may be None
5. **Use reason correctly** - Start/Active/Changed for drag operations
6. **Clean up in onExit** - Reset state, hide drawables
