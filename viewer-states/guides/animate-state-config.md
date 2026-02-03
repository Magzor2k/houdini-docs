---
layout: default
title: Animate State Config
parent: Guides
grand_parent: Viewer States
nav_order: 4
description: Configuring SideFX's Animate State framework from HDA Python modules
permalink: /viewer-states/guides/animate-state-config/
---

# Animate State Configuration Guide

Learn how to configure SideFX's Animate State framework for animation and rigging tools.

---

## Quick Summary

SideFX's **Animate State** is a specialized viewer state framework used by animation tools like the Autorigger SOP and Animate SOP. It provides a configurable animation environment with tools, radial menus, and behaviors that can be customized via a config function in your HDA's Python module.

**Key Use Cases**:
- Animation and rigging tools that need interactive control placement
- Tools that need to layer custom behavior on top of the Animate State
- SOPs that require animation testing without saving keyframes

---

## When to Use Animate State vs. Custom Viewer States

| Use Animate State When | Use Custom Viewer State When |
|:------------------------|:-----------------------------|
| Building animation/rigging tools | Building general geometry editing tools |
| Need access to animation tools (constraints, dynamic motion, ragdoll) | Need complete control over state behavior |
| Want to layer behavior on existing framework | Creating entirely new interaction patterns |
| Working with APEX rig controls | Working with raw geometry or attributes |

See [HDA Integration](hda-integration.md) for creating custom viewer states from scratch.

---

## Configuration via HDA Python Module

Configuration is done via a `config` function added to your HDA's Python module. This function returns a dictionary controlling which tools are available, which tool to start with, and which features are enabled.

### Basic Example

Here's a minimal config from an HDA that uses the Animate State:

```python
def config():
    """Configure the Animate State for this HDA."""
    config_data = {}

    # Set which tool to auto-enter when state activates
    config_data["default_tool"] = "TOOL_CONSTRAINTS"

    # Disable the default animation toolbar
    config_data["disable_animation_tools"] = True

    return config_data
```

### Complete Autorigger Example

This example shows the pattern used by SideFX's Autorigger SOP:

```python
import hou

# Tool constants
TOOL_CONSTRAINTS = "constraints"
TOOL_DYNAMICMOTION = "dynamic_motion"
TOOL_RAGDOLL = "ragdoll"

def config():
    """Configure Animate State for Autorigger."""
    config_data = {}

    # Specify animation network node for geometry fetching
    config_data["animation_node"] = "ANIMATION"

    # Parameter that controls whether state is writable
    config_data["nosave_animation_parm"] = "disable_animation_tools"

    # Disable default animation toolbar
    config_data["disable_animation_tools"] = True

    # Override state preferences
    config_data["clickdrag"] = "locator_drag"  # Custom drag behavior

    # Auto-enter this tool on state activation
    config_data["default_tool"] = TOOL_CONSTRAINTS

    # Define radial menu layout
    config_data["radial_tool_menu"] = {
        TOOL_DYNAMICMOTION: hou.radialItemLocation.BottomRight,
        TOOL_RAGDOLL: hou.radialItemLocation.BottomLeft,
    }

    # Define available tools with custom labels
    config_data["tools"] = {
        TOOL_CONSTRAINTS: "Add Constraint",
        TOOL_DYNAMICMOTION: "Dynamic Motion",
        TOOL_RAGDOLL: "Ragdoll Physics",
    }

    return config_data
```

---

## Available Config Options

### `animation_node`

**Type**: `str`
**Description**: The node path inside the animation network to fetch animated scene geometry from.

```python
config_data["animation_node"] = "ANIMATION"
```

If not defined, the default first output of the HDA node is used for the scene.

---

### `nosave_animation_parm`

**Type**: `str`
**Description**: Name of a boolean parameter on your HDA that controls whether the Animate State is writable (can save keyframes).

```python
config_data["nosave_animation_parm"] = "disable_animation_tools"
```

**Use Case**: Testing animation behaviors without saving keyframes. Some users prefer auto-resetting behaviors (no keyframe saving), while others want to keep test keyframes.

**Workflow**:
1. Add a toggle parameter to your HDA (e.g., `disable_animation_tools`)
2. When **enabled**: State is read-only, keyframes are not saved
3. When **disabled**: State is writable, keyframes persist

---

### `disable_animation_tools`

**Type**: `bool`
**Description**: Disables the default animation toolbar and radial menu.

```python
config_data["disable_animation_tools"] = True
```

Use this when you want to provide only your custom tools without the default animation tools.

---

### `clickdrag` (and other state preferences)

**Type**: `str`
**Description**: Overrides the default click-and-drag control selection behavior.

```python
config_data["clickdrag"] = "locator_drag"
```

**Important**: **ALL** Animate State preferences can be overwritten this way. The dictionary key must match the preference name exactly.

Examples of overridable preferences:
- `clickdrag` - Click and drag behavior
- `select_mode` - Selection mode
- `auto_commit` - Auto-commit changes
- (Refer to Houdini's Animate State documentation for complete preference list)

---

### `default_tool`

**Type**: `str`
**Description**: The tool that will be automatically entered when the Animate State activates.

```python
config_data["default_tool"] = "TOOL_CONSTRAINTS"
```

This allows you to layer new state behavior on top of the Animate State. SideFX heavily relies on this pattern for rig tools - the state activates and immediately enters a specific tool mode.

**Workflow**:
1. Animate State activates (user presses Enter on HDA node)
2. State immediately enters the specified default tool
3. Tool handles all interaction until user switches or exits

---

### `radial_tool_menu`

**Type**: `dict[str, hou.radialItemLocation]`
**Description**: Dictionary mapping tool names to radial menu positions.

```python
config_data["radial_tool_menu"] = {
    "TOOL_DYNAMICMOTION": hou.radialItemLocation.BottomRight,
    "TOOL_RAGDOLL": hou.radialItemLocation.BottomLeft,
    "TOOL_CONSTRAINTS": hou.radialItemLocation.Top,
}
```

**Available Positions**:
- `hou.radialItemLocation.Top`
- `hou.radialItemLocation.TopRight`
- `hou.radialItemLocation.Right`
- `hou.radialItemLocation.BottomRight`
- `hou.radialItemLocation.Bottom`
- `hou.radialItemLocation.BottomLeft`
- `hou.radialItemLocation.Left`
- `hou.radialItemLocation.TopLeft`

---

### `tools`

**Type**: `dict[str, str]`
**Description**: Dictionary mapping tool identifiers to display labels.

```python
config_data["tools"] = {
    "TOOL_CONSTRAINTS": "Add Constraint",
    "TOOL_DYNAMICMOTION": "Dynamic Motion",
    "TOOL_RAGDOLL": "Ragdoll Physics",
}
```

This serves two purposes:
1. Defines which tools are available
2. Provides custom display labels for the toolbar/menus

Only tools listed here will be available in the Animate State. Use this to limit tools to only those relevant for your HDA.

---

## Integration Workflow

### Step 1: Create HDA with Animation Network

Your HDA should have an animation network node if it needs to display animated geometry:

```
your_hda/
├── input (geometry)
├── ANIMATION (animation network node)
└── output
```

### Step 2: Add Config Function to Python Module

1. Open your HDA's Type Properties
2. Go to the **Scripts** tab
3. Add the `config()` function to the **PythonModule** section
4. Define your config dictionary as shown in examples above

### Step 3: Set Default State

Add a `DefaultState` section to your HDA that references the Animate State:

```python
# Via hotl command
echo "animate" > expanded_hda/YourHDA/DefaultState

# Or programmatically
definition = node.type().definition()
definition.addSection("DefaultState", "animate")
definition.save(hda_path)
```

**Note**: The exact state name may vary depending on Houdini version. Check SideFX documentation for the correct Animate State identifier.

### Step 4: Test Configuration

1. Create an instance of your HDA
2. Press **Enter** in the viewport to activate the state
3. Verify:
   - Correct default tool activates
   - Radial menu shows configured tools
   - Toolbar visibility matches `disable_animation_tools` setting
   - Click/drag behavior matches preferences

---

## Implementing Custom Tools

Custom tools are loaded by the Animate State and provide interactive behaviors. This section covers patterns for creating tools that modify the APEX scene.

### Tool Class Structure

Tools are Python classes loaded via a `load()` entry point:

```python
def load(viewer_state):
    """Entry point - called by Animate State to load the tool."""
    return MyTool(viewer_state)

class MyTool:
    def __init__(self, state):
        self.state = state
        self.scene = state.scene  # APEX Scene object

    def label(self):
        return "My Tool"

    def onActivate(self, kwargs=None):
        pass

    def onDeactivate(self):
        pass

    def onMouseEvent(self, kwargs):
        return False  # Return True to consume event

    def onKeyEvent(self, kwargs):
        return False
```

### Scene Reload Handling

**Critical**: When `reloadFromGeometry()` is called, the tool exits UNLESS it has reload event hooks.

```python
def onPreReloadEvent(self):
    """Called before scene reload - prevents tool exit."""
    pass

def onPostReloadEvent(self):
    """Called after scene reload - refresh references."""
    # CRITICAL: scene may be recreated during reload
    self.scene = self.state.scene
```

### Scene Reference Refresh Pattern

After `reloadFromGeometry()`, `self.scene` becomes stale. Always refresh:

```python
# After any reload operation:
self.state.writeToSopNode()
self.state.reloadFromGeometry(force=True)

# MUST refresh scene reference
self.scene = self.state.scene

# MUST update control manager for hover to work
if hasattr(self.state, 'control_manager') and self.state.control_manager:
    self.state.control_manager.update(self.scene)

if hasattr(self.state, 'runSceneCallbacks'):
    self.state.runSceneCallbacks()

# Force viewport redraw
self.state.scene_viewer.curViewport().draw()
```

### Adding Controls to APEX Scene

Pattern for adding TransformObject controls that appear in the viewport:

```python
import apex
import apex.scene_2 as apexscene

# 1. Build graph as geometry with proper structure
geo = hou.Geometry()
geo.addAttrib(hou.attribType.Point, 'name', '')
geo.addAttrib(hou.attribType.Point, 'callback', '')
geo.addAttrib(hou.attribType.Point, 'parms', {}, create_local_variable=False)
geo.addAttrib(hou.attribType.Point, 'properties', {}, create_local_variable=False)
geo.addAttrib(hou.attribType.Vertex, 'portname', '')

# Create __parms__ node
pt_parms = geo.createPoint()
pt_parms.setAttribValue('name', '__parms__')
pt_parms.setAttribValue('callback', '__parms__')

# Create TransformObject node
pt_ctrl = geo.createPoint()
pt_ctrl.setAttribValue('name', 'ctrl_0')
pt_ctrl.setAttribValue('callback', 'TransformObject')
pt_ctrl.setAttribValue('properties', {
    'control': {
        'shape': 'circle_wires',
        'shapescale': [2.0, 2.0, 2.0],
        'color': [1.0, 0.8, 0.2],
    }
})

# Create __output__ node (required for control graph)
pt_output = geo.createPoint()
pt_output.setAttribValue('name', 'output')
pt_output.setAttribValue('callback', '__output__')

# Create wire from __parms__ to control's t port
wire = geo.createPolygon()
wire.setIsClosed(False)
v0 = wire.addVertex(pt_parms)
v1 = wire.addVertex(pt_ctrl)
v0.setAttribValue('portname', 'ctrl_0')  # Promoted param name
v1.setAttribValue('portname', 't')       # TransformObject t input

# Create wire from control's xform to __output__
wire2 = geo.createPolygon()
wire2.setIsClosed(False)
v0 = wire2.addVertex(pt_ctrl)
v1 = wire2.addVertex(pt_output)
v0.setAttribValue('portname', 'xform')
v1.setAttribValue('portname', 'ctrl_0_xform')

# 2. Load into apex.Graph
graph = apex.Graph()
graph.loadFromGeometry(geo)

# 3. Set default parameters
default_parms = graph.getDefaultParameters()
default_parms['ctrl_0'] = hou.Vector3(0, 0, 0)
graph.setDefaultParms(default_parms)

# 4. Store in scene
RIG_PATH = "/Controls.tool/Base.rig"
rig = apexscene.SceneGraph(RIG_PATH, graph)
self.scene.loadRigGraph(RIG_PATH, rig)
self.scene.setData(f"{RIG_PATH}/graph", graph.freeze())
self.scene.addControlGraphForRig(RIG_PATH)

# 5. Save and reload
self.state.writeToSopNode()
self.state.reloadFromGeometry(force=True)
self.scene = self.state.scene  # Refresh!
```

### State Attributes Reference

| Attribute | Type | Description |
|:----------|:-----|:------------|
| `state.scene` | `apex.scene_2.Scene` | The APEX scene object |
| `state.scene_viewer` | `hou.SceneViewer` | The viewport |
| `state.control_manager` | `ControlManager` | Handles control hover/selection |
| `state.hotkeys` | Object | Hotkey bindings (e.g., `state.hotkeys.tool_key_h`) |

### State Methods Reference

| Method | Description |
|:-------|:------------|
| `writeToSopNode()` | Save in-memory scene changes to SOP geometry |
| `reloadFromGeometry(force=True)` | Reload scene from SOP geometry |
| `runSceneCallbacks()` | Update drawables and visual state |
| `_getControlSelectionList()` | Get list of selected control paths |

---

## Troubleshooting

### Config function not called

**Symptoms**: State uses default behavior, ignores your config

**Solutions**:
- Verify function is named exactly `config()` (case-sensitive)
- Check function is in the HDA's PythonModule, not ViewerStateModule
- Ensure function returns a dictionary
- Check Houdini console for Python errors

### Tools not appearing

**Symptoms**: Custom tools missing from toolbar/radial menu

**Solutions**:
- Verify tool identifiers match between `tools` dict and `radial_tool_menu` dict
- Check tool identifiers are valid Animate State tools
- Ensure `disable_animation_tools` is not hiding all tools

### Default tool doesn't activate

**Symptoms**: State enters but doesn't auto-enter the specified tool

**Solutions**:
- Verify tool identifier in `default_tool` matches a key in `tools` dict
- Check tool is a valid Animate State tool
- Look for errors in Houdini console during state entry

### State preferences not applied

**Symptoms**: Click/drag behavior doesn't match config

**Solutions**:
- Verify preference name matches exactly (case-sensitive)
- Check preference value is valid for that preference type
- Some preferences may require specific Houdini versions

---

## Related Documentation

- [HDA Integration](hda-integration.md) - Creating custom viewer states
- [Viewer States Reference](../reference/state-class.md) - State class structure and lifecycle
- [Testing Viewer States](testing.md) - Debugging strategies

---

## Additional Resources

- SideFX Autorigger SOP - Reference implementation
- Houdini Animate State documentation - Complete preference list and tool identifiers
- APEX Character Rigging documentation - Animation workflow patterns
