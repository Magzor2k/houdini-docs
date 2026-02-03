---
layout: default
title: Tool Events
parent: Reference
grand_parent: APEX
nav_order: 7
description: Event callbacks for APEX animate state tools
permalink: /apex/reference/tool-events/
---

# APEX Tool Events Reference

> Houdini 21.0 | Source: SideFX Internal Documentation

This document describes the event callbacks available when building custom animation tools for the APEX Animate State. Implement these events in your tool class to respond to user actions.

**Related:** [Controls](controls.md) - Control system architecture and API

---

## Activation/Deactivation/Saving

| Event | Called From | Description |
|:------|:------------|:------------|
| `onActivate(kwargs)` | `setActiveTool()` | Activates a tool. **Inputs:** `kwargs['settings_hud_tab_info']` - Tuple of (hud window, tab widget name, tab name); `kwargs['channel_hud_tab_info']` - Tuple of (channel hud window, tab widget name, tab name) |
| `onDeactivate()` | `setActiveTool()`, `_unloadTools()` | Deactivates the tool |
| `onPreSave()` | `_hipFileCB()` | Called just before hip file saves. Make modifications necessary to save/reload without issue |
| `onPostSave()` | `_hipFileCB()` | Called after hip file saves. Revert any changes made during `onPreSave()` |
| `onPreReloadEvent()` | `reloadFromGeometry()` | Called before reloading scene from geometry. If not implemented, tool will exit on reload |
| `onPostReloadEvent()` | `reloadFromGeometry()` | Called after reloading scene from geometry. If not implemented, tool will exit on reload |

---

## Animation Layers

| Event | Called From | Description |
|:------|:------------|:------------|
| `onPreActiveLayerChanged()` | `onPreActiveLayerChanged()` | Called just before active animation layer changes |
| `onActiveLayerChanged(kwargs)` | `onActiveLayerChanged()` | Called after active layer changes but before animation evaluator responds. **Inputs:** `kwargs['prev_active']` - Name of previously active layer |
| `onPostActiveLayerChanged()` | `onPostActiveLayerChanged()` | Called after active layer has been changed |
| `onLayerAnimChanged()` | `handleLayerAnimChanged()` | Called when animation layers stack changes (locking, muting, soloing, adding layers, switching layers, etc.) |

---

## Baking

| Event | Called From | Description |
|:------|:------------|:------------|
| `onBakeEvent(kwargs)` | `bakeKeys()` | Bake keys according to Bake settings HUD options. **Inputs:** `kwargs` - Dictionary of parameters from bake menu |
| `onPostBakeEvent(kwargs)` | `bakeKeys()` | Called after bake event completes. **Inputs:** `kwargs['handled_by_tool']` - Boolean if active tool handled the bake; `kwargs` - Dictionary of bake menu parameters |
| `onStartRecordingEvent(kwargs)` | `startRecordingPoses()` | Called when tool should start recording poses. **Inputs:** `kwargs` - Dictionary of bake menu parameters |
| `onStopRecordingEvent(kwargs)` | `stopRecordingPoses()` | Called when tool should stop recording poses. **Inputs:** `kwargs` - Dictionary of bake menu parameters |

---

## Channels

| Event | Called From | Description |
|:------|:------------|:------------|
| `onChannelsChangeStartEvent(collection_name, channel_names)` | `_handleActiveChannelsChanged()` | Called at start of channel change from outside Animate State (typically from Animation Editor) |
| `onChannelsChangeEvent(collection_name, channel_names)` | `_handleActiveChannelsChanged()` | Called while channels are changed from outside Animate State |
| `onChannelsChangeEndEvent(collection_name, channel_names)` | `_handleFinishedChannelsChanged()` | Called at end of channel change from outside Animate State |
| `onPushChannelsEvent(collection_name, channel_names)` | `_handleFinishedChannelsChanged()` | Called when geometry channels are pushed to animation editor |
| `clearPendingChannels()` | `_clearPending()` | Clears pending keys from all channels |
| `getPendingChannels()` | `_updatePendingFromBindings()` | Returns custom channel primitives with pending keys |
| `getPinnedChannels()` | `_scopeSelected()` | Returns custom channel primitives that are pinned |
| `getPinnedControls()` | `_setPinnedControls()` | Returns list of custom control paths that are pinned |
| `getScopedChannels()` | `_scopeSelected()` | Returns custom channel primitives that are scoped |
| `getScopedControls()` | `_getScopedControls()` | Returns custom control paths that are scoped |
| `scopeChannels(channel_list, scoped_channels, pinned_channels)` | `_scopeSelected()` | Scopes the given lists of channels |
| `addGeometryChannelCallbacks()` | `_setChannelList()` | Adds tool functions to geometry channel changed callbacks |
| `removeGeometryChannelCallbacks()` | `_setChannelList()`, `_scopeSelected()` | Removes tool functions from geometry channel changed callbacks |

---

## Evaluation

| Event | Called From | Description |
|:------|:------------|:------------|
| `onPreEvaluationEvent()` | `_runSceneCallbacks()` | Updates evaluation parameters before evaluation. **Warning:** Calls to `runSceneCallbacks()` from within this event will not run. **Note:** For changes to take effect, the animation binding must be muted |
| `onPostEvaluationEvent()` | `_runSceneCallbacks()` | Evaluates geometry drawables not captured by scene callbacks. **Warning:** Calls to `runSceneCallbacks()` from within this event will not run |

---

## HUDs

| Event | Called From | Description |
|:------|:------------|:------------|
| `hudTemplate()` | `updateHud()` | Returns custom rows for the HUD template for a tool |

---

## Posing

| Event | Called From | Description |
|:------|:------------|:------------|
| `onPoseStartEvent(kwargs)` | `_startControlInteraction()`, `_onXformGadgetStartEvent()` | Called at end of functions that start control movement. Use for work needed at pose start. **Inputs:** `kwargs` - Dictionary from `onMouseEvent(kwargs)` |
| `onPoseStartPreInitEvent(kwargs)` | `_startControlInteraction()`, `_onXformGadgetStartEvent()` | Called before scheduler switches to interactive mode. Opportunity to add/remove constraints |
| `onPoseEvent(kwargs)` | `_updateControlInteraction()`, `_onXformGadgetEvent()` | Called at end of functions responding to control posing. **Inputs:** `kwargs` - Dictionary from `onMouseEvent(kwargs)` |
| `onPoseEndPreKeyEvent(kwargs)` | `_endControlInteraction()`, `_onXformGadgetEndEvent()` | Called after scheduler switches back to playback mode but before new keys are set |
| `onPoseEndEvent(kwargs)` | `_endControlInteraction()`, `_onXformGadgetEndEvent()`, `updateControls()` | Called at very end of control release functions |
| `onXformGadgetStartEvent(kwargs)` | `_onXformGadgetStartEvent()` | Called when tool-managed xform gadget begins moving |
| `onXformGadgetEvent(kwargs)` | `_onXformGadgetEvent()` | Called when tool-managed xform gadget is moving |
| `onXformGadgetEndEvent(kwargs)` | `_onXformGadgetEndEvent()` | Called when tool-managed xform gadget ends movement |

---

## Selections

| Event | Called From | Description |
|:------|:------------|:------------|
| `onSelectionChanged(control_selection)` | `_selectControls()` | Called when selected control set changes from any source |
| `onPreSelectionSetsSelectControls(kwargs)` | `onCommand(): SELECT_CONTROLS` | Called when selecting controls from selection manager. Set `kwargs['skip_next_selection_sets_selection'] = True` to skip selection |
| `getPickWalkData(sel_controls)` | `getPickWalkData()` | Returns two dicts: rig paths to `PickWalkRig` data structures, and rig paths to control paths |

---

## Selection Sets

| Event | Called From | Description |
|:------|:------------|:------------|
| `saveToolSelectionSetData(model, config)` | `setActiveTool()`, `_unloadTools()` | Saves tool's selection set data |
| `loadToolSelectionSetData(model, config)` | `setActiveTool()` | Loads tool's selection set data |
| `getVisibleCharacters()` | `_getVisibleCharacters()` | Returns list of paths to all visible characters managed by tool |
| `getAllControlPaths()` | `_splitControlPaths()`, `_getAllControlPaths()` | Returns list of paths to all controls managed by tool |
| `getVisibleControlPaths()` | `_getVisibleControlsList()` | Returns list of paths to visible controls managed by tool |
| `getSelectedControlPaths()` | `_getControlSelectionTree()`, `_frameSelectedControls()` | Returns list of paths to selected controls managed by tool |
| `selectionSetsShowCharacters(char_paths, pick_modifier)` | `onCommand(): SHOW_CHARS` | Alters visibility of characters managed by tool |
| `selectionSetsPinControls(tool_ctrl_paths, pick_modifier)` | `onCommand(): PIN_CONTROLS` | Alters 'pinned' status of controls managed by tool |
| `selectionSetsShowControls(tool_ctrl_paths, pick_modifier)` | `onCommand(): SHOW_CONTROLS` | Alters visibility of controls managed by tool |
| `selectionSetsSelectControls(ctrl_paths, primary_control, pick_modifier)` | `onCommand(): SELECT_CONTROLS` | Selects controls managed by tool |

---

## State Events

| Event | Called From | Description |
|:------|:------------|:------------|
| `onResume(kwargs)` | `onResume(kwargs)` | Called during state's `onResume` event |
| `onInterrupt(kwargs)` | `onInterrupt(kwargs)` | Called during state's `onInterrupt` event |
| `onDraw(kwargs)` | `onDraw(kwargs)` | Called during state's `onDraw` event |
| `onDrawInterrupt(kwargs)` | `onDrawInterrupt(kwargs)` | Called during state's `onDrawInterrupt` event |
| `onMouseWheelEvent(kwargs)` | `onMouseWheelEvent(kwargs)` | Called during state's `onMouseWheelEvent` |
| `onMouseEvent(kwargs)` | `onMouseEvent(kwargs)` | Called when xform handle and controls haven't handled the event. Return `True` to mark handled |
| `onPreLocateEvent(kwargs)` | `onMouseEvent(kwargs)` | Called before control highlighting and selection. Can fully block mouse interactions |
| `onKeyEvent(kwargs)` | `onKeyEvent(kwargs)` | Called at beginning of state's `onKeyEvent`. Return `True` to mark handled |
| `onKeyTransitEvent(kwargs)` | `onKeyTransitEvent(kwargs)` | Called at beginning of state's `onKeyTransitEvent`. Return `True` to mark handled |
| `onVolatileClientEvent(kwargs)` | `onVolatileClientEvent(kwargs)` | Called at beginning of state's `onVolatileClientEvent`. Return `True` to mark handled |
| `onPrePlaybackChangeEvent(kwargs)` | `onPlaybackChangeEvent(kwargs)` | Called at beginning of state's `onPlaybackChangeEvent` |
| `onPlaybackChangeEvent(kwargs)` | `onPlaybackChangeEvent(kwargs)` | Called at end of state's `onPlaybackChangeEvent` |
| `onCommand(command, command_args)` | `onCommand(kwargs)` | Called for custom commands. **Inputs:** `command` - Command name; `command_args` - Keyword argument dictionary |

---

## Viewport

| Event | Called From | Description |
|:------|:------------|:------------|
| `frameSelectedControls()` | `_frameSelectedControls()` | Adds geometry from tool's selected controls to list of geometries to frame |

---

## HUD Widget

| Event | Called From | Description |
|:------|:------------|:------------|
| `onParmUpdateEndFromWidget()` | `ControlChannelWidget().onActionButtonFromWidget()` | Called at end of parameter change action from control channel widget |
| `onUpdateParmsAndControlsFromWidget` | `stateparmutils._updateParmsAndControls()` | Called during and after parameter change action from control channel widget |
| `onHUDParmEvent(kwargs)` | `onHUDParmEvent(kwargs)` | Called for Parameter HUD window events. **Inputs:** `kwargs['parm_name']`, `kwargs['parm_value']`, `kwargs['event']` (ParmEvent), `kwargs['type_name']`, `kwargs['tab_widget_name']`, `kwargs['tab_name']`, `kwargs['args']` |
| `onHUDSettingEvent(kwargs)` | `onHUDSettingEvent(kwargs)` | Called for Settings HUD window events. Same inputs as `onHUDParmEvent` |

---

## Interactive Regression Tests

| Event | Called From | Description |
|:------|:------------|:------------|
| `savePlaybackOutputGeometry(geo)` | `savePlaybackOutputGeometry` | Saves current tool state into geometry. Returns None |

---

## Example: Basic Tool Class

```python
class MyAnimationTool:
    """Example tool implementing common event callbacks."""

    def onActivate(self, kwargs):
        """Called when tool is activated."""
        self.settings_hud = kwargs.get('settings_hud_tab_info')
        self.channel_hud = kwargs.get('channel_hud_tab_info')
        # Initialize tool state

    def onDeactivate(self):
        """Called when tool is deactivated."""
        # Clean up tool state
        pass

    def onSelectionChanged(self, control_selection):
        """Called when selected controls change."""
        # Update internal state based on new selection
        self._selected_controls = control_selection

    def onPoseStartEvent(self, kwargs):
        """Called when user starts posing controls."""
        # Record initial state for undo
        pass

    def onPoseEvent(self, kwargs):
        """Called during control posing."""
        # Update visualization or constraints
        pass

    def onPoseEndEvent(self, kwargs):
        """Called when user finishes posing."""
        # Finalize the pose, set keys if needed
        pass

    def onKeyEvent(self, kwargs):
        """Handle keyboard input."""
        key = kwargs.get('ui_event').device().keyString()
        if key == 'r':
            # Custom behavior for 'r' key
            return True  # Mark as handled
        return False  # Let other handlers process
```

# How to trigger a tool in scene animate:

from apex.ui.statecommandutils import startTool
startTool(<Tool module name as string>, {})