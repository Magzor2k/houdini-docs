---
layout: default
title: HDA Integration
parent: Guides
grand_parent: Viewer States
nav_order: 1
description: Connecting Python viewer states to Houdini Digital Assets
permalink: /viewer-states/guides/hda-integration/
---

# HDA Integration Guide

Step-by-step guide to connecting a Python viewer state to an HDA.

---

## Quick Summary

1. Create a standalone viewer state file in `package/viewer_states/`
2. Add a `DefaultState` section to the HDA containing the state name
3. The state activates when you press **Enter** in the viewport after selecting the HDA node

---

## Quick Checklist

### Pre-Flight

- [ ] Viewer state file named `*_state.py` (e.g., `my_tool_state.py`)
- [ ] File in `package/viewer_states/` directory
- [ ] `createViewerStateTemplate()` function defined
- [ ] State name in template matches `DefaultState` value

### HDA Setup

- [ ] `definition.addSection("DefaultState", STATE_NAME)`
- [ ] `definition.setExtraFileOption("DefaultState", STATE_NAME)`
- [ ] Do NOT add `ViewerStateModule` section
- [ ] Save HDA: `definition.save(HDA_PATH)`

---

## Method: External State File + DefaultState Section

This is the recommended approach - it's simpler and more maintainable than embedding state code in the HDA.

### Step 1: Create the Viewer State File

Create `package/viewer_states/my_hda_state.py` (filename MUST end with `_state.py`):

```python
import hou

class MyState:
    def __init__(self, state_name, scene_viewer):
        self.state_name = state_name
        self.scene_viewer = scene_viewer

    def onEnter(self, kwargs):
        print("State entered!")

    def onExit(self, kwargs):
        print("State exited!")

def createViewerStateTemplate():
    template = hou.ViewerStateTemplate(
        "my_hda_state",           # State name - MUST match DefaultState
        "My HDA State",           # Display label
        hou.sopNodeTypeCategory() # SOP context
    )
    template.bindFactory(MyState)
    return template
```

### Step 2: Expand the HDA

```bash
hotl -X expanded_dir my_hda.hda
```

This creates:
```
expanded_dir/
├── houdini.hdalibrary
├── INDEX__SECTION
├── Sections.list
└── My_8_8Sop_1my__hda_8_81.0/
    ├── Contents.gz
    ├── CreateScript
    ├── DialogScript
    ├── ExtraFileOptions
    ├── InternalFileOptions
    └── Sections.list
```

### Step 3: Add DefaultState Section

Create a file named `DefaultState` in the HDA definition folder:

```bash
echo "my_hda_state" > expanded_dir/My_8_8Sop_1my__hda_8_81.0/DefaultState
```

### Step 4: Update Sections.list

Edit `expanded_dir/My_8_8Sop_1my__hda_8_81.0/Sections.list` to include the new section:

```
""
DialogScript	DialogScript
CreateScript	CreateScript
InternalFileOptions	InternalFileOptions
Contents.gz	Contents.gz
ExtraFileOptions	ExtraFileOptions
DefaultState	DefaultState
```

### Step 5: Rebuild the HDA

```bash
hotl -C expanded_dir my_hda.hda
```

### Step 6: Test

1. Launch Houdini with your package in HOUDINI_PATH
2. Create the HDA node
3. Select it and press **Enter** in viewport
4. The state should activate

---

## Programmatic Setup

Create HDA with DefaultState section via Python:

```python
import hou
import os

# Create HDA
obj = hou.node("/obj")
geo = obj.createNode("geo", "temp")
subnet = geo.createNode("subnet", "my_hda_base")

hda_node = subnet.createDigitalAsset(
    name="my::hda::1.0",
    hda_file_name="/path/to/my_hda.hda",
    description="My HDA",
)

node_type = hda_node.type()
definition = node_type.definition()

# Add DefaultState section
definition.addSection("DefaultState", "my_hda_state")

# Save
definition.save("/path/to/my_hda.hda")

# Cleanup
geo.destroy()
```

---

## Complete Automated HDA Creation

Working template for creating HDAs with viewer states via hython:

```python
"""
Rebuild HDA with Viewer State - Working Template
Run via: hython rebuild_my_hda.py
"""

import hou
import os

# Configuration
PACKAGE_DIR = "c:/path/to/package"
OTLS_DIR = os.path.join(PACKAGE_DIR, "otls")
HDA_PATH = os.path.join(OTLS_DIR, "sop_my.hda.1.0.hda")

HDA_NAME = "my::hda::1.0"
HDA_LABEL = "My HDA"
STATE_NAME = "my_hda"  # Must match createViewerStateTemplate() name

def rebuild_hda():
    # Ensure directories exist
    os.makedirs(OTLS_DIR, exist_ok=True)

    # Remove existing HDA if loaded
    node_type = hou.nodeType(hou.sopNodeTypeCategory(), HDA_NAME)
    if node_type:
        defn = node_type.definition()
        if defn:
            try:
                hou.hda.uninstallFile(defn.libraryFilePath())
            except:
                pass

    # Remove old HDA file
    if os.path.exists(HDA_PATH):
        os.remove(HDA_PATH)

    # Create fresh scene
    hou.hipFile.clear(suppress_save_prompt=True)

    # Build HDA internal network
    obj = hou.node("/obj")
    geo = obj.createNode("geo", "temp_geo")
    for child in geo.children():
        child.destroy()

    subnet = geo.createNode("subnet", "hda_base")
    # ... add internal nodes ...
    subnet.layoutChildren()

    # Create HDA definition
    hda_node = subnet.createDigitalAsset(
        name=HDA_NAME,
        hda_file_name=HDA_PATH,
        description=HDA_LABEL,
        min_num_inputs=0,
        max_num_inputs=0
    )

    node_type = hda_node.type()
    definition = node_type.definition()

    # Add parameters as needed
    ptg = definition.parmTemplateGroup()
    # ... add parameters ...
    definition.setParmTemplateGroup(ptg)

    # CRITICAL: Set DefaultState (external state file)
    # Do NOT add ViewerStateModule - use external file only
    definition.addSection("DefaultState", STATE_NAME)
    definition.setExtraFileOption("DefaultState", STATE_NAME)

    # Save
    definition.save(HDA_PATH)

    # Cleanup and reload
    geo.destroy()
    hou.hda.installFile(HDA_PATH)

    print(f"HDA created: {HDA_PATH}")
    print(f"DefaultState: {STATE_NAME}")
    print(f"Viewer state file: package/viewer_states/{STATE_NAME}_state.py")

rebuild_hda()
```

---

## Understanding DefaultState Behavior

**Important:** The `DefaultState` section specifies which state to use when entering **Handle mode** - it does NOT auto-activate on node selection.

**To activate the state:**
1. Select the HDA node
2. Press **Enter** in the viewport (enters Handle mode)
3. State activates via `onEnter`

This is intentional Houdini design - auto-activating states on selection would interrupt normal viewport navigation.

### Workarounds for Auto-Activation

If you truly need auto-activation when selecting a node:

**Option 1: OnCreated Script**

In HDA Type Properties > Scripts > OnCreated:
```python
import hou
viewer = hou.ui.paneTabOfType(hou.paneTabType.SceneViewer)
if viewer:
    viewer.setCurrentState("my_hda_state")
```

**Option 2: Button Parameter**

Add a button parameter to the HDA that activates the state:
```python
# Callback script for button
viewer = hou.ui.paneTabOfType(hou.paneTabType.SceneViewer)
viewer.setCurrentState("my_hda_state")
```

---

## Verification

### Check HDA Sections

```bash
# List all sections
hotl -l my_hda.hda
# Should have: DefaultState
# Should NOT have: ViewerStateModule
```

```python
import hou

node_type = hou.nodeType(hou.sopNodeTypeCategory(), "my::hda::1.0")
definition = node_type.definition()

print("Sections:", list(definition.sections().keys()))

if "DefaultState" in definition.sections():
    print("DefaultState:", definition.sections()["DefaultState"].contents())
```

### Check State Registration

```python
# Is state registered?
print(hou.ui.isRegisteredViewerState("my_hda_state"))

# Manually activate
viewer = hou.ui.paneTabOfType(hou.paneTabType.SceneViewer)
viewer.setCurrentState("my_hda_state")
```

---

## Common Mistakes

| Problem | Cause | Fix |
|:--------|:------|:----|
| State not registered | File not named `*_state.py` | Rename to `my_state.py` |
| State not activating | ViewerStateModule conflicts | Remove ViewerStateModule from HDA |
| State name mismatch | DefaultState != template name | Ensure names match exactly |
| No Enter key response | Wrong context | Navigate inside geo node first |

---

## HDA File Structure Reference

After setup, your expanded HDA should look like:

```
my_hda_definition/
├── Contents.gz           # Internal network
├── CreateScript          # Node creation script
├── DefaultState          # State name (e.g., "my_hda_state")
├── DialogScript          # Parameter UI definition
├── ExtraFileOptions      # JSON metadata
├── InternalFileOptions   # Internal options
├── Sections.list         # List of all sections
└── TypePropertiesOptions # Type properties (optional)
```

**Should NOT have:**
- `ViewerStateModule` - causes registration conflicts with external file

---

## Troubleshooting

See [Troubleshooting](../troubleshooting.md) for common issues including:
- State not registering (file naming `_state.py` requirement)
- State not activating (context, ViewerStateModule conflicts)
- DefaultState behavior (requires Enter key to activate)

---

## Editable Nodes (Writing to Stash from Viewer State)

When your viewer state needs to store data inside the HDA (e.g., in a stash node), you must mark that node as "editable" in the HDA definition. This allows writing to specific internal nodes without unlocking the entire HDA.

### The Wrong Way (NEVER do this)

```python
# WRONG - Never unlock HDA from viewer state!
def onEnter(self, kwargs):
    self.node = kwargs.get("node")
    if self.node and self.node.isLockedHDA():
        self.node.allowEditingOfContents()  # DON'T DO THIS!
```

This unlocks the entire HDA, which:
- Breaks asset integrity
- May cause issues with HDA updates
- Is not the intended workflow

### The Right Way: EditableNodes Section

Add an `EditableNodes` section to the HDA definition:

```python
# In your rebuild_hda.py script
definition = node_type.definition()

# Mark the stash node as editable (space-separated paths, NO trailing newline)
definition.addSection("EditableNodes", "controller_data")

# For multiple nodes:
definition.addSection("EditableNodes", "stash1 stash2 subnet/nested_stash")

definition.save(HDA_PATH)
```

**Critical format requirements:**
- Space-separated list of relative node paths
- NO trailing newline (use `"node_name"` not `"node_name\n"`)
- Paths are relative to the HDA root

### Verification

Check that the node is editable:

```python
import hou

# Create HDA instance
placer = geo.createNode('th::rig_controller_placer::1.0')

# Check the stash node
stash = placer.node('controller_data')
print(f"HDA is locked: {placer.isLockedHDA()}")
print(f"Stash is editable: {stash.isEditableInsideLockedHDA()}")

# Should print:
# HDA is locked: True
# Stash is editable: True
```

### Using the Editable Stash

From your viewer state, you can now write to the stash without unlocking:

```python
def _saveToStash(self):
    """Save data to HDA stash (marked as editable in definition)."""
    stash_node = self.node.node("controller_data")
    if stash_node is None:
        return

    # This works because stash is in EditableNodes
    geo = self._createMyGeometry()
    stash_node.parm("stash").set(geo)
```

### Complete HDA Creation Example

```python
def rebuild_hda():
    # ... create internal network ...

    # Create stash node (will be marked editable)
    stash = subnet.createNode("stash", "controller_data")

    # ... create HDA definition ...

    # CRITICAL: Set DefaultState for viewer state
    definition.addSection("DefaultState", STATE_NAME)
    definition.setExtraFileOption("DefaultState", STATE_NAME)

    # CRITICAL: Mark stash as editable inside locked HDA
    # Format: space-separated relative paths (NO trailing newline!)
    definition.addSection("EditableNodes", "controller_data")

    definition.save(HDA_PATH)
```

### Checking HDA Sections

```python
import hou

node_type = hou.nodeType(hou.sopNodeTypeCategory(), "my::hda::1.0")
definition = node_type.definition()
sections = definition.sections()

# Check EditableNodes
if "EditableNodes" in sections:
    content = sections["EditableNodes"].contents()
    print(f"EditableNodes: '{content}'")
    # Should print: EditableNodes: 'controller_data'
```

### Common EditableNodes Issues

| Problem | Cause | Fix |
|:--------|:------|:----|
| `Failed to modify node... permission error` | EditableNodes not set | Add `EditableNodes` section to definition |
| Stash shows `isEditable: False` | Trailing newline in section | Use `"node_name"` not `"node_name\n"` |
| Node not found | Wrong path | Use relative path from HDA root |
| Works after unlock, fails after reopen | Using `allowEditingOfContents()` | Remove it, use `EditableNodes` instead |

---

## Related Documentation

- [State Class Reference](../reference/state-class.md) - Complete state class structure and lifecycle
- [Animate State Configuration](animate-state-config.md) - Configure SideFX's Animate State framework for animation and rigging tools
- [Testing Viewer States](testing.md) - Debugging strategies and test workflows

> **Note**: If you're building animation or rigging tools, consider using SideFX's Animate State framework with config-based customization instead of creating a custom viewer state from scratch. See [Animate State Configuration](animate-state-config.md) for details.
