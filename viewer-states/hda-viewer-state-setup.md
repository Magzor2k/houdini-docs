---
parent: Viewer States
layout: default
title: HDA Viewer State Setup
nav_order: 7
---

# HDA Viewer State Setup
{: .fs-9 }

Step-by-step guide to connecting a Python viewer state to an HDA.
{: .fs-6 .fw-300 }

---

## Quick Summary

1. Create a standalone viewer state file in `package/viewer_states/`
2. Add a `DefaultState` section to the HDA containing the state name
3. The state auto-activates when the HDA node is selected (press Enter)

---

## Method: External State File + DefaultState Section

This is the recommended approach - it's simpler and more maintainable than embedding state code in the HDA.

### Step 1: Create the Viewer State File

Create `package/viewer_states/my_state.py`:

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
        "my_hda_state",           # State name
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

## Verifying the Setup

### Check HDA Sections

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

## Troubleshooting

### State Not Found

**Symptom:** `isRegisteredViewerState()` returns False

**Fix:** Ensure the viewer state file is in `viewer_states/` folder within HOUDINI_PATH:
```
your_package/
├── viewer_states/
│   └── my_state.py    # Must be here
└── ...
```

### State Fails to Activate

**Symptom:** `setCurrentState()` raises `hou.OperationFailed`

**Possible causes:**
1. Wrong context - SOP state requires being inside a Geometry node
2. Python error in state code - check Houdini console
3. Cached old version - unregister and re-register:
   ```python
   hou.ui.unregisterViewerState("my_hda_state")
   hou.ui.registerViewerStateFile("/path/to/my_state.py")
   ```

### DefaultState Not Working

**Symptom:** Node selected but state doesn't auto-activate

**Important:** This is expected Houdini behavior! The `DefaultState` section specifies which state to use when entering **Handle mode** - it does NOT auto-activate on node selection.

**To activate:**
1. Select the HDA node
2. Press **Enter** in the viewport (enters Handle mode)
3. State activates via `onEnter`

**Why this design?** Auto-activating states on selection would interrupt normal viewport navigation and could cause performance issues.

**Want true auto-activation?** See the workarounds in [troubleshooting.md](troubleshooting.md#workarounds-for-auto-activation).

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

### Sections.list Format

Tab-separated, first line is empty string:
```
""
DialogScript	DialogScript
CreateScript	CreateScript
InternalFileOptions	InternalFileOptions
Contents.gz	Contents.gz
ExtraFileOptions	ExtraFileOptions
DefaultState	DefaultState
TypePropertiesOptions	TypePropertiesOptions
```

---

## Alternative: Embedded ViewerStateModule

You can embed state code directly in the HDA, but this is more complex and harder to debug.

### Required Sections

- `ViewerStateModule` - Python state code
- `DefaultState` - State name

### Required ExtraFileOptions

```json
{
    "DefaultState": {
        "type": "string",
        "value": "my_hda_state"
    },
    "ViewerStateModule/IsPython": {
        "type": "bool",
        "value": true
    },
    "ViewerStateModule/IsScript": {
        "type": "bool",
        "value": true
    },
    "ViewerStateModule/IsViewerStateModule": {
        "type": "bool",
        "value": true
    }
}
```

**Note:** Embedded states can have registration issues and are harder to debug. The external file method is recommended.
