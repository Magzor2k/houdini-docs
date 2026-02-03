---
layout: default
title: Button Callbacks
parent: HDK
nav_order: 10
description: Implementing button parameter callbacks in verb-based SOPs with DS files
permalink: /hdk/button-callbacks/
---

# Button Callbacks

How to implement button parameter callbacks in HDK verb-based SOPs that use embedded DS files for parameter definitions.

---

## Overview

Button parameters (`PRM_CALLBACK` type) allow users to trigger actions from the parameter pane. In verb-based SOPs using embedded DS files, button callbacks require special handling through `PRM_TemplateBuilder::setCallback()`.

The key insight is that DS file callbacks (the `callback` field) don't work in compiled DSOs. Instead, you must register a C++ callback function after the template is built.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  DS File (embedded string)                          │
│  └── parm { name "reset" type button }              │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│  PRM_TemplateBuilder                                │
│  └── setCallback("reset", &MyNode::onReset)         │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│  Button Click → onReset() callback                  │
│  └── Modify parameters, trigger recook, etc.        │
└─────────────────────────────────────────────────────┘
```

## Implementation

### 1. Define Button in DS File

```cpp
static const char *theDsFile = R"THEDSFILE(
{
    name        parameters

    parm {
        name    "reset"
        label   "Reset Simulation"
        type    button
        help    "Reset the simulation to rest pose."
    }

    // Other parameters...
}
)THEDSFILE";
```

### 2. Declare Callback Method

In your node header file:

```cpp
class SOP_MyNode : public SOP_Node
{
public:
    static OP_Node* myConstructor(OP_Network* net, const char* name, OP_Operator* op);
    static PRM_Template* buildTemplates();
    static const UT_StringHolder theSOPTypeName;

    // Button callback declaration
    static int onReset(void* data, int index, fpreal64 time, const PRM_Template* tplate);

    // ... rest of class
};
```

### 3. Register Callback in buildTemplates()

```cpp
#include <PRM/PRM_TemplateBuilder.h>
#include <PRM/PRM_Callback.h>

PRM_Template* SOP_MyNode::buildTemplates()
{
    static PRM_TemplateBuilder templ("SOP_MyNode.cpp", theDsFile);

    // Register callback ONLY on first build
    if (templ.justBuilt())
    {
        templ.setCallback("reset", &SOP_MyNode::onReset);
    }

    return templ.templates();
}
```

### 4. Implement Callback Function

```cpp
/*static*/ int
SOP_MyNode::onReset(void* data, int /*index*/, fpreal64 time, const PRM_Template* /*tplate*/)
{
    SOP_MyNode* sop = static_cast<SOP_MyNode*>(data);
    if (!sop)
        return 0;

    // Modify parameters (disable undo for programmatic changes)
    {
        UT_AutoDisableUndos disableundos;
        sop->setChRefInt("myparam", 0, time, 0);
    }

    // Trigger a recook
    sop->forceRecook();

    return 1;  // Return 1 to refresh the UI
}
```

## Callback Signature

```cpp
typedef int (*PRM_Callback64)(void *data, int index, fpreal64 time, const PRM_Template *tplate);
```

| Parameter | Description |
|:----------|:------------|
| `data` | Pointer to the node instance (cast to your node type) |
| `index` | Parameter index (for multi-component params) |
| `time` | Current evaluation time |
| `tplate` | Pointer to the PRM_Template |
| **Return** | `1` to refresh UI, `0` otherwise |

## Complete Example: Reset Button

This example shows a reset button that disables live simulation and resets state:

### Header

```cpp
class SOP_MySolver : public SOP_Node
{
public:
    static PRM_Template* buildTemplates();
    static int onReset(void* data, int index, fpreal64 time, const PRM_Template* tplate);

    // Flag for communicating between callback and cook
    bool needsForceReset() const { return myForceReset; }
    void setForceReset(bool reset) { myForceReset = reset; }

private:
    bool myForceReset = false;
};
```

### Implementation

```cpp
PRM_Template* SOP_MySolver::buildTemplates()
{
    static PRM_TemplateBuilder templ("SOP_MySolver.cpp", theDsFile);
    if (templ.justBuilt())
    {
        templ.setCallback("reset", &SOP_MySolver::onReset);
    }
    return templ.templates();
}

/*static*/ int
SOP_MySolver::onReset(void* data, int /*index*/, fpreal64 time, const PRM_Template* /*tplate*/)
{
    SOP_MySolver* sop = static_cast<SOP_MySolver*>(data);
    if (!sop)
        return 0;

    // Set flag for cook() to detect
    sop->setForceReset(true);

    // Disable live simulation toggle
    {
        UT_AutoDisableUndos disableundos;
        sop->setChRefInt("livesim", 0, time, 0);
        sop->setChRefInt("tick", 0, time, 0);
    }

    // Trigger recook
    sop->forceRecook();

    return 1;
}
```

### Cook Logic

```cpp
void SOP_MySolverVerb::cook(const CookParms &cookparms) const
{
    const SOP_Node* node = cookparms.getNode();

    // Check for reset flag
    bool resetPressed = false;
    if (node) {
        SOP_MySolver* solverNode = const_cast<SOP_MySolver*>(
            static_cast<const SOP_MySolver*>(node));
        if (solverNode && solverNode->needsForceReset()) {
            resetPressed = true;
            solverNode->setForceReset(false);  // Clear flag
        }
    }

    if (resetPressed) {
        // Reset simulation state...
    }
}
```

## Common Issues

### Callback Not Firing

**Problem:** Button click does nothing.

**Solution:** Ensure `setCallback()` is called inside `justBuilt()` check:
```cpp
if (templ.justBuilt())
{
    templ.setCallback("reset", &MyNode::onReset);
}
```

### DS File Callbacks Don't Work

**Problem:** The `callback` field in DS files is ignored in compiled DSOs.

**Solution:** Don't use DS file callbacks - use `PRM_TemplateBuilder::setCallback()` instead.

```cpp
// WRONG - this doesn't work in compiled DSOs:
parm {
    name    "reset"
    type    button
    callback "python('...')"  // Ignored!
}

// RIGHT - register in C++ instead:
templ.setCallback("reset", &MyNode::onReset);
```

### Cast Errors

**Problem:** Can't cast `void* data` to your node type.

**Solution:** Cast through the base class first:
```cpp
// Direct cast
MyNode* node = static_cast<MyNode*>(data);

// Or through base class (safer)
OP_Node* opnode = static_cast<OP_Node*>(data);
MyNode* node = static_cast<MyNode*>(opnode);
```

### Parameter Changes Not Saved

**Problem:** Parameter changes made in callback aren't saved in hip file.

**Solution:** Use `setChRefInt/Float/String` instead of direct manipulation:
```cpp
// Use setChRefInt for persistent changes
sop->setChRefInt("myparam", 0, time, newValue);
```

## Required Includes

```cpp
#include <PRM/PRM_TemplateBuilder.h>
#include <PRM/PRM_Callback.h>
#include <UT/UT_UndoManager.h>  // For UT_AutoDisableUndos
```

## See Also

- [Verb Nodes](verb-nodes.md) - SOP_NodeVerb pattern overview
- [Parameters](parameters.md) - Parameter definition details
- [PRM_Callback.h](https://www.sidefx.com/docs/hdk/_p_r_m___callback_8h.html) - Official HDK docs
- [PRM_TemplateBuilder](https://www.sidefx.com/docs/hdk/class_p_r_m___template_builder.html) - Template builder reference
