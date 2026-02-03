---
layout: default
title: SOP Node
parent: HDK
nav_order: 2
description: Creating SOP nodes with HDK
permalink: /hdk/sop-node/
---

# SOP Node Creation
{: .fs-9 }

Building custom Surface Operator nodes with the Houdini Development Kit.
{: .fs-6 .fw-300 }

---

## SOP Node Architecture

A Houdini SOP node consists of:

```
┌─────────────────────────────────────────────┐
│              SOP_MyNode.cpp                  │
│  ┌────────────────────────────────────────┐ │
│  │         newSopOperator()               │ │
│  │    - Register node with Houdini        │ │
│  │    - Define parameters                 │ │
│  └────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────┐ │
│  │         SOP_MyNode class               │ │
│  │    - Constructor                       │ │
│  │    - cookMySop() - main logic          │ │
│  │    - inputLabel() - input names        │ │
│  └────────────────────────────────────────┘ │
└─────────────────────────────────────────────┘
```

---

## Minimal SOP Node

### Header File

```cpp
// SOP_MyNode.h
#pragma once

#include <SOP/SOP_Node.h>

class SOP_MyNode : public SOP_Node
{
public:
    // Factory method for Houdini
    static OP_Node* myConstructor(
        OP_Network* net,
        const char* name,
        OP_Operator* op
    );

    // Parameter templates
    static PRM_Template myTemplateList[];

    // Input labels
    const char* inputLabel(unsigned idx) const override;

protected:
    SOP_MyNode(OP_Network* net, const char* name, OP_Operator* op);
    ~SOP_MyNode() override = default;

    // Main cook method
    OP_ERROR cookMySop(OP_Context& context) override;

private:
    // Helper to read parameters
    int ITERATIONS(fpreal t) { return evalInt("iterations", 0, t); }
    fpreal STRENGTH(fpreal t) { return evalFloat("strength", 0, t); }
};
```

### Source File

```cpp
// SOP_MyNode.cpp
#include "SOP_MyNode.h"

#include <OP/OP_AutoLockInputs.h>
#include <OP/OP_Operator.h>
#include <OP/OP_OperatorTable.h>
#include <PRM/PRM_Include.h>
#include <UT/UT_DSOVersion.h>

// ============================================================================
// Node Registration
// ============================================================================

void newSopOperator(OP_OperatorTable* table)
{
    table->addOperator(new OP_Operator(
        "hdk_mynode",              // Internal name
        "My Custom Node",          // Display name
        SOP_MyNode::myConstructor, // Factory function
        SOP_MyNode::myTemplateList,// Parameters
        1,                         // Min inputs
        1,                         // Max inputs
        nullptr,                   // Local variables
        OP_FLAG_GENERATOR          // Flags
    ));
}

// ============================================================================
// Parameter Definitions
// ============================================================================

static PRM_Name iterationsName("iterations", "Iterations");
static PRM_Name strengthName("strength", "Strength");

static PRM_Default iterationsDefault(10);
static PRM_Default strengthDefault(1.0);

static PRM_Range iterationsRange(PRM_RANGE_RESTRICTED, 1, PRM_RANGE_UI, 100);
static PRM_Range strengthRange(PRM_RANGE_RESTRICTED, 0.0, PRM_RANGE_UI, 2.0);

PRM_Template SOP_MyNode::myTemplateList[] = {
    PRM_Template(PRM_INT_J, 1, &iterationsName, &iterationsDefault,
                 nullptr, &iterationsRange),
    PRM_Template(PRM_FLT_J, 1, &strengthName, &strengthDefault,
                 nullptr, &strengthRange),
    PRM_Template()  // Null terminator required
};

// ============================================================================
// Constructor
// ============================================================================

OP_Node* SOP_MyNode::myConstructor(
    OP_Network* net,
    const char* name,
    OP_Operator* op)
{
    return new SOP_MyNode(net, name, op);
}

SOP_MyNode::SOP_MyNode(
    OP_Network* net,
    const char* name,
    OP_Operator* op)
    : SOP_Node(net, name, op)
{
}

// ============================================================================
// Input Labels
// ============================================================================

const char* SOP_MyNode::inputLabel(unsigned idx) const
{
    switch (idx) {
        case 0: return "Geometry to Process";
        default: return "Input";
    }
}

// ============================================================================
// Main Cook Method
// ============================================================================

OP_ERROR SOP_MyNode::cookMySop(OP_Context& context)
{
    // Lock inputs to prevent modification during cook
    OP_AutoLockInputs inputs(this);
    if (inputs.lock(context) >= UT_ERROR_ABORT)
        return error();

    // Get current time
    fpreal t = context.getTime();

    // Read parameters
    int iterations = ITERATIONS(t);
    fpreal strength = STRENGTH(t);

    // Duplicate input geometry to output
    duplicateSource(0, context);

    // Get writable geometry detail
    GU_Detail* gdp = this->gdp;
    if (!gdp)
        return error();

    // Get point count
    GA_Size numPoints = gdp->getNumPoints();
    if (numPoints == 0)
        return error();

    // Get position attribute handle
    GA_RWHandleV3 posHandle(gdp->getP());
    if (!posHandle.isValid())
        return error();

    // Process points
    for (int iter = 0; iter < iterations; iter++) {
        GA_Offset ptoff;
        GA_FOR_ALL_PTOFF(gdp, ptoff) {
            UT_Vector3 pos = posHandle.get(ptoff);

            // Example: scale position by strength
            pos *= strength;

            posHandle.set(ptoff, pos);
        }
    }

    // Mark position attribute as modified
    gdp->getP()->bumpDataId();

    return error();
}
```

---

## Node Naming Convention

SideFX recommends a namespace pattern:

```cpp
// Format: namespace::name::version
"studio::mydeformer::1.0"

// Examples from houdini-cuda-deformers:
"cuda::deltamush::1.0"
"cuda::clothsolver::1.0"
"cuda::uvunwrap::1.0"
```

This prevents conflicts with other plugins and built-in nodes.

---

## Input Handling

### Multiple Inputs

```cpp
// In newSopOperator:
table->addOperator(new OP_Operator(
    "hdk_mynode",
    "My Node",
    SOP_MyNode::myConstructor,
    SOP_MyNode::myTemplateList,
    2,    // Min inputs (required)
    4,    // Max inputs (optional inputs)
    nullptr,
    0
));

// Input labels
const char* SOP_MyNode::inputLabel(unsigned idx) const
{
    switch (idx) {
        case 0: return "Geometry";
        case 1: return "Rest Geometry";
        case 2: return "Constraints (optional)";
        case 3: return "Collision Geometry (optional)";
        default: return "Input";
    }
}
```

### Accessing Inputs

```cpp
OP_ERROR SOP_MyNode::cookMySop(OP_Context& context)
{
    OP_AutoLockInputs inputs(this);
    if (inputs.lock(context) >= UT_ERROR_ABORT)
        return error();

    // Required input (index 0)
    const GU_Detail* inputGeo = inputGeo(0);
    if (!inputGeo)
        return error();

    // Optional input (index 2) - may be null
    const GU_Detail* constraintGeo = inputGeo(2);
    bool hasConstraints = (constraintGeo != nullptr);

    // Duplicate first input to output
    duplicateSource(0, context);

    // ...
}
```

---

## Integrating with CUDA Bridge

For GPU-accelerated nodes, use the bridge pattern:

```cpp
// SOP_CudaDeformer.h
#include "CudaBridge.h"

class SOP_CudaDeformer : public SOP_Node
{
    // ... standard declarations ...

private:
    CudaBridge m_bridge;  // GPU interface
};

// SOP_CudaDeformer.cpp
OP_ERROR SOP_CudaDeformer::cookMySop(OP_Context& context)
{
    OP_AutoLockInputs inputs(this);
    if (inputs.lock(context) >= UT_ERROR_ABORT)
        return error();

    fpreal t = context.getTime();
    duplicateSource(0, context);

    // Read parameters
    int iterations = ITERATIONS(t);
    float strength = STRENGTH(t);

    // Check for topology changes
    const GU_Detail* inputDetail = inputGeo(0);
    GA_DataId topoId = inputDetail->getTopology().getDataId();

    if (topoId != m_lastTopoId) {
        // Topology changed - rebuild GPU data
        m_bridge.uploadTopology(gdp);
        m_lastTopoId = topoId;
    }

    // Upload current positions
    m_bridge.uploadPositions(gdp);

    // Run GPU computation
    m_bridge.compute(iterations, strength);

    // Download results
    m_bridge.downloadPositions(gdp);

    gdp->getP()->bumpDataId();
    return error();
}
```

---

## Error Handling

```cpp
OP_ERROR SOP_CudaDeformer::cookMySop(OP_Context& context)
{
    // ... setup ...

    try {
        m_bridge.compute(iterations, strength);
    }
    catch (const std::exception& e) {
        addError(SOP_MESSAGE, e.what());
        return error();
    }

    // Check for GPU errors (NaN, explosion, etc.)
    if (m_bridge.hasError()) {
        addWarning(SOP_MESSAGE, m_bridge.getErrorMessage().c_str());
    }

    return error();
}
```

---

## Operator Flags

Common flags for `OP_Operator`:

| Flag | Description |
|:-----|:------------|
| `OP_FLAG_GENERATOR` | Node generates geometry (no input required) |
| `0` | Standard modifier (requires input) |
| `OP_FLAG_UNORDERED` | Inputs can be connected in any order |

```cpp
// Generator (like Grid SOP)
table->addOperator(new OP_Operator(
    "hdk_mygenerator", "My Generator",
    ...,
    0,    // Min inputs = 0
    0,    // Max inputs = 0
    nullptr,
    OP_FLAG_GENERATOR
));

// Modifier (like Smooth SOP)
table->addOperator(new OP_Operator(
    "hdk_mymodifier", "My Modifier",
    ...,
    1,    // Min inputs = 1
    1,    // Max inputs = 1
    nullptr,
    0     // No special flags
));
```

---

## Best Practices

1. **Always lock inputs** using `OP_AutoLockInputs`
2. **Check for null geometry** before processing
3. **Bump data IDs** after modifying attributes
4. **Use parameter macros** for clean parameter access
5. **Cache topology** to avoid redundant GPU uploads
6. **Handle errors gracefully** with `addError()` and `addWarning()`
