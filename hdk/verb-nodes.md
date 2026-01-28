---
layout: default
title: Verb Nodes
parent: HDK
nav_order: 7
description: Modern SOP_NodeVerb pattern for compilable, cacheable geometry nodes
permalink: /hdk/verb-nodes/
---

# Verb Nodes

The SOP_NodeVerb pattern is the modern way to implement SOP nodes in Houdini's HDK. It separates computation logic from the node itself, enabling compiled block support, better caching, and parallel execution.

---

## Overview

A verb-based SOP consists of two classes:

1. **Verb Class** (`SOP_NodeVerb`) - Singleton that performs the actual computation
2. **Node Class** (`SOP_Node`) - Wrapper that delegates cooking to the verb

This separation allows:
- **Compiled SOPs** - Verbs can run inside compiled blocks
- **Efficient caching** - Parameters and cache are separate from the singleton verb
- **Parallel execution** - Multiple node instances can cook simultaneously
- **Dynamic discovery** - Verbs are registered and discoverable by name

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  SOP_Node (your node class)                         │
│  ├── cookVerb() → returns verb singleton            │
│  └── cookMySop() → calls cookMyselfAsVerb()         │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│  SOP_NodeVerb (your verb class - singleton)         │
│  ├── allocParms() → creates parameter object        │
│  ├── cookMode() → COOK_INPLACE, COOK_GENERIC, etc.  │
│  └── cook(CookParms) → performs actual computation  │
└─────────────────────────────────────────────────────┘
```

## Usage

### Verb Class Definition

```cpp
class SOP_MyNodeVerb : public SOP_NodeVerb
{
public:
    SOP_MyNodeVerb() {}
    virtual ~SOP_MyNodeVerb() {}

    // Required: Create parameter object (auto-generated from .ds file)
    virtual SOP_NodeParms *allocParms() const
    {
        return new SOP_MyNodeParms();
    }

    // Required: Return verb identifier
    virtual UT_StringHolder name() const
    {
        return SOP_MyNode::theSOPTypeName;
    }

    // Required: How geometry is processed
    virtual CookMode cookMode(const SOP_NodeParms *parms) const
    {
        return COOK_GENERIC;
    }

    // Required: Main computation function
    virtual void cook(const CookParms &cookparms) const;

    // Auto-registration at library load
    static const SOP_NodeVerb::Register<SOP_MyNodeVerb> theVerb;
};

// Static registration (must be outside class definition)
const SOP_NodeVerb::Register<SOP_MyNodeVerb> SOP_MyNodeVerb::theVerb;
```

### Node Class Definition

```cpp
class SOP_MyNode : public SOP_Node
{
public:
    static PRM_Template *buildTemplates();
    static OP_Node *myConstructor(OP_Network *net, const char *name, OP_Operator *op)
    {
        return new SOP_MyNode(net, name, op);
    }

    static const UT_StringHolder theSOPTypeName;

    // Return the verb singleton
    const SOP_NodeVerb *cookVerb() const override
    {
        return SOP_MyNodeVerb::theVerb.get();
    }

protected:
    SOP_MyNode(OP_Network *net, const char *name, OP_Operator *op)
        : SOP_Node(net, name, op)
    {
        // Critical: All verb SOPs must manage data IDs
        mySopFlags.setManagesDataIDs(true);
    }

    ~SOP_MyNode() override {}

    // Delegate cooking to the verb
    OP_ERROR cookMySop(OP_Context &context) override
    {
        return cookMyselfAsVerb(context);
    }
};
```

### Cook Implementation

```cpp
void SOP_MyNodeVerb::cook(const CookParms &cookparms) const
{
    // Get typed parameters (auto-generated class)
    auto &&sopparms = cookparms.parms<SOP_MyNodeParms>();

    // Get output geometry (writable)
    GU_Detail *detail = cookparms.gdh().gdpNC();

    // Get input geometry (read-only) - for filter SOPs
    const GEO_Detail *input = cookparms.inputGeo(0);

    // Access parameter values
    int divisions = sopparms.getDivs();
    float radius = sopparms.getRadius();

    // Perform geometry operations...
    detail->clearAndDestroy();
    // ... create or modify geometry ...

    // Bump data IDs when adding geometry
    detail->bumpDataIdsForAddOrRemove(true, true, true);

    // Add warnings/errors
    if (divisions < 2)
        cookparms.sopAddWarning(SOP_MESSAGE, "Need at least 2 divisions");
}
```

## Cook Modes

| Mode | Description | Use Case |
|:-----|:------------|:---------|
| `COOK_INPLACE` | Modify input geometry directly | Filter SOPs (e.g., transform, smooth) |
| `COOK_DUPLICATE` | Copy input, then modify | When original must be preserved |
| `COOK_GENERATOR` | Create geometry from scratch | Generator SOPs (e.g., sphere, grid) |
| `COOK_GENERIC` | General-purpose processing | Complex operations |
| `COOK_INSTANCE` | Instance existing geometry | Instancing operations |
| `COOK_PASSTHROUGH` | Pass input through | Minimal modification |

## Parameter Definition

Parameters are defined using DS file format (inline raw string):

```cpp
static const char *theDsFile = R"THEDSFILE(
{
    name        parameters
    parm {
        name    "divs"
        label   "Divisions"
        type    integer
        default { "5" }
        range   { 2! 50 }
        export  all
    }
    parm {
        name    "radius"
        label   "Radius"
        type    float
        default { "1.0" }
        range   { 0! 10 }
    }
    parm {
        name    "center"
        label   "Center"
        type    vector
        size    3
        default { "0" "0" "0" }
    }
    parm {
        name    "plane"
        label   "Orientation"
        type    ordinal
        default { "0" }
        menu    {
            "xy"    "XY Plane"
            "yz"    "YZ Plane"
            "zx"    "ZX Plane"
        }
    }
}
)THEDSFILE";
```

### CMake Proto Header Generation

Add to CMakeLists.txt to auto-generate the parameter class:

```cmake
houdini_generate_proto_headers(FILES SOP_MyNode.cpp)
```

This generates `SOP_MyNode.proto.h` containing `SOP_MyNodeParms` with typed accessors.

## CookParms Reference

| Method | Description |
|:-------|:------------|
| `parms<T>()` | Get typed parameter object |
| `gdh().gdpNC()` | Get writable output geometry |
| `inputGeo(idx)` | Get read-only input geometry |
| `nInputs()` | Number of connected inputs |
| `cache()` | Get per-cook cache object |
| `getNode()` | Get cooking node (null in compiled mode) |
| `getCookTime()` | Current frame/time |
| `sopAddWarning()` | Add warning message |
| `sopAddError()` | Add error message |

## Examples

### Generator SOP (SOP_Star)

```cpp
void SOP_StarVerb::cook(const CookParms &cookparms) const
{
    auto &&sopparms = cookparms.parms<SOP_StarParms>();
    GU_Detail *detail = cookparms.gdh().gdpNC();

    exint npoints = sopparms.getDivs() * 2;

    // Optimize: reuse geometry if point count unchanged
    if (detail->getNumPoints() != npoints)
    {
        detail->clearAndDestroy();

        // Create polygon with vertices
        GA_Offset start_vtxoff;
        detail->appendPrimitivesAndVertices(GA_PRIMPOLY, 1, npoints, start_vtxoff, true);

        // Create points
        GA_Offset start_ptoff = detail->appendPointBlock(npoints);

        // Wire vertices to points
        for (exint i = 0; i < npoints; ++i)
            detail->setVertexPoint(start_vtxoff + i, start_ptoff + i);

        detail->bumpDataIdsForAddOrRemove(true, true, true);
    }
    else
    {
        detail->getP()->bumpDataId();  // Only bumping P
    }

    // Set point positions...
}
```

### Filter SOP (COOK_INPLACE)

```cpp
virtual CookMode cookMode(const SOP_NodeParms *parms) const
{
    return COOK_INPLACE;  // Modify input directly
}

void SOP_SplitPointsVerb::cook(const CookParms &cookparms) const
{
    auto &&sopparms = cookparms.parms<SOP_SplitPointsParms>();
    GU_Detail *gdp = cookparms.gdh().gdpNC();  // Input is already here

    // Parse group
    GOP_Manager gop;
    const GA_ElementGroup *group = nullptr;
    // ... parse group from parameter ...

    // Modify geometry in place
    GEOsplitPoints(gdp, group);
}
```

### With Cache

```cpp
class SOP_MyCache : public SOP_NodeCache
{
public:
    SOP_MyCache() : SOP_NodeCache() {}

    // Store data between cooks
    UT_Array<GA_Offset> myStoredOffsets;
};

class SOP_MyVerb : public SOP_NodeVerb
{
    virtual SOP_NodeCache *allocCache() const
    {
        return new SOP_MyCache();
    }

    virtual void cook(const CookParms &cookparms) const
    {
        auto sopcache = (SOP_MyCache *)cookparms.cache();
        // Use sopcache->myStoredOffsets...
    }
};
```

## Common Issues

### Data IDs Not Managed

**Problem:** Node doesn't work correctly in compiled blocks or with caching.

**Solution:** Always set in constructor:
```cpp
mySopFlags.setManagesDataIDs(true);
```

### Missing Data ID Bumps

**Problem:** Downstream nodes don't update when geometry changes.

**Solution:** Bump data IDs after modifying geometry:
```cpp
// After adding points/prims/vertices
detail->bumpDataIdsForAddOrRemove(true, true, true);

// After modifying only P
detail->getP()->bumpDataId();

// After modifying specific attribute
myAttrib->bumpDataId();
```

### Verb Not Registered

**Problem:** Node not found or crashes on load.

**Solution:** Ensure static registration is defined outside class:
```cpp
const SOP_NodeVerb::Register<SOP_MyVerb> SOP_MyVerb::theVerb;
```

## HDK Sample Locations

Complete verb examples in your Houdini installation:

```
$HFS/toolkit/samples/SOP/
├── SOP_Star/           # Simple generator
├── SOP_SplitPoints/    # COOK_INPLACE filter
├── SOP_CopyPacked/     # Complex with caching
├── SOP_CopyToPoints/   # Multi-input
├── SOP_Sweep/          # Sweep operation
└── SOP_OrientAlongCurve/
```

## See Also

- [SOP Node Basics](sop-node.md) - Traditional SOP_Node pattern
- [Parameters](parameters.md) - Parameter definition details
- [CMake Setup](cmake-setup.md) - Build configuration
- [SOP_NodeVerb Reference](https://www.sidefx.com/docs/hdk/class_s_o_p___node_verb.html) - Official HDK docs
- [CookParms Reference](https://www.sidefx.com/docs/hdk/class_s_o_p___node_verb_1_1_cook_parms.html) - CookParms API
