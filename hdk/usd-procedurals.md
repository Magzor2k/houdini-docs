---
layout: default
title: USD Procedurals
parent: HDK
nav_order: 9
description: Creating custom render-time procedurals for Houdini's Solaris/USD pipeline
permalink: /hdk/usd-procedurals/
---

# USD Procedurals

Creating custom render-time geometry generation for Houdini's Solaris/USD pipeline.

---

## Overview

Houdini has **two distinct procedural systems** for render-time geometry generation:

1. **HoudiniProceduralAPI** - SideFX's renderer-agnostic system using Python + SOP verbs
2. **BRAY_Procedural** - Karma-specific C++ API for custom geometry/intersection

Most built-in procedurals (Hair, Ocean, RBD, Crowd, Feather) use HoudiniProceduralAPI. The BRAY system is for advanced C++ development.

### Procedural System Comparison

| Aspect | HoudiniProceduralAPI | BRAY_Procedural |
|:-------|:---------------------|:----------------|
| **Language** | Python + SOP verbs | C++ |
| **Renderers** | All Hydra delegates | Karma CPU only |
| **Use Case** | SOP network execution at render time | Custom geometry/ray intersection |
| **Built-in Examples** | Hair, Ocean, RBD, Crowd, Feather | sphere, file, vdb |
| **Complexity** | Medium (Python scripting) | High (HDK development) |

### Renderer Compatibility

| Renderer | HoudiniProceduralAPI | BRAY Procedurals |
|:---------|:---------------------|:-----------------|
| **Karma CPU** | Yes | Yes |
| **Karma XPU** | Yes | No |
| **Houdini VK (Storm)** | Preview only | No |
| **Other Hydra Delegates** | Yes | No |

---

## HoudiniProceduralAPI System

This is SideFX's custom USD schema that enables **SOP networks to execute at render time** via Python scripts. It's the system behind all the built-in Houdini Procedural LOPs.

### Architecture

```
                    At Author Time (LOPs)
┌─────────────────────────────────────────────────────────────────┐
│  1. SOP Network (e.g., Ocean, Hair, RBD setup)                  │
│            ↓                                                     │
│  2. Attribute from Parameters SOP → Serialize to .bgeo          │
│            ↓                                                     │
│  3. LOP Procedural Node → Create USD prim with:                 │
│       - HoudiniProceduralAPI schema applied                     │
│       - path = "@invokegraph.py@"                               │
│       - args = { "graph": "/path/to/network.bgeo", ... }        │
└─────────────────────────────────────────────────────────────────┘

                    At Render Time (husk)
┌─────────────────────────────────────────────────────────────────┐
│  4. husk finds prims with HoudiniProceduralAPI                  │
│            ↓                                                     │
│  5. Executes Python script (invokegraph.py) with hou API        │
│            ↓                                                     │
│  6. Script loads .bgeo, runs invokegraph verb                   │
│            ↓                                                     │
│  7. Returns geometry → merged into render stage as sublayer     │
└─────────────────────────────────────────────────────────────────┘
```

### USD Schema Attributes

When `HoudiniProceduralAPI` is applied to a prim, it gains these attributes:

| Attribute | Purpose |
|:----------|:--------|
| `houdiniProcedural:NAME:houdini:procedural:path` | Path to Python script (e.g., `@invokegraph.py@`) |
| `houdiniProcedural:NAME:houdini:procedural:args` | Dictionary of arguments |
| `houdiniProcedural:NAME:houdini:procedural:type` | Procedural type identifier |
| `houdiniProcedural:NAME:houdini:active` | Enable/disable |
| `houdiniProcedural:NAME:houdini:animated` | Whether time-varying |
| `houdiniProcedural:NAME:houdini:priority` | Execution order |

### Key Components

#### invokegraph.py

Located at `$HFS/houdini/husdplugins/houdiniprocedurals/invokegraph.py`

This is the core execution engine. When husk encounters a prim with HoudiniProceduralAPI:

1. Loads a `.bgeo` file containing a serialized SOP network as geometry
2. Uses the `invokegraph` SOP verb to execute the network
3. Returns the resulting geometry to the renderer

```python
def procedural(prim, args):
    # Load the compiled SOP graph from .bgeo file
    graph = hou.Geometry()
    graph_path = hou.text.expandString(args['graph'])
    graph.loadFromFile(graph_path)

    # Import USD inputs as geometry
    for input in args['inputs']:
        geo = hou.Geometry()
        geo.importUsdStage(stage, rule, purpose='guide default render')
        geos.append(geo)

    # Execute the graph using the invokegraph verb
    invoke = verbs['invokegraph']
    invoke.execute(result, geos)

    return result
```

#### Attribute from Parameters SOP

Converts a SOP network into geometry:
- **Nodes → Points** with parameter dictionaries as attributes
- **Connections → Polylines** between points
- **Result → .bgeo file** that can be executed later

#### Invoke Graph SOP

Executes **SOP verbs** (not actual nodes) from geometry data. This is key because:
- Verbs are compiled, stateless execution units
- They run without creating node instances
- They work within husk's Python environment

### Built-in Procedurals

| Procedural | Implementation | Use Case |
|:-----------|:---------------|:---------|
| **Hair** | invokegraph.py | Generate/deform hair curves |
| **Ocean** | invokegraph.py | Render-time ocean displacement |
| **RBD** | invokegraph.py | Fracture piece transforms |
| **Feather** | invokegraph.py | Feather generation |
| **Crowd** | houdinicrowdprocedural.py | Agent instancing optimization |

### Creating Custom HoudiniProceduralAPI Procedurals

See [Hacking Houdini Solaris Procedurals](https://www.marcelruegenberg.com/blog/2025/01/14/hacking-houdini-solaris-procedurals) for a detailed guide.

#### Step 1: Create a Compile Block Network

Use `compile_begin` and `compile_end` nodes to wrap your SOP network:

```python
# Create compilable SOP network
geo = obj.createNode("geo", "procedural_builder")

# Start of compilable block
compile_begin = geo.createNode("compile_begin", "input")

# Your geometry operations
grid = geo.createNode("grid", "point_grid")
wrangle = geo.createNode("attribwrangle", "modify")
wrangle.setInput(0, grid)

# End of compilable block
compile_end = geo.createNode("compile_end", "output")
compile_end.setInput(0, wrangle)
```

#### Step 2: Serialize to .bgeo with Attribute from Parameters

```python
# Convert network to geometry graph
attrib_from_parm = geo.createNode("attribfromparm", "network_to_graph")
attrib_from_parm.parm("method").set(2)  # "Points from Compiled Block"
attrib_from_parm.parm("nodepath").set(compile_end.path())

# Save the graph
attrib_from_parm.cook(force=True)
graph_geo = attrib_from_parm.geometry()
graph_geo.saveToFile("/path/to/graph.bgeo")
```

#### Step 3: Create USD with HoudiniProceduralAPI

The API is **multi-apply** with an instance name:

```python
from pxr import Usd, UsdGeom, Sdf

# Create prim and apply the API
prim = stage.DefinePrim("/World/my_procedural", "Mesh")
instance_name = "my_instance"

# Apply multi-apply API schema
prim.GetPrim().ApplyAPI("HoudiniProceduralAPI", instance_name)

# Set procedural path (note the instance name in attribute path)
path_attr = prim.CreateAttribute(
    f"houdiniProcedural:{instance_name}:houdini:procedural:path",
    Sdf.ValueTypeNames.Asset
)
path_attr.Set("@invokegraph.py@")

# Set args as string (Python dict format)
args_attr = prim.CreateAttribute(
    f"houdiniProcedural:{instance_name}:houdini:procedural:args",
    Sdf.ValueTypeNames.String
)
args_attr.Set("{'graph': '/path/to/graph.bgeo'}")

# Set animated flag
anim_attr = prim.CreateAttribute(
    f"houdiniProcedural:{instance_name}:houdini:animated",
    Sdf.ValueTypeNames.Bool
)
anim_attr.Set(False)
```

#### Step 4: Test with Preview Houdini Procedurals LOP

Use the **Houdini Preview Procedurals** LOP to test in the viewport before rendering:

```python
preview = stage.createNode("houdinipreviewprocedurals", "preview")
preview.setInput(0, python_lop)
preview.setDisplayFlag(True)
```

#### Resulting USD Structure

When exported, the USD looks like:

```usda
def Mesh "my_procedural" (
    prepend apiSchemas = ["HoudiniProceduralAPI:my_instance"]
)
{
    uniform bool houdiniProcedural:my_instance:houdini:animated = 0
    string houdiniProcedural:my_instance:houdini:procedural:args = "{'graph': '/path/to/graph.bgeo'}"
    asset houdiniProcedural:my_instance:houdini:procedural:path = @invokegraph.py@
}
```

#### Working Example

This repository includes a working example in `apex_procedural/`:

| File | Purpose |
|:-----|:--------|
| `sphere_grid_graph.bgeo` | Serialized SOP network (compile block) |
| `procedural_example.hip` | Scene showing SOP creation |
| `procedural_lop_test.hip` | LOP scene with HoudiniProceduralAPI |
| `procedural_test.usda` | Exported USD for rendering |

To create your own:
```bash
hython scripts/create_procedural_example.py
hython scripts/create_procedural_lop_test.py
```

---

## BRAY_Procedural System (C++)

For advanced users who need custom ray intersection or want Karma-specific optimizations. This is a **Karma CPU only** system.

### Key Difference: BRAY vs HdGp

| Aspect | HdGpGenerativeProcedural (Standard USD) | BRAY_Procedural (Karma) |
|--------|----------------------------------------|------------------------|
| Header | `pxr/imaging/hdGp/generativeProcedural.h` | `BRAY/BRAY_Procedural.h` |
| Registration | `TF_REGISTRY_FUNCTION(TfType)` | `BRAYregisterProcedural()` |
| USD Prim | `GenerativeProcedural` | `Points` with `primvars:karma_procedural` |
| Invocation | Hydra scene index processing | Karma render delegate directly |
| Output DSO | USD plugin path | `$HOUDINI_DSO_PATH/karma/` |

**Important:** Karma does NOT use `HdGpGenerativeProcedural` even though the headers exist in Houdini's USD libraries.

## BRAY API Headers

Located in `$HFS/toolkit/include/BRAY/`:

| Header | Purpose |
|:-------|:--------|
| `BRAY_Procedural.h` | Base class for ray intersection procedurals |
| `BRAY_ProceduralScene.h` | Simplified base for geometry generation |
| `BRAY_ProceduralFactory.h` | Factory pattern for procedural registration |
| `BRAY_AttribList.h` | Parameter definition structures |
| `BRAY_Interface.h` | Scene and object creation API |

## Implementation Pattern

### 1. Factory Class

The factory registers your procedural and defines its parameters:

```cpp
#include <BRAY/BRAY_ProceduralFactory.h>
#include <BRAY/BRAY_AttribList.h>

class MyProceduralFactory : public BRAY_ProceduralFactory {
public:
    MyProceduralFactory()
        : BRAY_ProceduralFactory("my_procedural"_sh)  // Token name
    {}

    BRAY_Procedural* create() const override;
    const BRAY_AttribList* paramList() const override;
};
```

### 2. Procedural Scene Class

Derive from `BRAY_ProceduralScene` for geometry generation:

```cpp
#include <BRAY/BRAY_ProceduralScene.h>
#include <GT/GT_PrimitiveBuilder.h>

class MyProcedural : public BRAY_ProceduralScene {
public:
    bool updateScene() override {
        // Reset and get scene
        this->reset();
        BRAY::ScenePtr scene = this->scene();

        // Create geometry using GT_PrimitiveBuilder
        UT_BoundingBox bbox(-1, -1, -1, 1, 1, 1);
        GT_BuilderStatus err;
        auto prim = GT_PrimitiveBuilder::box(err, bbox);

        // Convert to BRAY object
        BRAY::ObjectPtr obj = scene.createGeometry(prim);
        obj.setMaterial(scene,
                       scene.createMaterial("default"_sh),
                       scene.objectProperties());

        // Instance the object
        UT_Matrix4D xform;
        xform.identity();
        UT_Array<BRAY::SpacePtr> xforms;
        xforms.append(BRAY::SpacePtr(xform));

        BRAY::ObjectPtr inst = scene.createInstance(obj, "my_inst"_sh);
        inst.setInstanceTransforms(scene, xforms);
        scene.updateObject(inst, BRAY_EVENT_NEW);

        return true;
    }

    void doSetParameter(const UT_StringRef& key,
                        const fpreal64* values, int n) override {
        if (key == "mypos" && n == 3) {
            myPos = UT_Vector3D(values[0], values[1], values[2]);
        }
    }

    // Implement all doSetParameter overloads...
    void doSetParameter(const UT_StringRef&, const int32*, int) override {}
    void doSetParameter(const UT_StringRef&, const int64*, int) override {}
    void doSetParameter(const UT_StringRef&, const fpreal32*, int) override {}
    void doSetParameter(const UT_StringRef&, const UT_StringHolder*, int) override {}

    void update(BRAY_EventType event) override {}

private:
    UT_Vector3D myPos;
};
```

### 3. Registration

Export the `BRAYregisterProcedural()` function:

```cpp
static MyProceduralFactory& getFactory() {
    static MyProceduralFactory theFactory;
    return theFactory;
}

void BRAYregisterProcedural() {
    getFactory();
}
```

### 4. Parameter List

Define parameters that can be set from USD attributes:

```cpp
const BRAY_AttribList* MyProceduralFactory::paramList() const {
    static BRAY_AttribList::Attrib parms[] = {
        {
            "mypos"_sh,                      // Parameter name
            3,                                // Component count
            BRAY_AttribList::ATTRIB_POINT,   // Attribute owner
            GA_STORE_REAL64,                  // Storage type
            false                             // Is array
        }
    };
    static BRAY_AttribList list(parms, SYSarraySize(parms), false);
    return &list;
}
```

## USD Invocation

To invoke a BRAY procedural from USD, create a `Points` prim with the `primvars:karma_procedural` attribute.

### Using Inline USD Node (Recommended)

In Houdini LOPs, use an **Inline USD** node with USDA content:

```usda
#usda 1.0

def Points "proc_point" (
    kind = "component"
)
{
    point3f[] points = [(0, 0, 0)]
    string primvars:karma_procedural = "cube_grid"

    # Optional parameters as primvars
    float primvars:cubesize = 1.0
    int primvars:gridsize = 3
    float primvars:spacing = 2.0
}
```

### Using Python

```python
# In a LOP Python node or script
from pxr import Sdf, UsdGeom

stage = node.editableStage()
points = UsdGeom.Points.Define(stage, "/MyProcedural/points")
points.CreatePointsAttr([(0, 0, 0)])

# Set the procedural type - must match factory token
points.GetPrim().CreateAttribute(
    "primvars:karma_procedural",
    Sdf.ValueTypeNames.TokenArray
).Set(["my_procedural"])

# Set custom parameters as primvars
points.GetPrim().CreateAttribute(
    "primvars:mypos",
    Sdf.ValueTypeNames.Double3Array
).Set([(1.0, 2.0, 3.0)])
```

## Built-in Procedural Types

| Token | Description |
|:------|:------------|
| `sphere` | Render spheres at point locations |
| `file` | Load geometry from file at each point |
| `vdb` | Generate VDB iso-surface geometry |

## CMake Configuration

**Critical:** BRAY procedural DSOs must be in a `karma` subdirectory, and `libBRAY.lib` must be linked explicitly:

```cmake
cmake_minimum_required(VERSION 3.18)
project(KarmaProcedurals LANGUAGES CXX)

# Find Houdini
if(DEFINED HFS)
elseif(DEFINED ENV{HFS})
    set(HFS $ENV{HFS})
else()
    message(FATAL_ERROR "HFS not set.")
endif()

list(APPEND CMAKE_PREFIX_PATH "${HFS}/toolkit/cmake")
find_package(Houdini REQUIRED)

# Create the procedural library
add_library(BRAY_MyProcedural SHARED src/BRAY_MyProcedural.cpp)
houdini_configure_target(BRAY_MyProcedural)

# CRITICAL: Link libBRAY explicitly - not part of default Houdini target
target_link_libraries(BRAY_MyProcedural PRIVATE
    Houdini
    "${HFS}/custom/houdini/dsolib/libBRAY.lib"
)

# CRITICAL: Output to karma subdirectory
set(PACKAGE_DSO_KARMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../package/dso/karma")
set_target_properties(BRAY_MyProcedural PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PACKAGE_DSO_KARMA_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${PACKAGE_DSO_KARMA_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${PACKAGE_DSO_KARMA_DIR}"
)
```

## Examples

### This Repository: Cube Grid Procedural

A working example that generates a 3D grid of cubes at render time:

```
karma_procedural/
├── src/
│   ├── BRAY_CubeGrid.h          # Factory class
│   └── BRAY_CubeGrid.cpp        # Procedural implementation
├── CMakeLists.txt               # Build configuration
└── test_cube_grid.hip           # Test scene
```

**Parameters:**
- `cubesize` (float) - Size of each cube (default: 1.0)
- `gridsize` (int) - Grid dimension NxNxN (default: 3)
- `spacing` (float) - Distance between cubes (default: 2.0)

**Build:**
```bash
cd karma_procedural
cmake -B build -G "Visual Studio 17 2022"
cmake --build build --config Release
```

**Output:** `package/dso/karma/BRAY_CubeGrid.dll`

### HDK Sample

SideFX provides examples in:
```
$HFS/toolkit/samples/karma_procedurals/
├── BRAY_HdBox.C      # ProceduralScene example (geometry generation)
├── BRAY_HdSphere.C   # Full Procedural example (custom intersection)
└── README            # Documentation
```

The `BRAY_HdBox` sample demonstrates:
- Factory pattern implementation
- Parameter handling
- Geometry creation with `GT_PrimitiveBuilder`
- Scene instancing

The `BRAY_HdSphere` sample demonstrates:
- Custom ray intersection (`intersect()`)
- Attribute evaluation (`attribVal()`)
- Bounding box computation (`bounds()`)

## Common Issues

### Procedural Not Found

**Problem:** Karma doesn't recognize your procedural token.

**Solution:**
1. Verify DSO is in `$HOUDINI_DSO_PATH`
2. Check `BRAYregisterProcedural()` is exported
3. Verify token name matches exactly (case-sensitive)

### HdGpGenerativeProcedural Not Working

**Problem:** Standard USD `GenerativeProcedural` prims don't render in Karma.

**Solution:** Karma uses BRAY_Procedural, not HdGpGenerativeProcedural. Convert to using `primvars:karma_procedural` on Points prims instead.

### Parameters Not Received

**Problem:** `doSetParameter()` not being called.

**Solution:**
1. Ensure parameter is defined in `paramList()`
2. Check USD attribute name matches (with `primvars:` prefix)
3. Verify data type matches (float vs double, etc.)

## See Also

### Internal Docs
- [SOP Nodes](sop-node.md) - Creating custom SOP geometry nodes
- [DSO Packaging](dso-packaging.md) - Output and distribution setup
- [CMake Setup](cmake-setup.md) - Build configuration basics

### External Resources
- [Hacking Houdini Solaris Procedurals](https://www.marcelruegenberg.com/blog/2025/01/14/hacking-houdini-solaris-procedurals) - Excellent deep-dive into HoudiniProceduralAPI
- [Karma Procedural LOP](https://www.sidefx.com/docs/houdini/nodes/lop/karmaprocedural.html) - Official SideFX docs
- [Houdini Procedural: Hair](https://www.sidefx.com/docs/houdini/nodes/lop/houdinihairprocedural.html) - Hair procedural reference
- [Houdini Procedural: Ocean](https://www.sidefx.com/docs/houdini/solaris/houdini_ocean_procedural.html) - Ocean procedural reference
- [Houdini Procedural: RBD](https://www.sidefx.com/docs/houdini/solaris/houdini_rbd_procedural.html) - RBD procedural reference
- [Houdini Procedural: Crowd](https://www.sidefx.com/docs/houdini/solaris/houdini_crowd_procedural.html) - Crowd procedural reference
- [Preview Houdini Procedurals LOP](https://www.sidefx.com/docs/houdini/nodes/lop/houdinipreviewprocedurals.html) - Interactive preview
- [HDK Karma Procedurals Sample](https://www.sidefx.com/docs/hdk/karma_procedurals_2_b_r_a_y__hd_box_8_c-example.html) - BRAY C++ example
- [SideFX Forum: Husk Procedural Hair](https://www.sidefx.com/forum/topic/85494/) - Forum discussion with SideFX insights
