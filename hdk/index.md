---
layout: default
title: HDK
nav_order: 5
has_children: true
description: Houdini Development Kit documentation
permalink: /hdk/
---

# Houdini HDK Guide
{: .fs-9 }

Building custom Houdini plugins with the HDK (Houdini Development Kit) and CMake.
{: .fs-6 .fw-300 }

---

## What is the HDK?

The **Houdini Development Kit (HDK)** is a C++ SDK that lets you create custom:

- **SOP nodes** - Custom geometry operations
- **DOP nodes** - Custom simulation components
- **VOP nodes** - Custom shader/VEX nodes
- **APEX nodes** - Custom APEX graph operations
- **And more** - ROPs, SOHOs, custom panels, etc.

Plugins are compiled into **DSO files** (Dynamic Shared Objects) that Houdini loads at startup.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                   Your Plugin (.dll)                     │
│  ┌─────────────────────────────────────────────────┐    │
│  │              SOP Node (C++)                      │    │
│  │  ┌─────────────┐  ┌─────────────┐               │    │
│  │  │ Parameters  │  │ cookMySop() │               │    │
│  │  │ PRM_Template│  │  Main logic │               │    │
│  │  └─────────────┘  └─────────────┘               │    │
│  └─────────────────────────────────────────────────┘    │
│                          │                               │
│  ┌─────────────────────────────────────────────────┐    │
│  │         Optional: CUDA Library (.lib)            │    │
│  │  ┌─────────────┐  ┌─────────────┐               │    │
│  │  │   Kernels   │  │   Bridge    │               │    │
│  │  │  (GPU code) │  │  (transfer) │               │    │
│  │  └─────────────┘  └─────────────┘               │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

## Project Structure

A typical HDK plugin project:

```
my_hdk_plugin/
├── CMakeLists.txt           # Build configuration
├── build.bat                # Windows build script
├── cmake/
│   └── HoudiniCommon.cmake  # Shared CMake helpers
├── src/
│   ├── SOP_MyNode.cpp       # Node implementation
│   ├── SOP_MyNode.h         # Node header
│   ├── MyBridge.cpp         # Optional: GPU bridge
│   ├── MyBridge.h
│   └── cuda/                # Optional: CUDA kernels
│       ├── MyKernels.cu
│       └── MyKernels.cuh
└── package/
    ├── dso/                 # Output DSO files
    └── otls/                # Optional: HDA files
```

## Key Components

| Component | Purpose |
|:----------|:--------|
| **SOP_Node** | Base class for geometry operators |
| **PRM_Template** | Define parameters (UI controls) |
| **GU_Detail** | Geometry container (points, prims, attribs) |
| **GA_Attribute** | Point/vertex/primitive attributes |
| **OP_Operator** | Node registration and metadata |

## Quick Start Example

Minimal SOP node that copies input to output:

```cpp
// SOP_MyNode.h
#include <SOP/SOP_Node.h>

class SOP_MyNode : public SOP_Node {
public:
    static OP_Node* myConstructor(OP_Network* net,
                                   const char* name,
                                   OP_Operator* op);
    static PRM_Template myTemplateList[];

protected:
    SOP_MyNode(OP_Network* net, const char* name, OP_Operator* op);
    OP_ERROR cookMySop(OP_Context& context) override;
};
```

```cpp
// SOP_MyNode.cpp
#include "SOP_MyNode.h"
#include <OP/OP_OperatorTable.h>

PRM_Template SOP_MyNode::myTemplateList[] = {
    PRM_Template()  // Null terminator
};

OP_Node* SOP_MyNode::myConstructor(OP_Network* net,
                                    const char* name,
                                    OP_Operator* op) {
    return new SOP_MyNode(net, name, op);
}

SOP_MyNode::SOP_MyNode(OP_Network* net, const char* name, OP_Operator* op)
    : SOP_Node(net, name, op) {}

OP_ERROR SOP_MyNode::cookMySop(OP_Context& context) {
    if (lockInputs(context) >= UT_ERROR_ABORT)
        return error();

    // Copy input to output
    duplicateSource(0, context);

    unlockInputs();
    return error();
}

// Registration
void newSopOperator(OP_OperatorTable* table) {
    table->addOperator(new OP_Operator(
        "myns::mynode::1.0",      // Internal name
        "My Node",                 // UI name
        SOP_MyNode::myConstructor,
        SOP_MyNode::myTemplateList,
        1, 1                       // Min/max inputs
    ));
}
```

## Documentation Sections

| Section | Description |
|:--------|:------------|
| [CMake Setup](cmake-setup.html) | CMakeLists.txt configuration for HDK |
| [SOP Nodes](sop-node.html) | Creating custom SOP nodes |
| [Parameters](parameters.html) | PRM_Template parameter definitions |
| [DSO Packaging](dso-packaging.html) | Output configuration and distribution |
| [Build Scripts](build-scripts.html) | Automated build with build.bat |
| [Verb Nodes](verb-nodes.html) | Modern SOP_NodeVerb pattern for compilable, cacheable geometry nodes |
| [Houdini Package](houdini-package.html) | Package folder structure for deploying HDAs, DSOs, and Python scripts |
| [USD Procedurals](usd-procedurals.html) | Render-time procedurals (HoudiniProceduralAPI and BRAY) |
| [Button Callbacks](button-callbacks.html) | Implementing button parameter callbacks in verb-based SOPs with DS files |

## Prerequisites

- **Houdini** with HDK installed (included with Houdini)
- **CMake 3.18+** - Build system generator
- **Visual Studio 2022** - Windows compiler (or GCC on Linux)
- **Optional: CUDA Toolkit** - For GPU-accelerated plugins

## Environment

This documentation is based on:
- **Houdini 21.0.559**
- **CMake 3.18+**
- **Visual Studio 2022**

## Key Paths

| Path | Description |
|:-----|:------------|
| `$HFS` | Houdini install directory |
| `$HFS/toolkit/cmake` | Houdini CMake modules |
| `$HFS/toolkit/include` | HDK headers |
| `$HOUDINI_USER_PREF_DIR/dso` | User DSO directory |
