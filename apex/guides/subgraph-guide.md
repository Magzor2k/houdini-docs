---
layout: default
title: Creating APEX Subgraphs
parent: Guides
grand_parent: APEX
nav_order: 3
description: Building, packaging, and deploying reusable APEX subgraphs
permalink: /apex/guides/subgraph-guide/
---

# Creating APEX Subgraphs

Guide to building, packaging, and deploying reusable APEX subgraphs in Houdini 21+.

---

## Overview

A **subgraph** is a locked, reusable APEX graph that behaves like an HDA for APEX networks. Subgraphs let you:

- **Build once, reuse everywhere** -- define logic in one place and use it across multiple rigs and graphs
- **Stay in sync** -- modifying a subgraph updates all instances automatically
- **Share with teams** -- distribute `.bgeo` library files so others can use your nodes

Subgraphs are stored in `.bgeo` geometry files and registered with the APEX Registry. Once registered, they appear in the **Tab menu** inside any APEX network view, just like built-in nodes.

**Common use cases:**
- Visibility toggles
- Custom IK solvers
- Repeated rig patterns (FK chains, control groups)
- Utility operations (math helpers, attribute processors)
- Custom deformers

---

## Subgraph vs Subnet

| | Subgraph | Subnet |
|:--|:---------|:-------|
| **Locked** | Yes -- contents not directly editable | No -- fully editable |
| **Reusable** | Across files and projects | Local to the current graph only |
| **Instances** | All instances stay in sync | Each copy is independent |
| **Analogy** | Like a locked HDA | Like an unlocked subnet |
| **Storage** | `.bgeo` library file on disk | Embedded in the graph |
| **Use for** | Shared, production-ready logic | Local organization |

---

## Three Methods to Create Subgraphs

| Method | Best For | Tool |
|:-------|:---------|:-----|
| [APEX Graph SOP](#method-1-apex-graph-sop-visual) | Visual, interactive creation | Save to Disk button |
| [APEX Script SOP](#method-2-apex-script-sop-programmatic) | Programmatic, code-based creation | `@subgraph` decorator |
| [Manual SOP Pipeline](#method-3-manual-sop-pipeline-pack-folder) | Multiple subgraphs in one library, incremental updates | Pack Folder + ROP Geometry |

---

## Method 1: APEX Graph SOP (Visual)

The simplest approach -- create a subnet visually, then save it as a subgraph.

### Step 1: Create a Subnet

1. Open an **APEX Graph** SOP
2. Build your graph logic (add nodes, wire them)
3. Box-select the nodes you want to package
4. Right-click > **Collapse to Subnet**

### Step 2: Configure the Subnet

1. Dive into the subnet
2. Middle-click port names on the graph input/output node to rename them (only subports can be renamed)
3. Verify inputs and outputs match your intended interface

### Step 3: Save as Subgraph

In the APEX Graph SOP parameters, find the **Subgraphs** section:

| Parameter | Value | Notes |
|:----------|:------|:------|
| Output File | `$HIP/apexgraph/my_subgraphs.bgeo` | Path to save the library |
| Name Space | (optional) | Prefix like `th` creates `th::subgraph_name` |

Click **Save to Disk**.

### Step 4: Reload

```python
import apex
apex.Registry().reloadSubgraphs()
```

Your subgraph now appears in the Tab menu.

---

## Method 2: APEX Script SOP (Programmatic)

Use the `@subgraph` decorator to define subgraphs in code.

### Basic Example

```python
@subgraph('$HIP/apexgraph/my_tools.bgeo', name='my_ns::add_vectors::1.0')
def add_vectors(a: Vector3, b: Vector3):
    c: Vector3 = a + b
    return c
```

### Decorator Arguments

| Argument | Type | Description |
|:---------|:-----|:------------|
| File path | String | Where to save (e.g., `'$HIP/apexgraph/tools.bgeo'`) |
| `name` | String | Registered name (e.g., `'my_ns::my_node::1.0'`) |
| `hidden` | Bool | `True` hides from Tab menu |
| Custom kwargs | Any | Added to the graph's `properties` attribute |

### Workflow

1. In an **APEX Script** SOP, enable the **Subgraphs** parameter
2. Write your function with the `@subgraph` decorator in the **Subgraph** snippet
3. Set the **Geometry File** parameter (or pass the path in the decorator)
4. Click **Save Subgraphs**
5. Reload the registry:

```python
import apex
apex.Registry().reloadSubgraphs()
```

### Using Saved Subgraphs as Functions

After saving, the subgraph is available as a global function in any APEX Script SOP:

```python
# Call by namespace.function_name
result = my_ns.add_vectors(vec_a, vec_b)
```

It also appears as a node in the APEX network Tab menu (search for `add_vectors` or `my_ns`).

---

## Method 3: Manual SOP Pipeline (Pack Folder)

This approach gives full control over library management and is ideal for maintaining multiple subgraphs in a single `.bgeo` file with incremental updates.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              SUBGRAPH AUTHORING PIPELINE                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌───────────────┐      ┌─────────────────────┐            │
│  │    file1      │      │   my_subgraph       │            │
│  │  (File SOP)   │      │ (APEX Edit Graph)   │            │
│  │               │      │                     │            │
│  │ loads existing│      │  Input ──► logic    │            │
│  │ .bgeo library │      │  ──► Output         │            │
│  └───────┬───────┘      └──────────┬──────────┘            │
│          │                         │                        │
│          │      ┌──────────────────▼───────────┐            │
│          └─────►│       packfolder             │            │
│                 │    (Pack Folder SOP)          │            │
│                 │  name: my_ns::my_subgraph    │            │
│                 │  Pack Output: ON              │            │
│                 └──────────────┬────────────────┘            │
│                                │                             │
│                 ┌──────────────▼────────────────┐            │
│                 │      rop_geometry             │            │
│                 │  (ROP Geometry Output)         │            │
│                 │  saves to .bgeo               │            │
│                 └───────────────────────────────┘            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Step 1: Set Up the SOP Network

Create four nodes and wire them in sequence:

1. **File SOP** (`file1`) -- loads the existing subgraph library
2. **APEX Edit Graph** (`my_subgraph`) -- where you build the subgraph logic
3. **Pack Folder SOP** (`packfolder`) -- packages everything into a library
4. **ROP Geometry Output** (`rop_geometry`) -- saves the library to disk

If starting from scratch with no existing library, you can skip the File SOP or leave it disconnected.

### Step 2: Build the Subgraph Logic

Inside the **APEX Edit Graph** node:

1. Add an **Input** node -- this defines the subgraph's input ports
2. Add an **Output** node -- this defines the subgraph's output ports
3. Build the logic between them

The ports you define on the Input/Output nodes become the visible ports when the subgraph is used as a node. For example, a visibility toggle might have:

| Input Ports | Output Ports |
|:------------|:-------------|
| `visible` (Bool) | `out` (Geometry) |
| `geo` (Geometry) | `next` (passthrough) |
| `group` (String) | |
| `next` (passthrough) | |

**Important:** You need Input and Output nodes to define the interface. You do **not** need to manually collapse nodes to a subnet -- the Input/Output nodes handle the subgraph boundary.

### Step 3: Configure the Pack Folder SOP

This is the critical packaging step.

**Main Parameters:**

| Parameter | Value | Notes |
|:----------|:------|:------|
| Parent Folder | `/` | Root of the packed hierarchy |
| Replace Method | `Add and Replace` | Merges new subgraphs with existing ones |
| **Pack Output** | **ON** | Required -- packs the subgraphs into geometry |
| **Only Pack Unpacked** | **ON** | Avoids re-packing already-packed data from file1 |

**Input Operators Table:**

| Row | Name | Input Operator | Purpose |
|:----|:-----|:---------------|:--------|
| 1 | (from file) | `file1` | Existing subgraphs from the library |
| 2 | `my_ns::my_subgraph` | `my_subgraph` | The new subgraph being added |

The **Name** column value becomes the subgraph's registered name. Leave the **Type** column empty for subgraphs.

### Step 4: Save to Disk

Configure the **ROP Geometry Output** node:

- Set the output path to your `apexgraph` directory (e.g., `$HIP/apexgraph/my_library.bgeo`)
- Click **Save to Disk** (or Render)

### Step 5: Reload the Registry

```python
import apex
apex.Registry().reloadSubgraphs()
```

The subgraph now appears in the Tab menu inside APEX Graph nodes.

---

## Subgraph Library Structure

Subgraph libraries use `.bgeo` geometry files in one of two formats:

### Unpacked Format (Single Subgraph)

A single unpacked APEX graph where the subgraph name comes from the `name` **detail** attribute.

**To create:**
1. Build your graph in an APEX Edit Graph node
2. Add an **Attribute Create** SOP after it:
   - Class: Detail
   - Name: `name`
   - Value: your subgraph name (e.g., `my_ns::my_node`)
3. Save with **ROP Geometry Output** to `$HIP/apexgraph/my_node.bgeo`

### Packed Format (Multiple Subgraphs)

Multiple packed geometry primitives, each containing an APEX graph. The subgraph name comes from the `name` **primitive** attribute.

**To create:** Use the [Pack Folder SOP approach](#method-3-manual-sop-pipeline-pack-folder) described above. This is the recommended format for libraries containing multiple subgraphs.

---

## Discovery and Registration

Houdini automatically discovers subgraph `.bgeo` files from `@/apexgraph` directories, where `@` expands to each directory in `HOUDINI_PATH`.

### Search Locations

| Location | Notes |
|:---------|:------|
| `$HOUDINI_USER_PREF_DIR/apexgraph/` | User preferences directory |
| `$HIP/apexgraph/` | Hip file directory (see caveat below) |
| `$HOME/apexgraph/` | Default fallback |
| Package `apexgraph/` directories | Via Houdini packages on `HOUDINI_PATH` |

**Caveat:** `$HIP/apexgraph/` is only discovered if Houdini is launched by opening the `.hip` file directly. Launching Houdini from the Start Menu and then loading the file may not pick it up. Use Houdini packages or `$HOUDINI_USER_PREF_DIR` for reliable discovery.

### Reloading After Changes

After saving or updating a subgraph library, run in the Python Shell:

```python
import apex
apex.Registry().reloadSubgraphs()
```

This forces Houdini to re-scan all `apexgraph` directories and update the registry.

---

## Naming Conventions

Subgraph names follow the format `namespace::name` or `namespace::name::version`:

| Example | Namespace | Name | Version |
|:--------|:----------|:-----|:--------|
| `th::set_visibility` | `th` | `set_visibility` | (none) |
| `my_ns::add_vectors::1.0` | `my_ns` | `add_vectors` | `1.0` |
| `rig::ControlSpline::3.0` | `rig` | `ControlSpline` | `3.0` |
| `skel::SetPointTransforms` | `skel` | `SetPointTransforms` | (none) |

- **Namespace** groups related subgraphs (e.g., `th` for your studio, `rig` for SideFX built-ins)
- **Version** is optional but recommended for production use
- The full name appears in the Tab menu when searching

---

## Portability and Sharing

Subgraphs only work if the `.bgeo` library file exists on the target machine. To share rigs that use custom subgraphs:

1. **Include the `apexgraph/` folder** with your project files
2. **Use Houdini packages** to add custom paths to `HOUDINI_PATH` so the `apexgraph/` directory is discovered automatically
3. **For team workflows**, store subgraph libraries in a shared network location and configure all machines to include that path

Without the library file present, any APEX graph referencing the subgraph will fail to cook.

---

## Tips and Gotchas

- **Always reload after saving** -- subgraphs are cached in the registry. Run `apex.Registry().reloadSubgraphs()` after every save or Houdini won't see the changes.

- **Input/Output nodes are required** (Method 3) -- without them, the subgraph has no interface and no ports will be visible when used as a node.

- **Pack Output must be ON** -- forgetting to enable Pack Output on the Pack Folder SOP is a common mistake. The subgraph won't be packed without it.

- **Names must be unique** -- if two subgraphs share the same registered name, the later one replaces the earlier one silently.

- **Test before saving** -- cook the Pack Folder SOP and inspect its output geometry in the Geometry Spreadsheet to verify the subgraph is correctly packed before writing to disk.

- **Back up your library** -- since the File SOP and ROP Geometry both point to the same `.bgeo` file (Method 3), a bad save can overwrite good data. Keep backups.

- **Port renaming** -- inside subnets, middle-click port names on the graph input/output node to rename them. Only subports can be renamed, not regular ports.

- **`$HIP` discovery caveat** -- Houdini launched from the Start Menu may not find `$HIP/apexgraph/`. Use Houdini packages or `$HOUDINI_USER_PREF_DIR` for reliable discovery.

---

## Python API Reference

```python
import apex

# Reload all subgraph libraries from apexgraph directories
apex.Registry().reloadSubgraphs()

# Get a subgraph by its registered callback name
subgraph = apex.Registry().getSubGraph('my_ns::my_node')

# Check if a hou.ApexNode represents a subgraph
is_sg = node.isSubgraph()  # Returns True for subgraph nodes

# Get the callback name for a node in a graph
name = graph.callbackName()
```

---

## See Also

- [APEX Quick Reference](../reference/quick-reference.md) -- `@subgraph` decorator syntax
- [APEX Patterns](../reference/patterns.md) -- Pack Folder and Scene Animate workflow patterns
- [APEX Functions](../reference/functions.md) -- Graph building API
- [SideFX: APEX Graph Basics](https://www.sidefx.com/docs/houdini/character/kinefx/apexgraphbasics.html) -- Official subgraph documentation
- [SideFX: Functions in APEX Script](https://www.sidefx.com/docs/houdini/character/kinefx/apexscriptfunctions.html) -- `@subgraph` decorator details
- [SideFX: APEX Graph SOP](https://www.sidefx.com/docs/houdini/nodes/sop/apex--graph.html) -- Save to Disk workflow
- [SideFX: Pack Folder SOP](https://www.sidefx.com/docs/houdini/nodes/sop/packfolder.html) -- Pack Folder parameter reference
- [SideFX: APEX Rigging Masterclass](https://www.sidefx.com/tutorials/apex-rigging-masterclass/) -- Video covering components and subgraphs
