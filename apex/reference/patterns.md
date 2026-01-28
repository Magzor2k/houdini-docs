---
layout: default
title: Patterns
parent: Reference
grand_parent: APEX
nav_order: 4
description: APEX Script patterns and best practices
permalink: /apex/reference/patterns/
---

# APEX Script Patterns

> Houdini 21.0.559 | Last updated: 2025-12-26

## Graph Output Pattern

Every APEX Script that builds a graph must end with:

```
graph.sort(True)
geo = graph.saveToGeometry()
BindOutput(geo)
```

**Required SOP Settings:**

| Parameter | Value |
|:----------|:------|
| `headertemplate` | 1 (Graph) |
| `invocation` | 1 |
| `bindoutputgeo` | 1 |
| `apexgeooutput` | `output:geo` |

**Note:** In Houdini 20.5, use `graph.writeToGeo()` instead of `graph.saveToGeometry()`

---

## Port Naming Convention

**IMPORTANT**: When accessing ports in APEX Script, use the `_in` or `_out` suffix:

| Port Type | Suffix | Example |
|:----------|:-------|:--------|
| Input ports | `_in` | `node.geo_in`, `node.r_in`, `node.parent_in` |
| Output ports | `_out` | `node.geo_out`, `node.xform_out` |

**Exception**: `Value<T>` nodes use `.parm` and `.value` (no suffix):
```
value_node = graph.addNode('val', 'Value<Geometry>')
value_node.parm.promoteInput('input_geo')   # parm is INPUT
value_node.value  # value is OUTPUT (use in addWire)
```

---

## Graph Import/Export Pattern

APEX graphs must be converted to/from geometry to pass between SOP nodes.

### Loading an Existing Graph

```
geo: Geometry = BindInput()
graph = geo.graph.loadFromGeometry()
```

### Saving a Modified Graph

```
graph.sort(True)
out = graph.saveToGeometry()
BindOutput(result=out)  # CRITICAL: Use keyword argument!
```

### SOP Configuration for Graph Import

```python
apex_node.parm("headertemplate").set(0)                # BASIC template (not Graph!)
apex_node.parm("invocation").set(1)
apex_node.parm("inputbindings1").set(1)
apex_node.parm("_bindasgeoinput1").set(1)
apex_node.parm("_apexgeoparm1").set("geo")             # Must match variable name
apex_node.parm("bindoutputgeo").set(1)
apex_node.parm("apexgeooutput").set("output:result")   # Must match BindOutput keyword
```

---

## FK Rig Pattern

### Dynamic Build-Time Skeleton Reading

Read skeleton geometry at build time and bake transforms into the rig graph:

```
skel_geo: Geometry = BindInput()

def _trObj_from_skel(graph: ApexGraphHandle, name: String, skel: Geometry):
    restlocal: Matrix4 = skel.getPointLocalTransform(name)
    trobj: ApexNodeID = graph.addNode(name, 'TransformObject', parms={'restlocal': restlocal})
    return graph, trobj

skel_geo = apex.skel.sort(skel_geo)
graph = ApexGraphHandle()
parms = graph.addNode('parms', '__parms__')
output = graph.addNode('output', '__output__')

for point in skel_geo.points():
    name: String = skel_geo.pointAttribValue(point, 'name')
    graph, trobj = _trObj_from_skel(graph, name, skel_geo)

    # Wire FK hierarchy
    parent_point = skel_geo.getParent(point)
    if parent_point != -1:
        parent_name: String = skel_geo.pointAttribValue(parent_point, 'name')
        parent_node = graph.findNode(parent_name)
        parent_node.xform_out.connect(trobj.parent_in)
        parent_node.localxform_out.connect(trobj.parentlocal_in)

    # Promote controls
    trobj.t_in.promoteInput(f'{name}_t')
    trobj.r_in.promoteInput(f'{name}_r')
    trobj.s_in.promoteInput(f'{name}_s')
```

**Required SOP Settings:**
```python
apex_node.parm("headertemplate").set(1)
apex_node.parm("invocation").set(1)
apex_node.parm("inputbindings1").set(1)
apex_node.parm("_bindasgeoinput1").set(1)
apex_node.parm("_apexgeoparm1").set("skel_geo")  # Must match variable name!
apex_node.parm("bindoutputgeo").set(1)
apex_node.parm("apexgeooutput").set("output:geo")
```

### TransformObject Ports

| Port | Access Name | Direction | Type | Purpose |
|:-----|:------------|:----------|:-----|:--------|
| `t` | `t_in` | Input | Vector3 | Translation offset |
| `r` | `r_in` | Input | Vector3 | Euler rotation (degrees) |
| `s` | `s_in` | Input | Vector3 | Scale |
| `restlocal` | `restlocal_in` | Input | Matrix4 | Rest position/orientation |
| `parent` | `parent_in` | Input | Matrix4 | Parent's world transform |
| `parentlocal` | `parentlocal_in` | Input | Matrix4 | Parent's local transform |
| `xform` | `xform_out` | Output | Matrix4 | World transform |
| `localxform` | `localxform_out` | Output | Matrix4 | Local transform |

### FK Hierarchy Wiring

```
graph.addWire(parent.xform_out, child.parent_in)
graph.addWire(parent.localxform_out, child.parentlocal_in)
```

---

## Variadic Port Pattern (skel::SetPointTransforms)

For variadic ports, use `findAndConnectInput`:

```
spt = graph.addNode('spt', 'skel::SetPointTransforms')
skel_input.value.connect(spt.geo_in)

# Connect each joint's world transform
for point in skel_geo.points():
    name: String = skel_geo.pointAttribValue(point, 'name')
    node: ApexNodeID = xform_nodes[name]
    graph.findAndConnectInput(srcnode=node, srcname='xform', dstnode=spt, dstname='transforms', dstalias=name)

spt.geo_out.promoteOutput('Base.skel')
```

### Alternative: Using getSubPort (cleaner syntax)

```
node = graph.addOrUpdateNode(callback='rig::SampleSplineTransforms::3.0')
names: StringArray = ['one', 'two', 'three']
for name in names:
    valnode = graph.addOrUpdateNode(callback='Value<Matrix4>', name=f'{name}_xform')
    subport = node.xforms_out.getSubPort(name)
    subport.connect(valnode.parm_in)
```

### Alternative: Using findOrAddPort

```
out_port: ApexPortID = graph.findOrAddPort(node, f'{dstname}[{new_port_name}]')
out_port.connect(out_node.portName_in)
```

---

## Bone Deform Pattern

### sop::bonedeform Port Names

| Port | Wire To | Type | Description |
|:-----|:--------|:-----|:------------|
| `geo0` | `geo0_in` | Geometry | Rest mesh (with boneCapture) |
| `geo1` | `geo1_in` | Geometry | Capture skeleton (rest) |
| `geo2` | `geo2_in` | Geometry | Animated skeleton |
| `geo0` | `geo0_out` | Geometry | Deformed mesh (output) |

### Adding Bone Deform to Existing Rig

```
# Load existing rig graph
geo: Geometry = BindInput()
graph = geo.graph.loadFromGeometry()

# Add mesh input
shp_input = graph.addNode('shp_input', 'Value<Geometry>')
shp_input.parm.promoteInput('Base.shp')

# Add bone deform
bone_deform = graph.addNode('bone_deform', 'sop::bonedeform')

# Wire connections (Value<T> nodes use .value, not .value_out)
skel_input_nodes = graph.matchNodes('skel_input')
spt_nodes = graph.matchNodes('spt')

graph.addWire(shp_input.value, bone_deform.geo0_in)
for node in skel_input_nodes:
    graph.addWire(node.value, bone_deform.geo1_in)
for node in spt_nodes:
    graph.addWire(node.geo_out, bone_deform.geo2_in)

bone_deform.geo0_out.promoteOutput('Base.shp')

graph.sort(True)
out = graph.saveToGeometry()
BindOutput(result=out)
```

---

## Scene Animate Workflow

**Important**: Do NOT use `apex::invokegraph` with Scene Animate. Use the full scene workflow:

```
skeleton
    |
    +-----------------+
    v                 v
apex::script     (shape geometry)
(builds rig)          |
    |                 |
    v                 v
packfolder <----------+
    |
    v
apex::sceneaddcharacter  <- Character to Input 2 (index 1)!
    |
    v
apex::sceneanimate
    |
    v
apex::sceneinvoke::2.0   <- outputmode=1 for unpacked
    |
    v
OUT
```

### Pack Folder Configuration

```python
pack_folder.parm("names").set(3)     # Set count FIRST
pack_folder.parm("name1").set("Base")
pack_folder.parm("type1").set("rig")  # suffix in type, not name!
pack_folder.parm("name2").set("Base")
pack_folder.parm("type2").set("skel")
pack_folder.parm("name3").set("Base")
pack_folder.parm("type3").set("shp")
```

### Scene Add Character

```python
# Character goes to Input 2 (index 1), NOT Input 1!
scene_add.setInput(1, pack_folder, 0)
scene_add.parm("charactername").set("MyCharacter")
```

### Scene Invoke

```python
scene_invoke.parm("enableanimation").set(1)
scene_invoke.parm("outputcharactershapes").set(1)
scene_invoke.parm("outputmode").set(1)  # 0=Packed, 1=Unpacked
```

---

## Geometry Transform Pattern

For geometry processing, build a graph that uses geometry callbacks:

```
graph = ApexGraphHandle()
parms = graph.addNode('parms', '__parms__')
output = graph.addNode('output', '__output__')

# Input geometry
geo_input = graph.addNode('geo_input', 'Value<Geometry>')
geo_input.parm.promoteInput('geo_in')

# Transform matrix input
xform_input = graph.addNode('xform_input', 'Value<Matrix4>')
xform_input.parm.promoteInput('xform')

# geo::Transform node
xform = graph.addNode('xform', 'geo::Transform')
graph.addWire(geo_input.value, xform.geo_in)
graph.addWire(xform_input.value, xform.xform_in)

xform.geo_out.promoteOutput('geo_out')

graph.sort(True)
geo = graph.saveToGeometry()
BindOutput(geo)
```

### Available Geometry Callbacks

| Callback | Purpose |
|:---------|:--------|
| `geo::Transform` | Apply Matrix4 to geometry |
| `geo::Lattice` | Lattice deformation |
| `geo::SetPointAttribValuesByName<T>` | Set point attributes |
| `sop::blendshapes::2.0` | Blend geometry shapes |
| `sop::attribvop` | Run VEX on geometry |
| `sop::bonedeform` | Skeletal deformation |

---

## Python Introspection

Discover port names for any APEX node:

```python
import apex

registry = apex.callbackRegistry()
sig = registry.getSignature("sop::bonedeform")

print("Inputs:", sig.inputs())
print("Outputs:", sig.outputs())
```
