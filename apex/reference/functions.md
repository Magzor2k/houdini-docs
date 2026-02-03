---
layout: default
title: Functions
parent: Reference
grand_parent: APEX
nav_order: 3
description: APEX Script functions reference
permalink: /apex/reference/functions/
---

# APEX Script Functions Reference

> Houdini 21.0.559 | Last updated: 2025-12-26

## Calling Conventions

Functions can be called in two styles:

```python
# Namespace style
result = apex.sop.copytopoints(geo)

# Object-oriented style
result = geo.copytopoints()
```

---

## Parameter Binding

### BindInput
Creates a parameter on the APEX Script SOP node.

```python
# Syntax
value = BindInput(type, 'param_name', default_value)

# Examples
a = BindInput(Float, 'amplitude', 1.0)
name = BindInput(String, 'object_name', 'default')
pos = BindInput(Vector3, 'position', Vector3(0, 0, 0))
```

### BindOutput
Exposes a value as graph output.

```python
# Syntax
BindOutput(value, 'output_name')

# Examples
BindOutput(result, 'computed_value')
BindOutput(geo, 'output_geo')
```

---

## Graph Building

### ApexGraphHandle
Creates and manipulates APEX graphs programmatically.

```python
graph = ApexGraphHandle()

# Add nodes
node = graph.addNode('node_name', 'callback_type')

# Add node with default port values (BAKES values into node)
val = graph.addNode('my_val', 'Value<Float>', parms={'value': 1.0})
xform = graph.addNode('joint', 'TransformObject', parms={'restlocal': Matrix4(...)})

# Special node types
parms = graph.addNode('parms', '__parms__')      # Input parameters
output = graph.addNode('output', '__output__')    # Output
```

### Node Operations

```python
# Set node properties
node.setName('new_name')
node.setColor((1, 0, 0))        # RGB tuple
node.setPos((100, 200, 0))      # Position in graph

# Set node parameters after creation (alternative to parms dict)
node.setParms({'param1': value1, 'param2': value2})
```

### Baking Values vs Wiring

Use `parms` dict to bake constant values at graph build time:
```python
# BAKED: Value is constant, no runtime lookup
hip = graph.addNode('hip', 'TransformObject', parms={'restlocal': Matrix4(...)})

# WIRED: Value comes from another node at runtime
graph.addWire(some_node.output, hip.restlocal)
```

Baking is preferred when values are known at build time (faster evaluation).

### Port Operations

```python
# Promote as input/output
node.port_name.promoteInput('exposed_name')
node.port_name.promoteOutput('exposed_name')

# Connect ports
graph.addWire(source.output_port, dest.input_port)
node1.out.connect(node2.in_port)

# Subports (for variadic inputs)
subport = node.array_in.subport('element_name')
```

### Graph Queries

```python
# Find nodes by pattern
nodes = graph.matchNodes('prefix_*')
nodes = graph.matchNodes('*_suffix')

# Layout
graph.sort(True)  # Auto-layout with hierarchy
```

### Graph Output

```python
# Convert graph to geometry (Houdini 21.0+)
geo = graph.saveToGeometry()
BindOutput(geo)

# In Houdini 20.5, use:
# geo = graph.writeToGeo()
# BindOutput(geo)
```

**Required SOP Settings:** For the graph to appear on Output 1:
- `invocation` = 1
- `bindoutputgeo` = 1
- `apexgeooutput` = `output:geo`

---

## Geometry Functions

### Point Operations

```python
# Iterate points
for pt in geo.points():
    # pt is a point handle
    pass

# Get point count
count = geo.numPoints()

# Get position
pos = geo.getPos3(pt)

# Set position
geo.setPos3(pt, Vector3(x, y, z))
```

### Attribute Access

```python
# Read point attribute
value = geo.pointAttribValue(pt, 'attrib_name', valuetype=Float)
value = geo.pointAttribValue_Float(pt, 'attrib_name')

# Write point attribute
geo.setPointAttribValue(pt, 'attrib_name', value, valuetype=Float)
geo.setPointAttribValue_Float(pt, 'attrib_name', value)

# Read prim attribute
value = geo.primAttribValue(prim, 'name', valuetype=String)

# Write prim attribute
geo.setPrimAttribValue(prim, 'name', value, valuetype=String)
```

### Primitive Operations

```python
# Iterate primitives
for prim in geo.prims():
    pass

# Get prim count
count = geo.numPrims()
```

### Transform

```python
# Apply transform to geometry
geo.transform(matrix4)

# Compute transform from attributes
xform = geo.computeTransform()
```

---

## Math Functions

### Basic Math

```python
# Absolute value
result = abs(x)

# Min/Max
result = min(a, b)
result = max(a, b)

# Clamp
result = clamp(value, min_val, max_val)

# Floor/Ceil
result = floor(x)
result = ceil(x)

# Power/Sqrt
result = pow(base, exp)
result = sqrt(x)
```

### Trigonometry

```python
result = sin(angle)
result = cos(angle)
result = tan(angle)
result = asin(x)
result = acos(x)
result = atan(x)
result = atan2(y, x)
```

### Interpolation

```python
result = lerp(a, b, t)  # Linear interpolation
```

---

## Vector Functions

```python
# Length
length = len(vector)

# Normalize
result = normalize(vector)

# Dot product
result = dot(v1, v2)

# Cross product
result = cross(v1, v2)

# Distance
result = distance(v1, v2)
```

---

## Matrix Functions

```python
# Create identity
m = Matrix4()

# Invert
inv = invert(matrix)

# Transpose
t = transpose(matrix)

# Extract components
pos = matrix.getTranslates()
rot = matrix.getRotation()

# Set components
matrix.setTranslates(Vector3(x, y, z))
```

---

## Skeleton/Rig Functions

### TransformObject
Core node for skeleton transforms.

```python
to = graph.addNode('joint_name', 'TransformObject')

# Common ports
to.r_in          # Rotation input
to.t_in          # Translation input
to.s_in          # Scale input
to.parent        # Parent transform input
to.parentlocal   # Parent local transform
to.restlocal     # Rest local transform
to.xform_out     # World transform output
to.localxform_out # Local transform output
```

### SetPointTransforms
Applies transforms from graph to skeleton geometry.

```python
spt = graph.addNode('spt', 'skel::SetPointTransforms')
spt.geo_in.promoteInput('skeleton')
spt.geo_out.promoteOutput('skeleton')
spt.transforms_in.subport('joint_name')  # Per-joint input
```

---

## Utility Functions

### Logging

```python
print('Message')
print(f'Value: {x}')

warning('Warning message')

raise error('Error message')
```

### Array Operations

```python
length = len(array)
array.append(item)
combined = array1 + array2
```

### Range

```python
for i in range(10):          # 0 to 9
    pass

for i in range(5, 10):       # 5 to 9
    pass

for i in range(0, 10, 2):    # 0, 2, 4, 6, 8
    pass
```

---

## Special Node Arguments

All APEX nodes support these metadata parameters:

| Argument | Type | Purpose |
|----------|------|---------|
| `__name` | String | Node identifier |
| `__color` | Vector3 | Node color (RGB) |
| `__pos` | Vector3 | Graph position |
| `__properties` | Dict | Custom metadata |
| `__tags` | StringArray | Categorical labels |

```python
node = graph.addNode('my_node', 'SomeCallback',
    __color=(1, 0, 0),
    __pos=(100, 200, 0))
```

---

## Templated Functions

For functions that work with multiple types, use `valuetype` parameter:

```python
# Explicit type
geo.setPointAttribValue(pt, 'name', 'value', valuetype=String)

# Or use typed version
geo.setPointAttribValue_String(pt, 'name', 'value')
geo.setPointAttribValue_Float(pt, 'value', 1.0)
geo.setPointAttribValue_Vector3(pt, 'P', Vector3(0, 0, 0))
```

---

## Version Management

```python
# Set version at script start
HoudiniVersion('21.0')

# Or use 'newest' for latest
HoudiniVersion('newest')

# Access specific function version
result = some_function.v2_0()
```
