---
layout: default
title: Quick Reference
parent: Reference
grand_parent: APEX
nav_order: 1
description: APEX Script syntax cheat sheet
permalink: /apex/reference/quick-reference/
---

# APEX Script Quick Reference

> Houdini 21.0.559 | Last updated: 2025-12-26

## Variables and Types

```python
# Type inference
x = 1.5                     # Float
name = 'hello'              # String
flag = True                 # Bool

# Explicit type annotation
a: Float = 1.5
b: IntArray = []
c: String = 'test'
```

### Primitive Types

| Type | Description | Example |
|:-----|:------------|:--------|
| `Int` | Integer | `x = 42` |
| `Float` | Floating point | `x = 3.14` |
| `Bool` | Boolean | `x = True` |
| `String` | Text | `x = 'hello'` |

### Vector/Matrix Types

| Type | Description | Example |
|:-----|:------------|:--------|
| `Vector2` | 2D vector | `v = Vector2(1, 2)` |
| `Vector3` | 3D vector | `v = Vector3(1, 2, 3)` |
| `Vector4` | 4D vector | `v = Vector4(1, 2, 3, 4)` |
| `Matrix3` | 3x3 matrix | `m = Matrix3()` |
| `Matrix4` | 4x4 transform | `m = Matrix4()` |

### Array Types

| Type | Description | Example |
|:-----|:------------|:--------|
| `IntArray` | Integer array | `a = [1, 2, 3]` |
| `FloatArray` | Float array | `a = [1.0, 2.0]` |
| `StringArray` | String array | `a = ['a', 'b']` |
| `Vector3Array` | Vector3 array | `a = Vector3Array()` |

### Special Types

| Type | Description |
|:-----|:------------|
| `Geometry` | Houdini geometry |
| `Dict` | Key-value dictionary |
| `FloatRamp` | Float ramp curve |
| `ColorRamp` | Color ramp |

---

## Type Conversion

```python
a = 1.5
b = Int(a)          # Float -> Int (truncates)
c = Float(b)        # Int -> Float
d = Bool(b)         # Int -> Bool (0=False, else True)
s = String(a)       # Any -> String

# Vector3 to Float components
myVector: Vector3
x, y, z = myVector.vector3ToFloat()

# Int/Float to String using f-strings
index: Int = 0
index_s: String = f'{index}'
```

---

## Operators

### Arithmetic

| Operator | Description | Example |
|:---------|:------------|:--------|
| `+` | Addition | `a + b` |
| `-` | Subtraction | `a - b` |
| `*` | Multiplication | `a * b` |
| `/` | Division | `a / b` |
| `**` | Power | `a ** 2` |

### Comparison

| Operator | Description | Example |
|:---------|:------------|:--------|
| `==` | Equal | `a == b` |
| `!=` | Not equal | `a != b` |
| `is` | Identity | `a is b` |
| `is not` | Not identity | `a is not b` |
| `<`, `>` | Less/greater | `a < b` |
| `<=`, `>=` | Less/greater or equal | `a <= b` |

### Logical

| Operator | Description | Example |
|:---------|:------------|:--------|
| `and` | Logical AND | `a and b` |
| `or` | Logical OR | `a or b` |
| `not` | Logical NOT | `not a` |
| `in` | Membership | `x in array` |
| `not in` | Not member | `x not in array` |

### Assignment

| Operator | Description | Example |
|:---------|:------------|:--------|
| `=` | Assign | `a = 5` |
| `+=` | Add assign | `a += 1` |
| `-=` | Subtract assign | `a -= 1` |
| `*=` | Multiply assign | `a *= 2` |
| `/=` | Divide assign | `a /= 2` |

---

## Control Flow

### If/Elif/Else
```python
if x < 0:
    result = 'negative'
elif x > 0:
    result = 'positive'
else:
    result = 'zero'
```

### Ternary Expression
```python
result = 'yes' if condition else 'no'
```

### For Loops
```python
# Range loop
for i in range(10):
    total = total + i

# Array iteration
for item in my_array:
    process(item)

# Enumerated iteration
for index, value in enumerate(array):
    print(f'{index}: {value}')

# Geometry iteration
for pt in geo.points():
    pos = geo.pointAttribValue(pt, 'P', valuetype=Vector3)

for prim in geo.prims():
    name = geo.primAttribValue(prim, 'name', valuetype=String)
```

---

## Strings

```python
# Creation
s = 'hello'
s = "hello"
multi = """Multi-line
string here"""

# Concatenation
result = a + ' ' + b

# F-strings (formatting)
msg = f'Value is {x}'
msg = f'Result: {value:+.2f}'  # Format specifiers

# Length
n = len(s)
```

---

## Arrays

```python
# Creation
a = IntArray()
a = [1, 2, 3]
a: FloatArray = []

# Access
first = a[0]
last = a[-1]

# Modification
a[0] = 10
a.append(5)

# Operations
length = len(a)
combined = a + b  # Concatenation
```

---

## Vectors

```python
v = Vector3(1, 2, 3)

# Access by index
x = v[0]

# Access by component
x = v.x()
y = v.y()
z = v.z()

# Modification
v[0] = 5.0
```

---

## Matrices

```python
# Create 3x3 matrix (row-major)
m = Matrix3(1,2,3, 4,5,6, 7,8,9)

# Create 4x4 matrices
m = Matrix4()   # Zero matrix (all zeros)
m = Matrix4(1)  # Identity matrix

# Access element
val = m[row][col]

# Set row
m[1] = (10, 10, 10)

# Build transform with offset
restlocal = Matrix4(1).translate((0.0, 0.5, 0.0))
```

---

## Dictionaries

```python
# Creation
d = {'key1': 1.0, 'key2': 'hello'}

# Access (type annotation often needed)
s: String = d['key2']

# Modification
d['key3'] = 5.0

# Length
n = len(d)
```

---

## Functions

```python
# Basic function
def my_function(a: Int, b: Float) -> Float:
    return a * b

# Multiple returns
def get_values() -> (Int, Float):
    return 1, 2.5

# Default arguments
def greet(name: String = 'World') -> String:
    return f'Hello {name}'

# Keyword arguments
result = my_function(b=2.0, a=3)
```

---

## Comments

```python
# Single line comment

x = 5  # Inline comment

# Multi-line comments use multiple #
# This is line 1
# This is line 2
```

---

## Debugging

```python
print('Debug message')
print(f'Value: {x}')

warning('Warning message')

raise error('Error message')
raise error(f'Failed: {reason}')

# Debug trick: use node name to display debug string in graph
graph.addNode(name=debug_string, callback='Value<String>')
```

---

## Parameter Binding

```python
# Create input parameters (appears on node UI)
value = BindInput(Float, 'my_value', 1.0)
name = BindInput(String, 'name', 'default')

# Create output (exposed from graph)
BindOutput(result, 'output_name')
```

---

## Graph Building

```python
graph = ApexGraphHandle()

# Add nodes
parms = graph.addNode('parms', '__parms__')
output = graph.addNode('output', '__output__')
xform = graph.addNode('my_xform', 'TransformObject')

# Add or update node (creates if not exists, updates if exists)
node = graph.addOrUpdateNode(name='my_node', callback='TransformObject')
node = graph.addOrUpdateNode(callback='rig::SampleSplineTransforms::3.0')

# Promote inputs/outputs
# IMPORTANT: Use _in suffix for input ports, _out suffix for output ports
xform.r_in.promoteInput('rotation')      # r_in is the input port
xform.xform_out.promoteOutput('transform') # xform_out is the output port

# Promote to a specific parms node
geo_parms: ApexNodeID = graph.addOrUpdateNode('geo_parms', '__parms__')
myNode.geo_in.promoteInput(parmname='Base.shp', parmnodeid=geo_parms)

# Wire nodes - always use _in/_out suffixes
graph.addWire(node1.value_out, node2.value_in)

# Wire nodes between subnets
graph.addWirePath(parent_node.xform_out, child_node.a_in)

# Find or add port (ensures port exists)
geo_parms: ApexNodeID = graph.addOrUpdateNode('geo_parms', '__parms__')
baseShp_port = graph.findOrAddPort(geo_parms, "Base.shp")

# Check if node exists
node: ApexNodeID = graph.findFirstNode('nodeName')
if Bool(node):
    # Node exists, do stuff
    pass

# Alternative existence check
if len(node) > 0:
    # Node exists
    pass

# Find nodes by pattern
for n in graph.matchNodes('point_*'):
    n.r_in.promoteInput('r')

# Update node parameters after creation
ctrl_node.updateNodeParms(parms={'restlocal': restlocal})

# Layout graph
graph.sort(True)

# CRITICAL: Output the graph
geo = graph.saveToGeometry()
BindOutput(geo)
```

### Port Naming Convention

**IMPORTANT**: When accessing ports in APEX Script, always use the `_in` or `_out` suffix:

| Port Type | Suffix | Example |
|:----------|:-------|:--------|
| Input ports | `_in` | `node.value_in`, `node.geo_in`, `node.r_in` |
| Output ports | `_out` | `node.value_out`, `node.geo_out`, `node.xform_out` |

Common node port examples:
```python
# TransformObject - uses _in/_out suffix
xform.t_in          # Translation input
xform.r_in          # Rotation input
xform.s_in          # Scale input
xform.restlocal_in  # Rest local transform input
xform.parent_in     # Parent world transform input
xform.xform_out     # World transform output
xform.localxform_out # Local transform output

# Value<Geometry> - EXCEPTION: no _in/_out suffix!
value_node.parm     # Input port (where data comes in)
value_node.value    # Output port (where data goes out)

# sop::bonedeform - uses _in/_out suffix
bonedeform.geo0_in  # Rest mesh input
bonedeform.geo1_in  # Capture skeleton input
bonedeform.geo2_in  # Animated skeleton input
bonedeform.geo0_out # Deformed mesh output

# skel::SetPointTransforms - uses _in/_out suffix
spt.geo_in          # Skeleton geometry input
spt.geo_out         # Deformed skeleton output
```

**Note**: `Value<T>` nodes (e.g., `Value<Geometry>`, `Value<Float>`) are an exception - they use `.parm` and `.value` without the `_in`/`_out` suffix.

### Required APEX Script SOP Settings

For the graph to appear on Output 1, configure these parameters:

| Parameter | Value |
|:----------|:------|
| `invocation` | 1 |
| `bindoutputgeo` | 1 |
| `apexgeooutput` | `output:geo` |
| `dictbindings` | 1 |
| `apexoutputgroup1` | `output` |
| `outputattrib1` | `results` |

**Note:** In Houdini 20.5, use `graph.writeToGeo()` instead of `graph.saveToGeometry()`

---

## Decorators

```python
@subgraph
def my_component():
    # Saved as reusable subgraph (.bgeo)
    pass

@namespace
def utils():
    # Allows calling as utils.function()
    pass

@safeguard_inputs  # Default: enabled
def safe_function():
    # Makes copies of inputs to prevent issues
    pass
```
