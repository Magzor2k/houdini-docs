---
layout: default
title: Types
parent: Reference
grand_parent: APEX
nav_order: 2
description: APEX Script data types
permalink: /apex/reference/types/
---

# APEX Script Data Types

> Houdini 21.0.559 | Last updated: 2025-12-26

## Type System Overview

APEX Script is **statically typed**. Once a variable is assigned a type, it cannot change.

```python
x = 1.5      # x is now Float
x = 'hello'  # ERROR: Cannot change type
```

---

## Primitive Types

### Int
Integer numbers (whole numbers).

```python
a = 42
b = -10
c: Int = 0

# Operations
result = a + b
result = a * 2
result = a // b  # Integer division
result = a % b   # Modulo
```

### Float
Floating-point numbers (decimals).

```python
a = 3.14
b = -0.5
c: Float = 0.0

# Operations
result = a + b
result = a * 2.0
result = a / b
result = a ** 2  # Power
```

### Bool
Boolean values (True/False).

```python
flag = True
enabled = False
check: Bool = True

# Operations
result = a and b
result = a or b
result = not a
```

### String
Text strings.

```python
s = 'hello'
s = "world"
multi = """Multi-line
text here"""

# Operations
combined = a + ' ' + b
length = len(s)
formatted = f'Value: {x}'
```

---

## Vector Types

### Vector2
2D vector (x, y).

```python
v = Vector2(1.0, 2.0)
v = Vector2()  # (0, 0)

# Access
x = v[0]
y = v[1]
x = v.x()
y = v.y()

# Modify
v[0] = 5.0
```

### Vector3
3D vector (x, y, z). Most common for positions, directions.

```python
v = Vector3(1.0, 2.0, 3.0)
v = Vector3()  # (0, 0, 0)

# Access
x = v[0]
y = v[1]
z = v[2]
x = v.x()
y = v.y()
z = v.z()

# Unpack to floats
x, y, z = v.vector3ToFloat()

# Modify
v[0] = 5.0

# Common operations
length = len(v)
normalized = normalize(v)
d = dot(v1, v2)
c = cross(v1, v2)
dist = distance(v1, v2)
```

### Vector4
4D vector (x, y, z, w). Used for colors with alpha, quaternions.

```python
v = Vector4(1.0, 2.0, 3.0, 1.0)
v = Vector4()  # (0, 0, 0, 0)

# Access
x = v[0]
w = v[3]
```

---

## Matrix Types

### Matrix3
3x3 rotation/scale matrix.

```python
# Row-major construction
m = Matrix3(
    1, 0, 0,  # Row 0
    0, 1, 0,  # Row 1
    0, 0, 1   # Row 2
)

# Identity
m = Matrix3()

# Access element
val = m[row][col]

# Set row
m[1] = (1.0, 0.0, 0.0)

# Operations
inv = invert(m)
t = transpose(m)
```

### Matrix4
4x4 transformation matrix. Used for full transforms (translate, rotate, scale).

```python
# Zero matrix (all zeros)
m = Matrix4()

# Identity matrix
m = Matrix4(1)

# Access
val = m[row][col]

# Get/Set translation
pos = m.getTranslates()
m.setTranslates(Vector3(x, y, z))

# Build transform with offset
restlocal = Matrix4(1).translate((0.0, 0.5, 0.0))

# Operations
inv = invert(m)
combined = m1 * m2  # Matrix multiplication
```

---

## Array Types

Arrays hold multiple values of the same type.

### IntArray
```python
a = IntArray()
a = [1, 2, 3]
a: IntArray = []

# Operations
a.append(4)
length = len(a)
first = a[0]
last = a[-1]
combined = a + b
```

### FloatArray
```python
a = FloatArray()
a = [1.0, 2.0, 3.0]
a: FloatArray = []
```

### StringArray
```python
a = StringArray()
a = ['hello', 'world']
a: StringArray = []
```

### Vector3Array
```python
a = Vector3Array()
a.append(Vector3(1, 2, 3))
```

### Matrix4Array
```python
a = Matrix4Array()
a.append(Matrix4())
```

---

## Dictionary Type

Key-value storage with string keys.

```python
d = {'key1': 1.0, 'key2': 'hello'}
d = Dict()

# Access (type annotation often required)
val: Float = d['key1']
s: String = d['key2']

# Modify
d['key3'] = 5.0

# Length
count = len(d)
```

**Note**: When reading from a dictionary, you often need type annotations because the dictionary can hold mixed types.

---

## Ramp Types

### FloatRamp
Curve mapping input (0-1) to output values.

```python
r = FloatRamp(
    basis=['Linear'],
    keys=[0.0, 0.5, 1.0],
    values=[0.0, 0.8, 1.0]
)

# Evaluate
result = r.lookup(0.5)
```

### ColorRamp
Curve mapping input to colors.

```python
r = ColorRamp(
    basis=[3, 1, 4],  # Interpolation types
    keys=[0.0, 0.5, 1.0],
    values=[(1,0,0), (0,1,0), (0,0,1)],
    colortype='HSV'  # or 'RGB'
)
```

---

## Geometry Type

Represents Houdini geometry (points, prims, attributes).

```python
geo = BindInput(Geometry, 'input_geo')

# Query
num_pts = geo.numPoints()
num_prims = geo.numPrims()

# Iterate
for pt in geo.points():
    pass

for prim in geo.prims():
    pass

# Transform
geo.transform(matrix4)
```

---

## Type Conversion

### Explicit Conversion
```python
# Int <-> Float
i = Int(3.7)      # Result: 3 (truncates)
f = Float(5)      # Result: 5.0

# To Bool
b = Bool(0)       # False
b = Bool(1)       # True
b = Bool('')      # False
b = Bool('x')     # True

# To String
s = String(42)    # '42'
s = String(3.14)  # '3.14'
```

### Implicit Conversion
Some contexts allow implicit conversion:

```python
# Int to Float in arithmetic
result = 5 + 1.5  # 5 becomes 5.0

# Numeric to String in f-strings
msg = f'Count: {42}'
```

---

## Type Annotations

Explicit type declarations help with clarity and are sometimes required.

```python
# Variable declaration
x: Float = 1.0
name: String = 'default'
items: IntArray = []

# Function parameters
def process(a: Int, b: Float) -> String:
    return f'{a} + {b}'

# Dictionary access
d = {'val': 1.0}
x: Float = d['val']  # Type hint needed
```

---

## Special Types

### Geometry Handles
```python
pt   # Point handle (from geo.points())
prim # Primitive handle (from geo.prims())
```

### Graph Handles
```python
graph = ApexGraphHandle()  # Graph builder
node  # Node handle (from addNode)
port  # Port handle (node.port_name)
```

### VariadicArg
For functions accepting variable number of typed arguments:
```python
# Used in C++ APEX callbacks
VariadicArg<Matrix4>  # Array of Matrix4 with names
```
