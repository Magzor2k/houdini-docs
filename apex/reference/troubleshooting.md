---
layout: default
title: Troubleshooting
parent: Reference
grand_parent: APEX
nav_order: 5
description: APEX Script troubleshooting guide
permalink: /apex/reference/troubleshooting/
---

# APEX Script Troubleshooting

> Houdini 21.0.559 | Last updated: 2025-12-26

## Common Errors and Solutions

---

### "Only additive expressions are supported for string expressions"

**Error message:**
```
Line X: :: c = ``a - b``
Only additive expressions are supported for string expressions
```

**Cause:**
You're trying to do arithmetic (`-`, `*`, `/`) on values returned by `BindInput()`.

`BindInput()` returns a **port reference** (string-like), not a numeric value. Only `+` works (for string concatenation).

**Wrong:**
```python
a = BindInput('Float', 'input_a', 1.0)
b = BindInput('Float', 'input_b', 2.0)
c = a - b   # ERROR!
c = a * b   # ERROR!
```

**Solutions:**

1. **Use direct values for prototyping:**
```python
a: Float = 1.5
b: Float = 2.5
c = a - b   # Works!
c = a * b   # Works!
```

2. **Use graph building for parameterized math:**
```python
graph = ApexGraphHandle()
parms = graph.addNode('parms', '__parms__')
output = graph.addNode('output', '__output__')

# Use math nodes
sub_node = graph.addNode('sub1', 'Subtract<Float>')
sub_node.a_in.promoteInput('input_a')
sub_node.b_in.promoteInput('input_b')
sub_node.diff_out.promoteOutput('difference')
```

---

### "Invalid type or non-constant expression used in BindInput()"

**Error message:**
```
Line X: :: value = BindInput(``Float, 'my_value', 1.0``)
Invalid type or non-constant expression used in BindInput().
```

**Cause:**
Using unquoted type name in `BindInput()`.

**Wrong:**
```python
value = BindInput(Float, 'my_value', 1.0)   # Float without quotes
value = BindInput(type::Float, 'name', 1.0) # type:: prefix
```

**Correct:**
```python
value = BindInput('Float', 'my_value', 1.0)  # Type as string
value = BindInput('Int', 'count', 0)
value = BindInput('String', 'name', 'default')
```

---

### "SyntaxError: invalid syntax" with type::

**Error message:**
```
SyntaxError: invalid syntax
    value = BindInput(type::Float, 'my_value', 1.0)
                          ^
```

**Cause:**
`type::Float` is not valid Python/APEX Script syntax.

**Correct:**
```python
value = BindInput('Float', 'my_value', 1.0)
```

---

### Script cooks but Output 1 has 0 points, 0 prims

**Symptom:**
Script completes successfully but Output 1 has 0 points, 0 prims.

**Cause:**
Missing `graph.saveToGeometry()` + `BindOutput(geo)` OR incorrect SOP settings.

**Solutions:**

1. **Add the output pattern at the end of your script:**
```python
graph.sort(True)
geo = graph.saveToGeometry()
BindOutput(geo)
```

2. **Configure APEX Script SOP parameters:**
   - `invocation` = 1
   - `bindoutputgeo` = 1
   - `apexgeooutput` = `output:geo`
   - `dictbindings` = 1
   - `apexoutputgroup1` = `output`
   - `outputattrib1` = `results`

**Note:** In Houdini 20.5, use `graph.writeToGeo()` instead of `graph.saveToGeometry()`

---

### Output 2 has geometry but Output 1 is empty

**Symptom:**
Output 2 (the meta-graph) has points/prims but Output 1 is empty.

**Cause:**
Output 2 shows the *script execution graph* (the graph that builds your graph).
Output 1 shows the *actual APEX graph* you created.

**Solution:**
Add at the end of your script:
```python
geo = graph.saveToGeometry()
BindOutput(geo)
```

And ensure `bindoutputgeo = 1` and `apexgeooutput = "output:geo"` on the SOP.

---

### Function parameters not recognized

**Error:**
```
Unknown parameter 'x' in function call
```

**Cause:**
Missing type annotations on function parameters.

**Wrong:**
```python
def add(x, y):      # No type hints
    return x + y
```

**Correct:**
```python
def add(x: Float, y: Float) -> Float:
    return x + y
```

---

### Loop variable type mismatch

**Error when adding loop counter to Float:**
```
Type mismatch: cannot add Int to Float
```

**Cause:**
Range loop variables are Int, need explicit conversion.

**Solution:**
```python
total: Float = 0.0
for i in range(5):
    total = total + Float(i)  # Convert i to Float
```

---

### Cannot find node type / callback

**Error:**
```
Unknown callback type: SomeNode
```

**Cause:**
The callback type name is wrong or doesn't exist.

**Common callback types:**
- `Add<Float>`, `Add<Int>`, `Add<Vector3>`
- `Subtract<Float>`, `Multiply<Float>`, `Divide<Float>`
- `TransformObject` (for rigging)
- `__parms__` (input parameters node)
- `__output__` (output node)

**Tip:** Use Houdini's APEX Graph node to discover available callbacks via the Tab menu.

---

### Dictionary access returns wrong type

**Error:**
```
Type mismatch when accessing dictionary
```

**Cause:**
Dictionary values are dynamically typed; need type annotation.

**Solution:**
```python
d = {'val': 1.0, 'name': 'test'}
x: Float = d['val']      # Type annotation required
s: String = d['name']    # Type annotation required
```

---

## Debugging Tips

### Enable Traceback
On APEX Script SOP, enable **Show Traceback** to see full error stack.

### Inspect Line
Set **Inspect Line** parameter to highlight a specific line in red.

### Log Viewer
Use **View Log** button or Window > Log Viewer to see `print()` output.

### Check Both Outputs
1. **First output** - Invoked result (after APEX Invoke Graph)
2. **Second output** - The APEX graph itself (useful for graph-building scripts)

---

## Template Mode Differences

| Mode | Purpose | When to Use |
|:-----|:--------|:------------|
| Basic (0) | Simple computations | Learning, prototyping |
| Graph (1) | Build APEX graphs | Creating parameterized tools |
| Component (2) | Rig components | KineFX rigging workflows |
| Custom (3) | Custom header/footer | Advanced use cases |

The template mode affects what header code is prepended to your script.

---

## Key Concept: APEX Script Builds Graphs, Doesn't Execute Directly

**APEX Script is a GRAPH-BUILDING language, not a direct execution language like VEX.**

### What This Means

When you write:
```python
geo = BindInput(Geometry())
for pt in geo.points():
    # process points
BindOutput(geo)
```

This creates an APEX **graph** that describes the processing, but doesn't execute it immediately.

- **Output 1**: The invoked result (empty until graph is invoked)
- **Output 2**: The meta-graph (script execution graph)

### How to Actually Process Geometry

1. **Build the graph** with APEX Script SOP
2. **Invoke the graph** with APEX Invoke Graph SOP
3. **Provide inputs** via the Invoke Graph's input connections

### Available Geometry Callbacks

For geometry processing in APEX graphs, use these callbacks:

| Callback | Purpose |
|:---------|:--------|
| `geo::Transform` | Apply Matrix4 to geometry |
| `geo::Lattice` | Lattice deformation |
| `geo::SetPointAttribValuesByName<T>` | Set point attributes |
| `sop::blendshapes::2.0` | Blend geometry shapes |
| `sop::attribvop` | Run VEX on geometry |

### geo::Transform Port Names

```python
xform = graph.addNode('xform', 'geo::Transform')
# xform.geo_in   - Geometry input
# xform.xform_in - Matrix4 transform input
# xform.geo_out  - Transformed geometry output
```

---

### Vector component access error

**Error:**
```
Can only resolve attribute types for 'ApexNodeID' or 'ApexGraphHandle'
```

**Cause:**
Trying to access Vector3 components incorrectly.

**Wrong:**
```python
pos = geo.pointAttribValue(pt, 'P', valuetype=Vector3)
x = pos.x * scale  # ERROR - can't use .x directly
```

**Correct:**
```python
pos = geo.pointAttribValue(pt, 'P', valuetype=Vector3)
x: Float = pos[0]  # Use index access
y: Float = pos[1]
z: Float = pos[2]
```

Or use method syntax:
```python
x: Float = pos.x()  # Method call, not property
```

---

### "The given function 'subport' does not exist" / "The given function 'connect' does not exist"

**Error message:**
```
Line X: :: ``spt.transforms.subport('hip').connect(hip.xform_out)``
The given function 'subport' does not exist for the variable 'spt.transforms' of type 'ApexGraphHandle'
```

**Cause:**
Trying to chain `.subport().connect()` - the `.connect()` method does not exist on subports.

**Wrong:**
```python
# This does NOT work!
spt.transforms.subport('hip').connect(hip.xform_out)
```

**Correct:**
```python
# Create subport reference first, then use graph.addWire()
hip_port = spt.transforms.subport('hip')
graph.addWire(hip.xform_out, hip_port)
```

This pattern applies to all variadic ports like `skel::SetPointTransforms.transforms`.

---

### Variadic Port Connection Pattern

For nodes with variadic ports (like `skel::SetPointTransforms`), you must:

1. Create a subport reference with a unique name matching the skeleton joint
2. Use `graph.addWire()` to connect to that subport

```python
spt = graph.addNode('spt', 'skel::SetPointTransforms')
graph.addWire(skel_input.value, spt.geo_in)

# For each joint - create subport then wire
hip_port = spt.transforms.subport('hip')      # Name must match skeleton joint name
spine_port = spt.transforms.subport('spine')
chest_port = spt.transforms.subport('chest')

graph.addWire(hip.xform_out, hip_port)
graph.addWire(spine.xform_out, spine_port)
graph.addWire(chest.xform_out, chest_port)
```

---

### TransformObject rigging - joints don't move

**Symptom:**
FK rig builds but joints don't respond to rotation inputs.

**Cause:**
Missing parent hierarchy wiring or rest transform setup.

**Solution:**
For FK chains, wire both `xform_out` and `localxform_out` to children:

```python
# Parent -> Child FK hierarchy
graph.addWire(parent.xform_out, child.parent_in)
graph.addWire(parent.localxform_out, child.parentlocal_in)
```

Also consider setting `restlocal` from the skeleton's initial transforms.

---

### "Callback with the name '...parm[in]...' does not exist"

**Error message:**
```
Callback with the name 'skel[in]put.parm[in]::promoteInput' does not exist
```

**Cause:**
APEX Script parser interprets `parm_in` as `parm[in]` due to `in` being a Python keyword. The parser sees `_in` as an indexing operation.

**Wrong:**
```python
value_node = graph.addNode('val', 'Value<Geometry>')
value_node.parm_in.promoteInput('geo')   # ERROR - parm_in parsed as parm[in]
```

**Correct:**
```python
value_node = graph.addNode('val', 'Value<Geometry>')
value_node.parm.promoteInput('geo')   # Works - Value<T> nodes use .parm and .value
```

**Note:** This is a special case for `Value<T>` nodes. They use `.parm` (input) and `.value` (output) without the `_in`/`_out` suffix. All other nodes should use the `_in`/`_out` suffix convention.
