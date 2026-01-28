---
layout: default
title: Python Checklist
parent: Code Review
nav_order: 2
description: Python-specific code review checklist for Houdini scripts and viewer states
permalink: /code-review/checklists/python/
---

# Python Code Review Checklist

Apply these checks to Python code including Houdini scripts, viewer states, and APEX scripts.

## Type Hints

| Check | Description |
|:------|:------------|
| Public Functions Typed | All public function signatures have type hints |
| Return Types Specified | Functions declare their return type |
| Complex Types Annotated | Lists, dicts, optionals properly typed |
| Type Aliases Used | Complex types have readable aliases |

### Examples
```python
# Good
def get_selected_points(geo: hou.Geometry) -> list[int]:
    ...

# With type alias
PointIndex = int
def get_neighbors(geo: hou.Geometry, point: PointIndex) -> list[PointIndex]:
    ...

# Bad - no hints
def process_data(data, options):
    ...
```

## Resource Cleanup

| Check | Description |
|:------|:------------|
| Context Managers Used | Files, connections use `with` statements |
| Explicit Cleanup | Resources released in finally blocks if needed |
| No Leaked Handles | File handles, sockets properly closed |
| Undo Blocks Closed | Houdini `hou.undos` blocks always end |

### Patterns
```python
# Good - context manager
with open(path) as f:
    data = f.read()

# Good - Houdini undo block
with hou.undos.group("My Operation"):
    node.parm("tx").set(1.0)

# Bad - leaked handle
f = open(path)
data = f.read()
# f never closed
```

## Imports

| Check | Description |
|:------|:------------|
| Organized Imports | stdlib, third-party, local grouped and sorted |
| No Unused Imports | Every import is used |
| Absolute Imports | Prefer `from package.module` over relative |
| No Wildcard Imports | Avoid `from module import *` |

### Import Order
```python
# 1. Standard library
import os
import sys
from pathlib import Path

# 2. Third-party
import numpy as np

# 3. Houdini
import hou

# 4. Local
from .utils import helper_function
```

## Exception Handling

| Check | Description |
|:------|:------------|
| Specific Exceptions | Catch specific types, not bare `except:` |
| No Silent Swallowing | Caught exceptions logged or re-raised |
| Exception Chaining | Use `raise ... from e` for context |
| Custom Exceptions | Domain errors use custom exception classes |

### Anti-Patterns
```python
# Bad - catches everything including KeyboardInterrupt
try:
    do_something()
except:
    pass

# Bad - loses stack trace
try:
    do_something()
except Exception as e:
    raise RuntimeError("Failed")  # Lost original traceback

# Good
try:
    do_something()
except ValueError as e:
    logger.error(f"Invalid value: {e}")
    raise ProcessingError("Data validation failed") from e
```

## Houdini API

| Check | Description |
|:------|:------------|
| Node References Valid | Check `node is not None` before use |
| Parameter Existence | Verify parms exist before getting/setting |
| Geometry Modifications | Use proper GA/GU patterns |
| Cook Dependencies | Proper input dependencies declared |
| Thread Safety | No hou calls from non-main thread |

### Common Issues
```python
# Bad - might be None
node = hou.node("/obj/geo1")
node.parm("tx").set(1)  # Crash if node doesn't exist

# Good - defensive
node = hou.node("/obj/geo1")
if node is not None:
    parm = node.parm("tx")
    if parm is not None:
        parm.set(1)

# Or with error handling
node = hou.node("/obj/geo1")
if node is None:
    raise ValueError("Node /obj/geo1 not found")
```

## Viewer States

| Check | Description |
|:------|:------------|
| Event Handlers Complete | All relevant events handled |
| Drawable Cleanup | Drawables removed in `onExit` |
| State Registration | State registered in `createViewerStateTemplate` |
| Selection Modes | Proper selection handling |
| Undo Integration | Changes wrapped in undo blocks |

### Required Methods
```python
class MyState(object):
    def __init__(self, state_name, scene_viewer):
        self.state_name = state_name
        self.scene_viewer = scene_viewer
        self._drawable = None

    def onEnter(self, kwargs):
        # Initialize drawables, state
        pass

    def onExit(self, kwargs):
        # CRITICAL: Clean up drawables
        if self._drawable:
            self._drawable = None

    def onMouseEvent(self, kwargs):
        # Handle mouse interaction
        return False  # or True to consume event
```

### Drawable Lifecycle
```python
# Create in onEnter or lazily
def onEnter(self, kwargs):
    self._geo_drawable = hou.GeometryDrawable(
        self.scene_viewer,
        hou.drawableGeometryType.Line,
        "my_lines"
    )

# Update in handlers
def onMouseMove(self, kwargs):
    self._geo_drawable.setGeometry(self._build_geo())
    self._geo_drawable.show(True)

# Clean up in onExit
def onExit(self, kwargs):
    self._geo_drawable = None
```

## Performance

| Check | Description |
|:------|:------------|
| Avoid Repeated Lookups | Cache node/parm references |
| Batch Operations | Combine multiple small operations |
| Generator Expressions | Use generators for large sequences |
| Avoid String Concatenation | Use f-strings or join() |

### Optimization Patterns
```python
# Bad - repeated lookups
for i in range(1000):
    hou.node("/obj/geo1").parm("tx").set(i)

# Good - cached reference
node = hou.node("/obj/geo1")
parm = node.parm("tx")
for i in range(1000):
    parm.set(i)

# Bad - string concatenation in loop
result = ""
for item in items:
    result += str(item)

# Good - join
result = "".join(str(item) for item in items)
```

## Summary Checklist

Quick reference for Python review:

- [ ] **Type Hints**: Present on public functions
- [ ] **Resource Cleanup**: Context managers, proper disposal
- [ ] **Imports**: Organized, no unused imports
- [ ] **Exception Handling**: Specific exceptions, not bare `except:`
- [ ] **Houdini API**: Proper hou module usage, node references
- [ ] **Viewer States**: Event handling, drawable cleanup
- [ ] **Performance**: Cached references, batch operations
