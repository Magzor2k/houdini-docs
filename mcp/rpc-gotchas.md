---
layout: default
title: RPC Gotchas
parent: MCP/RPC
nav_order: 1
description: Common pitfalls when using Houdini RPC
permalink: /mcp/rpc-gotchas/
---

# RPC Gotchas
{: .fs-9 }

Common pitfalls when using Houdini RPC and how to avoid them.
{: .fs-6 .fw-300 }

---

## Geometry Proxy Objects

### The Problem

When you get a geometry object via RPC, it's a **proxy** that always reads the current state:

```python
# Frame 1
hou.setFrame(1)
geo1 = node.geometry()
pos1 = geo1.pointFloatAttribValues('P')  # Frame 1 positions

# Frame 5 - geo1 now returns frame 5 data!
hou.setFrame(5)
node.cook(force=True)
pos1_again = geo1.pointFloatAttribValues('P')  # Frame 5 positions!
```

Even though `geo1` was obtained at frame 1, calling methods on it after cooking returns the **current** geometry state.

### The Solution

Extract data to local Python objects immediately:

```python
import numpy as np

# Frame 1 - extract to numpy array (local copy)
hou.setFrame(1)
geo1 = node.geometry()
positions_f1 = np.array(geo1.pointFloatAttribValues('P')).reshape(-1, 3)

# Frame 5 - positions_f1 is unchanged
hou.setFrame(5)
node.cook(force=True)
positions_f5 = np.array(node.geometry().pointFloatAttribValues('P')).reshape(-1, 3)

# Now you can compare
delta = positions_f5 - positions_f1  # Correct!
```

### Impact on Analysis Code

The cloth analysis module stores positions in `ClothMetrics.positions` as a numpy array. When computing frame-to-frame velocity, always use the stored positions from the previous metrics object:

```python
# Correct - use stored positions
prev_positions = prev_metrics.positions if prev_metrics else None

# Wrong - re-extracting from geometry proxy returns current frame
prev_positions = extract_positions(prev_geo)  # Always returns current!
```

---

## Module Reloading

### The Problem

When you modify Python code and want to test it via RPC, `importlib.reload()` doesn't fully update imported functions:

```python
import importlib
import cloth_analysis.metrics
importlib.reload(cloth_analysis.metrics)

from cloth_analysis import compute_frame_metrics  # Still old version!
```

### The Solution

Remove modules from `sys.modules` before reimporting:

```python
import sys

# Force full reimport
for mod_name in list(sys.modules.keys()):
    if 'cloth_analysis' in mod_name:
        del sys.modules[mod_name]

# Now import fresh
from cloth_analysis import compute_frame_metrics  # New version
```

---

## JSON Serialization

### The Problem

Numpy types aren't JSON serializable:

```python
import numpy as np
import json

result = {'is_stable': np.bool_(True)}  # numpy.bool_
json.dumps(result)  # TypeError!
```

### The Solution

Convert numpy types to Python natives:

```python
result = {
    'is_stable': bool(is_stable),        # np.bool_ -> bool
    'max_value': float(np.max(arr)),     # np.float64 -> float
    'count': int(np.sum(mask)),          # np.int64 -> int
}
```

---

## execute_python Return Values

### The Problem

The `mcp__houdini__execute_python` tool doesn't capture print output or expression results:

```python
# This returns null
mcp__houdini__execute_python("print('hello')")

# This also returns null
mcp__houdini__execute_python("1 + 1")
```

### The Solution

Write results to a file and read it back:

```python
code = '''
import json
result = {"value": 1 + 1}
with open('/tmp/result.json', 'w') as f:
    json.dump(result, f)
'''
mcp__houdini__execute_python(code)

# Then read the file
result = Read('/tmp/result.json')
```

Or use the dedicated MCP tools when possible:
- `mcp__houdini__get_parameter` for parameter values
- `mcp__houdini__get_geometry_info` for geometry stats
- `mcp__houdini__get_scene_info` for scene info

---

## Connection State

### The Problem

The RPC connection can drop if Houdini becomes unresponsive or is restarted:

```python
mcp__houdini__create_node(...)  # Error: Not connected
```

### The Solution

Check connection status before operations:

```python
status = mcp__houdini__get_connection_status()
if not status['connected']:
    mcp__houdini__connect(port=YOUR_PORT)  # Use YOUR port from session file
```

The MCP tools generally handle reconnection, but long-running scripts should check periodically.

---

## Multi-Agent Session Conflicts (NEW)

### The Problem

When multiple agents run simultaneously, they share a single MCP server process. When Agent A connects to port 18811 and Agent B connects to port 18812, the "current" connection switches. If Agent A then calls MCP tools without reconnecting, they operate on Agent B's Houdini instance.

```python
# Agent A connects
mcp__houdini__connect(port=18811)

# ... Agent B connects to their port ...
# Agent B: mcp__houdini__connect(port=18812)

# Agent A tries to create a node - but it goes to port 18812!
mcp__houdini__create_node(...)  # Wrong Houdini instance!
```

### The Solution

**Always reconnect to YOUR port before each batch of operations:**

```python
# Get your port from your session file
my_port = 18811  # From .claude/sessions/{your_session_id}.json

# ALWAYS reconnect before doing work
mcp__houdini__connect(port=my_port)

# Now do your operations
mcp__houdini__create_node(...)
mcp__houdini__set_parameter(...)
```

### Session Ownership Rules

1. Each agent gets a unique session file: `.claude/sessions/{sessionId}.json`
2. Your session file contains your assigned port
3. ONLY connect to the port in YOUR session file
4. NEVER connect to arbitrary ports from `list_sessions()`
5. The deprecated `current-session.json` may contain another agent's data

See `.claude/rules/houdini-mcp.md` for complete session isolation rules.

---

## Summary

| Issue | Solution |
|:------|:---------|
| Geometry proxy returns current state | Extract to numpy/list immediately |
| Module changes not picked up | Delete from sys.modules before import |
| Numpy types not JSON serializable | Cast to Python natives: `bool()`, `float()`, `int()` |
| execute_python returns null | Write to file, or use dedicated tools |
| Connection dropped | Check status and reconnect |
| Multi-agent session conflicts | Always reconnect to YOUR port before operations |
