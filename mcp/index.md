---
layout: default
title: MCP/RPC
nav_order: 7
has_children: true
description: Houdini MCP server for remote Python execution
permalink: /mcp/
---

# Houdini MCP Server
{: .fs-9 }

Remote Python execution and scene control via Model Context Protocol (MCP).
{: .fs-6 .fw-300 }

---

## Overview

The Houdini MCP server enables Claude Code (and other MCP clients) to interact with a running Houdini session via RPC. This allows:

- **Scene inspection** - query nodes, parameters, geometry
- **Scene modification** - create nodes, set parameters, wire connections
- **Python execution** - run arbitrary Python code in Houdini's environment
- **Simulation control** - set frames, cook nodes, run analyses

## Architecture

```
┌─────────────┐     MCP Protocol     ┌─────────────┐     hrpyc/rpyc     ┌─────────────┐
│ Claude Code │ ◄──────────────────► │  MCP Server │ ◄────────────────► │   Houdini   │
└─────────────┘      (stdio)         └─────────────┘    (port 18811)    └─────────────┘
```

1. **Claude Code** calls MCP tools (e.g., `mcp__houdini__create_node`)
2. **MCP Server** translates to Houdini Python API calls
3. **hrpyc/rpyc** executes commands in Houdini's Python environment

## Quick Start

1. **Start RPC in Houdini**: Use shelf tool or Python Shell:
   ```python
   import hrpyc; hrpyc.start_server()
   ```

2. **Connect from Claude**: The MCP server auto-connects on first tool call

3. **Use tools**: Create nodes, set parameters, execute Python

## Documentation

| Topic | Description |
|:------|:------------|
| [RPC Gotchas](rpc-gotchas.md) | Common pitfalls when using RPC |

## Key Concepts

### RPC Proxy Objects

When accessing Houdini objects via RPC, you get **proxy objects** that reference the remote state. These proxies always return the **current** state, not a snapshot.

```python
# This is a proxy - it always reads current geometry
geo = node.geometry()

# If you cook the node, geo now returns different data!
node.cook(force=True)
positions = geo.pointFloatAttribValues('P')  # Returns NEW positions
```

**Solution**: Extract data to local Python objects (numpy arrays, lists) immediately:

```python
positions = np.array(geo.pointFloatAttribValues('P'))  # Local copy
```

See [RPC Gotchas](rpc-gotchas.md) for more details.

### Session Discovery

Multiple Houdini instances can run RPC servers on different ports:

```python
# Scan for active sessions
sessions = mcp__houdini__list_sessions()
# Returns: [{"port": 18811, "hip_file": "scene.hip", ...}]
```

### Multi-Agent Session Isolation

**CRITICAL:** When multiple agents run simultaneously, each has its own isolated session.

- Each agent launch creates: `.claude/sessions/{sessionId}.json`
- Agents must connect ONLY to ports from their own session file
- The MCP server is shared - always reconnect before operation batches
- `current-session.json` is deprecated - do not use it

See `.claude/rules/houdini-mcp.md` for detailed session ownership rules.

## Shelf Tools

The "Claude MCP" shelf in `package/toolbar/` provides:

| Tool | Description |
|:-----|:------------|
| Start RPC | Start the RPC server on port 18811 |
| Stop RPC | Stop the RPC server |
| RPC Status | Check if server is running |
| MCP Logs | Open log viewer UI |
