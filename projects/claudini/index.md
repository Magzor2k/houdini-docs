---
layout: default
title: Claudini
parent: Projects
nav_order: 13
description: Claude Code integration plugin for Houdini
permalink: /projects/claudini/
---

# Claudini

Claude Code integration plugin for SideFX Houdini - MCP servers, session management, and workflow automation.

---

## Overview

Claudini provides seamless integration between Claude Code and Houdini, enabling AI-assisted Houdini development workflows.

## Location

`claudini/` (git submodule)

## Components

### MCP Server

The Houdini MCP server enables Claude to:
- Connect to running Houdini sessions via RPC
- Create, modify, and query nodes
- Set parameters and cook nodes
- Capture viewport screenshots
- Execute Python code in Houdini

### Session Manager

The Session Manager daemon coordinates Houdini sessions:
- Tracks active Houdini instances
- Manages session ownership for multi-agent workflows
- Coordinates build operations (DLL compilation)
- Auto-saves work before session termination

## Related Documentation

- [MCP/RPC Documentation](../../mcp/) - RPC patterns and gotchas
- [MCP RPC Gotchas](../../mcp/rpc-gotchas/) - Common pitfalls

## Skills

| Skill | Description |
|:------|:------------|
| `/launch-houdini` | Launch Houdini with session tracking |
| `/kill-houdini` | Gracefully terminate sessions |
| `/hou-status` | Check MCP connection status |
| `/session-manager` | Start/check Session Manager daemon |
