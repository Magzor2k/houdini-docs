---
layout: default
title: APEX
nav_order: 3
has_children: true
description: APEX scripting and tools for Houdini
permalink: /apex/
---

# APEX
{: .fs-9 }

APEX Script reference and tools for Houdini 21+.
{: .fs-6 .fw-300 }

---

## Documentation Structure

| Section | Description |
|:--------|:------------|
| [Reference](reference/) | Language reference - syntax, types, functions, patterns, troubleshooting |
| [Guides](guides/) | Tutorials and walkthroughs - skeleton builder, spline rigs |

---

## Reference Highlights

| Document | Description |
|:---------|:------------|
| [Quick Reference](reference/quick-reference.md) | Syntax cheat sheet |
| [Types](reference/types.md) | Data types - primitives, vectors, matrices, arrays |
| [Functions](reference/functions.md) | Built-in functions and graph building |
| [Patterns](reference/patterns.md) | FK rigs, bone deform, spline rigs |
| [Troubleshooting](reference/troubleshooting.md) | Common errors and solutions |

## Guides Highlights

| Guide | Description |
|:------|:------------|
| [Animate State Tools](../projects/apex-tools/animate-state-tools.md) | Building custom tools for sceneanimate |
| [Spline Rig Setup](guides/spline-rig-setup.md) | Spline-based rigs with CV controls |
| [Creating APEX Subgraphs](guides/subgraph-guide.md) | Reusable subgraph packaging and deployment |

---

## Related Tools

Looking for ready-to-use APEX HDAs and interactive tools?

**[APEX Tools Project](../projects/apex-tools/)** - Collection of HDAs and tools for APEX rigging:
- **Skeleton Placer** - Interactive joint placement for sceneanimate
- **Rig Controller Placer** - Interactive control shape placement
- **Cloth Analysis** - Simulation quality metrics

See the [Animate State Tools Guide](../projects/apex-tools/animate-state-tools.md) to learn how to build your own tools.

These are production tools built ON TOP of APEX scripting (this documentation covers the underlying APEX language).

---

## Animate State Reference

| Document | Description |
|:---------|:------------|
| [Tool Events](reference/tool-events.md) | Event callbacks for animation tools |
| [Controls](reference/controls.md) | Control system architecture and API |

---

## Quick Start

```python
# Basic APEX Script graph building
graph = ApexGraphHandle()
parms = graph.addNode('parms', '__parms__')
output = graph.addNode('output', '__output__')

# Create a transform node
hip = graph.addNode('hip', 'TransformObject')
hip.t_in.set(Vector3(0.0, 1.0, 0.0))
hip.xform_out.promoteOutput('hip_xform')

# Output the graph
graph.sort(True)
geo = graph.saveToGeometry()
BindOutput(geo)
```

## Environment

- **Houdini 21.0.559**
- **APEX Script SOP** (`apex::script`)

---

## External Resources

- [SideFX APEX Script Reference](https://www.sidefx.com/docs/houdini/character/kinefx/apexscriptlanguage.html)
- [CGWiki APEX](https://tokeru.com/cgwiki/HoudiniApex.html)
