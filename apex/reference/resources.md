---
layout: default
title: Resources
parent: Reference
grand_parent: APEX
nav_order: 6
description: APEX Script external resources and links
permalink: /apex/reference/resources/
---

# APEX Script Resources

> Houdini 21.0.559 | Last updated: 2025-12-26

## Official SideFX Documentation

### Core Documentation
- [APEX Script Language Reference](https://www.sidefx.com/docs/houdini/character/kinefx/apexscriptlanguage.html)
  - Complete syntax guide: variables, types, control flow, operators

- [Functions in APEX Script](https://www.sidefx.com/docs/houdini/character/kinefx/apexscriptfunctions.html)
  - Built-in functions, decorators, templated functions

- [APEX Script SOP](https://www.sidefx.com/docs/houdini/nodes/sop/apex--script.html)
  - Node parameters, workflow, usage

### Related Node Documentation
- [APEX Graph SOP](https://www.sidefx.com/docs/houdini/nodes/sop/apex--graph.html)
- [APEX Invoke Graph SOP](https://www.sidefx.com/docs/houdini/nodes/sop/apex--invokegraph.html)
- [APEX Autorig Component](https://www.sidefx.com/docs/houdini/nodes/sop/apex--autorigcomponent.html)
- [APEX Rigscript Component](https://www.sidefx.com/docs/houdini/nodes/sop/apex--rigscriptcomponent.html)

---

## Official Tutorials

### Effective APEX Scripting (Recommended for Beginners)
- **URL**: https://www.sidefx.com/tutorials/effective-apex-scripting/
- **Level**: Intermediate
- **Houdini Version**: 20.5
- **Author**: Cameron Skinner
- **Topics**: Squash & stretch, mesh deformation, API discovery
- **Format**: 16-part video series

### APEX Rigging Masterclass
- **URL**: https://www.sidefx.com/tutorials/apex-rigging-masterclass/
- **Level**: Masterclass
- **Author**: William Harley (SideFX)
- **Topics**: Components, subgraphs, character rigging
- **Format**: Video masterclass

---

## Community Resources

### CGWiki by Matt Estela
- **URL**: https://tokeru.com/cgwiki/HoudiniApex.html
- **Topics**: Practical examples, graph building, TransformObject, matchNodes
- **Quality**: Excellent, hands-on approach

---

## Local Examples

### Houdini Installation Examples
Located at `$HFS/houdini/help/examples/nodes/sop/`:

| Path | Description |
|------|-------------|
| `apex--graph/APEXGraphExamples.hda` | Basic numerical operations, animate state |
| `apex--autorigcomponent/NeckComponent.hda` | Neck rig component example |
| `apex--autorigcomponent/RootComponent.hda` | Root rig component |
| `apex--autorigcomponent/RigCharacter.hda` | Complete character rig |

### HDK/Toolkit Examples
Located at `$HFS/toolkit/samples/APEX/`:

| File | Description |
|------|-------------|
| `apex_external_test.C` | C++ APEX callback examples |
| `CMakeLists.txt` | Build configuration for C++ plugins |

---

## APEX C++ Development

### Header Files
Located at `$HFS/toolkit/include/APEX/`:

| Header | Purpose |
|--------|---------|
| `APEX_Include.h` | Main types (Bool, Int, Float, Vector3, Matrix4, etc.) |
| `APEX_Callback.h` | Base class for callbacks |
| `APEX_Generic.h` | Generic function templates |
| `APEX_Types.h` | Type system definitions |
| `APEX_Registry.h` | Node registration |
| `APEX_CallbackPlugin.h` | Plugin architecture |

### Scene API Headers
Located at `$HFS/toolkit/include/APEXA/`:

| Header | Purpose |
|--------|---------|
| `APEXA_SceneInvoke.h` | Scene invocation |
| `APEXA_SceneUtils.h` | Scene utilities |

---

## Tips for Learning

### Development Setup
1. Use Sublime Text with LSP/LSP-Pyright for autocomplete
2. Enable "View Log" on APEX Script SOP for debugging
3. Use "Inspect Line" parameter to highlight specific lines
4. Decompile existing components with "Convert To Snippet"

### Workflow
1. Start with the APEX Script SOP node
2. Write code in the Snippet parameter
3. Click "Reload Setup Parms" after adding BindInput
4. Connect to APEX Invoke Graph to execute
5. Check the second output for the raw graph

### Debugging
- Use `print()` for logging (appears in log viewer)
- Use `warning()` for important messages
- Use `raise error()` to halt execution with message
- Check node info window for errors

---

## Version Notes

### Houdini 21.0
- APEX Script is production-ready
- Extensive KineFX integration
- C++ callback API stable

### Houdini 20.5
- Major APEX Script improvements
- Joint tagging system
- Better component architecture

---

## Useful Houdini Commands

```python
# In Python Shell / hython
import hou

# List APEX node types
for t in hou.sopNodeTypeCategory().nodeTypes():
    if 'apex' in t.lower():
        print(t)

# Find APEX callbacks
# (Available in APEX Script autocomplete)
```
