# Documentation Submission Guide

This folder is the staging area for new documentation created by agents.

## How to Submit Documentation

1. Create your `.md` file with proper frontmatter (see `templates/`)
2. Add `target_path` to frontmatter specifying destination
3. Save to `docs/_submissions/pending/your-file.md`
4. The docs agent will validate and integrate

## Required Frontmatter

```yaml
---
layout: default
title: Your Page Title
parent: Parent Section Name    # Must match existing section exactly
nav_order: 5                   # Check existing files to avoid conflicts
description: Single line description under 100 characters
target_path: apex/my-new-doc.md  # Where this should go in docs/
---
```

## Valid Parent Values

| Parent | For docs in |
|:-------|:------------|
| `APEX` | `docs/apex/` |
| `CUDA` | `docs/cuda/` |
| `HDK` | `docs/hdk/` |
| `OpenCL` | `docs/opencl/` |
| `Viewer States` | `docs/viewer-states/` |
| `Reference` | `docs/viewer-states/reference/` (also needs `grand_parent: Viewer States`) |
| `Guides` | `docs/viewer-states/guides/` (also needs `grand_parent: Viewer States`) |

## File Naming

- Use **kebab-case**: `my-new-feature.md`
- Be descriptive: `cuda-memory-pools.md` not `memory.md`
- No spaces or underscores

## Content Guidelines

Your documentation should include:

1. **Clear title** matching the frontmatter `title`
2. **Brief introduction** - what this page covers
3. **Sections with headers** - use `##` and `###`
4. **Code examples** - with language specified (```python, ```cpp, etc.)
5. **See Also section** - links to related documentation

## Example Submission

```markdown
---
layout: default
title: Memory Pools
parent: CUDA
nav_order: 5
description: GPU memory pool management for efficient allocation
target_path: cuda/memory-pools.md
---

# Memory Pools

This guide covers GPU memory pool management for efficient allocation in CUDA.

---

## Overview

Memory pools reduce allocation overhead by...

## Usage

```cpp
// Code example here
```

## See Also

- [Memory Management](memory-management.md) - General GPU memory concepts
- [Bridge Pattern](bridge-pattern.md) - CPU to GPU data transfer
```

## What Happens Next

1. Run `/integrate-docs` to process pending submissions
2. Valid files are moved to their target location
3. Invalid files get a `.errors` file explaining issues
4. Parent index.md is updated with link to new page

## Validation Errors

If your submission has issues, a `.errors` file will be created. Common issues:

- Missing required frontmatter fields
- Invalid parent name (must match exactly)
- Target directory doesn't exist
- Filename not in kebab-case
