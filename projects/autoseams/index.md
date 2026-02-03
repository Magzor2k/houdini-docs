---
layout: default
title: Auto Seams
parent: Projects
nav_order: 11
description: Automatic UV seam detection for Houdini
permalink: /projects/autoseams/
---

# Auto Seams

Automatic UV seam detection tool for Houdini.

---

## Overview

Auto Seams analyzes mesh topology to automatically detect optimal UV seam placement. Uses curvature analysis and edge cost evaluation to find natural seam locations.

## Location

`houdini-cuda-deformers/autoseams/`

## Node

**Node Type:** `th::cuda_autoseams::1.0`

## Features

- Automatic seam detection based on curvature
- Edge cost evaluation
- Support for complex mesh topologies
- Integration with UV unwrapping workflow

## Related Documentation

- [UV Unwrap](../uvunwrap/) - Companion UV unwrapping tool
- [CUDA Development](../../cuda/) - CUDA HDK patterns
