---
layout: default
title: UV Unwrap
parent: Projects
nav_order: 12
description: GPU-accelerated UV unwrapping for Houdini
permalink: /projects/uvunwrap/
---

# UV Unwrap

GPU-accelerated UV unwrapping tool for Houdini.

---

## Overview

UV Unwrap provides fast, GPU-accelerated UV unwrapping using CUDA. Supports conformal and area-preserving parameterization methods.

## Location

`houdini-cuda-deformers/uvunwrap/`

## Node

**Node Type:** `th::cuda_uvunwrap::1.0`

## Features

- GPU-accelerated unwrapping
- Conformal parameterization
- Area-preserving options
- Works with Auto Seams for complete UV workflow

## Related Documentation

- [Auto Seams](../autoseams/) - Automatic seam detection
- [CUDA Development](../../cuda/) - CUDA HDK patterns
