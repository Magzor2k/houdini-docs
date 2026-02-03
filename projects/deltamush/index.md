---
layout: default
title: Delta Mush
parent: Projects
nav_order: 9
description: CUDA-accelerated delta mush deformer for Houdini
permalink: /projects/deltamush/
---

# Delta Mush

CUDA-accelerated delta mush deformer for character skinning in Houdini.

---

## Overview

Delta Mush is a smoothing deformer that preserves detail while smoothing out linear blend skinning artifacts. This CUDA implementation provides GPU-accelerated performance for real-time playback.

## Location

`houdini-cuda-deformers/deltamush/`

## Node

**Node Type:** `th::cuda_deltamush::1.0`

## Features

- GPU-accelerated Laplacian smoothing
- Preserves high-frequency detail
- Real-time performance on GPU
- Compatible with standard Houdini skinning workflow

## Related Documentation

- [CUDA Development](../../cuda/) - CUDA HDK patterns
- [HDK Development](../../hdk/) - Building Houdini plugins
