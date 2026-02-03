---
layout: default
title: Schwarz Cloth
parent: Projects
nav_order: 10
description: CUDA Schwarz-based cloth solver for Houdini
permalink: /projects/schwarz/
---

# Schwarz Cloth

CUDA-accelerated cloth solver using Schwarz domain decomposition for Houdini.

---

## Overview

Schwarz Cloth implements a parallel cloth simulation algorithm using Schwarz domain decomposition. This approach enables efficient GPU parallelization while maintaining physical accuracy.

## Location

`houdini-cuda-deformers/schwarz/`

## Features

- Domain decomposition for parallel solving
- GPU-accelerated constraint projection
- Collision handling
- Integration with Houdini simulation workflow

## Related Documentation

- [CUDA Development](../../cuda/) - CUDA HDK patterns
- [VBD Solver](../vbd/) - Related CUDA cloth/soft body solver
