---
layout: default
title: OpenCL
nav_order: 6
has_children: true
description: OpenCL development for Houdini
permalink: /opencl/
---

# OpenCL in Houdini

OpenCL integration for GPU-accelerated geometry processing in Houdini SOPs.

---

## Quick Topic Lookup

| Looking for... | Go to |
|:---------------|:------|
| Learning OpenCL basics | [learning_resources.md](learning_resources.md) |
| Binding attributes | [binding_guide.md](binding_guide.md) |

---

## Documentation

| Section | Description |
|:--------|:------------|
| [Learning Resources](learning_resources.md) | Tutorials and references for OpenCL |
| [Binding Guide](binding_guide.md) | How to bind Houdini attributes to OpenCL |

---

## What is OpenCL in Houdini?

OpenCL (Open Computing Language) allows you to write kernels that run on the GPU for parallel computation. In Houdini, OpenCL SOPs let you:

- Process geometry in parallel on the GPU
- Perform complex calculations faster than VEX
- Access point, primitive, and vertex attributes
- Write custom deformers and simulations

---

## Basic OpenCL SOP Workflow

1. Add an OpenCL SOP to your network
2. Write your kernel code
3. Bind input/output attributes
4. Configure work groups

---

## Environment

- **Houdini**: 21.0.559
- **OpenCL**: 1.2+
