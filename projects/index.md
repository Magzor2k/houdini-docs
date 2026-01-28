---
layout: default
title: Projects
nav_order: 8
has_children: true
description: Documentation for individual projects in this workspace
permalink: /projects/
---

# Projects

Project-specific documentation for tools and plugins in this workspace.

Each project in this repository can have its own `docs/` folder containing project-specific documentation. These are automatically discovered and integrated into this documentation site.

---

## Available Projects

| Project | Description |
|:--------|:------------|
| [Karma Procedural](karma-procedural/) | Custom Karma procedural geometry using BRAY API |
| [VBD Solver](vbd/) | CUDA cloth/soft body simulation with Vertex Block Descent |
| [Hydra Procedural](hydra-procedural/) | C++ Hydra 2.0 Generative Procedural |
| [APEX Script](apex-script/) | APEX scripting examples and tutorials |
| [APEX Tools](apex-tools/) | Utility HDAs and tools for APEX rigging |
| [APEX USD Nodes](apex-usd-nodes/) | Custom APEX callback nodes for USD generation |
| [APEX Procedural](apex-procedural/) | Python USD procedural using APEX graphs |
| [TumbleheadRig](TumbleheadRig/) | Tumblehead APEX rigging tools and HDAs |

---

## Adding Documentation

All project documentation now lives in this `docs/projects/` folder. To add documentation for a new project, create a folder here (e.g., `docs/projects/my-project/`) with an `index.md` file.

### Project Docs Structure

```
your-project/
├── project.json
├── docs/
│   ├── index.md          # Required: Project overview
│   ├── getting-started.md
│   ├── reference/
│   │   └── api.md
│   └── guides/
│       └── tutorial.md
└── ...
```

### Front Matter Template

For your project's `docs/index.md`:

```yaml
---
layout: default
title: Your Project Name
parent: Projects
has_children: true
description: Brief description (under 100 chars)
---
```

For child pages:

```yaml
---
layout: default
title: Page Title
parent: Your Project Name
grand_parent: Projects
---
```
