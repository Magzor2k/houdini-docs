---
layout: default
title: Code Review
nav_order: 10
has_children: true
description: Code review checklists and guidelines for this repository
permalink: /code-review/
---

# Code Review

Domain-specific checklists for reviewing code in this repository.

## Checklists

| Checklist | Use For |
|:----------|:--------|
| [General](checklists/general.md) | All code - design, naming, error handling |
| [Python](checklists/python.md) | Python, viewer states, APEX scripts |
| [CUDA](checklists/cuda.md) | CUDA kernels and GPU code |
| [HDK](checklists/hdk.md) | HDK/C++ SOP nodes |

## Using the `/code-review` Skill

Run `/code-review` after discussing code to get an automated review:

```
/code-review                    # Summary review (default)
/code-review comprehensive      # Detailed audit with full checklist
/code-review --category security   # Force security category
```

The skill will:
1. Extract relevant files from conversation context
2. Detect code types and apply appropriate checklists
3. Generate a review report
4. Store the review in `[project]/codereview/[category]/`
5. Update the changelog at `[project]/codereview/CHANGELOG.md`

## Review Categories

Reviews are automatically categorized based on context:

| Category | Folder | Triggered By |
|:---------|:-------|:-------------|
| Security | `security/` | Security concerns, vulnerabilities |
| Performance | `performance/` | Optimization, profiling |
| Refactoring | `refactoring/` | Code cleanup, restructuring |
| Feature | `feature/` | New feature implementation |
| General | `general/` | Default catch-all |

## Storage Structure

Reviews are stored per-project with versioning:

```
[project]/codereview/
├── CHANGELOG.md          # Summary of all reviews
├── debug/                # Debug session logs
│   └── 2025-01-15_v1.md
├── security/
│   └── 2025-01-15_v1.md
├── performance/
├── refactoring/
├── feature/
└── general/
```

## Changelog

Every code review appends a summary to `CHANGELOG.md`:

```markdown
## 2025-01-15

### v1 - performance/2025-01-15_v1.md
- **Files**: VBDCore.cu (800 lines)
- **Verdict**: Pass with Notes
- **Key Issues**: Memory coalescing warning
```

This provides a quick history of all reviews for a project.

## Using the `/debug` Skill

Run `/debug` when investigating bugs. It uses review history to find root causes:

```
/debug "CUDA kernel crashes on frame 50"
/debug                              # Uses conversation context
```

The debug workflow:
1. **Zoom in**: Check recent reviews for same files, extract known issues
2. **Form hypothesis**: Correlate symptoms with past findings
3. **Zoom out**: If stuck, expand scope to all reviews, look for patterns
4. **Repeat**: Form new hypothesis, zoom back in
5. **Report**: Present findings with investigation trail
6. **Optional storage**: Save debug session if requested

Debug sessions are stored in `[project]/codereview/debug/` with the same versioning format.
