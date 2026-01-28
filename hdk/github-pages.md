---
layout: default
title: GitHub Pages
parent: HDK
nav_order: 6
description: Publishing documentation to GitHub Pages
permalink: /hdk/github-pages/
---

# Publishing Documentation to GitHub Pages
{: .fs-9 }

How to set up Jekyll-based documentation with hierarchical navigation.
{: .fs-6 .fw-300 }

---

## Overview

This guide covers publishing documentation to GitHub Pages using Jekyll with the "Just the Docs" theme. It focuses on:

- Setting up hierarchical navigation (parent/child/grandchild pages)
- Automating deployment with batch scripts
- Front matter patterns for multi-level navigation

---

## Jekyll Front Matter Basics

Every markdown file needs YAML front matter between `---` markers:

```yaml
---
layout: default
title: Page Title
nav_order: 1
---
```

### Key Attributes

| Attribute | Purpose |
|:----------|:--------|
| `layout` | Page template (`default` for content pages, `home` for landing) |
| `title` | Display name in navigation |
| `nav_order` | Sort order (lower = higher in list) |
| `has_children` | Set to `true` if page has child pages |
| `parent` | Title of parent page (must match exactly) |
| `grand_parent` | Title of grandparent page (for 3-level hierarchy) |

---

## Navigation Hierarchy Patterns

### Two-Level Hierarchy (Parent → Children)

**Parent page** (`hdk/index.md`):
```yaml
---
layout: default
title: HDK
nav_order: 4
has_children: true
---
```

**Child page** (`hdk/cmake-setup.md`):
```yaml
---
layout: default
title: CMake Setup
parent: HDK
nav_order: 1
---
```

Result:
```
HDK
├── CMake Setup
├── SOP Node Creation
└── ...
```

### Three-Level Hierarchy (Grandparent → Parent → Children)

**Grandparent page** (`apex/index.md`):
```yaml
---
layout: default
title: APEX
nav_order: 2
has_children: true
---
```

**Parent page** (`apex/script/index.md`):
```yaml
---
layout: default
title: Script
parent: APEX
nav_order: 1
has_children: true
---
```

**Child page** (`apex/script/functions.md`):
```yaml
---
layout: default
title: Functions
parent: Script
grand_parent: APEX
nav_order: 5
---
```

Result:
```
APEX
├── Script
│   ├── Reference
│   ├── Functions
│   └── ...
└── Tools
    └── Skeleton Builder
```

### Critical Rules

1. **`parent` must match `title` exactly** - Case-sensitive string match
2. **`grand_parent` required for 3-level** - Without it, page appears at wrong level
3. **Each level needs `has_children: true`** - Or children won't nest properly
4. **Front matter must be valid YAML** - Each attribute on its own line

---

## Batch Script for Deployment

### Basic Structure

```batch
@echo off
setlocal enabledelayedexpansion

set "STAGING=%TEMP%\docs-staging"
set "GITHUB_REPO=https://github.com/user/repo.git"

:: Clean staging
if exist "%STAGING%" rmdir /s /q "%STAGING%"
mkdir "%STAGING%"

:: Copy docs
xcopy "source\docs\*.md" "%STAGING%\section\" /s /q

:: Create _config.yml
:: ... (see below)

:: Update front matter
:: ... (see below)

:: Git push
cd /d "%STAGING%"
git init
git add -A
git commit -m "Update documentation"
git remote add origin %GITHUB_REPO%
git branch -M main
git push -u origin main --force

endlocal
```

### Creating Jekyll Config

```batch
(
echo remote_theme: just-the-docs/just-the-docs
echo title: My Docs
echo color_scheme: dark
echo search_enabled: true
echo nav_sort: case_insensitive
) > "%STAGING%\_config.yml"
```

### Creating Index Pages with Batch

Use `echo` with proper escaping for special characters:

```batch
(
echo ---
echo layout: default
echo title: Section Name
echo parent: Parent Name
echo nav_order: 1
echo has_children: true
echo ---
echo.
echo # Section Title
echo {: .fs-9 }
echo.
echo Description here.
echo {: .fs-6 .fw-300 }
echo.
echo ---
echo.
echo ^| Column 1 ^| Column 2 ^|
echo ^|:--------^|:--------^|
echo ^| [Link](page.html^) ^| Description ^|
) > "%STAGING%\section\index.md"
```

**Escaping rules:**
- `^|` for pipe characters in tables
- `^)` for closing parentheses in links
- `echo.` for blank lines

---

## PowerShell Front Matter Injection

The tricky part is updating existing files to add `parent` and `grand_parent` attributes with proper newlines.

### The Problem

Simple string replacement puts everything on one line:
```yaml
---
parent: Script grand_parent: APEX  # WRONG - renders as visible text!
layout: default
```

### The Solution

Use regex capture groups to preserve original newlines:

```powershell
# Pattern explanation:
# ^---(\r?\n)  - Match opening --- and capture the newline
# $1           - Reuse captured newline in replacement

$content = $content -replace '^---(\r?\n)', "---`$1parent: Script`$1grand_parent: APEX`$1"
```

### Complete PowerShell Command (for batch file)

```batch
powershell -ExecutionPolicy Bypass -Command "Get-ChildItem '%STAGING%\section\*.md' | Where-Object { $_.Name -ne 'index.md' } | ForEach-Object { $c = Get-Content $_.FullName -Raw; $c = $c -replace '(\r?\n)parent:[^\r\n]*', ''; $c = $c -replace '(\r?\n)grand_parent:[^\r\n]*', ''; $c = $c -replace '^---(\r?\n)', \"---`$1parent: ParentTitle`$1grand_parent: GrandparentTitle`$1\"; Set-Content $_.FullName $c -NoNewline }"
```

**What this does:**
1. Find all `.md` files except `index.md`
2. Read file content as raw string
3. Remove any existing `parent:` lines
4. Remove any existing `grand_parent:` lines
5. Insert new parent/grand_parent after opening `---`
6. Write back without adding extra newline

### For Two-Level Hierarchy (no grand_parent)

```batch
powershell -ExecutionPolicy Bypass -Command "Get-ChildItem '%STAGING%\section\*.md' | Where-Object { $_.Name -ne 'index.md' } | ForEach-Object { $c = Get-Content $_.FullName -Raw; $c = $c -replace '(\r?\n)parent:[^\r\n]*', ''; $c = $c -replace '^---(\r?\n)', \"---`$1parent: ParentTitle`$1\"; Set-Content $_.FullName $c -NoNewline }"
```

---

## Common Issues

### Navigation Shows Flat Instead of Nested

**Cause:** Missing or incorrect `parent`/`grand_parent` attributes.

**Fix:** Ensure:
- Parent page has `has_children: true`
- Child page `parent:` matches parent's `title:` exactly
- For 3-level, child has both `parent:` and `grand_parent:`

### Front Matter Appears as Visible Text

**Cause:** Attributes on same line or malformed YAML.

**Wrong:**
```yaml
---
parent: Script grand_parent: APEX
```

**Correct:**
```yaml
---
parent: Script
grand_parent: APEX
```

**Fix:** Use PowerShell regex with captured newlines (see above).

### Pages in Wrong Order

**Cause:** Missing or conflicting `nav_order` values.

**Fix:** Assign unique `nav_order` to each page at same level.

### Changes Not Appearing

**Cause:** GitHub Pages cache or build delay.

**Fix:**
- Wait 2-5 minutes after push
- Check Actions tab for build status
- Hard refresh browser (Ctrl+Shift+R)

---

## Directory Structure Example

```
docs-staging/
├── _config.yml
├── index.md                 # Home page
├── apex/
│   ├── index.md             # APEX parent (has_children: true)
│   ├── script/
│   │   ├── index.md         # Script parent (parent: APEX, has_children: true)
│   │   ├── reference.md     # Child (parent: Script, grand_parent: APEX)
│   │   └── functions.md     # Child (parent: Script, grand_parent: APEX)
│   └── tools/
│       ├── index.md         # Tools parent (parent: APEX, has_children: true)
│       └── skeleton.md      # Child (parent: Tools, grand_parent: APEX)
├── hdk/
│   ├── index.md             # HDK parent (has_children: true)
│   ├── cmake-setup.md       # Child (parent: HDK)
│   └── sop-node.md          # Child (parent: HDK)
└── gpu/
    ├── index.md             # GPU parent (has_children: true)
    ├── cuda/
    │   ├── index.md         # CUDA parent (parent: GPU, has_children: true)
    │   └── kernels.md       # Child (parent: CUDA, grand_parent: GPU)
    └── opencl/
        ├── index.md         # OpenCL parent (parent: GPU, has_children: true)
        └── bindings.md      # Child (parent: OpenCL, grand_parent: GPU)
```

---

## Reference

- [Just the Docs Navigation](https://just-the-docs.github.io/just-the-docs/docs/navigation-structure/)
- [Jekyll Front Matter](https://jekyllrb.com/docs/front-matter/)
- [GitHub Pages Documentation](https://docs.github.com/en/pages)
