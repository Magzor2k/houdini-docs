---
layout: default
title: Dialog-Free Startup
parent: Guides
grand_parent: Viewer States
nav_order: 3
description: Environment variables to suppress Houdini startup dialogs for automated testing
permalink: /viewer-states/guides/dialog-free-startup/
---

# Dialog-Free Startup

Environment variables to suppress Houdini startup dialogs, enabling fully automated testing of viewer states.

---

## Overview

When launching Houdini for automated testing (e.g., via scripts or Claude), startup dialogs block execution and require manual interaction. Setting these environment variables before launch suppresses all popups.

---

## Environment Variables

| Variable | Value | Suppresses |
|:---------|:------|:-----------|
| `HOUDINI_NO_SPLASH` | `1` | Splash screen |
| `HOUDINI_NO_START_PAGE_SPLASH` | `1` | "Start Here" page + statistics opt-in dialog |
| `HOUDINI_LMINFO_VERBOSE` | `0` | License expiration warnings (popup + terminal) |
| `HOUDINI_ANONYMOUS_STATISTICS` | `0` | Usage statistics collection |
| `HOUDINI_DISABLE_CONSOLE` | `1` | Floating console window (Windows) |

### Additional Variables

| Variable | Value | Effect |
|:---------|:------|:-------|
| `HOUDINI_NOHKEY` | `1` | Prevents license administrator from launching on failure |
| `HOUDINI_PROMPT_ON_CRASHES` | `0` | Silent crash reporting (no dialog) |
| `HOUDINI_DISABLE_FILE_LOAD_WARNINGS` | `1` | Suppresses file load warning dialogs |

---

## Windows Examples

### Batch Script

```batch
@echo off
set HOUDINI_NO_SPLASH=1
set HOUDINI_NO_START_PAGE_SPLASH=1
set HOUDINI_LMINFO_VERBOSE=0
set HOUDINI_ANONYMOUS_STATISTICS=0
set HOUDINI_DISABLE_CONSOLE=1

"C:\Program Files\Side Effects Software\Houdini 21.0.599\bin\houdini.exe" -waitforui test_script.py
```

### PowerShell

```powershell
$env:HOUDINI_NO_SPLASH = "1"
$env:HOUDINI_NO_START_PAGE_SPLASH = "1"
$env:HOUDINI_LMINFO_VERBOSE = "0"
$env:HOUDINI_ANONYMOUS_STATISTICS = "0"
$env:HOUDINI_DISABLE_CONSOLE = "1"

& "C:\Program Files\Side Effects Software\Houdini 21.0.599\bin\houdini.exe" -waitforui test_script.py
```

---

## Combined with -waitforui

For fully automated viewer state testing:

1. Set environment variables to suppress dialogs
2. Use `-waitforui` to ensure `hou.ui` is available
3. Pass your test script after the flag

This enables Houdini to launch, run your test, and (optionally) exit without any manual intervention.

---

## See Also

- [Testing](testing.md) - General testing and debugging strategies
- [SideFX Environment Variables](https://www.sidefx.com/docs/houdini/ref/env.html) - Complete reference
