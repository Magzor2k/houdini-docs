# Documentation Agent Instructions

You are working in the Houdini R&D documentation folder. This documentation is **optimized for LLM navigation**.

## Your Role

When reviewing or integrating docs, ensure they follow these principles for LLM-friendly documentation.

## LLM-Friendly Documentation Principles

1. **Topic Lookup Tables** - Every index.md should have a "Looking for X? → Go to Y" table
2. **Single Source of Truth** - Each concept lives in ONE place; others link to it
3. **Descriptive Titles** - Titles should answer "what will I learn here?"
4. **Frontmatter Descriptions** - Every file needs a `description` field for search/discovery
5. **Cross-References** - Link related topics so LLMs can navigate between them

## When Integrating New Docs

Check `docs/_submissions/pending/` for new documentation files. For each file:

1. **Validate frontmatter** has required fields:
   - `layout: default`
   - `title:` (display name)
   - `parent:` (must match existing section exactly)
   - `nav_order:` (check for conflicts with existing files)
   - `description:` (under 100 characters)
   - `target_path:` (where to move the file)

2. **Verify target section exists** - The parent folder must already exist

3. **Check nav_order doesn't conflict** - Read existing files in target folder

4. **Move file to target location** - Remove `target_path` field from frontmatter

5. **Update parent index.md** - Add link to new page in the navigation table

6. **Update docs/index.md** - Add to topic lookup table if it's a major topic

## Quality Checklist

Before integrating, ensure the doc:

- [ ] Has clear section headers (`##`, `###`)
- [ ] Includes code examples where appropriate
- [ ] Uses tables for quick reference data
- [ ] Links to related documentation
- [ ] Follows kebab-case filename convention
- [ ] Has description under 100 characters
- [ ] Content matches the `target_path` section

## Current Structure

```
docs/
├── index.md              # Main hub with topic lookup
├── _submissions/         # Staging area for new docs
│   ├── pending/          # New submissions go here
│   └── templates/        # Templates for different doc types
├── viewer-states/        # Python viewport interaction
│   ├── reference/        # API docs (state-class, events, drawables)
│   └── guides/           # How-to tutorials (hda-integration, testing)
├── apex/                 # APEX scripting and tools
├── cuda/                 # GPU/CUDA development
├── hdk/                  # Houdini Development Kit
└── opencl/               # OpenCL computing
```

## Valid Parent Values

| Parent | Target Directory | Notes |
|:-------|:-----------------|:------|
| `APEX` | `docs/apex/` | |
| `CUDA` | `docs/cuda/` | |
| `HDK` | `docs/hdk/` | |
| `OpenCL` | `docs/opencl/` | |
| `Viewer States` | `docs/viewer-states/` | |
| `Reference` | `docs/viewer-states/reference/` | Also needs `grand_parent: Viewer States` |
| `Guides` | `docs/viewer-states/guides/` | Also needs `grand_parent: Viewer States` |

## Handling Errors

If a submission has issues:

1. Create a `.errors` file alongside the submission listing the problems
2. Do NOT move invalid files to their target location
3. Report the errors so the submitting agent can fix them

Common errors:
- Missing required frontmatter fields
- Invalid parent name (doesn't match exactly)
- Target directory doesn't exist
- nav_order conflicts with existing file
- Filename not in kebab-case (has underscores or spaces)

## After Integration

1. Delete the submission file from `pending/`
2. Delete any `.validated` or `.errors` files
3. Commit the changes with a message like: "docs: integrate [filename] into [section]"
