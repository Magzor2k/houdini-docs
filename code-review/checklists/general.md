---
layout: default
title: General Checklist
parent: Code Review
nav_order: 1
description: Universal code review checklist applicable to all code types
permalink: /code-review/checklists/general/
---

# General Code Review Checklist

Apply these checks to all code regardless of language or domain.

## Design

| Check | Description |
|:------|:------------|
| Single Responsibility | Each function/class does one thing well |
| Clear Purpose | Intent is obvious from structure and naming |
| Appropriate Abstraction | Not over-engineered, not under-abstracted |
| Loose Coupling | Components can change independently |
| YAGNI | No speculative features or unused code |

### Questions to Ask
- Could another developer understand this in 5 minutes?
- Is there a simpler way to achieve the same result?
- Would removing any part break the core functionality?

## Naming

| Check | Description |
|:------|:------------|
| Descriptive Names | Names reveal intent (`getUserById` not `get`) |
| Consistent Conventions | Same style throughout (camelCase, snake_case) |
| No Abbreviations | Avoid cryptic shorthand (`idx` → `index`) |
| Accurate Names | Name matches what the code actually does |
| Searchable | Can grep for it meaningfully |

### Red Flags
- Single-letter variables (except loop counters)
- Generic names: `data`, `temp`, `result`, `item`
- Misleading names that don't match behavior

## Error Handling

| Check | Description |
|:------|:------------|
| Explicit Handling | Errors caught and handled, not ignored |
| Meaningful Messages | Error messages help diagnose the issue |
| Graceful Degradation | Failures don't crash the system |
| Resource Cleanup | Resources released even on error paths |
| No Silent Failures | Errors logged or reported, never swallowed |

### Anti-Patterns
- Empty catch blocks
- Catching all exceptions generically
- Returning null/None instead of throwing
- Error messages that don't help ("Error occurred")

## Security

| Check | Description |
|:------|:------------|
| Input Validation | All external input validated and sanitized |
| No Hardcoded Secrets | No passwords, API keys, or tokens in code |
| Principle of Least Privilege | Minimal permissions requested |
| Safe Defaults | Secure by default, explicit opt-in for risky |
| Dependency Safety | No known vulnerabilities in dependencies |

### Critical Checks
- [ ] User input not directly used in file paths
- [ ] User input not directly used in SQL/queries
- [ ] Credentials loaded from environment/config, not hardcoded
- [ ] Sensitive data not logged

## Documentation

| Check | Description |
|:------|:------------|
| Why, Not What | Comments explain reasoning, not mechanics |
| Updated Comments | Comments match current code behavior |
| Public API Docs | Public functions have clear documentation |
| Complex Logic Explained | Non-obvious algorithms have explanations |
| No Redundant Comments | Code is self-documenting where possible |

### Good Comment Examples
```python
# Use binary search because dataset is always sorted and can be 10M+ items
# Retry with exponential backoff to handle transient network failures
# This offset compensates for the header row in the CSV
```

### Bad Comment Examples
```python
# Increment counter
# Get the user
# Loop through items
```

## Testing

| Check | Description |
|:------|:------------|
| New Code Tested | New functionality has corresponding tests |
| Edge Cases Covered | Boundary conditions and error paths tested |
| Tests Are Isolated | Tests don't depend on each other |
| Meaningful Assertions | Tests verify actual behavior, not implementation |
| No Flaky Tests | Tests pass consistently |

### Coverage Guidelines
- Happy path: Always tested
- Error paths: Tested for critical failures
- Edge cases: Null/empty inputs, boundary values
- Integration points: External API calls, file I/O

## Complexity

| Check | Description |
|:------|:------------|
| Appropriate Length | Functions under ~50 lines, files under ~500 lines |
| Limited Nesting | Max 3-4 levels of nesting |
| Few Parameters | Functions take 0-4 parameters |
| Low Cyclomatic Complexity | Few branches per function |
| DRY Applied Judiciously | Duplication removed when it aids clarity |

### Simplification Strategies
- Extract method for repeated logic
- Early returns to reduce nesting
- Replace conditionals with polymorphism
- Use guard clauses at function start

## Summary Checklist

Quick reference for code review:

- [ ] **Design**: Single responsibility, clear purpose
- [ ] **Naming**: Descriptive, consistent conventions
- [ ] **Error Handling**: Graceful failures, meaningful messages
- [ ] **Security**: Input validation, no hardcoded secrets
- [ ] **Documentation**: Comments explain "why", not "what"
- [ ] **Testing**: Coverage for new functionality
- [ ] **Complexity**: Could be simpler without losing functionality
