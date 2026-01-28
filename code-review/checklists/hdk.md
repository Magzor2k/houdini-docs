---
layout: default
title: HDK Checklist
parent: Code Review
nav_order: 4
description: HDK/C++ code review checklist for Houdini SOP nodes and plugins
permalink: /code-review/checklists/hdk/
---

# HDK Code Review Checklist

Apply these checks to HDK C++ code including SOP nodes, custom operators, and verb implementations.

## Memory Safety

| Check | Description |
|:------|:------------|
| RAII Used | Resources managed by constructors/destructors |
| Smart Pointers | `UT_UniquePtr`, `UT_SharedPtr` for ownership |
| No Raw `new`/`delete` | Avoid manual memory management |
| Buffer Overflows Prevented | Array bounds always checked |
| No Dangling Pointers | Pointers nulled after delete |

### RAII Patterns
```cpp
// Good - RAII with Houdini types
void processGeometry(GEO_Detail* gdp) {
    // GA_RWHandleF manages attribute lifecycle
    GA_RWHandleF pos_h(gdp->findFloatTuple(GA_ATTRIB_POINT, "P", 3));
    if (!pos_h.isValid()) return;

    // Handle automatically cleaned up at scope exit
}

// Good - smart pointers
UT_UniquePtr<GEO_Detail> temp_gdp(new GEO_Detail());

// Bad - manual memory management
GEO_Detail* temp = new GEO_Detail();
// ... code that might throw ...
delete temp;  // Never reached on exception
```

### Safe Array Access
```cpp
// Good - bounds checked
GA_Offset ptoff;
GA_FOR_ALL_PTOFF(gdp, ptoff) {
    // ptoff is always valid within loop
    pos_h.set(ptoff, value);
}

// Bad - unchecked index
for (int i = 0; i < num_points; i++) {
    GA_Offset ptoff = gdp->pointOffset(i);  // Assumes valid index
}
```

## Thread Safety (Verb Cooking)

| Check | Description |
|:------|:------------|
| No Static Mutable State | Verbs must be stateless |
| Local Variables Only | Don't store state between cooks |
| Thread-Local Storage | Use `tbb::enumerable_thread_specific` if needed |
| Const Correctness | Mark read-only access as const |
| No Global Modification | Don't modify shared data structures |

### Thread-Safe Verb Pattern
```cpp
class SOP_MyNodeVerb : public SOP_NodeVerb {
public:
    // MUST be stateless - no mutable member variables
    void cook(const CookParms& cookparms) const override {
        // All state is local to this function
        const GEO_Detail* input = cookparms.inputGeo(0);
        GEO_Detail* output = cookparms.gdh().gdpNC();

        // Process without storing state
        processGeometry(input, output, cookparms);
    }

private:
    // OK: const helpers
    void processGeometry(const GEO_Detail* in,
                        GEO_Detail* out,
                        const CookParms& parms) const;
};
```

### Thread-Local Storage
```cpp
// For caching expensive computations per-thread
#include <tbb/enumerable_thread_specific.h>

tbb::enumerable_thread_specific<MyCache> thread_cache;

void cook(const CookParms& parms) const override {
    MyCache& cache = thread_cache.local();
    // Use thread-local cache
}
```

## Parameter Handling

| Check | Description |
|:------|:------------|
| Defaults Set | All parameters have sensible defaults |
| Validation Done | Parameter values validated before use |
| Ranges Specified | Min/max for numeric parameters |
| Help Strings | Parameters have documentation |
| Conditional Visibility | Hidden parameters use conditionals |

### Parameter Definition
```cpp
static PRM_Name prm_names[] = {
    PRM_Name("iterations", "Iterations"),
    PRM_Name("strength", "Strength"),
    PRM_Name(0)
};

static PRM_Default prm_defaults[] = {
    PRM_Default(10),      // iterations default
    PRM_Default(1.0f),    // strength default
};

static PRM_Range prm_ranges[] = {
    PRM_Range(PRM_RANGE_RESTRICTED, 1, PRM_RANGE_UI, 100),  // iterations: 1-100
    PRM_Range(PRM_RANGE_RESTRICTED, 0, PRM_RANGE_UI, 2),    // strength: 0-2
};

PRM_Template SOP_MyNode::myTemplateList[] = {
    PRM_Template(PRM_INT, 1, &prm_names[0], &prm_defaults[0], 0, &prm_ranges[0]),
    PRM_Template(PRM_FLT, 1, &prm_names[1], &prm_defaults[1], 0, &prm_ranges[1]),
    PRM_Template()
};
```

### Parameter Validation
```cpp
void cook(const CookParms& parms) const override {
    int iterations = parms.evalInt("iterations", 0, 0);
    float strength = parms.evalFloat("strength", 0, 0);

    // Validate even with PRM_RANGE
    iterations = SYSmax(1, iterations);
    strength = SYSclamp(strength, 0.0f, 10.0f);

    // Now safe to use
}
```

## Error Reporting

| Check | Description |
|:------|:------------|
| Errors Reported | Use `addError()` for failures |
| Warnings Appropriate | Use `addWarning()` for non-fatal issues |
| Messages Helpful | Error text explains the problem |
| No Silent Failures | All error paths report something |
| Error Codes Used | Standard SOP error codes preferred |

### Error Reporting Patterns
```cpp
void cook(const CookParms& parms) const override {
    const GEO_Detail* input = parms.inputGeo(0);

    // Error for critical failures
    if (!input) {
        parms.sopAddError(SOP_MESSAGE, "Missing input geometry");
        return;
    }

    // Warning for recoverable issues
    if (input->getNumPoints() == 0) {
        parms.sopAddWarning(SOP_MESSAGE, "Input has no points");
        return;
    }

    GA_ROHandleV3 pos_h(input->findFloatTuple(GA_ATTRIB_POINT, "P", 3));
    if (!pos_h.isValid()) {
        parms.sopAddError(SOP_ATTRIBUTE_INVALID, "P");
        return;
    }
}
```

### Standard Error Codes
```cpp
// Common SOP error codes
SOP_MESSAGE          // Generic message
SOP_ATTRIBUTE_INVALID // Attribute not found or wrong type
SOP_ERR_FILEGEO      // File I/O error
SOP_ERR_BADGROUP     // Invalid group name
```

## Performance

| Check | Description |
|:------|:------------|
| References Used | Pass large objects by reference |
| Copies Avoided | Don't copy when not needed |
| Move Semantics | Use std::move for transfers |
| Cache Attributes | Don't re-lookup attributes in loops |
| Parallel Ready | Use UTparallelFor where applicable |

### Efficient Patterns
```cpp
// Good - cached attribute handle
GA_RWHandleV3 pos_h(gdp->findFloatTuple(GA_ATTRIB_POINT, "P", 3));
GA_FOR_ALL_PTOFF(gdp, ptoff) {
    UT_Vector3 pos = pos_h.get(ptoff);
    pos_h.set(ptoff, transform(pos));
}

// Bad - lookup in loop
GA_FOR_ALL_PTOFF(gdp, ptoff) {
    GA_RWHandleV3 pos_h(gdp->findFloatTuple(GA_ATTRIB_POINT, "P", 3));  // SLOW
    ...
}
```

### Parallel Processing
```cpp
#include <UT/UT_ParallelUtil.h>

void cook(const CookParms& parms) const override {
    GEO_Detail* gdp = parms.gdh().gdpNC();
    GA_RWHandleV3 pos_h(gdp->findFloatTuple(GA_ATTRIB_POINT, "P", 3));

    // Process points in parallel
    UTparallelFor(GA_SplittableRange(gdp->getPointRange()),
        [&](const GA_SplittableRange& range) {
            GA_Offset start, end;
            for (GA_Iterator it(range); it.blockAdvance(start, end);) {
                for (GA_Offset ptoff = start; ptoff < end; ++ptoff) {
                    UT_Vector3 pos = pos_h.get(ptoff);
                    pos_h.set(ptoff, processPoint(pos));
                }
            }
        }
    );
}
```

## Houdini API Usage

| Check | Description |
|:------|:------------|
| GA Offsets Correct | Use GA_Offset, not indices |
| Attribute Types Right | Match storage type to data |
| Groups Handled | Respect point/prim groups |
| Detail Copying | Copy attributes when duplicating |
| Time Handling | Use fpreal for time, frames |

### GA_Offset vs Index
```cpp
// Good - GA_Offset for element access
GA_Offset ptoff = gdp->pointOffset(point_index);
UT_Vector3 pos = pos_h.get(ptoff);

// Good - iterate with offset
GA_FOR_ALL_PTOFF(gdp, ptoff) {
    // ptoff is GA_Offset
}

// Bad - treating offset as index
for (GA_Offset i = 0; i < gdp->getNumPoints(); i++) {
    // WRONG: offset != index when elements deleted
}
```

### Attribute Creation
```cpp
// Create point attribute
GA_Attribute* attr = gdp->addFloatTuple(
    GA_ATTRIB_POINT,    // Class
    "myattr",           // Name
    3,                  // Tuple size
    GA_Defaults(0.0f)   // Default value
);

// With type info
GA_RWHandleV3 handle(gdp->addFloatTuple(
    GA_ATTRIB_POINT, "Cd", 3, GA_Defaults(1.0f)
));
handle.getAttribute()->setTypeInfo(GA_TYPE_COLOR);
```

## Input/Output Handling

| Check | Description |
|:------|:------------|
| Inputs Validated | Check all required inputs exist |
| Output Initialized | Clear or initialize output properly |
| Pass-Through Working | Non-modified inputs passed through |
| Cooking Efficient | Only process what's needed |

### Input Validation Pattern
```cpp
void cook(const CookParms& parms) const override {
    // Check required inputs
    const GEO_Detail* input0 = parms.inputGeo(0);
    if (!input0) {
        parms.sopAddError(SOP_MESSAGE, "First input required");
        return;
    }

    // Optional input
    const GEO_Detail* input1 = parms.inputGeo(1);  // May be null

    GEO_Detail* output = parms.gdh().gdpNC();

    // Copy input to output
    output->replaceWith(*input0);

    // Modify output
    processGeometry(output, input1);
}
```

## Summary Checklist

Quick reference for HDK review:

- [ ] **Memory Safety**: No leaks, proper RAII
- [ ] **Thread Safety**: Verb cooking is thread-safe
- [ ] **Parameter Handling**: Defaults, validation, ranges
- [ ] **Error Reporting**: addError/addWarning usage
- [ ] **Performance**: Avoid unnecessary copies, use references
- [ ] **Houdini API**: Proper GA_Offset, GEO_Detail usage
- [ ] **Input/Output**: Proper validation and initialization
