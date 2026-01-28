---
layout: default
title: Parameters
parent: HDK
nav_order: 3
description: HDK parameter setup and configuration
permalink: /hdk/parameters/
---

# Parameter Setup
{: .fs-9 }

Defining node parameters with PRM_Template.
{: .fs-6 .fw-300 }

---

## Parameter Architecture

Parameters in HDK are defined using `PRM_Template` arrays:

```
PRM_Template[] ──┬── PRM_Name     (internal name, label)
                 ├── PRM_Default  (default value)
                 ├── PRM_Range    (min/max constraints)
                 ├── PRM_Type     (int, float, toggle, etc.)
                 └── PRM_ChoiceList (dropdown menus)
```

---

## Basic Parameter Types

### Integer Parameter

```cpp
static PRM_Name iterationsName("iterations", "Iterations");
static PRM_Default iterationsDefault(10);
static PRM_Range iterationsRange(PRM_RANGE_RESTRICTED, 1, PRM_RANGE_UI, 100);

PRM_Template(
    PRM_INT_J,           // Type: integer
    1,                   // Vector size (1 = scalar)
    &iterationsName,     // Name
    &iterationsDefault,  // Default value
    nullptr,             // Choice list (none)
    &iterationsRange     // Range
)
```

### Float Parameter

```cpp
static PRM_Name strengthName("strength", "Strength");
static PRM_Default strengthDefault(1.0);
static PRM_Range strengthRange(PRM_RANGE_RESTRICTED, 0.0, PRM_RANGE_UI, 2.0);

PRM_Template(
    PRM_FLT_J,           // Type: float
    1,                   // Vector size
    &strengthName,
    &strengthDefault,
    nullptr,
    &strengthRange
)
```

### Toggle (Checkbox)

```cpp
static PRM_Name enableName("enable", "Enable");
static PRM_Default enableDefault(1);  // 1 = on, 0 = off

PRM_Template(
    PRM_TOGGLE,
    1,
    &enableName,
    &enableDefault
)
```

### Vector3 Parameter

```cpp
static PRM_Name gravityName("gravity", "Gravity");
static PRM_Default gravityDefault[] = {
    PRM_Default(0.0),   // X
    PRM_Default(-9.81), // Y
    PRM_Default(0.0)    // Z
};

PRM_Template(
    PRM_XYZ_J,           // Type: 3-component vector
    3,                   // Vector size = 3
    &gravityName,
    gravityDefault       // Array of defaults
)
```

### String Parameter

```cpp
static PRM_Name groupName("group", "Point Group");
static PRM_Default groupDefault(0, "");  // Empty string default

PRM_Template(
    PRM_STRING,
    1,
    &groupName,
    &groupDefault
)
```

---

## Range Types

The `PRM_Range` constructor takes four arguments:

```cpp
PRM_Range(minType, minValue, maxType, maxValue)
```

| Type | Description |
|:-----|:------------|
| `PRM_RANGE_RESTRICTED` | Hard limit - value cannot exceed |
| `PRM_RANGE_UI` | Soft limit - slider stops but value can exceed |
| `PRM_RANGE_FREE` | No limit on that end |

```cpp
// Hard limit 0-1
PRM_Range(PRM_RANGE_RESTRICTED, 0.0, PRM_RANGE_RESTRICTED, 1.0)

// Soft slider 0-100, but can type higher values
PRM_Range(PRM_RANGE_RESTRICTED, 0, PRM_RANGE_UI, 100)

// Positive only, no upper limit
PRM_Range(PRM_RANGE_RESTRICTED, 0.0, PRM_RANGE_FREE, 1.0)
```

---

## Complete Parameter Template Example

From a cloth solver:

```cpp
// Parameter names
static PRM_Name substepsName("substeps", "Substeps");
static PRM_Name iterationsName("iterations", "Solver Iterations");
static PRM_Name dampingName("damping", "Damping");
static PRM_Name gravityName("gravity", "Gravity");
static PRM_Name stretchStiffnessName("stretchstiffness", "Stretch Stiffness");
static PRM_Name bendStiffnessName("bendstiffness", "Bend Stiffness");
static PRM_Name enableCollisionName("enablecollision", "Enable Collision");
static PRM_Name collisionRadiusName("collisionradius", "Collision Radius");

// Parameter defaults
static PRM_Default substepsDefault(4);
static PRM_Default iterationsDefault(20);
static PRM_Default dampingDefault(0.99);
static PRM_Default gravityDefault[] = {
    PRM_Default(0.0),
    PRM_Default(-9.81),
    PRM_Default(0.0)
};
static PRM_Default stretchStiffnessDefault(1.0);
static PRM_Default bendStiffnessDefault(0.1);
static PRM_Default enableCollisionDefault(1);
static PRM_Default collisionRadiusDefault(0.01);

// Parameter ranges
static PRM_Range substepsRange(PRM_RANGE_RESTRICTED, 1, PRM_RANGE_UI, 16);
static PRM_Range iterationsRange(PRM_RANGE_RESTRICTED, 1, PRM_RANGE_UI, 100);
static PRM_Range dampingRange(PRM_RANGE_RESTRICTED, 0.0, PRM_RANGE_RESTRICTED, 1.0);
static PRM_Range stiffnessRange(PRM_RANGE_RESTRICTED, 0.0, PRM_RANGE_UI, 10.0);
static PRM_Range radiusRange(PRM_RANGE_RESTRICTED, 0.0, PRM_RANGE_UI, 0.1);

// Separators for UI organization
static PRM_Name solverSeparator("solver_sep", "Solver");
static PRM_Name materialSeparator("material_sep", "Material");
static PRM_Name collisionSeparator("collision_sep", "Collision");

// Template array
PRM_Template SOP_ClothSolver::myTemplateList[] = {
    // Solver section
    PRM_Template(PRM_SEPARATOR, 1, &solverSeparator),
    PRM_Template(PRM_INT_J, 1, &substepsName, &substepsDefault,
                 nullptr, &substepsRange),
    PRM_Template(PRM_INT_J, 1, &iterationsName, &iterationsDefault,
                 nullptr, &iterationsRange),
    PRM_Template(PRM_FLT_J, 1, &dampingName, &dampingDefault,
                 nullptr, &dampingRange),
    PRM_Template(PRM_XYZ_J, 3, &gravityName, gravityDefault),

    // Material section
    PRM_Template(PRM_SEPARATOR, 1, &materialSeparator),
    PRM_Template(PRM_FLT_J, 1, &stretchStiffnessName, &stretchStiffnessDefault,
                 nullptr, &stiffnessRange),
    PRM_Template(PRM_FLT_J, 1, &bendStiffnessName, &bendStiffnessDefault,
                 nullptr, &stiffnessRange),

    // Collision section
    PRM_Template(PRM_SEPARATOR, 1, &collisionSeparator),
    PRM_Template(PRM_TOGGLE, 1, &enableCollisionName, &enableCollisionDefault),
    PRM_Template(PRM_FLT_J, 1, &collisionRadiusName, &collisionRadiusDefault,
                 nullptr, &radiusRange),

    // Null terminator - REQUIRED
    PRM_Template()
};
```

---

## Reading Parameters in Code

### Parameter Macros

Define convenience macros in your header:

```cpp
// SOP_ClothSolver.h
private:
    // Integer parameters
    int SUBSTEPS(fpreal t) { return evalInt("substeps", 0, t); }
    int ITERATIONS(fpreal t) { return evalInt("iterations", 0, t); }

    // Float parameters
    fpreal DAMPING(fpreal t) { return evalFloat("damping", 0, t); }
    fpreal STRETCH_STIFFNESS(fpreal t) { return evalFloat("stretchstiffness", 0, t); }
    fpreal BEND_STIFFNESS(fpreal t) { return evalFloat("bendstiffness", 0, t); }
    fpreal COLLISION_RADIUS(fpreal t) { return evalFloat("collisionradius", 0, t); }

    // Toggle parameters
    bool ENABLE_COLLISION(fpreal t) { return evalInt("enablecollision", 0, t) != 0; }

    // Vector parameters
    UT_Vector3 GRAVITY(fpreal t) {
        return UT_Vector3(
            evalFloat("gravity", 0, t),
            evalFloat("gravity", 1, t),
            evalFloat("gravity", 2, t)
        );
    }
```

### Using in cookMySop

```cpp
OP_ERROR SOP_ClothSolver::cookMySop(OP_Context& context)
{
    fpreal t = context.getTime();

    // Read all parameters
    int substeps = SUBSTEPS(t);
    int iterations = ITERATIONS(t);
    float damping = DAMPING(t);
    UT_Vector3 gravity = GRAVITY(t);
    float stretchStiffness = STRETCH_STIFFNESS(t);
    float bendStiffness = BEND_STIFFNESS(t);
    bool enableCollision = ENABLE_COLLISION(t);
    float collisionRadius = COLLISION_RADIUS(t);

    // Pass to solver
    m_bridge.setParams(
        substeps, iterations, damping,
        gravity.x(), gravity.y(), gravity.z(),
        stretchStiffness, bendStiffness,
        enableCollision, collisionRadius
    );

    // ...
}
```

---

## Dropdown Menus

```cpp
// Menu items
static PRM_Name solverModeItems[] = {
    PRM_Name("jacobi", "Jacobi"),
    PRM_Name("gaussseidel", "Gauss-Seidel"),
    PRM_Name("pcg", "PCG"),
    PRM_Name(0)  // Null terminator
};

static PRM_ChoiceList solverModeMenu(PRM_CHOICELIST_SINGLE, solverModeItems);

static PRM_Name solverModeName("solvermode", "Solver Mode");
static PRM_Default solverModeDefault(0);  // First item

// In template array
PRM_Template(
    PRM_ORD,             // Ordinal (dropdown)
    1,
    &solverModeName,
    &solverModeDefault,
    &solverModeMenu      // Choice list
)
```

Reading dropdown value:

```cpp
int SOLVER_MODE(fpreal t) { return evalInt("solvermode", 0, t); }

// In cook:
int mode = SOLVER_MODE(t);
switch (mode) {
    case 0: // Jacobi
        break;
    case 1: // Gauss-Seidel
        break;
    case 2: // PCG
        break;
}
```

---

## Common Parameter Types

| Type | Description | Example Use |
|:-----|:------------|:------------|
| `PRM_INT_J` | Integer | Iterations, counts |
| `PRM_FLT_J` | Float | Strength, stiffness |
| `PRM_TOGGLE` | Checkbox | Enable/disable features |
| `PRM_XYZ_J` | Vector3 | Gravity, direction |
| `PRM_RGB_J` | Color | Visualization colors |
| `PRM_STRING` | Text | Group names, file paths |
| `PRM_ORD` | Dropdown | Mode selection |
| `PRM_SEPARATOR` | Visual separator | UI organization |
| `PRM_FILE` | File path | Input/output files |

---

## Important Notes

1. **Null Terminator Required**: Template arrays must end with `PRM_Template()`
2. **Static Storage**: PRM_Name, PRM_Default, PRM_Range must be static
3. **Parameter Names**: Use lowercase, no spaces for internal names
4. **Time Parameter**: Always pass `fpreal t` for animated parameters
5. **Vector Index**: Second argument to `evalFloat()` is vector component (0, 1, 2)
