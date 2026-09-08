# Explicit Standard Library Imports

**Status:** Implemented
**Author:** Joe Loser
**Created:** December 2025

## Summary

This document proposes requiring explicit imports from the Mojo
standard library (`std`) to improve import ergonomics and
consistency. Currently, the implicit availability of `std` modules
creates a counterintuitive asymmetry where local modules require
more syntax than external ones.

## Problem Statement

Mojo's current import system implicitly makes all `std` modules
available without qualification. While users *can* work around
naming conflicts using relative imports, the resulting UX is
counterintuitive and error-prone.

### Current Behavior

Consider a user project with this structure:

```text
my_project/
├── __init__.mojo
├── main.mojo
└── memory.mojo
```

**memory.mojo:**

```mojo
struct MyAllocator:
    def allocate(self, size: Int) -> UnsafePointer[UInt8]:
        # Custom allocation logic
        ...
```

**main.mojo:**

```mojo
from memory import MyAllocator  # Fails! Resolves to std.memory

def main():
    var alloc = MyAllocator()
```

This fails because `memory` resolves to `std.memory`, not the
local `memory.mojo`.

### The Workaround

Users can use relative imports to access their local module:

```mojo
from .memory import MyAllocator      # Works! Relative import
```

### The Problem: Counterintuitive Asymmetry

While the workaround exists, it creates an awkward asymmetry:

| Import Target       | Syntax Required                                 |
|---------------------|-------------------------------------------------|
| Local `memory.mojo` | `from .memory import ...` (requires `.` prefix) |
| `std.memory`        | `from memory import ...` (no prefix needed)     |

This is backwards - the "closer" local module requires **more**
syntax than the "distant" standard library. This leads to:

1. **Surprise**: Users may expect bare names to resolve locally
   first
2. **Wrong module resolution**: Forgetting `.` silently gives you
   `std.memory` instead of your local `memory.mojo`
3. **Cognitive load**: Must remember to use `.` for local imports
   that might conflict
4. **Inconsistency**: Doesn't match how Rust, Zig, or modern
   Python work, creating additional learning/friction for little
   value add in the current approach

Note: This behavior **does** match modern Python (PEP 328), where
bare names resolve to installed packages and `.` is required for
relative imports. However, this was controversial in the Python
community, and languages designed after Python (Rust, Zig) chose
differently. Since Mojo has the opportunity to learn from Python's
experience, we should consider whether Python's choice is one worth
emulating or one worth improving upon.

### Affected Module Names

The following `std` module names are common enough that users will
inevitably want to use them:

- `memory` - Very common for custom allocators, memory pools
- `algorithm` - Generic algorithms, sorting, searching
- `collections` - Custom data structures
- `utils` - Utility functions (extremely common)
- `os` - OS-specific functionality
- `sys` - System interfaces
- `io` - Input/output handling
- `time` - Timing utilities
- `random` - Random number generation
- `math` - Mathematical operations
- `testing` - Test frameworks
- `pathlib` - Path manipulation
- `hashlib` - Hashing utilities

There is also an argument to be made about additional (perhaps
common in name) future modules added to `std` over time.

## Survey of Other Languages

### Rust

Rust solves this with explicit path prefixes:

```rust
// Standard library - must be explicit
use std::collections::HashMap;

// Current crate root
use crate::memory::MyAllocator;

// Current module
use self::helper::process;

// Parent module
use super::common::Config;

// External crate
use serde::Serialize;
```

Rust's approach:

- **No implicit imports** from `std` (except the prelude, which
  is minimal)
- **`crate::`** prefix always refers to the current crate root
- **`self::`** and **`super::`** keywords for relative navigation
- Clear, unambiguous, explicit

### Zig

Zig uses `@import` with explicit paths:

```zig
// Standard library
const std = @import("std");

// Local file
const memory = @import("./memory.zig");
const MyAllocator = memory.MyAllocator;

// Package dependency
const json = @import("json");
```

Zig's approach:

- **Everything is explicit** - no implicit imports
- **Relative paths** (starting with `./` or `../`) for local files
- **Bare names** for packages and std
- Zero ambiguity by design

### C++

C++ has evolved multiple mechanisms:

```cpp
// System/library headers (searches system paths)
#include <memory>
#include <vector>

// Local headers (searches local paths first)
#include "memory.h"
#include "utils/helper.h"

// C++20 modules
import std;           // Standard library module
import my_module;     // User module
```

C++'s approach:

- **Angle brackets** `<>` for system/library headers
- **Quotes** `""` for local headers (falls back to system)
- **Namespaces** (`std::`, `my_namespace::`) for runtime
  disambiguation

### Python

Python faced this exact problem and addressed it with PEP 328
(absolute/relative imports):

```python
# Absolute import - searches sys.path (installed packages, stdlib)
from collections import OrderedDict

# Relative imports - always local
from . import memory  # Current package
from .memory import MyAllocator  # Current package's memory module
from .. import utils  # Parent package
from ..utils import helper  # Parent package's utils module
```

Python's approach:

- **Bare names** search `sys.path` (external/installed packages
  first)
- **Dot prefix** (`.`, `..`) forces relative/local resolution
- Required `from __future__ import absolute_import` in Python 2.7
  for transition
- Now the default in Python 3

**Mojo's current behavior matches Python's model.** This is worth
noting since Mojo aims for Python familiarity. However, this
design was controversial in the Python community:

- Many consider it a "gotcha" that trips up developers
- Some argue the "closer is harder to access" asymmetry is
  counterintuitive
- The transition (Python 2 → 3) was painful for existing codebases
- Languages designed after Python (Rust, Zig, Go) chose explicit
  external imports instead

The question for Mojo is: should we match Python's
familiar-but-controversial choice, or learn from it and adopt the
model that newer languages have converged on?

### Swift

Swift uses explicit module imports:

```swift
// Standard library (implicitly available but can be explicit)
import Swift
import Foundation

// User module
import MyFramework

// Specific symbol
import struct MyFramework.MyAllocator
```

Swift's approach:

- Module names must be unique
- Framework/package namespacing
- No relative imports needed because module names are globally
  unique (this is a bit restrictive I'd argue)

## Proposed Solutions

### Option A: Require Explicit `std.` Prefix (Recommended)

Remove implicit `std` imports and require explicit qualification:

```mojo
# Standard library - explicit
from std.memory import UnsafePointer
from std.collections import List, Dict
import std.os

# Local modules - bare names work because std is not implicit
from memory import MyAllocator
from .utils import helper  # Relative also works
```

**Pros:**

- Clear and unambiguous
- Simple mental model: bare names are local/relative
- Minimal syntax changes

**Cons:**

- Breaking change for existing code (but is mechanical and easy
  to update)
- More verbose for std imports

### Option B: Keep Current Behavior (Status Quo)

This is the current state - implicit std imports with relative
syntax available:

```mojo
# Standard library - implicit (current behavior)
from memory import UnsafePointer  # Gets std.memory

# Local modules - require relative prefix
from .memory import MyAllocator
from ..utils import helper
```

**Pros:**

- No breaking changes
- Relative imports already work

**Cons:**

- Inconsistent: std is implicit, local requires prefix
- Counterintuitive: "closer" things require more syntax than
  "distant" things
- Users must remember to use `.` for local imports or face
  silent bugs
- Doesn't match Rust, Zig, (although it does match Python)

### Option C: Hybrid with `pkg` or `self` Keyword

Introduce a keyword for the current package:

```mojo
# Standard library
from std.memory import UnsafePointer

# Current package root
from pkg.memory import MyAllocator
from self.memory import MyAllocator

# Relative
from .utils import helper
```

**Pros:**

- Very explicit
- Clear distinction between std, package root, and relative

**Cons:**

- New keyword to learn
- More verbose

### Recommendation

**Option A (Require Explicit `std.` Prefix)** is recommended
because:

1. **Intuitive**: Local/closer modules are easier to import than
   distant ones
2. **Clarity**: Code is self-documenting about where imports come
   from
3. **Consistency**: Same mental model - explicit paths for
   external code
4. **Future-proof**: Works well as the ecosystem grows with more
   packages
5. **Precedent**: Rust's success with this approach demonstrates
   viability

With Option A, users get natural behavior:

```mojo
# Bare names resolve locally first (intuitive)
from memory import MyAllocator       # Local memory.mojo
from utils import helper             # Local utils.mojo

# Std requires explicit qualification
from std.memory import UnsafePointer
from std.collections import List

# Relative imports still work for explicitness
from .memory import MyAllocator      # Explicitly local
```

The existing relative import support (`.module`) remains available
as a complement for cases where explicitness is desired.

### Third-Party Packages

An important consideration is how this interacts with third-party
packages. If a user installs a package called `awesome_lib` that
has a `memory` module, how do they import it?

**Proposed behavior**: All external packages require explicit
qualification:

```mojo
# Local module (bare name)
from memory import MyAllocator

# Standard library (std. prefix)
from std.memory import UnsafePointer

# Third-party package (package name prefix)
from awesome_lib.memory import TheirAllocator
```

This matches Rust's model where:

- `crate::` = current crate (local)
- `std::` = standard library
- `package_name::` = external dependency

The rule becomes simple: **bare names are local, everything
external needs a prefix**.

If there's no local `memory` module, attempting
`from memory import X` would produce an error rather than falling
back to external packages. This prevents surprising implicit
resolution and makes dependencies explicit in the code.

```mojo
# Error: no local module 'memory' found
# Did you mean: from std.memory import ... ?
#           or: from awesome_lib.memory import ... ?
from memory import UnsafePointer
```

This explicitness is valuable because:

1. **Readable**: You can see all external dependencies at the
   import site
2. **Predictable**: Adding a local `memory.mojo` doesn't silently
   shadow an external import
3. **Tooling-friendly**: Static analysis can resolve imports
   without package metadata

## Implementation Experience

### Compiler Changes

The implementation requires changes to
`Mojo/lib/MojoLLDB/SharedState.cpp` and related import resolution
code:

1. **Remove implicit std injection**: Stop automatically adding
   `std` to the import search path
2. **Update error messages**: Guide users to the new syntax

The compiler changes themselves are localized. The main
implementation work involves:

- Modifying `getStdlibPackageName()` behavior
- Updating import resolution order

### Migration Effort

### Affected Packages

The `std` package and other packages built on top (kernels, etc.)
would need to be updated to use explicit imports internally. This
is mostly mechanical:

- `from module` → `from std.module`

for each module that is a top-level module in `std`.

### User Code Migration

For users migrating existing code:

**Simple case (no conflicts):**

```mojo
# Before
from collections import List

# After
from std.collections import List
```

**With local modules:**

```mojo
# Before (broken if user has memory.mojo)
from memory import UnsafePointer

# After (works correctly)
from std.memory import UnsafePointer
from .memory import MyAllocator  # Local module
```

A migration tool could automate most changes by:

1. Scanning for `from X import` where X is a std module name
2. Adding `std.` prefix
3. Flagging ambiguous cases for manual review

### Backward Compatibility

To ease migration, we could:

1. **Deprecation period**: Warn on implicit std imports for one
   release cycle
2. **Migration tool**: Provide automated code migration
3. **Documentation**: Clear upgrade guide with examples

## Timeline and Mojo 1.0

As discussed in
[The Path to Mojo 1.0](https://www.modular.com/blog/the-path-to-mojo-1-0),
breaking changes should be made before the 1.0 release to
establish stable semantics. This change should be prioritized
because:

1. **Import semantics are fundamental**: They affect every Mojo
   program
2. **Breaking change**: Must happen before stability guarantees
3. **Ecosystem health**: Third-party packages need clear import
   rules
4. **User expectations**: Systems programmers expect to name
   modules freely

Recommended timeline:

- **Pre-1.0**: Implement explicit std imports as the default
- **Migration period**: Provide tooling and documentation
- **1.0 release**: Stable import semantics with no implicit std

## Open Questions

1. **Import resolution order**: With explicit `std.`, should bare
   names check local packages first, or should there be a defined
   search order?
2. **Error messages**: How do we guide users who try the old
   implicit syntax? Should we have a deprecation period with
   warnings?

## References

- [Rust Module System](https://doc.rust-lang.org/book/ch07-00-managing-growing-projects-with-packages-crates-and-modules.html)
- [Zig @import](https://ziglang.org/documentation/master/#import)
- [PEP 328 - Python Absolute/Relative Imports](https://peps.python.org/pep-0328/)
- [C++20 Modules](https://en.cppreference.com/w/cpp/language/modules)
- [The Path to Mojo 1.0](https://www.modular.com/blog/the-path-to-mojo-1-0)
