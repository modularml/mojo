# `@stable` Decorator for API Stability Markers

**Status**: Accepted, partially implemented.

**Implementation notes** (as of 2026):

- `@stable` on structs, traits, functions, and comptime values (top-level and
  members), with `--warn-on-unstable-apis` flag: **implemented**.
- `@stable(since="version")`: syntax parsed and stored; version string is not
  yet enforced at use-sites.
- Always-on authoring warnings (stable member in unstable parent, stable trait
  inheriting from unstable): **implemented**.
- `@stable(recursive=True)` on `from ... import X` statements: **implemented**
  (see the import section below).
- `@stable(recursive=True)` on `comptime` aliases: **not implemented**. See the
  comptime section below for the reason.

Author: Denis Gurchenkov

Date: January 2026

## Summary

This proposal introduces the `@stable`
decorator to mark API stability in the Mojo standard library. By default, all
stdlib APIs are unstable.

The `--warn-on-unstable-apis` compiler flag lets users ensure they only depend
on stable APIs. Third-party package support will be added in the future.

## Scope

This feature is initially scoped to the **Mojo standard library only**.
Third-party package support may be added later as an opt-in mechanism. That
design is not defined yet.

## Motivation

As Mojo evolves, some APIs will be more mature than others. Users building
production applications need to identify which APIs are stable. They want:

1. APIs that won’t change without warning in future releases
2. Clear visibility when opting into experimental features
3. Build-time warnings (or errors) when unstable APIs creep into their code

Without explicit markers, users must rely on hints in the documentation or
discover unstable APIs through trial and error.

## Unit of stability

The `@stable` decorator can be applied to:

- **Top-level symbols:** traits, structs, comptime values, functions
- **Struct and trait members:** methods, comptime values, fields
- **Import statements**

`@stable` does not apply to local variables or comptime values inside functions
or methods. It also does not apply to parameters or arguments in a function or
struct/trait signature, or any other names except the ones mentioned above.

Example

```mojo
## Stdlib example
# Design of List is mostly stabilized
@stable(since="1.0")
struct List[T: Copyable & Movable](
    Boolable, Copyable, Defaultable, Iterable, Movable, Sized
):
    # unstable by default
    var _data: UnsafePointer[Self.T, MutOrigin.external]

    @stable
    var capacity: Int

    @stable
    def __init__(out self):

    # unstable by default
    def steal_data(mut self) -> UnsafePointer[Self.T, MutOrigin.external]:

## User code example:
# Even if FileDescriptor is not marked as stable, this annotation
# prevents stability warnings when FileDescriptor or its members are
# used in this file.
@stable(recursive=True)
from std.io import FileDescriptor
..
use(FileDescriptor.create()) # no warning
```

## Intent

By marking something as "stable," the API author promises to avoid
backward-incompatible changes except in major releases of the package.
"Backward-incompatible" is intentionally defined loosely.

This includes changes such as renaming, removing, moving, or changing input
types. There are also more subtle ways to break compatibility (for example,
changing behavior while keeping the interface intact), and it is the API
author's responsibility to avoid such changes.

The documentation page for the decorator should describe common ways stability
can be broken. For example, adding a new function overload can change overload
resolution and break existing code.

### Package Opt-in

The standard library (`std` package) will be opted in, with all types
**unstable by default**. Some language keywords and magic functions are also
opted in. For the purposes of this feature, they are treated as part of the
standard library.

Other packages (third-party packages and kernels) will not be opted in for
Mojo 1.0. Symbols in those packages are treated as stable and produce no
warnings.

Extending this feature to third-party packages will be considered in the
future.

The reason for deferring third-party opt-in is that Mojo does not yet have a
place or syntax for a package author to mark a package as "opted in", such as
package definition files. This remains to be designed.

### Default Stability

In opted-in packages, all APIs are **unstable by default**:

1. **Explicit guarantees**: Stability promises should be deliberate, not
   accidental.
2. **Natural evolution**: APIs start unstable and graduate to stable over time.
3. **Avoiding accidents**: If stability were opt-out, missing an annotation
   could create an implicit promise that is difficult to retract.

### Compiler Flag: `-warn-on-unstable-apis`

A new compiler flag enables warnings when user code directly uses unstable APIs:

```bash
mojo build --warn-on-unstable-apis myprogram.mojo
```

When enabled, the compiler emits warnings like:

```text
myprogram.mojo:42:5: warning: using unstable API 'std.ExperimentalBuffer'
    var buf = ExperimentalBuffer(1024)
              ^~~~~~~~~~~~~~~~~~
```

### When the warning is shown

Put simply: if user code references an unstable API, the compiler should emit
a warning.

- Call or reference a function or method
- Create an instance of a struct, or use struct/trait name to refer to one of
  its members
- Cast a value to an unstable trait
- Use an unstable struct or trait member
- Implement an unstable trait

Note that all these cases can be checked in the Mojo parser. None of the cases
require checking during or after generic instantiation / elaboration phase.
This is by design.

The warning does **not** trigger for:

- Transitive use through stable APIs (e.g., a stable function internally uses
an unstable helper)
- Use within the same package that defines the unstable API
- Importing an unstable API but not using it anywhere.

### Always-on authoring warnings (independent of `--warn-on-unstable-apis`)

In opted-in packages (initially `std`), the compiler also emits warnings for
*inconsistent stability declarations*, even if `--warn-on-unstable-apis` is
not enabled. These warnings help API authors avoid accidentally promising
stability while depending on unstable pieces.

Always-on warnings include:

- A stable function should not expose unstable types in its signature
  (parameters or return type).
- A stable struct implementing a stable trait must satisfy its requirements
  using stable methods.
- A stable trait must not inherit from an unstable trait.

### Interaction with `Werror`

Like other warnings, `--warn-on-unstable-apis` can be combined with
`-Werror` to treat these warnings as errors, enabling strict stability
enforcement in CI pipelines.

### Interaction with @deprecated decorator

These two decorators are mutually exclusive.

## Decorator Syntax

The `@stable` decorator takes two optional arguments: `since="version_string"`
and `recursive=True`

```mojo
@stable
def foo(): ...

@stable(since="1.2")
def bar(): ...

@stable(recursive=True)
from std import FileDescriptor
```

**The `since` argument is parsed and stored but not yet enforced at use-sites.
It will be fully implemented once the Mojo and package versioning scheme is
settled.**

**The `recursive` argument is only allowed on `comptime` and `import`.
`@stable(recursive=True)` on `import` is implemented. `@stable(recursive=True)`
on `comptime` is not yet implemented — see the comptime section below.**

### Meaning of `since`

`since` indicates the earliest **version of the package** in which the API
(as shown in documentation) is available.

If a stable API's signature is extended (for example, a new parameter is added
with a default), the `since` value should be updated to the version that
introduced the extended signature.

Example:

```mojo
@stable(since="1.0")
def foo(): ...

# later extended
@stable(since="1.3")
def foo(b: Int = 7): ...
```

This proposal does not dictate a specific process for choosing version strings
yet. In the future, when stability markers are extended beyond stdlib, the
`since` version will likely refer to the *package version*, which requires a
broader versioning design.

### Meaning of `recursive=True`

Recursive flag is only allowed on `comptime` and `import`, and will be
*required* on `import`. This flag marks this symbol **and all of its members**
as stable.

For comptime values, the user can choose to mark either the symbol or
symbol-and-members as stable. For import, to avoid ambiguity, we decided that
only recursive override can be used. The next two sections describe this flag
in detail.

## Changing stability status via comptime aliases

> **Implementation status**: `@stable` on a comptime (bare, without `recursive`)
> is implemented. `@stable(recursive=True)` on a comptime is
> **not yet implemented**. The reason is that `@stable` on a comptime does not
> create a new type — the alias and the original type share the same `ASTDecl`.
> The current suppression mechanism works by name: use-sites check whether the
> referenced name was imported with `@stable(recursive=True)`. A comptime alias
> is a different name binding, but since it resolves to the same underlying
> type, member accesses through it (e.g. `StableStruct.some_method()`) resolve
> the receiver type to the original type's decl, not the alias. Therefore, the
> alias name cannot be used as the suppression key in the current model. A type
> wrapper or a more expressive suppression model is needed to support this case.

Creating a comptime alias, parametric or not (`comptime NewName = OldName`)
creates a new symbol that can have different stability status.

This proposal supports two modes:

- `@stable` on an alias makes the **alias name** stable (but does not
  recursively affect member accesses).
- `@stable(recursive=True)` on an alias makes the alias name stable and also
  suppresses warnings for **member accesses through that alias**.

Examples:

```mojo
# Unstable
struct UnstableStruct[T: AnyType]:
  pass

# Non-recursive alias: only the name is treated as stable
@stable
comptime StableName = UnstableStruct[Int]

# Recursive alias: name + member accesses through StableStruct are warning-free
@stable(recursive=True)
comptime StableStruct = UnstableStruct[Int]
```

### Why this exists

This is (1) an escape hatch / override mechanism for end users, and (2) a
compatibility tool for stdlib authors. It can be used by the stdlib team to
preserve a stable surface API while changing an underlying implementation
shape:

```mojo
# had
@stable
StableStruct[T, P] ...

# now underlying implementation changes
UnstableStruct[P, Q, T] ...

# preserve stability via a stable alias
@stable
comptime StableStruct[T, P] = UnstableStruct[P, Int, T]
```

## Changing stability status via imports

> **Implementation status**: Implemented.

Import statements can be annotated with `@stable(recursive=True)`.

- `@stable(recursive=True)` on an import is a **warning suppression override**
  for the imported binding and all member accesses through it.
- For imports, `recursive=True` is required in this proposal (to avoid
  ambiguous expectations about whether members are affected).

```mojo
@stable(recursive=True)
from std import Dict

# No warning on the use of Dict, no warning on the use of members through Dict
_ = Dict[String, Int].REMOVED
```

Why this exists: same motivation as recursive comptime aliases. We want to give
end users complete control over the warning surface when they explicitly opt
into it.

### Limitation: override does not follow aliased types

The override covers the imported name and member accesses *through* that name.
It does not transitively cover other unstable types *exposed* by those members.
If a member is a comptime alias for a different unstable type, using that type
independently still warns:

```mojo
@stable(recursive=True)
from std import UnstableA  # UnstableA has: comptime B = UnstableB

var B = UnstableA.B  # no warning — B is a member of UnstableA ✓
B.static_method()    # warning — UnstableB is not in the stable override set ✗
```

To suppress the second warning, import `UnstableB` separately:

```mojo
@stable(recursive=True)
from std import UnstableA
@stable(recursive=True)
from std import UnstableB
```

**Re-exports:** creating a fully stable re-export (where downstream consumers
see no warnings for member accesses) is not supported in the current model.
A `@stable` alias suppresses the name-level warning but does not suppress
member-access warnings for downstream consumers, because the stable import
override set is file-scoped and does not propagate through re-exports.

## Keywords and magic functions

Some Mojo built-in keywords and magic functions are unstable (example:
`__get_mvalue_as_litref()`). This feature should also warn when users reference
those symbols. From an end user's perspective, it should not matter whether a
symbol is defined in the standard library or in the compiler. The warning
behavior should be consistent.

## Alternatives Considered

### Default Stable (Opt-Out Stability)

We considered making all APIs stable by default, requiring authors to mark
unstable APIs with `@unstable`:

```mojo
@unstable
def experimental():
    pass
```

This would be very desirable for end users, because they can follow the
principle of least surprise — the API is stable unless mentioned otherwise.

**Rejected because**:

1. API authors must remember to
mark every experimental API, and forgetting to do so creates an implicit
stability promise.
2. Retracting a stability promise is much harder than making one. If an author
forgets `@unstable`, users will depend on the API assuming it’s stable.
3. The natural evolution of stdlib APIs is from unstable to stable, and we
   foresee most of the stdlib being unstable to begin with. Introducing this
   feature to the standard library should not require a massive amount of
   unstable decorators.

### Hierarchical stability

We considered making `@stable` and `@unstable` decorators propagate through
nested types and members, so that marking a struct as stable would mark its
fields as well.

Rejected because:

- This conflicts with the version markers, new members won't have the "stable
  since=" information.
- This is error prone — API author may accidentally add a new APIs to a stable
  struct and fail to mark the new API as unstable

We ended up using hierarchical stability in the escape hatches: in comptime and
import overrides.

### Two decorators, @stable and @unstable

v0.2 of this proposal had both stable and unstable decorators. The @unstable
was removed for simplicity. We might add it in the future, if there is a use
case.

## FAQ

### Q: What about transitive dependencies on unstable APIs?

A: The warning only fires for direct use in user code. If a stable API
internally uses an unstable helper, that’s an implementation detail and doesn’t
warn.

### Q: Should tests be able to use unstable APIs without warnings?

A: Yes, test code often needs to exercise unstable APIs. Users should not
enable the flag when building tests.

We defer to a future time discussion about more fine-grained opt-out mechanism
for tests.

### Q: How does this interact with documentation?

A: Documentation generator should prominently print stability status. This
could include badges for stable APIs, separate sections for stable vs. unstable
APIs, or filtering options to show only stable APIs.

### Q: What if I want to use an unstable API but suppress the warning?

A: Re-export via a stable alias, this erases the unstable annotation

```mojo
@stable
comptime StableStruct = UnstableStruct
```

A: Use the @stable decorator on an import statement

```mojo
@stable(recursive=True)
from std import UnstableStruct
```

### Q: Can @stable be used in the third-party packages that did not opt into this feature?

A: No. Using this decorator in a non-opted-in package will cause a compiler
warning. We think this case is a programmer's mistake and Mojo compiler should
warn them.

### Q: How this feature interacts with struct extensions? If a struct is unstable, can the extension be stable, and the other way round?

A: Stability does not propagate from struct to extensions or back. For
simplicity, it will not be possible to have a stable extension for an unstable
struct.

### Q: Are there any other errors or error cases?

A: The compiler warns if a stable member appears inside an unstable struct.
Why? Because whenever the end user refers to that member, they would always get
a warning first from creating the instance or referring to the struct. We
consider this case (a stable member inside an unstable struct) to be the API
author's mistake and want to warn them early.

### Q: Is that guaranteed that all uses of all unstable APIs issue warnings?

A: No. Mojo compiler may prune warnings and show just some of those, to avoid a
massive wall of text.

If the program uses any unstable APIs, the compiler guarantees it will emit at
least one warning. It is not guaranteed that *all* warnings will be shown:

- If an unstable symbol is used, its first use should warn, but subsequent ones
  may be silent.
- If an unstable type is used, it would warn, but subsequent uses of its
  unstable members may be silent.

The user's workflow should be: if no warnings shown, your code does not use any
unstable APIs. Otherwise, if warnings are shown, fix those, and rerun the
compiler to see if there are more warnings.
