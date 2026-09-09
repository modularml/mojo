# Non-Nullable UnsafePointer

## Summary

Make `UnsafePointer` non-nullable by default.

This proposal removes the null default constructor and `__bool__` from
`UnsafePointer`, and expresses nullability explicitly with
`Optional[UnsafePointer[T, origin]]`.

Under this design:

- `UnsafePointer[T, origin]` means the pointer is **non-null**.
- `Optional[UnsafePointer[T, origin]]` means the pointer **may be null**.

This makes pointer nullability explicit in the type system and removes a common
source of undefined behavior.

## Motivation

In 2009, Tony Hoare called the invention of the null reference his
"billion-dollar mistake," noting that it has "led to innumerable errors,
vulnerabilities, and system crashes." Mojo can avoid repeating that mistake
for pointers.

Mojo's current `UnsafePointer` inherits C-style nullability. Any
`UnsafePointer` can be null, but the type does not communicate that fact.
Whether a null value is valid must be inferred from local conventions,
documentation, or defensive checks. Today, `UnsafePointer` has two features
that reinforce this model:

- A default constructor (`__init__(out self)`) that produces a null pointer.
- A `__bool__` conversion that allows `if ptr:` checks.

This leads to patterns like:

```mojo
var ptr = UnsafePointer[Int]()  # null pointer
# ...code
ptr[]                           # undefined behavior at runtime
```

and:

```mojo
if ptr:
    use(ptr)
```

The problem is not that these patterns exist. The problem is that the type
system does not distinguish between:

- a pointer that is guaranteed to be present
- a pointer that is optional and must be checked before use

As a result:

- Null becomes the implicit default.
- Pointer fields are often initialized into an invalid intermediate state.
- APIs do not clearly communicate whether null is allowed.
- Null checks become ad hoc and easy to forget.

This is a place where the type system can do better. If nullability is part of
the type, then the contract becomes explicit and the compiler can enforce it.

### Non-goals

This proposal does not attempt to make raw pointers generally safe.
`UnsafePointer` remains unsafe in the usual ways: it may still dangle, alias
incorrectly, point to invalid memory, or violate lifetime assumptions. This
proposal addresses only one dimension of pointer unsafety: nullability.

## Proposed Design

### 1. Make `UnsafePointer` non-nullable

Remove the default constructor that creates a null pointer:
`__init__(out self)`. After this change, `UnsafePointer` no longer conforms to
`Defaultable`. A value of type `UnsafePointer[T, origin]` must always contain a
non-null pointer.

### 2. Remove `__bool__` from `UnsafePointer`

After this change, `UnsafePointer` no longer conforms to `Boolable`. On a
non-nullable pointer type, `__bool__` is no longer meaningful. Keeping it would
imply that `UnsafePointer` still has a nullable state and would encourage a
style built around runtime null checks rather than explicit types.

### 3. Express nullability with `Optional`

Code that needs a nullable pointer should use
`Optional[UnsafePointer[T, origin]]`. For example:

```mojo
var x = 42
var maybe_ptr: Optional[UnsafePointer[Int, origin_of(x)]] = None

if maybe_ptr:
    maybe_ptr.value()[]  # ptr is known to be non-null here
    #         ^      ^
    #         |      | Then dereference the pointer
    #         | First unwrap the Optional
```

This makes nullable cases explicit and keeps non-nullable cases simple.

### 4. Allocation

`UnsafePointer.alloc()` continues to return `UnsafePointer` (non-null). If the
underlying allocator fails, the program traps rather than returning a null
pointer. This matches the behavior of most modern allocators (including the
default system allocator on Linux, which overcommits) and avoids forcing every
allocation site to handle a nullable return.

- Note: We can provide a fallible allocator interface via `try_alloc`, however
  most mojo programs today assuming infallible allocation, so this is a fine
  default.
- Code that needs to handle allocation failure explicitly should call the
  underlying allocator (e.g. `malloc`) directly and use
  `Optional[UnsafePointer]` for the result.

## Layout Requirement

This proposal requires that `Optional[UnsafePointer[T, origin]]` have the same
size and layout as `UnsafePointer[T, origin]`. The expected representation uses
the null pointer value as the `None` discriminant, so nullable pointers remain
zero-overhead. This matters for performance and FFI.

## Prior Art

Other modern systems languages have made similar choices.

### Swift

Swift's `UnsafePointer<T>` is non-nullable. The optional form
`UnsafePointer<T>?` is used when null is allowed. Imported C APIs map nullable
and non-nullable pointers to distinct types.

### Rust

Rust's raw pointers (`*const T`, `*mut T`) are nullable by default. The
standard library provides `NonNull<T>` as an opt-in non-null wrapper, with
`Option<NonNull<T>>` for the nullable variant. `Option<NonNull<T>>` is
guaranteed to have the same layout as a raw pointer via null pointer
optimization. Mojo goes further by making non-null the default rather than
opt-in.

### Zig

In Zig, `*T` is non-null by default and `?*T` is the nullable form. Optional
pointers use the null address as the null representation, so there is no extra
storage overhead.

## C Interop

This model works well with C APIs.

C functions that may return `NULL` should be wrapped with `Optional`. This works
because `Optional[UnsafePointer[T, origin]]` is guaranteed to have the same
layout as `UnsafePointer`. As C's `NULL` is guaranteed to represent the `None`
state of the `Optional`.

For example, `malloc` returns `NULL` on allocation failure:

```mojo
def c_malloc(
    size: Int
    out result: Optional[UnsafePointer[UInt8], UntrackedOrigin[mut=True]],
):
    return external_call["malloc", type_of(result)](size)

var ptr = c_malloc(16)
if not ptr: # Same as `if (ptr == NULL)` in C
    print("malloc failure!")
```

C functions that guarantee a non-null return should use `UnsafePointer`
directly. For example, a user-defined c method that always returns a global
pointer:

```c
static int GLOBAL_INT = 42;

const int* get_global_int() {
  return &GLOBAL_INT;
}
```

```mojo
from std.ffi import c_int

def c_get_global_int() -> UnsafePointer[c_int, StaticConstantOrigin]:
    return external_call[
      "get_global_int",
      UnsafePointer[c_int, StaticConstantOrigin],
    ]()
```

Likewise, nullable pointer parameters should be modeled as optional pointers,
while required pointer parameters should use non-null pointers. This makes C
contracts visible in Mojo signatures instead of leaving them implicit.

## Source Compatibility and Migration

This is a source-breaking change. The migration is mostly mechanical.

### Migration timeline

| Release          | Change                                                                                                                               |
|------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| **26.3**         | Add `UnsafeNullablePointer` as a transitional type. Remove nullability from `UnsafePointer` (no default constructor, no `__bool__`). |
| **26.4 nightly** | Deprecate `UnsafeNullablePointer` with a compiler warning.                                                                           |
| **26.4**         | Remove `UnsafeNullablePointer` from the standard library.                                                                            |

### `UnsafeNullablePointer` transitional type

In the 26.3 release, `UnsafeNullablePointer` is added to the standard library
as a drop-in replacement for the current (nullable) `UnsafePointer` semantics.
It preserves the default null constructor and `__bool__` conversion.

Migration steps:

1. **Find code that relies on nullable `UnsafePointer` behavior**: default
   construction, null sentinel returns, or `if ptr:` checks.
2. **Replace `UnsafePointer` with `UnsafeNullablePointer`** in those locations
   to keep existing behavior and unblock compilation.
3. **Migrate to `Optional[UnsafePointer]`** at your own pace, converting
   `UnsafeNullablePointer` usages to the idiomatic form.
4. **Remove all `UnsafeNullablePointer` usage** before the 26.4 release, when
   the type is removed.

`UnsafeNullablePointer` is temporary. It exists to decouple "update my code so
it compiles" from "adopt the new idiom," giving users a release cycle to
complete the migration without doing both at once.

### Default-initialized pointer fields

Code that currently relies on a null default must use `Optional` instead.

Before:

```mojo
struct MyBuffer:
    var data: UnsafePointer[UInt8, UntrackedOrigin[mut=True]]

    def __init__(out self):
        self.data = UnsafePointer[UInt8, UntrackedOrigin[mut=True]]()
```

After:

```mojo
struct MyBuffer:
    var data: Optional[UnsafePointer[UInt8, UntrackedOrigin[mut=True]]]

    def __init__(out self):
        self.data = None
```

### Null sentinel returns

Code that uses a null pointer as a sentinel return value should return
`Optional[UnsafePointer[T, origin]]` instead.

Before:

```mojo
def find(
    haystack: Span[Int, _], needle: Int
) -> UnsafePointer[Int, haystack.origin]:
    # ...
    return {}  # not found
```

After:

```mojo
def find(
    haystack: Span[Int, _], needle: Int
) -> Optional[UnsafePointer[Int, haystack.origin]]:
    # ...
    return None  # not found
```

### Null checks

Checks like `if ptr:` now only apply to optional pointers.

### Non-nullable code

Code that already constructs valid pointers and never uses null typically needs
no change.

```mojo
var ptr = UnsafePointer(to=value)
ptr[]  # still valid
```

## Benefits

### Nullability is explicit

Function signatures and field types now clearly express whether null is allowed.
That makes APIs easier to read and harder to misuse.

### Safer defaults

The easiest pointer type to use is now the safer one. Developers must opt in to
nullable behavior instead of getting it by default.

### Fewer defensive checks

Code receiving `UnsafePointer[T]` does not need to defensively check for null.
Only nullable cases require branching.

### Controlled access paths replace undefined behavior

When nullability is expressed in the type, every access path through
`Optional[UnsafePointer]` has a defined outcome. There is no silent undefined
behavior, the caller chooses how much safety overhead they accept:

```mojo
var maybe_ptr: Optional[UnsafePointer[Int, origin]] = None

maybe_ptr[][]
#        ^ ^
#        | | Dereference the pointer
#        |
#        Raises `EmptyOptional` — a catchable error. Call sites can handle
#        or propagate the failure.

maybe_ptr.value()[]
#         ^^^^^
#         Runtime abort with source location on null access. Uncatchable,
#         but reports exactly where the null was accessed.

maybe_ptr.unsafe_value()[]
#         ^^^^^^^^^^^^
#         Asserts in debug; zero overhead in release. Explicit opt-in:
#         the caller takes responsibility for ensuring the value is present.
```

With the old nullable `UnsafePointer`, a null dereference is undefined behavior:
the program may segfault, silently corrupt data, or continue running in an
invalid state. With `Optional[UnsafePointer]`, bugs that would have manifested
as a segfault or corrupted data are instead caught at the point of access —
and the only way to get zero-overhead access is to explicitly spell it out.

### Better initialization patterns

Removing the null default constructor discourages partially initialized states
where a pointer field exists but is not yet valid.

### Zero-overhead nullable pointers

Using `Optional[UnsafePointer[T]]` does not impose extra layout or runtime
cost.

### Familiar model

Developers coming from Rust, Swift, or Zig will already expect this design.

## Alternatives Considered

### Keep `UnsafePointer` nullable and add a separate non-null pointer type

This keeps the current behavior but makes the safer choice less obvious. The
default type tends to be used most often. If `UnsafePointer` remains nullable,
the unsafe behavior is the path of least resistance. This is the approach Rust
took with `NonNull<T>`, and in practice most Rust code uses raw nullable
pointers rather than opting in to the safer wrapper.

### Introduce a dedicated `NullablePointer` type

This would work, but it adds another pointer type to the language for a problem
that `Optional` already solves. `Optional[UnsafePointer[T]]` is simpler and
more general.
