# `UnsafePointer` v2

**Status**: Implemented.

## Motivation

As Mojo’s standard library matures, a few foundational types need to be
stabilized — one of the most important being `UnsafePointer`. It is a
fundamental building block of many low-level abstractions and data structures,
however its current API has several flaws.

## Current Issues with `UnsafePointer`

1. **Unsafe implicit conversions**
   - `immutable` → `mutable`
     ([GitHub issue #4386](https://github.com/modular/modular/issues/4386))
   - `origin_of(a)` → `origin_of(b)`
   - `AnyOrigin` → `origin_of(a)`

2. **Defaulted origin (`AnyOrigin`)**
   When an `UnsafePointer` is introduced with its defaulted `AnyOrigin`, any use
   of it extends *all* lifetimes and bypasses Mojo’s ASAP destruction rules.
   While sometimes desirable, such “escape hatches” should always be explicit.

3. **Defaulted mutability**
   Combining a defaulted `mut=True` with implicit casting from immutable to
   mutable has spread unsafe conversions throughout the codebase, especially in
   C FFI and kernel code.

Overall, Mojo’s current `UnsafePointer` is arguably *less safe* than C++ raw
pointers.

### Example Comparison

**C++ (errors on unsafe cast):**

```cpp
void foo(int* ptr) {}

int main() {
    const int y = 42;
    foo(&y); // Error: invalid conversion from 'const int*' to 'int*'
}
```

**Mojo (currently compiles):**

```mojo
def foo(ptr: UnsafePointer[mut=True, Int]): pass

def main():
    var y = 42
    foo(UnsafePointer(to=y).as_immutable())
    # ^^ Compiles without an error :(
```

## Path Forward

Two main fixes are needed:

1. Prevent unsafe implicit conversions.
2. Remove defaulted parameters for mutability and origin, aligning with other
types (`Span`, `LayoutTensor`, etc.).

The proposal introduces a new `UnsafePointerV2` type that corrects these issues
and provides a migration path. During transition, `v1` and `v2` pointers will
support implicit conversions to avoid breaking existing code.

## `UnsafePointer` API (current)

```mojo
struct UnsafePointer[
    type: AnyType,
    *,
    address_space: AddressSpace = AddressSpace.GENERIC,
    mut: Bool = True,  # ⚠️ Defaulted to mutable
    origin: Origin[mut] = Origin[mut].cast_from[MutAnyOrigin],  # ⚠️ Defaulted to AnyOrigin
](TrivialRegisterPassable):
    ...
```

**Issues:**

- `mut` defaults to `True`, making pointers mutable by default
- `origin` defaults to `MutAnyOrigin`, bypassing lifetime tracking
- Allows unsafe implicit conversions (immutable → mutable, origin casts)

## `UnsafePointer` (v2) API

```mojo
struct UnsafePointer[
    mut: Bool, //, # ✅ Inferred mutability, no default
    type: AnyType,
    origin: Origin[mut], # ✅ Non-defaulted origin, must be explicit
    *,
    address_space: AddressSpace = AddressSpace.GENERIC,
]:
    ...

alias MutUnsafePointer[...] = UnsafePointer[mut=True, ...]
alias ImmutUnsafePointer[...] = UnsafePointer[mut=False, ...]
```

**Improvements:**

- `mut` is inferred (using `//` marker) and has no default value
- `origin` must be explicitly specified or parameterized
- Prevents unsafe implicit conversions (compile-time errors)

### Cross-language Comparison

| Mojo                    | C++        | Rust       |
|-------------------------|------------|------------|
| `ImmutUnsafePointer[T]` | `const T*` | `*const T` |
| `MutUnsafePointer[T]`   | `T*`       | `*mut T`   |

---

## Why `V2`?

`UnsafePointer` is deeply integrated across the codebase.
Changing its interface directly would break a large amount of code, both
internally and in the community. `UnsafePointer` (v2) provides a transition
path, allowing incremental migration and validation before replacing
`UnsafePointer` entirely.

---

## Tentative Migration Timeline

### **Nightly (current)**

- Rename `UnsafePointer` to `LegacyUnsafePointer`.
- Introduce `UnsafePointerV2` and eventually rename to `UnsafePointer`.

### **25.7 (late November 2025)**

- Rename `UnsafePointer` to `LegacyUnsafePointer`.
- Introduce the new `UnsafePointer` for general use.

### **26.1 (Jan 2026)**

- Deprecate `LegacyUnsafePointer` (and then eventually remove).

---

## Alternative Approaches Considered

1. **Using `@implicit(deprecated=True)`**
   - Not feasible due to the complexity of `UnsafePointer` constructors.
   - The current conversion constructor would require multiple overloads, some
   deprecated, some not, to separate safe and unsafe conversions.
   - Managing ~7+ overloads while avoiding ambiguity would be error-prone and
   still wouldn’t address inferred mutability or defaulted origins.
   - Starting fresh with `UnsafePointerV2` ensures correctness and clarity.

2. **Using `alias UnsafePointerV2 = ...`**
   - Would only help reorder parameters, not change constructor behavior.
   - Since the problem extends beyond API shape to semantics (implicit casting
   and defaults), an alias alone isn’t sufficient.
   - Mojo’s alias system also prevents initializer syntax for aliases, making
   this option impractical.

---

## Migration Guide: From `LegacyUnsafePointer` to the New `UnsafePointer`

This guide walks through updating your Mojo codebase to use the new, safer
`UnsafePointer` API.

### Step 1 — Rename Old `UnsafePointer` to `LegacyUnsafePointer`

Rename every usage of the old `UnsafePointer` to `LegacyUnsafePointer`.
This preserves prior behavior and prevents mixing old and new pointer APIs.

This is a mechanical rename and does not change runtime semantics.

### Step 2 — Migrate Code to the New `UnsafePointer`

Once the old type is renamed, begin replacing `LegacyUnsafePointer` with the new
`UnsafePointer`.
The new pointer type requires you to explicitly specify the mutability and
origin to make pointer behavior clearer and safer.

#### Using `UnsafePointer` as a Function Argument

Function arguments that accept pointers must now state their mutability using
`mut=True` or `mut=False`.

```mojo
def read_pointer(ptr: UnsafePointer[mut=False, Int]):
    var n = ptr[]

def write_pointer(ptr: UnsafePointer[mut=True, Int]):
    ptr[] = 42
```

- `mut=False` means the pointee cannot be modified through this pointer.
- `mut=True` means the pointee may be mutated.

#### Using `UnsafePointer` as a Return Type

Returning a pointer requires specifying its origin, which tells the compiler
where the pointer came from and who manages its lifetime. [Read more about
lifetimes here](https://mojolang.org/docs/manual/values/lifetimes/).

```mojo
def pointer_to(
    mut string: String,
) -> UnsafePointer[String, origin_of(string)]:
    return UnsafePointer(to=string)
```

#### Storing an `UnsafePointer` to a heap allocation

Heap-allocated memory must use an explicit `external` origin.
This indicates that the lifetime of the memory is managed manually rather than
by Mojo.

```mojo
struct MyList:
    var _data: UnsafePointer[Int, MutOrigin.external]
    var _len: Int

    def __init__(out self, *, length: Int):
        self._data = alloc[Int](length)
        self._len = length

    def __del__(deinit self):
        # Always free external memory you allocate.
        self._data.free()
```

- `MutOrigin.external` communicates that memory is externally managed.

#### Exposing an `UnsafePointer` to a heap allocation

When returning a pointer to a heap allocation from inside a struct, the origin
must reflect that the memory belongs to a member of `self.`

When possible, prefer using the safe `Pointer` or `Span` types instead of an
`UnsafePointer`.

```mojo
struct MyList:
    var _data: UnsafePointer[Int, MutOrigin.external]
    var _len: Int

    def unsafe_ptr[
        mut: Bool,
        origin: Origin[mut], //
    ](ref [origin] self) -> UnsafePointer[Int, origin]:
        return self._data
                .mut_cast[mut]()
                .unsafe_origin_cast[origin]()
```

#### Using `UnsafePointer` in FFI

Most FFI calls can pass pointers directly using `external_call`.

```mojo
def wrap_c_func(
    read_ptr: UnsafePointer[mut=False, Int32],
    write_ptr: UnsafePointer[mut=True, UInt],
):
    # C signature:
    # void c_func(const int32_t*, size_t*)
    external_call["c_func", NoneType](read_ptr, write_ptr)
```

If an FFI function returns a pointer, its origin should often be `external`
since the memory comes from outside Mojo.

```mojo
def wrap_c_func(
    length: UInt,
    out result: UnsafePointer[Int32, ImmutOrigin.external],
):
    # C signature:
    # const char* c_func(size_t)
    result = external_call["c_func", type_of(result)](length)
```

#### Using `UnsafePointer` in GPU kernels

Similar to that of `LayoutTensor`, the origin must be specified.

```mojo
def kernel(
    ptr: UnsafePointer[Float32, MutAnyOrigin]
):
    ...

def main():
    with DeviceContext() as ctx:
        var size = 128
        var buf = ctx.enqueue_create_buffer[DType.float32](size)
        ctx.enqueue_function[kernel](buf)

        # ...
```
