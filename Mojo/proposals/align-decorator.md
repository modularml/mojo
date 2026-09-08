# `@align(N)` Decorator for Struct Alignment

**Status**: Implemented.

Author: Joe Loser

Date: January 5, 2026

## Summary

This design doc/RFC proposes adding an `@align(N)` decorator to Mojo that
specifies a minimum alignment for struct types, similar to C++'s `alignas` and
Rust's `#[repr(align(N))]`.

```mojo
@align(64)
struct CacheAligned:
    var data: SIMD[DType.float32, 16]

# align_of[CacheAligned]() returns 64
```

## Motivation

### Problem Statement

Mojo currently lacks a way to specify explicit alignment requirements for struct
types. This forces developers to use verbose workarounds when alignment is
critical for correctness or performance.

### Real-World Use Cases in the Codebase

### 1. TensorMap Descriptors

NVIDIA's TMA (Tensor Memory Accelerator) requires descriptors to be 64-byte
aligned for hardware compatibility:

```mojo
# Current comment in code:
# "It should be 64-byte aligned both on the host and the device"

# Current workaround:
var tensormap = stack_allocation[1, TensorMap, alignment=64]()[0]
```

With `@align`:

```mojo
@align(64)
struct TensorMap:
    # ... fields ...

# No workaround needed - allocation respects alignment automatically
var tensormap = TensorMap()
```

### 2. TMA Descriptors

Same pattern - 64-byte alignment required by hardware:

```mojo
# Current workaround:
var tma_descriptor = stack_allocation[1, TMADescriptor, alignment=64]()[0]
```

### 3. False Sharing Prevention in Concurrent Code

**File:** `stdlib/std/utils/lock.mojo`

```mojo
struct BlockingSpinLock(Defaultable):
    var counter: Atomic[Int64]
```

Without cache-line alignment, multiple `BlockingSpinLock` instances in an array
could share cache lines, causing false sharing and severe performance
degradation in multi-threaded code.

With `@align`:

```mojo
@align(64)  # Cache line size
struct BlockingSpinLock(Defaultable):
    var counter: Atomic[Int64]
```

### Current Workarounds

1. **`stack_allocation[..., alignment=N]()`** - Verbose, requires manual
   unpacking
2. **Manual padding fields** - Error-prone, wastes mental overhead
3. **Explicit `alignment` parameters to `alloc`** - Doesn't help with stack
   allocations

## Design

### Syntax

```mojo
@align(N)
struct MyStruct:
    var field: SomeType
```

Where `N` is a compile-time constant that must be a positive power of 2.

### Semantics

1. **Minimum Alignment**: `N` specifies a *minimum* alignment. The actual
   alignment is `max(N, natural_alignment)` where `natural_alignment` is the
   maximum alignment of all fields.
2. **Cannot Lower Alignment**: Unlike C's `#pragma pack`, `@align(N)` cannot
   reduce alignment below the natural alignment. This matches C++ `alignas` and
   Rust `#[repr(align(N))]` behavior.
3. **Propagates to All Allocations**: The alignment applies to:
   - Stack allocations
   - Heap allocations via `align_of[T]()`
   - Struct embeddings (a struct containing an aligned struct inherits the
     alignment requirement)
4. **Reflected in `align_of[T]()`**: The library `align_of` function returns
   the effective alignment including the decorator.

### Examples

```mojo
@align(64)
struct CacheLineAligned:
    var x: Int  # Natural alignment: 8

# align_of[CacheLineAligned]() == 64 (decorator wins)

@align(4)
struct TryToReduceAlignment:
    var x: Int64  # Natural alignment: 8

# align_of[TryToReduceAlignment]() == 8 (natural wins, cannot reduce)

struct Container:
    var aligned: CacheLineAligned
    var other: Int

# align_of[Container]() == 64 (inherits from field)
```

### Error Cases

```mojo
@align(3)  # Error: must be power of 2
struct Bad1:
    var x: Int

@align(-1)  # Error: must be positive
struct Bad2:
    var x: Int

@align  # Error: requires exactly one argument
struct Bad3:
    var x: Int

@align(n)  # Error (for now): requires compile-time constant
struct Bad4[n: Int]:
    var x: Int
```

## Implementation Experience

I have a working implementation for review.

## Future Work (not in PR above)

### 1. Parametric Alignment

Support alignment as a struct parameter:

```mojo
@align(alignment)
struct AlignedBuffer[alignment: Int]:
    var data: SIMD[DType.uint8, 64]
```

This requires propagating alignment through the parametric type system.

### 2. Field-Level Alignment

Support alignment on individual struct fields:

```mojo
struct MixedAlignment:
    @align(64)
    var hot_data: Atomic[Int64]
    var cold_data: Int
```

### 3. `@packed` Decorator

Complement `@align` with `@packed` for reducing alignment/padding:

```mojo
@packed
struct CompactData:
    var a: UInt8
    var b: UInt32  # No padding before this
```

### 4. Standard Library Updates

Once `@align` is available, update:

- `TensorMap` and `TMADescriptor` to use `@align(64)`
- Consider `BlockingSpinLock` for cache-line alignment
- Evaluate `_ArcPointerInner` for false-sharing prevention

## Alternatives Considered

### 1. Alignment as a Struct Parameter

```mojo
struct Aligned[alignment: Int = 0]:
    var x: Int
```

**Rejected:** This would require all structs to have an alignment parameter,
which is overly verbose for the common case.

### 2. Only Support via `stack_allocation`

Keep the current workaround pattern.

**Rejected:** Verbose, doesn't help heap allocations automatically, alignment
requirement not visible in type definition.

## References

- [C++ alignas specifier](https://en.cppreference.com/w/cpp/language/alignas)
- [Rust
  #[repr(align(N))]](https://doc.rust-lang.org/reference/type-layout.html#the-alignment-modifiers)
- [NVIDIA TMA Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#tensor-memory-access)
