---
title: Mojo nightly
---

This version is still a work in progress.

## Highlights

- Code that performs many implicit conversions, most visibly large collection
  literals, compiles faster: the compiler no longer runs parameter inference on
  constructors that cannot be used for an implicit conversion in the first
  place. Files that are mostly data, such as the standard library's Unicode
  lookup tables, compile about 1.3x faster.

## Documentation

## Language enhancements

- Unknown declaration errors now suggest a unique near-miss spelling from the
  enclosing scopes (for example `coun` → `count`), with a replace-token fixit.

- Mojo now supports contextually inferred member references: a leading-dot form
  such as `.red` or `.float64` resolves against the expected type of the
  expression, so you can omit a redundant type name when context already
  supplies it. Static methods, parametric static methods, parentheses, attribute
  chains, and typed collection literals all work:

  ```mojo
  struct Color(ImplicitlyCopyable):
      comptime red = Color(...)
      comptime green = Color(...)

      @staticmethod
      def hsb_to_rgb(h: Int, s: Int, b: Int) -> Color:
          return Color(...)

      def opacity(self, amount: Float64) -> Color:
          return Color(...)
      def __init__(out self, ...):

  def takes_color(c: Color):
  def takes_colors(colors: List[Color]):

  takes_color(.green)
  takes_color(.hsb_to_rgb(120, 100, 50))
  takes_color(.red.opacity(0.5))
  var x: Color = .red
  takes_colors([.red, .green])
  ```

  Without a contextual type, `.member` is an error.

- A `thin` function type can now carry trailing `where` clauses, constraining
  the parameters it declares. This lets a generic algorithm state what it
  promises the function it is handed, instead of leaving the constraint to be
  restated at every binding site.

  ```mojo
  comptime Kernel = def[w: Int](Int) thin -> None where (
      w > 0, "width must be positive"
  )

  def apply[F: Kernel](x: Int):
      F[4](x)     # ok
      F[0](x)     # error: violated constraint
  ```

  The clause binds to the innermost function type, so a declaration-level
  `where` that follows a function-type result needs that result parenthesized:

  ```mojo
  def make[n: Int]() -> (def() thin -> None) where n > 0: ...
  ```

## Language changes

- A walrus expression now always yields its right-hand side, uniformly for
  every kind of target.

- Binding a constrained function to a function type that declares no matching
  `where` clause is now an error, instead of silently dropping the constraint.
  Declare the obligation on the function type (now that a `thin` function type
  can carry a trailing `where` clause) or bind a function that does not require
  it. Passing an unconstrained function where a constrained type is expected is
  still allowed and still free.

- Renamed the `@parameter` decorator on parametric closures to
  `@__parameter`. The deprecated `@parameter if` / `@parameter for`
  forms are unchanged; prefer `comptime if` / `comptime for` for
  compile-time control flow.

- The module & package system:

  - Directories may now have "namespace" semantics; a single directory name may
    resolve across distinct locations on disk which share that name.

    ```mojo
    # .
    # ├── one
    # │   └── foo
    # │       └── bar.mojo
    # └── two
    #     └── foo
    #         └── baz.mojo
    #
    # Compiles with -Ione -Itwo
    import foo.bar
    import foo.baz
    ```

  - Importing functions with the same name from different modules, combining
    them into one overload set, is now an error, following a period of
    deprecation.

  - Intra-package accesses without explicit `import`s are now an error,
    following a period of deprecation.

- Use of the `read` argument convention is now a hard error, following a period
  of deprecation; use `imm` instead.

## Library stabilizations

- String
  - `def __init__(out self):`
  - `def __init__(out self, *, capacity_bytes: Int):`
  - `def reserve_bytes(mut self, new_capacity_bytes: Int, /):`
- List
  - `def append(mut self, var value: Self.T, /):`

## Library performance improvements

- Files with many t-string (`t"..."`) literals compile faster: the
  compile-time step that encodes each literal's format string (part of
  elaboration, not the whole compile) is about 7x faster. The effect on
  total build time scales with how many t-string literals a file has.

## Library changes

- `Coord` has a new `replace[at](value)` method that returns a `Coord` with
  the element at `at` swapped for `value`, keeping the other elements' types. A
  statically known element (`ComptimeInt`) has no runtime storage to assign
  into, so overwriting one with a runtime value yields a `Coord` of a different
  type rather than mutating in place. Unlike `make_dynamic()`, which converts
  every element to a `Scalar`, the untouched dimensions keep their compile-time
  values:

  ```mojo
  var c = Coord(ComptimeInt[3](), ComptimeInt[4]())
  var moved = c.replace[1](Int64(7))  # Coord(ComptimeInt[3](), Int64(7))
  ```

- Python functions exposed through `PythonModuleBuilder.def_function()`,
  `PythonTypeBuilder.def_method()`, and `PythonTypeBuilder.def_staticmethod()`
  no longer have a library-imposed limit on positional arguments.

- `List.extend` and `List.resize` now grow geometrically, so repeatedly
  extending or resizing by a small increment is no longer quadratic. As a
  result `capacity()` can report more than was asked for. `reserve` is
  unchanged and still allocates exactly what you request.

- `CompilationTarget` has a new `is_arm()` predicate, and `is_x86()` now
  reports the architecture rather than SSE4 availability. Both read the
  architecture from the target triple, so they no longer vary with
  `--target-cpu`. This changes `is_x86()` on x86 targets without SSE4.1 — most
  visibly the baseline `x86-64` CPU, where it used to return `False`. Use
  `has_sse4()`, `has_avx2()`, and friends to gate code on a specific
  instruction set.

- `Hasher.update` now takes a `ImmSpan[Byte, _]` instead of
  `Some[Hashable]`. Code such as `hasher.update(some_subobject)`
  in `__hash__` implementations should be changed to
  `some_subobject.__hash__(hasher)`.

- `CompilationTarget` can now describe RISC-V targets: `is_riscv()`,
  `is_rv32()`, and `is_rv64()` report the architecture, and
  `has_riscv_extension["m"]()` reports a single ISA extension by its lowercase
  LLVM name. An extension implied by another counts as present, so a target
  built with `d` also reports `f`. It is always `False` on a non-RISC-V target,
  and rejects an uppercase name at compile time.

  Selecting a RISC-V CPU or ISA string now resolves the extensions it implies,
  so `--target-cpu=sifive-e31` and `--march=rv32imac` both report `m`, `a`, and
  `c`. Previously either one reported only the base integer ISA.

- `Bencher.bench_function()` now takes a raising closure.

- The zero-argument `Bench.bench_function()` overload now takes a raising
  closure as a runtime argument. The compile-time parameter form
  `bench_function[fn]()` for a raising zero-argument body has been removed.

- The remaining compile-time parameter forms of `Bench.bench_function()` and
  `Bencher.iter()` have been removed. Pass the closure as a runtime argument.

- `Bencher.iter_preproc()` now takes its closures as runtime arguments instead
  of compile-time parameters, along with an explicit state value that is passed
  mutably to both: the preprocessing function prepares the state before each
  timed call of the benchmarked function, so state no longer has to be shuttled
  through mutable captures.

- `Bencher.bench_with_input()` now takes its benchmark closure as a runtime
  argument. Its register-passable overload accepts both non-raising and raising
  closures.

- `Bencher.iter_custom()` now only takes its closure as a runtime argument. The
  compile-time parameter form has been removed.

- `std.python.numpy` now handles multi-dimensional NumPy arrays, not just 1-D:

  - `copy_to_numpy_tensor()` copies a `Span` into a new NumPy array of a given
    shape. The shape is a `Coord`, so extents may be compile-time (`Idx[N]`) or
    runtime (`Int`) in any mix.

  - `from_numpy_tensor()` borrows an N-D C-contiguous array as a `NumPyView`,
    which holds the buffer and its shape together and indexes as
    `view[i, j]`.

  ```mojo
  from std.python.numpy import copy_to_numpy_tensor, from_numpy_tensor
  from std.utils.coord import Coord, Idx

  var values: List[Float64] = [0, 1, 2, 3, 4, 5]
  var arr = copy_to_numpy_tensor(values, Coord(Idx[2], Idx[3]))

  var view = from_numpy_tensor[DType.float64, 2](arr)
  var value = view[1, 2]
  ```

  The existing 1-D `copy_to_numpy_array()` and `from_numpy_array()` are
  unchanged.

- `Array` now conforms to `Comparable` when its element type does, adding `<`,
  `<=`, `>`, and `>=`. The ordering is **lexicographic**: the first differing
  element decides, so `[1, 5] < [2, 3]` is `True`.

- `StringDict` now conforms to `Writable` when its value type is `Writable`,
  matching the existing behavior of `Dict`. This lets you `print()` a
  `StringDict` or convert it to a `String`.

- The `chars` argument of `strip()`, `lstrip()` and `rstrip()` on `StringSpan`,
  `String` and `StringLiteral` is now an `ImmStringSpan`, so a mutable string
  is accepted as `chars`, including the string being stripped (`s.strip(s)`).

- `StringDict.__getitem__()` now accepts a `StringSpan`, so you can index a
  `StringDict` with a borrowed string view without first allocating a
  `String` just to perform the lookup.

- Renamed the variadic type-list parameter on `Tuple` and `VariadicPack` to
  `Ts`, standardizing the naming convention used across the standard library.
  The old name, `element_types`, remains as a deprecated alias.

- Added experimental `DType.float6_e2m3fn` and `DType.float6_e3m2fn`, the two
  6-bit encodings from the
  [Open Compute microscaling specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf).
  Both are finite-only, so neither has an inf nor a NaN encoding.

  These are experimental storage formats for packed weights rather than
  general-purpose numeric types, and standard library support is deliberately
  partial. As with the existing `DType.float4_e2m1fn`, they are excluded from
  `is_numeric()`, arithmetic is not implemented, and converting to or from
  another floating-point type is unsupported on every target, so values cannot
  be printed either.

- `Array` now conforms to `Defaultable` when its type `T` is also `Defaultable`.

- `Array` now supports concatenation with the `concat` method when its type `T`
  is `Movable`. Both operands are consumed and their elements are moved into the
  new array, whose length is the sum of the operands' lengths.

- `Array` now supports repetition with the `repeat` method when its type `T` is
  `Copyable`. The array is consumed: its elements are copied into all but the
  last repetition and moved into the last one.

- Deprecated `is_trivially_movable()`, `is_trivially_copyable()`, and
  `is_trivially_deletable()` in `std.memory` in favor of
  `IsTriviallyMovable[T]`, `IsTriviallyCopyable[T]`, and
  `IsTriviallyDeinitable[T]` in `std.traits`. The replacements are `comptime`
  predicates rather than functions, so drop the call parens at use sites, for
  example `IsTriviallyCopyable[T]` instead of `is_trivially_copyable[T]()`.

- Renamed `UnsafeMaybeUninit` to `MaybeUninit`. It conforms to `Movable`,
  `Copyable`/`ImplicitlyCopyable`, and `Deinitable` only when the contained
  type's own move, copy, or implicit deinitializer is trivial, since
  moving, copying, or destroying a `MaybeUninit` only touches its raw bits,
  never the contained value's own lifecycle methods. Gating conformance this way
  turns what would otherwise be silent memory-safety bugs into compile-time
  errors.

- Added `deinit()`, for any `Deinitable` type, to explicitly extend a value's
  lifetime up to a specific point and run its deinitializer there.

- `Atomic` is now parameterized on a value type `T` instead of a `DType`.
  Update call sites from `Atomic[DType.float32]` to `Atomic[Float32]`. The
  atomic operations (`load()`, `store()`, `fetch_add()`, `compare_exchange()`,
  and so on) still only support `Scalar` types.

- Added `Pointer[T].unsafe_write(def() -> T)`, which initializes the pointee
  with the value returned by a closure, constructing it directly in place rather
  than moving an already-constructed value there. Unlike `unsafe_write(var T)`,
  this does not require the pointee type to be `Movable`.

- `Array[T, N]` has a new `fill_with=` constructor that calls a function with
  each index in `[0, N)` and writes its result into that position, replacing
  the `Array(uninitialized=True)` plus manual fill-loop idiom.

- `List`'s element type is now bounded by `AnyType` instead of `Movable`.

- `List` has a new `fill_with=` constructor that calls a function with each
  index in `[0, length)` and writes its result into that position, without
  requiring the element type to be `Movable`.

- Added `write()` to `MaybeUninit` and `Pointer`, as a safe counterpart to
  `unsafe_write()` for types that are trivially deinitializable (for example
  `Int`). Since a trivial deinitializer is a no-op, overwriting a live value
  through `write()` can't leak a resource, so it's callable without first
  destroying the previous value. Prefer it over `unsafe_write()` whenever the
  pointee type is trivially deinitializable.

- `Pointer.mut_cast` is now deprecated. Developers should prefer using explicit
  mutabilites at the callsite via `MutPointer` or `ImmPointer`. If mut casting
  is needed (it should try to be avoided) - you can use `unsafe_mut_cast`.

- Added `ptr()` to `StringLiteral`, `CStringSlice`, `ArcPointer`, and
  `OwnedPointer`, deprecating their `unsafe_ptr()` methods. These types
  always hold a valid, live value, so a pointer to it is never unsafe.

- The following APIs have been migrated to unified closures: `sort`,
  `debug_assert`, `Span.apply`.

- Uncaught exceptions now print to `stderr`, not `stdout`.

## GPU programming

- The `max.gpu` package now mirrors everything reachable from `std.gpu`, making
  it a complete entry point for accelerator programming. Prefer `max.gpu`, which
  is becoming the only public source for these utilities.

- The `std.gpu` package is now private, as `std._gpu`. `max.gpu` is the only
  public source for the GPU primitives, and its API reference is the only
  published one; `/docs/std/gpu/...` pages redirect to `/api/mojo/max/gpu/...`.
  Replace `from std.gpu import ...` with `from max.gpu import ...`; a failed
  `std.gpu` import carries a note pointing at the new home.

## Tooling changes

- `mojo doc` now reports the condition of a conditional trait conformance, and
  the generated API docs show it alongside the trait. Previously the condition
  was dropped, making a conditional conformance indistinguishable from an
  unconditional one. Also fixed rendering of some `where` clauses.

- The new `@__doc_inline` decorator on an import statement documents the
  imported symbols in the importing module, so a package can document an API
  it re-exports from private modules. Wildcard (`import *`) and renamed
  (`import ... as ...`) imports are not supported.

  ```mojo
  @__doc_inline
  from ._impl import Widget, make_widget
  ```

## Removed

- Legacy constructs replaced in 1.0 were removed in this release, including
  the `fn`, `alias` and `__comptime_assert` keywords, `@parameter if` and
  `@parameter for` syntax.

This release completes the removal of APIs deprecated during the v1.0 cycle.

- Removed `is_npu()` from `max.gpu.host.info`. A custom op's `target` says
  whether the op was placed on the host or on a device, not which kind of
  device it landed on -- that comes from the build, via `_accelerator_arch()`,
  `get_gpu_target()` or `ctx.default_device_info`. `is_npu()` invited the
  target string to be asked a question it cannot answer, and nothing called
  it. `is_accelerator()` is unchanged in meaning for every target a graph
  produces: use it to ask whether an op is off the host.

- Implicit variable declaration now produces an error instead of a warning. The
  walrus operator also only overwrite existing values, not implicitly declare
  new ones.

- Removed the temporary `InlineArray` alias for `Array`, including its
  re-exports from `std.collections` and the prelude. Use `Array` directly.

- Removed redundant `Int` overloads across the standard library:
  `count_leading_zeros()`, `count_trailing_zeros()`, `bit_reverse()`,
  `byte_swap()`, `pop_count()`, `log2_ceil()`, `next_power_of_two()`, and
  `prev_power_of_two()` in `std.bit`; `broadcast()` in `std.gpu.primitives`
  (including the `UInt` overload); `readfirstlane()` in `std.sys`; and
  `umod()` in `std.math.uutils`. `Int` is an alias for `Scalar[DType.int]`,
  so the generic `SIMD` overloads already accept `Int` arguments and return
  `Int`; call sites need no changes. As a side effect, `broadcast()` on
  `Int`/`UInt` values now shuffles the full 64-bit value instead of silently
  truncating it to 32 bits.

- Removed the `Int` overloads of `rotate_bits_left()` and
  `rotate_bits_right()` in `std.bit`. The `SIMD` overloads now accept any
  integral element type instead of only unsigned ones — rotation is a pure
  bit-pattern operation, so signed and unsigned rotate identically — and
  therefore handle `Int` arguments directly. Call sites need no changes.

- Removed the `std.gpu.profiler` module and its `ProfileBlock` context manager.
  It timed host wall-clock, not GPU work, and reported the elapsed time with the
  operands reversed. Time a block of host code with
  [`perf_counter_ns()`](/docs/std/time/time/perf_counter_ns/) directly, and use
  a GPU profiler such as Nsight Systems or `rocprof` for device timings.

- Removed `memcmp` and its `std.memory` re-export. Use `unsafe_memcmp`
  instead.

- Removed `String.set_byte_length()`, an internal helper that set the length
  field without reserving capacity.

- Removed the `validate` parameter from
  [`b64decode()`](/docs/std/base64/base64/b64decode/), which now always
  validates. Passing `validate=False` did not skip any work on valid input; it
  only turned characters outside the base64 alphabet into silently corrupt
  output bytes. Drop `[validate=True]` from existing calls; calls that relied on
  the default now raise instead of returning garbage.

- Removed the origin aliases left over from the `Immut` to `Imm` and
  `External` to `Untracked` renames. Use the surviving spelling in each case:
  `ImmOrigin` for `ImmutOrigin`, `ImmUnsafeAnyOrigin` for
  `ImmutUnsafeAnyOrigin`, `ImmStaticOrigin` for `StaticConstantOrigin`,
  `UntrackedOrigin` for `ExternalOrigin`, `MutUntrackedOrigin` for
  `MutExternalOrigin`, and `ImmUntrackedOrigin` for both
  `ImmutUntrackedOrigin` and `ImmutExternalOrigin`.

- Removed the redundant `Int` overloads of `sqrt()`, `fma()`, `align_down()`,
  `align_up()`, `clamp()`, and `iota()` from `std.math`. `Int` is an alias for
  `Scalar[DType.int]`, so the generic `SIMD` overloads already accept `Int`
  arguments and return `Int`; call sites need no changes.

- Removed the pre-unification pointer aliases `MutUnsafePointer`,
  `ImmUnsafePointer`, `ImmutUnsafePointer`, `ImmutOpaquePointer`,
  `ImmutPointer`, and `OptionalUnsafePointer`. Use `MutPointer`, `ImmPointer`,
  `ImmOpaquePointer`, and `OptionalPointer` instead. `UnsafePointer` itself
  remains available, but is deprecated in favor of `Pointer`.

- Removed the raw memory functions superseded by their `unsafe_`-prefixed
  spellings: `memcpy`, `memset`, `memset_zero`, `uninit_move_n`,
  `uninit_copy_n`, and `destroy_n`. Use `unsafe_memcpy`, `unsafe_memset`,
  `unsafe_memset_zero`, `unsafe_uninit_move_n`, `unsafe_uninit_copy_n`, and
  `unsafe_destroy_n` instead.

- Removed the `size` aliases left from the `size` to `length` rename:
  `SIMD.size`, `Array.size`, `TypeList.size`, and the `SIMDSize` alias for
  `SIMDLength`. Use `length` and `SIMDLength`.

- Removed the `as_immutable()` and `get_immutable()` methods on `Pointer`,
  `Span`, and `StringSpan`. Use `as_imm()`.

- Removed the `ImmutSpan` alias. Use `ImmSpan`.

- Removed `String.as_string_slice()`. Construct a `StringSpan` from the string
  instead: `StringSpan(my_string)`.

- Removed the `ImplicitlyDestructible` and `ImplicitlyDeletable` aliases. Use
  `Deinitable`.

- Removed the deprecated ownership-transfer methods: `List.steal_data()` and
  `OwnedPointer.steal_data()` are now `unsafe_take_allocation()`,
  `OwnedPointer.take()` is `into_inner()`, and `Variant.take()` and
  `Variant.unsafe_take()` are `unwrap()` and `unsafe_unwrap()`.

- Removed the `Pointer` methods superseded by their `unsafe_`-prefixed
  spellings: `as_noalias_ptr()`, `destroy_pointee()`, `destroy_pointee_with()`,
  `init_pointee_move()`, `init_pointee_copy()`, and `init_pointee_move_from()`.
  Use `unsafe_as_noalias()`, `unsafe_deinit_pointee()`,
  `unsafe_deinit_pointee_with()`, `unsafe_write()`, and
  `unsafe_write_move_from()`. The `Pointer.type` alias for `Pointer.T` is gone
  as well.

- Removed the `ConditionalType` type function and the `std.utils.type_functions`
  module. Use the ternary expression `T if cond else U`.

- Removed `trait_downcast()`. Constrain on the trait instead, with
  `conforms_to(type_of(src), Trait)` in a `where` clause or a
  `comptime assert`.

- Removed the parametric `benchmark.run[func]()` overloads. Pass the function as
  an argument to `run(f)` instead, which accepts a unified closure.

- Removed `AnyCoroutine`, `Coroutine` and `RaisingCoroutine` from the prelude,
  and made the module that defines them private. Mojo's async support is
  unfinished, and these types being globally visible led people to build on an
  API that carries no stability guarantees. `async def` is unaffected: the
  compiler still synthesizes these types for you, so they continue to appear in
  inferred types and diagnostics. There is no supported way to name them
  directly.

- Removed the async task API from the public `std.runtime.asyncrt` module,
  which is now private. `initialize_runtime()` and `parallelism_level()` are
  unaffected and have moved up to the `std.runtime` package, so import them
  from `std.runtime` instead of `std.runtime.asyncrt`.

- Removed support for `.mojopkg` files after a period of deprecation. Use
  `.mojoc` files instead.

## Fixed

- `unsafe_uninit_move_n()` and `unsafe_uninit_copy_n()` with `overlapping=True`
  now handle an overlap in either direction when `T` is not trivially movable
  or copyable. They always walked front-to-back, so a `dest` above `src`
  overwrote elements that had not been moved or read yet.

- `SIMD.__init__(py=...)` now reads unsigned dtypes through the unsigned CPython
  entry point (`PyLong_AsSize_t`). Constructing an unsigned `SIMD` from a Python
  int in `[2**63, 2**64)` no longer overflows, and a negative Python int now
  raises instead of silently wrapping to the maximum value.

- A union whose widest member is a `SIMD[DType.bool, N]` with `N > 1`, such as
  `Optional[SIMD[DType.bool, 2]]`, now compiles.

- A `where` clause naming a type that an enclosing `where` clause constrained
  to a tighter trait can now be proven. Calling a method declared
  `where Ts.contains[T]()` with such a `T` failed with `lacking evidence to
  prove correctness`, even though `T` was plainly in `Ts`.

- `hash()` on a floating-point `SIMD` value now normalizes the sign of zero, so
  `hash(-0.0) == hash(0.0)`. Hashing the raw bit pattern broke the `Hashable`
  contract that equal values hash equally: a `Dict` or `Set` could hold both
  `-0.0` and `0.0` as separate keys even though they compare equal, and a
  lookup could then return a value stored under the other key.

- `mojo build` can cross-compile to RISC-V again. Emitting LLVM IR, assembly,
  or an object for a `riscv32` or `riscv64` triple failed with `target '...'
  is not supported by this build`.

- `mojo build --print-supported-targets` no longer lists targets that the
  compiler cannot generate code for.

- `mojo build --emit asm` and `--emit llvm` now always write the offload kernel
  files next to the host output file. Building a kernel that an earlier build
  had already compiled could write them into the earlier build's output
  directory, or skip them with no diagnostic.

- Parametric `raises` now accepts any primary expression as the thrown type in
  a function signature, matching the syntax positions where types otherwise
  appear. This most notably fixes `raises Self.SomeAssocType` on trait and
  struct methods, which would previously fail with an error. The parenthesized
  workaround (`raises (Self.DriveErrorType)`) is no longer required.

- An integer `range()` with a step of zero is now always empty. It previously
  used to be an infinite loop - iterating forever at runtime, and hanging the
  compiler at comptime.

- A strided `range()` no longer iterates forever when the element after the
  last one falls outside the element type, as in
  `range(UInt8(250), UInt8(255), UInt8(2))`. The cursor used to wrap past the
  type's limit and land back inside the range, so iteration restarted near the
  opposite limit and never agreed with `len()`. This affected signed and
  unsigned ranges in both step directions.

- `reversed()` on a scalar `range()` no longer yields an empty iterator when
  the range starts within one step of the element type's limit, as in
  `reversed(range(Int8.MIN, Int8.MIN + 8, Int8(1)))`. Unsigned ranges, and
  ranges whose span overflows their element type, are fixed by the same change.

  Reversing an already-reversed range, as in `reversed(reversed(range(10)))`,
  is now a compile-time error.

- Fixed `ceildiv()` returning `0` for unsigned operands near the type's
  maximum value. The unsigned code path computed `numerator + denominator -
  1`, which overflows and wraps for large operands; it now derives the
  ceiling from the floor division and remainder instead.

- `Counter.most_common(n)` now returns all elements when `n` exceeds the
  number of unique elements, matching Python, instead of aborting.

- `os.path.join()` now inserts separators based on the accumulated path rather
  than the first argument, so `join("/", "a", "b")` returns `/a/b` (previously
  `/ab`) and `join("a", "b/", "c")` returns `a/b/c` (previously `a/b//c`).

- `base64.b64decode()` now raises an error when the input length is not
  divisible by 4 instead of reading past the end of the input (or aborting
  when asserts are enabled).

- On macOS, `os.stat()` and `os.lstat()` no longer return a negative
  `st_mode` for regular files. The underlying `mode_t` and `nlink_t` C type
  aliases were declared as signed 16-bit integers, but macOS defines them as
  unsigned, so any mode with the `S_IFREG` bit set (every regular file)
  sign-extended into a negative `Int`.

- On macOS, `os.stat()` and `os.lstat()` now report file timestamps with the
  correct nanosecond values. `_CTimeSpec.as_nanoseconds()` previously treated
  the `timespec.tv_nsec` field as microseconds, inflating the subsecond
  component by a factor of up to 1,000.

- `PythonObject` no longer leaks a CPython reference per positional argument
  when calling a Python object, nor when setting an item, attribute, or set
  literal element.

- `atol()` (and therefore `Int(String)`) now raises for every value outside
  the `Int` range. Values just past `Int.MAX` (such as `Int.MAX + 1`) no
  longer wrap silently, and `Int.MIN` parses correctly by design rather than
  by wraparound.
  `atol()` will also now raise instead of aborting on a string that holds only
  whitespace, or only whitespace and a sign.

- Every value of a struct type whose `@align(N)` exceeds its natural
  alignment is now aligned to `N`, including every element of an array or a
  `List` of that type.

- [`b64decode()`](/docs/std/base64/base64/b64decode/) now ignores ASCII
  whitespace in its input, so base64 text wrapped across lines by a MIME
  encoder or the `base64` command line tool decodes without the caller
  stripping it first. Only the six ASCII whitespace bytes are ignored; unlike
  Python's `base64.b64decode()`, any other byte outside the base64 alphabet
  still raises. The "length must be divisible by 4" error now counts only the
  significant characters.
