# Mojo unreleased changelog

This is a list of UNRELEASED changes for the Mojo language and tools.

When we cut a release, these notes move to `changelog-released.md` and that's
what we publish.

[//]: # Here's the template to use when starting a new batch of notes:
[//]: ## UNRELEASED
[//]: ### ⭐️ New
[//]: ### 🦋 Changed
[//]: ### ❌ Removed
[//]: ### 🛠️ Fixed

## UNRELEASED

### 🔥 Legendary

### ⭐️ New

- Add a `sort` function for list of `ComparableCollectionElement`s.
  [PR #2609](https://github.com/modularml/mojo/pull/2609) by
  [@mzaks](https://github.com/mzaks)

- Mojo functions can return an auto-dereferenced refeference to storage with a
  new `ref` keyword in the result type specifier.  For example:

  ```mojo
  struct Pair:
    var first: Int
    var second: Int
    fn get_first_ref(inout self) -> ref[__lifetime_of(self)] Int:
      return self.first
  fn show_mutation():
    var somePair = ...
    get_first_ref(somePair) = 1
  ```

  This approach provides a general way to return an "automatically dereferenced"
  reference of a given type.  Notably, this eliminates the need for
  `__refitem__` to exist.  `__refitem__` has thus been removed and replaced with
  `__getitem__` that returns a reference.

- Mojo has introduced `@parameter for`, a new feature for compile-time
  programming. `@parameter for` defines a for loop where the sequence and the
  induction values in the sequence must be parameter values. For example:

  ```mojo
  fn parameter_for[max: Int]():
      @parameter
      for i in range(max)
          @parameter
          if i == 10:
              print("found 10!")
  ```

  Currently, `@parameter for` requires the sequence's `__iter__` method to
  return a `_StridedRangeIterator`, meaning the induction variables must be
  `Int`. The intention is to lift these restrictions in the future.

- Mojo added support for the inferred parameters. Inferred parameters must
  appear at the beginning of the parameter list and cannot be explicitly
  specified by the user. They are declared to the left of a `//` marker, much
  like positional-only parameters. This allows programmers to define functions
  with dependent parameters to be called without the caller specifying all the
  necessary parameters. For example:

  ```mojo
  fn parameter_simd[dt: DType, //, value: Scalar[dt]]():
      print(value)

  fn call_it():
      parameter_simd[Int32(42)]()
  ```

  In the above example, `Int32(42)` is passed directly into `value`, the first
  non-inferred parameter. `dt` is inferred from the parameter itself to be
  `DType.int32`.

  This also works with structs. For example:

  ```mojo
  struct ScalarContainer[dt: DType, //, value: Scalar[dt]]:
      pass

  fn foo(x: ScalarContainer[Int32(0)]): # 'dt' is inferred as `DType.int32`
      pass
  ```

  This should make working with dependent parameters more ergonomic.

- Mojo now allows functions overloaded on parameters to be resolved when forming
  references to, but not calling, those functions. For example, the following
  now works:

  ```mojo
  fn overloaded_parameters[value: Int32]():
      pass

  fn overloaded_parameters[value: Float32]():
      pass

  fn form_reference():
      alias ref = overloaded_parameters[Int32()] # works!
  ```

- Mojo now supports adding a `@deprecated` decorator on structs, functions,
  traits, aliases, and global variables. The decorator marks the attached decl
  as deprecated and causes a warning to be emitted when the deprecated decl is
  referenced in user code. The decorator requires a deprecation message to be
  specified as a string literal.

  ```mojo
  @deprecated("Foo is deprecated, use Bar instead")
  struct Foo:
      pass

  fn outdated_api(x: Foo): # warning: Foo is deprecated, use Bar instead
      pass

  @deprecated("use another function!")
  fn bar():
      pass

  fn techdebt():
      bar() # warning: use another function!
  ```

- Mojo has changed how `def` arguments are processed.  Previously, by default,
  arguments to a `def` were treated treated according to the `owned` convention,
  which makes a copy of the value, enabling that value to be mutable in the callee.
  This "worked", but was a major performance footgun, and required you to declare
  non-copyable types as `borrowed` explicitly.  Now Mojo takes a different approach:
  it takes the arguments as `borrowed` (consistent with `fn`s) but will make a local
  copy of the value **only if the argument is mutated** in the body of the function.
  This improves consistency, performance, and ease of use.

- `int()` can now take a string and a specified base to parse an integer from a
  string: `int("ff", 16)` returns `255`. Additionally, if a base of zero is
  specified, the string will be parsed as if it was an integer literal, with the
  base determined by whether the string contains the prefix `"0x"`, `"0o"`, or
  `"0b"`. ([PR #2273](https://github.com/modularml/mojo/pull/2273) by
  [@artemiogr97](https://github.com/artemiogr97), fixes
  [#2274](https://github.com/modularml/mojo/issues/2274))

- Mojo now supports types to opt in to use the `abs` and `round` functions by
  implementing the `__abs__` and `__round__` methods (i.e. by conforming to the
  new `Absable` and `Roundable` traits), respectively, e.g.:

  ```mojo
  from math import sqrt

  @value
  struct Complex(Absable, Roundable):
      var re: Float64
      var im: Float64

      fn __abs__(self) -> Self:
          return Self(sqrt(self.re * self.re + self.im * self.im), 0.0)

      fn __round__(self) -> Self:
          return Self(round(self.re), round(self.im))
  ```

- User defined types can now also opt in to use the `pow` function by
  implementing the `__pow__` method (and thus conforming to the new `Powable`
  trait). As before, these types will also benefit from being able to use the
  `**` operator.

- Mojo now allows types to opt in to use the `floor()`, `ceil()`, and `trunc()`
  functions in the `math` module by implementing the `__floor__()`,
  `__ceil__()`, and `__trunc__()` methods (and so conforming to the new
  `math.Floorable`, `math.Ceilable`, and `math.Truncable` traits, respectively).
  For example:

  ```mojo
    from math import Ceilable, Floorable, Truncable, ceil, floor, trunc

    @value
    struct Complex(Ceilable, Floorable, Truncable):
      var re: Float64
      var im: Float64

      fn __ceil__(self) -> Self:
          return Self(ceil(re), ceil(im))

      fn __floor__(self) -> Self:
          return Self(floor(re), floor(im))

      fn __trunc__(self) -> Self:
          return Self(trunc(re), trunc(im))
  ```

- You can now use the builtin `any()` and `all()` functions to check for truthy
  elements in a collection. Because `SIMD.__bool__()` is now constrained to
  `size=1`, You must explicity use these to get the truthy value of a SIMD
  vector. This avoids common bugs around implicit conversion of `SIMD` to
  `Bool`.
    ([PR #2600](https://github.com/modularml/mojo/pull/2600) by [@helehex](https://github.com/helehex))

  For example:

  ```mojo
    fn truthy_simd():
        var vec = SIMD[DType.int32, 4](0, 1, 2, 3)
        if any(vec):
            print("any elements are truthy")
        if all(vec):
            print("all elements are truthy")
  ```

- Add an `InlinedArray` type that works on memory-only types.
  Compare with the existing `StaticTuple` type, which is conceptually an array
  type, but only worked on `AnyTrivialRegType`.
    ([PR #2294](https://github.com/modularml/mojo/pull/2294) by [@lsh](https://github.com/lsh))

- Base64 decoding support has been added.
    ([PR #2364](https://github.com/modularml/mojo/pull/2364) by [@mikowals](https://github.com/mikowals))

- Add Base16 encoding and decoding support.
  ([PR #2584](https://github.com/modularml/mojo/pull/2584)
   by [@kernhanda](https://github.com/kernhanda))

- Add `repr()` function and `Representable` trait.
    ([PR #2361](https://github.com/modularml/mojo/pull/2361) by [@gabrieldemarmiesse](https://github.com/gabrieldemarmiesse))

- Add `SIMD.shuffle()` with `StaticIntTuple` mask.
    ([PR #2315](https://github.com/modularml/mojo/pull/2315) by [@mikowals](https://github.com/mikowals))

- Invoking `mojo package my-package -o my-dir` on the command line, where
  `my-package` is a Mojo package source directory, and `my-dir` is an existing
  directory, now outputs a Mojo package to `my-dir/my-package.mojopkg`.
  Previously, this had to be spelled out, as in `-o my-dir/my-package.mojopkg`.

- The Mojo Language Server now reports a warning when a local variable is unused.

- Implicit variable definitions in a `def` are more flexible: you can now
  implicitly declare variables as the result of a tuple return, using
  `a,b,c = foo()`, and can now shadow global immutable symbols using
  `slice = foo()` without getting a compiler error.

- The `math` module now has `CeilDivable` and `CeilDivableRaising` traits that
  allow users to opt into the `math.ceildiv` function.

- Mojo now allows methods to declare `self` as a `Reference` directly, which
  can be useful for advanced cases of parametric mutabilty and custom lifetime
  processing.  Previously it required the use of an internal MLIR type to
  achieve this.

- The `is_mutable` parameter of `Reference` and `AnyLifetime` is now a `Bool`,
  not a low-level `__mlir_type.i1` value.

  This improves the ergonomics of spelling out a
  `Reference` type explicitly. For example, to define a struct holding a
  `Reference`, you can now write:

  ```mojo
  struct Foo[is_mutable: Bool, lifetime: AnyLifetime[is_mutable].type]:
      var data: Reference[Int32, is_mutable, lifetime]
  ```

  Or to specify a field that is always immutable, `False` can be specified
  as the mutability:

  ```mojo
  struct Foo[lifetime: AnyLifetime[False].type]:
      var data: Reference[Int32, False, lifetime]
  ```

- `object` now implements all the bitwise operators.
    ([PR #2324](https://github.com/modularml/mojo/pull/2324) by [@LJ-9801](https://github.com/LJ-9801))

- A new `--validate-doc-strings` option has been added to `mojo` to emit errors
  on invalid doc strings instead of warnings.

- Several `mojo` subcommands now support a `--diagnostic-format` option that
  changes the format with which errors, warnings, and other diagnostics are
  printed. By specifying `--diagnostic-format json` on the command line, errors
  and other diagnostics will be output in a structured
  [JSON Lines](https://jsonlines.org) format that is easier for machines to
  parse.

  The full list of subcommands that support `--diagnostic-format` is as follows:
  `mojo build`, `mojo doc`, `mojo run`, `mojo package`, and `mojo test`.
  Further, the `mojo test --json` option has been subsumed into this new option;
  for the same behavior, run `mojo test --diagnostic-format json`.

  Note that the format of the JSON output may change; we don't currently
  guarantee its stability across releases of Mojo.

- A new decorator, `@doc_private`, was added that can be used to hide a decl
  from being generated in the output of `mojo doc`. It also removes the
  requirement that the decl has documentation (e.g. when used with
  --diagnose-missing-doc-strings).

- Added a new `Span` type for taking slices of contiguous collections.
  ([PR #2595](https://github.com/modularml/mojo/pull/2595) by [lsh](https://github.com/lsh))

- Added a new `StringSlice` type, to replace uses of the unsafe `StringRef` type
  in standard library code.

  `StringSlice` is a non-owning reference to encoded string data. Unlike
  `StringRef`, a `StringSlice` is safely tied to the lifetime of the data it
  points to.

  - Add `StringSlice` intializer from an `UnsafePointer` and a length in bytes.
  - Changed `Formatter.write_str()` to take a safe `StringSlice`.

- Added a new `as_bytes_slice()` method to `String` and `StringLiteral`, which
  returns a `Span` of the bytes owned by the string.

- Add new `ImmutableStaticLifetime` and `MutableStaticLifetime` helpers

- Add new `memcpy` overload for `UnsafePointer[Scalar[_]]` pointers.

- `Dict` now implements `get(key)` and `get(key, default)` functions.
    ([PR #2519](https://github.com/modularml/mojo/pull/2519) by [@martinvuyk](https://github.com/martinvuyk))

- Debugger users can now set breakpoints on function calls in O0 builds even if
  the call has been inlined by the compiler.

- The `os` module now provides functionality for adding and removing directories
  using `mkdir` and `rmdir`.
    ([PR #2430](https://github.com/modularml/mojo/pull/2430) by [@artemiogr97](https://github.com/artemiogr97))

- `Dict.__get_ref(key)`, allowing to get references to dictionary values.

- `String.strip()`, `lstrip()` and `rstrip()` can now remove custom characters
  other than whitespace.  In addition, there are now several useful aliases for
  whitespace, ASCII lower/uppercase, and so on.
    ([PR #2555](https://github.com/modularml/mojo/pull/2555) by [@toiletsandpaper](https://github.com/toiletsandpaper))

- `List` has a simplified syntax to call the `count` method: `my_list.count(x)`.
    ([PR #2675](https://github.com/modularml/mojo/pull/2675) by [@gabrieldemarmiesse](https://github.com/gabrieldemarmiesse))

- `Dict()` now supports `reversed` for `dict.items()` and `dict.values()`.
    ([PR #2340](https://github.com/modularml/mojo/pull/2340) by [@jayzhan211](https://github.com/jayzhan211))

- `Dict` now has a simplified conversion to `String` with `my_dict.__str__()`.
  Note that `Dict` does not conform to the `Stringable` trait so `str(my_dict)`
  is not possible yet.
    ([PR #2674](https://github.com/modularml/mojo/pull/2674) by [@gabrieldemarmiesse](https://github.com/gabrieldemarmiesse))

- `List()` now supports `__contains__`.
    ([PR #2667](https://github.com/modularml/mojo/pull/2667) by [@rd4com](https://github.com/rd4com/))

- `InlineList()` now supports `__contains__`, `__iter__`.
    ([PR #2703](https://github.com/modularml/mojo/pull/2703) by [@ChristopherLR](https://github.com/ChristopherLR))

- `List` now has an `index` method that allows one to find the (first) location
  of an element in a `List` of `EqualityComparable` types. For example:

  ```mojo
  var my_list = List[Int](2, 3, 5, 7, 3)
  print(my_list.index(3))  # prints 1
  ```

- `List` can now be converted to a `String` with a simplified syntax:

  ```mojo
  var my_list = List[Int](2, 3)
  print(my_list.__str__())  # prints [2, 3]
  ```

  Note that `List` doesn't conform to the `Stringable` trait yet so you cannot
  use `str(my_list)` yet.
    ([PR #2673](https://github.com/modularml/mojo/pull/2673) by [@gabrieldemarmiesse](https://github.com/gabrieldemarmiesse))

- Added the `Indexer` trait to denote types that implement the `__index__()`
  method which allows these types to be accepted in common `__getitem__` and
  `__setitem__` implementations, as well as allow a new builtin `index` function
  to be called on them. Most stdlib containers are now able to be indexed by
  any type that implements `Indexer`. For example:

  ```mojo
  @value
  struct AlwaysZero(Indexer):
      fn __index__(self) -> Int:
          return 0

  struct MyList:
      var data: List[Int]

      fn __init__(inout self):
          self.data = List[Int](1, 2, 3, 4)

      fn __getitem__[T: Indexer](self, idx: T) -> T:
          return self.data[index(idx)]

  print(MyList()[AlwaysZero()])  # prints `1`
  ```

  ([PR #2685](https://github.com/modularml/mojo/pull/2685) by [@bgreni](https://github.com/bgreni))

  Types conforming to the `Indexer` trait are implicitly convertible to Int.
  This means you can write generic APIs that take `Int` instead of making them
  take a generic type that conforms to `Indexer`, e.g.

  ```mojo
  @value
  struct AlwaysZero(Indexer):
      fn __index__(self) -> Int:
          return 0

  @value
  struct Incrementer:
      fn __getitem__(self, idx: Int) -> Int:
          return idx + 1

  var a = Incrementer()
  print(a[AlwaysZero()])  # works and prints 1
  ```

- `StringRef` now implements `strip()` which can be used to remove leading and
  trailing whitespaces. ([PR #2683](https://github.com/modularml/mojo/pull/2683)
  by [@fknfilewalker](https://github.com/fknfilewalker))

- The `bencher` module as part of the `benchmark` package is now public
  and documented. This module provides types such as `Bencher` which provides
  the ability to execute a `Benchmark` and allows for benchmarking configuration
  via the `BenchmarkConfig` struct.

- Added the `bin()` builtin function to convert integral types into their binary
  string representation. ([PR #2603](https://github.com/modularml/mojo/pull/2603)
  by [@bgreni](https://github.com/bgreni))

- Added `atof()` function which can convert a `String` to a `float64`.
  ([PR #2649](https://github.com/modularml/mojo/pull/2649) by [@fknfilewalker](https://github.com/fknfilewalker))

- `Tuple()` now supports `__contains__`. ([PR #2709](https://github.com/modularml/mojo/pull/2709)
  by [@rd4com](https://github.com/rd4com)) For example:

  ```mojo
  var x = Tuple(1, 2, True)
  if 1 in x:
      print("x contains 1")
  ```

- Added `os.getsize` function, which gives the size in bytes of a path.
    ([PR 2626](https://github.com/modularml/mojo/pull/2626) by [@artemiogr97](https://github.com/artemiogr97))

- `List` now has a method `unsafe_get` to get the reference to an
    element without bounds check or wraparound for negative indices.
    Note that this method is unsafe. Use with caution.
    ([PR #2800](https://github.com/modularml/mojo/pull/2800) by [@gabrieldemarmiesse](https://github.com/gabrieldemarmiesse))

- Added `fromkeys` method to `Dict` to return a `Dict` with the specified keys
  and value.
  ([PR 2622](https://github.com/modularml/mojo/pull/2622) by [@artemiogr97](https://github.com/artemiogr97))

- Added `clear` method  to `Dict`.
  ([PR 2627](https://github.com/modularml/mojo/pull/2627) by [@artemiogr97](https://github.com/artemiogr97))

- Added `os.path.join` function.
  ([PR 2792](https://github.com/modularml/mojo/pull/2792)) by [@artemiogr97](https://github.com/artemiogr97))

- `StringRef` now implements `startswith()` and `endswith()`.
    ([PR #2710](https://github.com/modularml/mojo/pull/2710) by [@fknfilewalker](https://github.com/fknfilewalker))

- The Mojo Language Server now supports renaming local variables.

- Added a new `tempfile` module, with `gettempdir` and `mkdtemp` functions.
  ([PR 2742](https://github.com/modularml/mojo/pull/2742) by [@artemiogr97](https://github.com/artemiogr97))

- Added `SIMD.__repr__` to get the verbose string representation of `SIMD` types.
([PR #2728](https://github.com/modularml/mojo/pull/2728) by [@bgreni](https://github.com/bgreni))

### 🦋 Changed

- Async function calls are no longer allowed to borrow non-trivial
  register-passable types. Because async functions capture their arguments but
  register-passable types don't have lifetimes (yet), Mojo is not able to
  correctly track the reference, making this unsafe. To cover this safety gap,
  Mojo has temporarily disallowed binding non-trivial register-passable types
  to borrowed arguments in async functions.

- `AnyRegType` has been renamed to `AnyTrivialRegType` and Mojo now forbids
  binding non-trivial register-passable types to `AnyTrivialRegType`. This
  closes a major safety hole in the language. Please use `AnyType` for generic
  code going forward.

- The `let` keyword has been completely removed from the language. We previously
  removed `let` declarations but still provided an error message to users. Now,
  it is completely gone from the grammar. Long live `var`!

- The `abs`, `round`, `min`, `max`, `pow`, and `divmod` functions have moved
  from `math` to `builtin`, so you no longer need to do
  `from math import abs, round, min, max, divmod, pow`.

- Many functions returning a pointer type have been unified to have a public
  API function of `unsafe_ptr()`.

- The `--warn-missing-doc-strings` flag for `mojo` has been renamed to
  `--diagnose-missing-doc-strings`.

- The `take` function in `Variant` and `Optional` has been renamed to
  `unsafe_take`.

- The `get` function in `Variant` has been replaced by `__refitem__`. That is,
  `v.get[T]()` should be replaced with `v[T]`.

- Various functions in the `algorithm` module are now moved to be
  builtin-functions.  This includes `sort`, `swap`, and `partition`.
  `swap` and `partition` will likely shuffle around as we're reworking
  our builtin `sort` function and optimizing it.

- `SIMD.bool()` is constrained only for when the `size` is `1` now. Instead,
  explicitly use `any()` or `all()`.
    ([PR #2502](https://github.com/modularml/mojo/pull/2502) by [@helehex](https://github.com/helehex))

- The `SIMD.reduce_or()` and `SIMD.reduce_and()` methods are now bitwise
  operations, and support integer types.
    ([PR #2671](https://github.com/modularml/mojo/pull/2671) by [@helehex](https://github.com/helehex))

- `ListLiteral` and `Tuple` now only require that element types be `Copyable`.
  Consequently, `ListLiteral` and `Tuple` are themselves no longer `Copyable`.

- Continued transition to `UnsafePointer` and unsigned byte type for strings:
  - `String.unsafe_ptr()` now returns an `UnsafePointer` (was `DTypePointer`)
  - `String.unsafe_uint8_ptr()` now returns `UnsafePointer` (was
    `DTypePointer`)
  - `StringLiteral.unsafe_ptr()` now returns an `UnsafePointer` (was
    `DTypePointer`).
  - `InlinedString.as_ptr()` has been renamed to `unsafe_ptr()` and now
    returns an `UnsafePointer[UInt8]` (was `DTypePointer[DType.int8]`).
  - `StringRef.data` is now an `UnsafePointer` (was `DTypePointer`)
  - `StringRef.unsafe_ptr()` now returns an `UnsafePointer[UInt8]` (was
    `DTypePointer[DType.int8]`).
  - Removed `StringRef.unsafe_uint8_ptr()`. The `unsafe_ptr()` method now has
    the same behavior.

- Added `String.isspace()` method conformant with Python's universal separators.

- Changed `isspace(..)` to take a `UInt8` and was made private (`_isspace(..)`),
  use `String.isspace()` instead.

- `String.split()` now defaults to whitespace and has pythonic behavior in that
  it removes all adjacent whitespaces by default.

- Added `UnsafePointer.offset()` method.

- The `math.bit` module has been moved to a new top-level `bit` module. The
  following functions in this module have been renamed:
  - `ctlz` -> `countl_zero`
  - `cttz` -> `countr_zero`
  - `bit_length` -> `bit_width`
  - `ctpop` -> `pop_count`
  - `bswap` -> `byte_swap`
  - `bitreverse` -> `bit_reverse`

- The `math.rotate_bits_left` and `math.rotate_bits_right` functions have been
  moved to the `bit` module.

- The implementation of the following functions have been moved from the `math`
  module to the new `utils.numerics` module: `isfinite`, `isinf`, `isnan`,
  `nan`, `nextafter`, and `ulp`. The functions continue to be exposed in the
  `math` module.

- `InlinedString` has been renamed to `InlineString` to be consistent with other
  types.

- The `Slice.__len__` function has been removed and `Slice` no longer conforms
  to the `Sized` trait. This clarifies the ambiguity of the semantics: the
  length of a slice always depends on the length of the object being sliced.
  Users that need the existing functionality can use the `Slice.unsafe_indices`
  method. This makes it explicit that this implementation does not check if the
  slice bounds are concrete or within any given object's length.

- `math.gcd` now works on negative inputs, and like Python's implementation,
  accepts a variadic list of integers. New overloads for a `List` or `Span`of
  integers are also added.
  ([PR #2777](https://github.com/modularml/mojo/pull/2777) by [@bgreni](https://github.com/bgreni))

### ❌ Removed

- The `@unroll` decorator has been deprecated and removed. The decorator was
  supposed to guarantee that a decorated loop would be unrolled, or else the
  compiler would error. In practice, this guarantee was eroded over time, as
  a compiler-based approach cannot be as robust as the Mojo parameter system.
  In addition, the `@unroll` decorator did not make the loop induction variables
  parameter values, limiting its usefulness. Please see `@parameter for` for a
  replacement!

- The method `object.print()` has been removed. Since now, `object` has the
  `Stringable` trait, you can use `print(my_object)` instead.

- The following functions have been removed from the math module:
  - `clamp`; use the new `SIMD.clamp` method instead.
  - `round_half_down` and `round_half_up`; these can be trivially implemented
    using the `ceil` and `floor` functions.
  - `add`, `sub`, `mul`, `div`, `mod`, `greater`, `greater_equal`, `less`,
    `less_equal`, `equal`, `not_equal`, `logical_and`, `logical_xor`, and
    `logical_not`; Instead, users should rely directly on the `+`, `-`, `*`,
    `/`, `%`, `>`, `>=`, `<`, `<=`, `==`, `!=`, `&`, `^`, and `~` operators,
    respectively.
  - `identity` and `reciprocal`; users can implement these trivially.
  - `select`; in favor of using `SIMD.select` directly.
  - `is_even` and `is_odd`; these can be trivially implemented using bitwise `&`
    with `1`.
  - `roundeven`; the new `SIMD.roundeven` method now provides the identical
    functionality.
  - `div_ceil`; use the new `ceildiv` function.
  - `rotate_left` and `rotate_right`; the same functionality is available in the
    builtin `SIMD.rotate_{left,right}` methods for `SIMD` types, and the
    `bit.rotate_bits_{left,right}` methods for `Int`.
  - an overload of `math.pow` taking an integer parameter exponent.
  - `align_down_residual`; it can be trivially implemented using `align_down`.

- The `math.bit.select` and `math.bit.bit_and` functions have been removed. The
  same functionality is available in the builtin `SIMD.select` and
  `SIMD.__and__` methods, respectively.

- The `math.limit` module has been removed. The same functionality is available
  as follows:
  - `math.limit.inf`: use `utils.numerics.max_or_inf`
  - `math.limit.neginf`: use `utils.numerics.min_or_neg_inf`
  - `math.limit.max_finite`: use `utils.numerics.max_finite`
  - `math.limit.min_finite`: use `utils.numerics.min_finite`

- The `tensor.random` module has been removed. The same functionality is now
  accessible via the `Tensor.rand` and `Tensor.randn` static methods.

- The builtin `SIMD` struct no longer conforms to `Indexer`; users must
  explicitly cast `Scalar` values using `int`.

### 🛠️ Fixed

- [#1837](https://github.com/modularml/mojo/issues/1837) Fix self-referential
  variant crashing the compiler.
- [#2363](https://github.com/modularml/mojo/issues/2363) Fix LSP crashing on
  simple trait definitions.
- [#1787](https://github.com/modularml/mojo/issues/1787) Fix error when using
  `//` on `FloatLiteral` in alias expression.
- Made several improvements to dictionary performance. Dicts with integer keys
  are most heavily affected, but large dicts and dicts with large values
  will also see large improvements.
- [#2692](https://github.com/modularml/mojo/issues/2692) Fix `assert_raises`
  to include calling location.
