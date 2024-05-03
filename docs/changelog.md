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

- The `abs, round, min, and max` functions have moved from `math` to `builtin`,
  so you no longer need to do `from math import abs, round, min, max`.

- Mojo now allows types to opt in to use the `floor()` and `ceil()` functions in
  the `math` module by implementing the `__floor__()` and `__ceil__()` methods
  (and so conforming to the new `math.Floorable` and `math.Ceilable` traits,
  respectively). For example:

  ```mojo
    from math import Ceilable, Floorable, ceil, floor

    @value
    struct Complex(Ceilable, Floorable):
      var re: Float64
      var im: Float64

      fn __ceil__(self) -> Self:
          return Self(ceil(re), ceil(im))

      fn __floor__(self) -> Self:
          return Self(floor(re), floor(im))
  ```

- Add an `InlinedArray` type that works on memory-only types.
  Compare with the existing `StaticTuple` type, which is conceptually an array
  type, but only worked on `AnyRegType`.
    ([PR #2294](https://github.com/modularml/mojo/pull/2294) by [@lsh](https://github.com/lsh))

- Base64 decoding support has been added.
    ([PR #2364](https://github.com/modularml/mojo/pull/2364) by [@mikowals](https://github.com/mikowals))

- Add `repr()` function and `Representable` trait.
    ([PR #2361](https://github.com/modularml/mojo/pull/2361) by [@mikowals](https://github.com/gabrieldemarmiesse))

- Add `SIMD.shuffle()` with `StaticIntTuple` mask.
    ([PR #2315](https://github.com/modularml/mojo/pull/2315) by [@mikowals](https://github.com/mikowals))

- Invoking `mojo package my-package -o my-dir` on the command line, where
  `my-package` is a Mojo package source directory, and `my-dir` is an existing
  directory, now outputs a Mojo package to `my-dir/my-package.mojopkg`.
  Previously, this had to be spelled out, as in `-o my-dir/my-package.mojopkg`.

### 🦋 Changed

- The `abs` and `round` functions have moved from `math` to `builtin`, so you no
  longer need to do `from math import abs, round`.

### ❌ Removed

- The method `object.print()` has been removed. Since now, `object` has the
  `Stringable` trait, you can use `print(my_object)` instead.

- The `math.clamp` function has been removed in favor of a new `SIMD.clamp`
  method.

- The `math.round_half_down` and `math.round_half_up` functions are removed.
  These can be trivially implemented using the `ceil` and `floor` functions.

- The `add`, `sub`, `mul`, `div`, and `mod` functions have been removed from the
  `math` module. Instead, users should rely on the `+`, `-`, `*`, `/`, and `%`
  operators, respectively.

- The `math.roundeven` function has been removed from the `math` module. The new
  `SIMD.roundeven` method now provides the identical functionality.

- The `math.div_ceil` function has been removed in favor of the `math.ceildiv`
  function.

### 🛠️ Fixed

- [#2363](https://github.com/modularml/mojo/issues/2363) Fix LSP crashing on
  simple trait definitions.
- [#1787](https://github.com/modularml/mojo/issues/1787) Fix error when using
  `//` on `FloatLiteral` in alias expression.
