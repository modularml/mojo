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

- Mojo now allows types to opt in to use the `abs()` function by implementing
  the `__abs__()` method, defined by the new `Absable`:

  ```mojo
  from math import sqrt

  struct Point(Absable):
      var x: Float64
      var y: Float64

      fn __abs__(self) -> Self:
          return sqrt(self.x * self.x + self.y * self.y)
  ```

- The `abs(), min(), and max()` functions have moved from `math` to `builtin`,
  so you no longer need to do `from math import abs, min, max`.

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

### 🦋 Changed

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

### 🛠️ Fixed

- [#2363](https://github.com/modularml/mojo/issues/2363) Fix LSP crashing on
  simple trait definitions.
- [#1787](https://github.com/modularml/mojo/issues/1787) Fix error when using
  `//` on `FloatLiteral` in alias expression.
