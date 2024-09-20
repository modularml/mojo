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

### ⭐️ New

- Mojo can now interpret simple LLVM intrinsics in parameter expressions,
  enabling things like `count_leading_zeros` to work at compile time:
  [Issue #933](https://github.com/modularml/mojo/issues/933).

- The VS Code Mojo Debugger now has a `buildArgs` JSON debug configuration
  setting that can be used in conjunction with `mojoFile` to define the build
  arguments when compiling the Mojo file.

- The VS Code extension now supports a `Configure Build and Run Args` command
  that helps set the build and run args for actions file `Run Mojo File` and
  `Debug Mojo File`. A corresponding button appears in `Run and Debug` selector
  in the top right corner of a Mojo File.

- Add the `Floatable` and `FloatableRaising` traits to denote types that can
  be converted to a `Float64` value using the builtin `float` function.
  - Make `SIMD` and `FloatLiteral` conform to the `Floatable` trait.

  ```mojo
  fn foo[F: Floatable](v: F):
    ...

  var f = float(Int32(45))
  ```

  ([PR #3163](https://github.com/modularml/mojo/pull/3163) by [@bgreni](https://github.com/bgreni))

- Add `DLHandle.get_symbol()`, for getting a pointer to a symbol in a dynamic
  library. This is more general purpose than the existing methods for getting
  function pointers.

- Introduce `TypedPythonObject` as a light-weight way to annotate `PythonObject`
  values with static type information. This design will likely evolve and
  change significantly.

- The `__type_of(x)` and `__lifetime_of(x)` operators are much more general now:
  they allow arbitrary expressions inside of them, allow referring to dynamic
  values in parameter contexts, and even allow referring to raising functions
  in non-raising contexts.  These operations never evaluate their expression, so
  any side effects that occur in the expression are never evaluated at runtime,
  eliminating concerns about `__type_of(expensive())` being a problem.

- Add `PythonObject.from_borrowed_ptr()`, to simplify the construction of
  `PythonObject` values from CPython 'borrowed reference' pointers.

  The existing `PythonObject.__init__(PyObjectPtr)` should continue to be used
  for the more common case of constructing a `PythonObject` from a
  'strong reference' pointer.

- The `rebind` standard library function now works with memory-only types in
  addition to `@register_passable("trivial")` ones, without requiring a copy.

### 🦋 Changed

- More things have been removed from the auto-exported set of entities in the `prelude`
  module from the Mojo standard library.
  - `UnsafePointer` has been removed. Please explicitly import it via
    `from memory import UnsafePointer`.
  - `StringRef` has been removed. Please explicitly import it via
    `from utils import StringRef`.

- A new `as_noalias_ptr` method as been added to `UnsafePointer`. This method
  specifies to the compiler that the resultant pointer is a distinct
  identifiable object that does not alias any other memory in the local scope.

- Restore implicit copyability of `Tuple` and `ListLiteral`.

- The aliases for C FFI have been renamed: `C_int` -> `c_int`, `C_long` -> `c_long`
  and so on.

### ❌ Removed

### 🛠️ Fixed

- Lifetime tracking is now fully field sensitive, which makes the uninitialized
  variable checker more precise.

- [Issue #3444](https://github.com/modularml/mojo/issues/3444) - Raising init
  causing use of uninitialized variable
