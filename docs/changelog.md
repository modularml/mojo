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

- The destructor insertion logic in Mojo is now aware that types that take an
  `AnyLifetime` as part of their signature could potentially access any live
  value that destructor insertion is tracking, eliminating a significant
  usability issue with unsafe APIs like `UnsafePointer`.  Consider a typical
  example working with strings before this change:

  ```mojo
  var str = String(...)
  var ptr = str.unsafe_ptr()
  some_low_level_api(ptr)
  _ = str^  # OLD HACK: Explicitly keep string alive until here!
  ```

  The `_ = str^` pattern was formerly required because the Mojo compiler has no
  idea what "ptr" might reference.  As a consequence, it had no idea that
  `some_low_level_api` might access `str` and therefore thought it was ok to
  destroy the `String` before the call - this is why the explicit lifetime
  extension was required.

  Mojo now knows that `UnsafePointer` may access the `AnyLifetime` lifetime,
  and now assumes that any API that uses that lifetime could use live values.
  In this case, it assumes that `some_low_level_api` might access `str` and
  because it might be using it, it cannot destroy `str` until after the call.
  The consequence of this is that the old hack is no longer needed for these
  cases!

- The `UnsafePointer` type now has a `lifetime` parameter that can be used when
  the `UnsafePointer` is known to point into some lifetime.  This lifetime is
  propagated through the `ptr[]` indirection operation.

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

  - Added `TypedPythonObject["Tuple].__getitem__` for accessing the elements of
    a Python tuple.

- Added `Python.add_object()`, to add a named `PythonObject` value to a Python
  'module' object instance.

- Added `Python.unsafe_get_python_exception()`, as an efficient low-level
  utility to get the Mojo `Error` equivalent of the current CPython error state.

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

- The `Dict.__getitem__` method now returns a reference instead of a copy of
  the value (or raises).  This improves the performance of common code that
  uses `Dict` by allowing borrows from the `Dict` elements.

- Autoparameterization of parameters is now supported. Specifying a parameter
  type with unbound parameters causes them to be implicitly added to the
  function signature as inferred parameters.

  ```mojo
  fn foo[value: SIMD[DType.int32, _]]():
    pass

  # Equivalent to
  fn foo[size: Int, //, value: SIMD[DType.int32, size]]():
    pass
  ```

- Function types now accept a lifetime set parameter. This parameter represents
  the lifetimes of values captured by a parameter closure. The compiler
  automatically tags parameter closures with the right set of lifetimes. This
  enables lifetimes and parameter closures to correctly compose.

  ```mojo
  fn call_it[f: fn() capturing [_] -> None]():
      f()

  fn test():
      var msg = String("hello world")

      @parameter
      fn say_hi():
          print(msg)

      call_it[say_hi]()
      # no longer need to write `_ = msg^`!!
  ```

  Note that this only works for higher-order functions which have explicitly
  added `[_]` as the capture lifetimes. By default, the compiler still assumes
  a `capturing` closure does not reference any lifetimes. This will soon change.

- The VS Code extension now has the `mojo.run.focusOnTerminalAfterLaunch`
  setting, which controls whether to focus on the terminal used by the
  `Mojo: Run Mojo File` command or on the editor after launch.
  [Issue #3532](https://github.com/modularml/mojo/issues/3532).

- The VS Code extension now has the `mojo.SDK.additionalSDKs` setting, which
  allows the user to provide a list of MAX SDKs that the extension can use when
  determining a default SDK to use. The user can select the default SDK to use
  with the `Mojo: Select the default MAX SDK` command.

- Added a new [`Box`](/mojo/stdlib/memory/box/Box) type as a safe, single-owner,
  non-nullable smart pointer with similar semantics to Rust's
  [`Box<>`](https://doc.rust-lang.org/std/boxed/struct.Box.html) and C++'s
  [`std::unique_ptr`](https://en.cppreference.com/w/cpp/memory/unique_ptr).

  ([PR #3524](https://github.com/modularml/mojo/pull/3524) by [@szbergeron](https://github.com/szbergeron))

- `ref` argument and result specifiers now allow providing a memory value
  directly in the lifetime specifier, rather than requiring the use of
  `__lifetime_of`.  It is still fine to use `__lifetime_of` explicitly though,
  and this is required when specifying lifetimes for parameters (e.g. to the
  `Reference` type). For example, this is now valid without `__lifetime_of`:

  ```mojo
  fn return_ref(a: String) -> ref [a] String:
      return a
  ```

- `StringRef` now implements `split()` which can be used to split a
  `StringRef` into a `List[StringRef]` by a delimiter.
  ([PR #2705](https://github.com/modularml/mojo/pull/2705) by [@fknfilewalker](https://github.com/fknfilewalker))

- Support for multi-dimensional indexing for `PythonObject`
  ([PR #3583](https://github.com/modularml/mojo/pull/3583) by [@jjvraw](https://github.com/jjvraw)).

    ```mojo
    var np = Python.import_module("numpy")
    var a = np.array(PythonObject([1,2,3,1,2,3])).reshape(2,3)
    print((a[0, 1])) # 2
    ```

- [`Arc`](/mojo/stdlib/memory/arc/Arc) now implements
  [`Identifiable`](/mojo/stdlib/builtin/identifiable/Identifiable), and can be
  compared for pointer equivalence using `a is b`.

### 🦋 Changed

- More things have been removed from the auto-exported set of entities in the `prelude`
  module from the Mojo standard library.
  - `UnsafePointer` has been removed. Please explicitly import it via
    `from memory import UnsafePointer`.
  - `StringRef` has been removed. Please explicitly import it via
    `from utils import StringRef`.

- The `Reference` type has been renamed to `Pointer`: a memory safe complement
  to `UnsafePointer`.  This change is motivated by the fact that `Pointer`
  is assignable and requires an explicit dereference with `ptr[]`.  Renaming
  to `Pointer` clarifies that "references" means `ref` arguments and results,
  and gives us a model that is more similar to what the C++ community would
  expect.

- A new `as_noalias_ptr` method as been added to `UnsafePointer`. This method
  specifies to the compiler that the resultant pointer is a distinct
  identifiable object that does not alias any other memory in the local scope.

- The `AnyLifetime` type (useful for declaring lifetime types as parameters) has
  been renamed to `Lifetime`.

- Restore implicit copyability of `Tuple` and `ListLiteral`.

- The aliases for C FFI have been renamed: `C_int` -> `c_int`, `C_long` -> `c_long`
  and so on.

- The VS Code extension now allows selecting a default SDK when multiple are available.

- `String.as_bytes_slice()` is renamed to `String.as_bytes_span()` since it
  returns a `Span` and not a `StringSlice`.

- The flag for turning on asserts has changed, e.g. to enable all checks:

  ```bash
  mojo -D ASSERT=all main.mojo
  ```

  The levels are:

  - none: all assertions off
  - warn: print assertion errors e.g. for multithreaded tests (previously -D
    ASSERT_WARNING)
  - safe: the default mode for standard CPU safety assertions
  - all: turn on all assertions (previously -D MOJO_ENABLE_ASSERTIONS)

  You can now also pass `Stringable` args to format a message, which will have
  no runtime penalty or IR bloat cost when assertions are off. Previously you
  had to:

  ```mojo
  x = -1
  debug_assert(
    x > 0, String.format_sequence(“expected x to be more than 0 but got: ”, x)
  )
  ```

  Which can't be optimized away by the compiler in release builds, you can now
  pass multiple args for a formatted message at no runtime cost:

  ```mojo
  debug_assert(x > 0, “expected x to be more than 0 but got: ”, x)
  ```

- The `StaticIntTuple` datastructure in the `utils` package has been renamed to
  `IndexList`. The datastructure now allows one to specify the index bitwidth of
  the elements along with whether the underlying indices are signed or unsigned.

- `String.as_bytes()` now returns a `Span[UInt8]` instead of a `List[Int8]`. The
  old behavior can be achieved by using `List(s.as_bytes())`.

### ❌ Removed

### 🛠️ Fixed

- Lifetime tracking is now fully field sensitive, which makes the uninitialized
  variable checker more precise.

- [Issue #3444](https://github.com/modularml/mojo/issues/3444) - Raising init
  causing use of uninitialized variable

- [Issue #3544](https://github.com/modularml/mojo/issues/3544) - Known
  mutable `ref` argument are not optimized as `noalias` by LLVM.

- [Issue #3559](https://github.com/modularml/mojo/issues/3559) - VariadicPack
  doesn't extend the lifetimes of the values it references.

- The VS Code extension now auto-updates its private copy of the MAX SDK.

- The variadic initializer for `SIMD` now works in parameter expressions.
