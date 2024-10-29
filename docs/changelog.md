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

- The `__type_of(x)` and `__origin_of(x)` operators are much more general now:
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
  `__origin_of`.  It is still fine to use `__origin_of` explicitly though,
  and this is required when specifying lifetimes for parameters (e.g. to the
  `Reference` type). For example, this is now valid without `__origin_of`:

  ```mojo
  fn return_ref(a: String) -> ref [a] String:
      return a
  ```

- `Slice.step` is now an `Optional[Int]`, matching the optionality of
  `slice.step` in Python.
  ([PR #3160](https://github.com/modularml/mojo/pull/3160) by
   [@bgreni](https://github.com/bgreni))

- `StringRef` now implements `split()` which can be used to split a
  `StringRef` into a `List[StringRef]` by a delimiter.
  ([PR #2705](https://github.com/modularml/mojo/pull/2705) by [@fknfilewalker](https://github.com/fknfilewalker))

- Support for multi-dimensional indexing for `PythonObject`
  ([PR #3583](https://github.com/modularml/mojo/pull/3583) by [@jjvraw](https://github.com/jjvraw)).

- Support for multi-dimensional indexing and slicing for `PythonObject`
  (PRs  [#3549](https://github.com/modularml/mojo/pull/3549),
  [#3583](https://github.com/modularml/mojo/pull/3583) by [@jjvraw](https://github.com/jjvraw)).

    ```mojo
    var np = Python.import_module("numpy")
    var a = np.array(PythonObject([1,2,3,4,5,6])).reshape(2,3)
    print((a[0, 1])) # 2
    print((a[1][::-1])) # [6 5 4]
   ```

  Note, that the syntax, `a[1, ::-1]`, is currently not supported.

- [`Arc`](/mojo/stdlib/memory/arc/Arc) now implements
  [`Identifiable`](/mojo/stdlib/builtin/identifiable/Identifiable), and can be
  compared for pointer equivalence using `a is b`.

- There is now a [`Byte`](/mojo/stdlib/builtin/simd/Byte) alias to better
  express intent when working with a pack of bits.
  ([PR #3670](https://github.com/modularml/mojo/pull/3670) by [@soraos](https://github.com/soraros)).

- The VS Code extension now supports setting [data breakpoints](https://code.visualstudio.com/docs/editor/debugging#_data-breakpoints)
  as well as [function breakpoints](https://code.visualstudio.com/docs/editor/debugging#_function-breakpoints).

- The Mojo LLDB debugger now supports symbol breakpoints, e.g. `b main` or
  `b my_module::main`.

- The VS Code extension now allows cancelling the installation of its private
  MAX SDK.

- The VS Code extension now opens the Run and Debug tab automatically whenever
  a debug session starts.

- The `mojo debug --vscode` command now support the `--init-command` and
  `--stop-on-entry` flags. Execute `mojo debug --help` for more information.

- The Mojo LLDB debugger on VS Code now supports inspecting the raw attributes
  of variables that are handled as synthetic types, e.g. `List` from Mojo or
  `std::vector` from C++.

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

- The `Formatter` struct has changed to a `Writer` trait to enable buffered IO,
  increasing print and file writing perf to the same speed as C. It's now more
  general purpose and can write any `Span[Byte]`. To align with this the
  `Formattable` trait is now named `Writable`, and the `String.format_sequence`
  static methods to initialize a new `String` have been renamed to
  `String.write`. Here's an example of using all the changes:

  ```mojo
  from utils import Span

  @value
  struct NewString(Writer, Writable):
      var s: String

      # Writer requirement to write a Span of Bytes
      fn write_bytes(inout self, bytes: Span[Byte, _]):
          self.s._iadd[False](bytes)

      # Writer requirement to take multiple args
      fn write[*Ts: Writable](inout self, *args: *Ts):
          @parameter
          fn write_arg[T: Writable](arg: T):
              arg.write_to(self)

          args.each[write_arg]()

      # Also make it Writable to allow `print` to write the inner String
      fn write_to[W: Writer](self, inout writer: W):
          writer.write(self.s)


  @value
  struct Point(Writable):
      var x: Int
      var y: Int

      # Pass multiple args to the Writer. The Int and StringLiteral types call
      # `writer.write_bytes` in their own `write_to` implementations.
      fn write_to[W: Writer](self, inout writer: W):
          writer.write("Point(", self.x, ", ", self.y, ")")

      # Enable conversion to a String using `str(point)`
      fn __str__(self) -> String:
          return String.write(self)


  fn main():
      var point = Point(1, 2)
      var new_string = NewString(str(point))
      new_string.write("\n", Point(3, 4))
      print(new_string)
  ```

  Point(1, 2)
  Point(3, 4)

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

- A new trait has been added `AsBytes` to enable taking a `Span[Byte]` of a
  type with `s.as_bytes()`. `String.as_bytes` and `String.as_bytes_slice` have
  been consolidated under `s.as_bytes` to return a `Span[Byte]`, you can convert
  it to a `List` if you require a copy with `List(s.as_bytes())`.

- `Lifetime` and related types has been renamed to `Origin` in the standard
  library to better clarify that parameters of this type indicate where a
  reference is derived from, not the more complicated notion of where a variable
  is initialized and destroyed.  Please see [the proposal](https://github.com/modularml/mojo/blob/main/proposals/lifetimes-keyword-renaming.md)
  for more information and rationale.  As a consequence `__lifetime_of` is now
  named `__origin_of`.

- You can now use the `+=` and `*` operators on a `StringLiteral` at compile
  time using the `alias` keyword:

  ```mojo
  alias original = "mojo"
  alias concat = original * 3
  assert_equal("mojomojomojo", concat)
  ```

  Or inside a `fn` that is being evaluated at compile time:

  ```mojo
  fn add_literal(
      owned original: StringLiteral, add: StringLiteral, n: Int
  ) -> StringLiteral:
      for _ in range(n):
          original += add
      return original


  fn main():
      alias original = "mojo"
      alias concat = add_literal(original, "!", 4)
      assert_equal("mojo!!!!", concat)
  ```

  These operators can't be evaluated at runtime, as a `StringLiteral` must be
  written into the binary during compilation.

  - You can now index into `UnsafePointer` using SIMD scalar integral types:

  ```mojo
  p = UnsafePointer[Int].alloc(1)
  i = UInt8(1)
  p[i] = 42
  print(p[i])
  ```

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

- [Issue #3627](https://github.com/modularml/mojo/issues/3627) - Compiler
  overlooked exclusivity violation caused by `ref [MutableAnyLifetime] T`

- The VS Code extension now auto-updates its private copy of the MAX SDK.

- The variadic initializer for `SIMD` now works in parameter expressions.

- The VS Code extension now downloads its private copy of the MAX SDK in a way
  that prevents ETXTBSY errors on Linux.

- The VS Code extension now allows invoking a mojo formatter from SDK
  installations that contain white spaces in their path.
