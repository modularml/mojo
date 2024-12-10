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

- `StringRef` is now representable so `repr(StringRef("hello"))` will return
  `StringRef('hello')`.

- Mojo can now interpret simple LLVM intrinsics in parameter expressions,
  enabling things like `count_leading_zeros` to work at compile time:
  [Issue #933](https://github.com/modularml/mojo/issues/933).

- The destructor insertion logic in Mojo is now aware that types that take an
  `MutableAnyOrigin` or `ImmutableAnyOrigin` as part of their signature could
  potentially access any live value that destructor insertion is tracking,
  eliminating a significant usability issue with unsafe APIs like
  `UnsafePointer`.  Consider a typical example working with strings before this
   change:

  ```mojo
  var str = String(...)
  var ptr = str.unsafe_ptr()
  some_low_level_api(ptr)
  _ = str^  # OLD HACK: Explicitly keep string alive until here!
  ```

  The `_ = str^` pattern was formerly required because the Mojo compiler has no
  idea what "ptr" might reference.  As a consequence, it had no idea that
  `some_low_level_api()` might access `str` and therefore thought it was ok to
  destroy the `String` before the call - this is why the explicit lifetime
  extension was required.

  Mojo now knows that `UnsafePointer` may access the `MutableAnyOrigin` origin,
  and now assumes that any API that uses that origin could use live values.
  In this case, it assumes that `some_low_level_api()` might access `str` and
  because it might be using it, it cannot destroy `str` until after the call.
  The consequence of this is that the old hack is no longer needed for these
  cases!

- Various improvements to origin handling and syntax have landed, including
  support for the ternary operator and allowing multiple arguments in a `ref`
  specifier (which are implicitly unions).  This enables expression of simple
  algorithms cleanly:

  ```mojo
  fn my_min[T: Comparable](ref a: T, ref b: T) -> ref [a, b] T:
    return a if a < b else b
  ```

  It is also nice that `my_min` automatically and implicitly propagates the
  mutability of its arguments, so things like `my_min(str1, str2) += "foo"` is
  valid.

- The `UnsafePointer` type now has an `origin` parameter that can be used when
  the `UnsafePointer` is known to point to a value with a known origin. This
  origin is propagated through the `ptr[]` indirection operation.

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

- Introduce `random.shuffle` for `List`.
  ([PR #3327](https://github.com/modularml/mojo/pull/3327) by [@jjvraw](https://github.com/jjvraw))

  Example:

  ```mojo
  from random import shuffle

  var l = List[Int](1, 2, 3, 4, 5)
  shuffle(l)
  ```

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

- Function types now accept an origin set parameter. This parameter represents
  the origins of values captured by a parameter closure. The compiler
  automatically tags parameter closures with the right set of origins. This
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
  added `[_]` as the capture origins. By default, the compiler still assumes
  a `capturing` closure does not reference any origins. This will soon change.

- The VS Code extension now has the `mojo.run.focusOnTerminalAfterLaunch`
  setting, which controls whether to focus on the terminal used by the
  `Mojo: Run Mojo File` command or on the editor after launch.
  [Issue #3532](https://github.com/modularml/mojo/issues/3532).

- The VS Code extension now has the `mojo.SDK.additionalSDKs` setting, which
  allows the user to provide a list of MAX SDKs that the extension can use when
  determining a default SDK to use. The user can select the default SDK to use
  with the `Mojo: Select the default MAX SDK` command.

- Added a new [`OwnedPointer`](/mojo/stdlib/memory/owned_pointer/OwnedPointer)
  type as a safe, single-owner, non-nullable smart pointer with similar
  semantics to Rust's
  [`Box<>`](https://doc.rust-lang.org/std/boxed/struct.Box.html) and C++'s
  [`std::unique_ptr`](https://en.cppreference.com/w/cpp/memory/unique_ptr).

  ([PR #3524](https://github.com/modularml/mojo/pull/3524) by [@szbergeron](https://github.com/szbergeron))

- `ref` argument and result specifiers now allow providing a memory value
  directly in the origin specifier, rather than requiring the use of
  `__origin_of()`.  It is still fine to use `__origin_of()` explicitly though,
  and this is required when specifying origins for parameters (e.g. to the
  `Pointer` type). For example, this is now valid without `__origin_of()`:

  ```mojo
  fn return_ref(a: String) -> ref [a] String:
      return a
  ```

- `ref` function arguments without an origin clause are now treated as
  `ref [_]`, which is more syntactically convenient and consistent:

  ```mojo
  fn takes_and_return_ref(ref a: String) -> ref [a] String:
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

- Expanded `os.path` with new functions (by [@thatstoasty](https://github.com/thatstoasty)):
  - `os.path.expandvars`: Expands environment variables in a path ([PR #3735](https://github.com/modularml/mojo/pull/3735)).
  - `os.path.splitroot`: Split a path into drive, root and tail.
  ([PR #3780](https://github.com/modularml/mojo/pull/3780)).

- Added a `reserve` method and new constructor to the `String` struct to
  allocate additional capacity.
  ([PR #3755](https://github.com/modularml/mojo/pull/3755) by [@thatstoasty](https://github.com/thatstoasty)).

- Introduced a new `Deque` (double-ended queue) collection type, based on a
  dynamically resizing circular buffer for efficient O(1) additions and removals
  at both ends as well as O(1) direct access to all elements.

  The `Deque` supports the full Python `collections.deque` API, ensuring that all
  expected deque operations perform as in Python.

  Enhancements to the standard Python API include `peek()` and `peekleft()`
  methods for non-destructive access to the last and first elements, and advanced
  constructor options (`capacity`, `min_capacity`, and `shrink`) for customizing
  memory allocation and performance. These options allow for optimized memory usage
  and reduced buffer reallocations, providing flexibility based on application requirements.

- A new `StringLiteral.get[some_stringable]()` method is available.  It
  allows forming a runtime-constant StringLiteral from a compile-time-dynamic
  `Stringable` value.

- `Span` now implements `__reversed__`. This means that one can get a
  reverse iterator over a `Span` using `reversed(my_span)`. Users should
  currently prefer this method over `my_span[::-1]`.

- `StringSlice` now implements `strip`, `rstrip`, and `lstrip`.

- Introduced the `@explicit_destroy` annotation, the `__disable_del` keyword,
  the `UnknownDestructibility` trait, and the `ImplicitlyDestructible` keyword,
  for the experimental explicitly destroyed types feature.

- Added associated types; we can now have aliases like `alias T: AnyType`,
  `alias N: Int`, etc. in a trait, and then specify them in structs that conform
  to that trait.

### 🦋 Changed

- The `inout` and `borrowed` argument conventions have been renamed to the `mut`
  and `read` argument conventions (respectively).  These verbs reflect
  declaratively what the callee can do to the argument value passed into the
  caller, without tying in the requirement for the programmer to know about
  advanced features like references.

- The argument convention for `__init__` methods has been changed from `inout`
  to `out`, reflecting that an `__init__` method initializes its `self` without
  reading from it.  This also enables spelling the type of an initializer
  correctly, which was not supported before:

  ```mojo
  struct Foo:
      fn __init__(out self): pass

  fn test():
      # This works now
      var fnPtr : fn(out x: Foo)->None = Foo.__init__

      var someFoo : Foo
      fnPtr(someFoo)  # initializes someFoo.
  ```

  The previous `fn __init__(inout self)` syntax is still supported in this
  release of Mojo, but will be removed in the future.  Please migrate to the
  new syntax.

- Similarly, the spelling of "named functions results" has switched to use `out`
  syntax instead of `-> T as name`.  Functions may have at most one named result
  or return type specified with the usual `->` syntax.  `out` arguments may
  occur anywhere in the argument list, but are typically last (except for
  `__init__` methods, where they are typically first).

  ```mojo
  # This function has type "fn() -> String"
  fn example(out result: String):
    result = "foo"
  ```

  The parser still accepts the old syntax as a synonym for this, but that will
  eventually be deprecated and removed.

  This was [discussed extensively in a public
  proposal](https://github.com/modularml/mojo/issues/3623).

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

- The `AnyLifetime` type (useful for declaring origin types as parameters) has
  been renamed to `Origin`.

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
  from memory import Span

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

- `Lifetime` and related types have been renamed to `Origin` in the standard
  library to better clarify that parameters of this type indicate where a
  reference is derived from, not the more complicated notion of where a variable
  is initialized and destroyed.  Please see [the proposal](https://github.com/modularml/mojo/blob/main/proposals/lifetimes-keyword-renaming.md)
  for more information and rationale.  As a consequence the `__lifetime_of()`
  operator is now named `__origin_of()`.

- `Origin` is now a complete wrapper around the MLIR origin type.

  - The `Origin.type` alias has been renamed to `_mlir_origin`. In parameter
    lists, you can now write just `Origin[..]`, instead of `Origin[..].type`.

  - `ImmutableOrigin` and `MutableOrigin` are now, respectively, just aliases
    for `Origin[False]` and `Origin[True]`.

  - `Origin` struct values are now supported in the brackets of a `ref [..]`
    argument.

  - Added `Origin.cast_from` for casting the mutability of an origin value.

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

- Float32 and Float64 are now printed and converted to strings with roundtrip
  guarantee and shortest representation:

  ```plaintext
  Value                       Old                       New
  Float64(0.3)                0.29999999999999999       0.3
  Float32(0.3)                0.30000001192092896       0.3
  Float64(0.0001)             0.0001                    0.0001
  Float32(0.0001)             9.9999997473787516e-05    0.0001
  Float64(-0.00001)           -1.0000000000000001e-05   -1e-05
  Float32(-0.00001)           -9.9999997473787516e-06   -1e-05
  Float32(0.00001234)         1.2339999557298142e-05    1.234e-05
  Float32(-0.00000123456)     -1.2345600453045336e-06   -1.23456e-06
  Float64(1.1234567e-320)     1.1235052786429946e-320   1.1235e-320
  Float64(1.234 * 10**16)     12340000000000000.0       1.234e+16
  ```

- Single argument constructors now require a `@implicit` decorator to allow
  for implicit conversions. Previously you could define an `__init__` that
  takes a single argument:

  ```mojo
  struct Foo:
      var value: Int

      fn __init__(out self, value: Int):
          self.value = value
  ```

  And this would allow you to pass an `Int` in the position of a `Foo`:

  ```mojo
  fn func(foo: Foo):
      print("implicitly converted Int to Foo:", foo.value)

  fn main():
      func(Int(42))
  ```

  This can result in complicated errors that are difficult to debug. By default
  this implicit behavior is now turned off, so you have to explicitly construct
  `Foo`:

  ```mojo
  fn main():
      func(Foo(42))
  ```

  You can still opt into implicit conversions by adding the `@implicit`
  decorator. For example, to enable implicit conversions from `Int` to `Foo`:

  ```mojo
  struct Foo:
      var value: Int

      @implicit
      fn __init__(out self, value: Int):
          self.value = value
  ```

- `Arc` has been renamed to `ArcPointer`, for consistency with `OwnedPointer`.

- `UnsafePointer` parameters (other than the type) are now keyword-only.

- Inferred-only parameters may now be explicitly bound with keywords, enabling
  some important patterns in the standard library:

  ```mojo
  struct StringSlice[is_mutable: Bool, //, origin: Origin[is_mutable]]: ...
  alias ImmStringSlice = StringSlice[is_mutable=False]
  # This auto-parameterizes on the origin, but constrains it to being an
  # immutable slice instead of a potentially mutable one.
  fn take_imm_slice(a: ImmStringSlice): ...
  ```

- Added `PythonObject.__contains__`.
  ([PR #3101](https://github.com/modularml/mojo/pull/3101) by [@rd4com](https://github.com/rd4com))

  Example usage:

  ```mojo
  x = PythonObject([1,2,3])
  if 1 in x:
     print("1 in x")

- `Span` has moved from the `utils` module to the `memory` module.

### ❌ Removed

- The `UnsafePointer.bitcast` overload for `DType` has been removed.  Wrap your
  `DType` in a `Scalar[my_dtype]` to call the only overload of `bitcast` now.

### 🛠️ Fixed

- Lifetime tracking is now fully field sensitive, which makes the uninitialized
  variable checker more precise.

- [Issue #1310](https://github.com/modularml/mojo/issues/1310) - Mojo permits
  the use of any constructor for implicit conversions

- [Issue #1632](https://github.com/modularml/mojo/issues/1632) - Mojo produces
  weird error when inout function is used in non mutating function

- [Issue #3444](https://github.com/modularml/mojo/issues/3444) - Raising init
  causing use of uninitialized variable

- [Issue #3544](https://github.com/modularml/mojo/issues/3544) - Known
  mutable `ref` argument are not optimized as `noalias` by LLVM.

- [Issue #3559](https://github.com/modularml/mojo/issues/3559) - VariadicPack
  doesn't extend the lifetimes of the values it references.

- [Issue #3627](https://github.com/modularml/mojo/issues/3627) - Compiler
  overlooked exclusivity violation caused by `ref [MutableAnyOrigin] T`

- [Issue #3710](https://github.com/modularml/mojo/issues/3710) - Mojo frees
  memory while reference to it is still in use.

- [Issue #3805](https://github.com/modularml/mojo/issues/3805) - Crash When
  Initializing !llvm.ptr.

- [Issue #3816](https://github.com/modularml/mojo/issues/3816) - Ternary
  if-operator doesn't propagate origin information.

- [Issue #3815](https://github.com/modularml/mojo/issues/3815) -
  [BUG] Mutability not preserved when taking the union of two origins.

- [Issue #3829](https://github.com/modularml/mojo/issues/3829) - Poor error
  message when invoking a function pointer upon an argument of the wrong origin

- [Issue #3830](https://github.com/modularml/mojo/issues/3830) - Failures
  emitting register RValues to ref arguments.

- The VS Code extension now auto-updates its private copy of the MAX SDK.

- The variadic initializer for `SIMD` now works in parameter expressions.

- The VS Code extension now downloads its private copy of the MAX SDK in a way
  that prevents ETXTBSY errors on Linux.

- The VS Code extension now allows invoking a mojo formatter from SDK
  installations that contain white spaces in their path.

- Error messages that include type names no longer include inferred or defaulted
  parameters when they aren't needed.  For example, previously Mojo complained
  about things like:

  ```plaintext
  ... cannot be converted from 'UnsafePointer[UInt, 0, _default_alignment::AnyType](), MutableAnyOrigin]' to 'UnsafePointer[Int, 0, _default_alignment[::AnyType](), MutableAnyOrigin]'
  ```

  it now complains more helpfully that:

  ```plaintext
  ... cannot be converted from 'UnsafePointer[UInt]' to 'UnsafePointer[Int]'
  ```

- Tooling now prints the origins of `ref` arguments and results correctly, and
  prints `self` instead of `self: Self` in methods.

- The LSP and generated documentation now print parametric result types
  correctly, e.g. showing `SIMD[type, simd_width]` instead of `SIMD[$0, $1]`.
