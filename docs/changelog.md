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

- Creating nested `PythonObject` from a list or tuple of python objects is
  possible now:

  ```mojo
  var np = Python.import_module("numpy")
  var a = np.array([1, 2, 3])
  var b = np.array([4, 5, 6])
  var arrays = PythonObject([a, b])
  assert_equal(len(arrays), 2)
  ```

  Also allowing more convenient call syntax:

  ```mojo
  var stacked = np.hstack((a, b))
  assert_equal(str(stacked), "[1 2 3 4 5 6]")
  ```

  ([PR 3264#](https://github.com/modularml/mojo/pull/3264) by
  [@kszucs](https://github.com/kszucs))

- `List[T]` values are now equality comparable with `==` and `!=` when `T` is
  equality comparable.
  ([PR 3195#](https://github.com/modularml/mojo/pull/3195) by
  [@kszucs](https://github.com/kszucs))

- `__setitem__` now works with variadic argument lists such as:

  ```mojo
  struct YourType:
      fn __setitem__(inout self, *indices: Int, val: Int): ...
  ```

  The Mojo compiler now always passes the "new value" being set using the last
  keyword argument of the `__setitem__`, e.g. turning `yourType[1, 2] = 3` into
  `yourType.__setitem__(1, 2, val=3)`.  This fixes
  [Issue #248](https://github.com/modularml/mojo/issues/248).

- The pointer variants (`DTypePointer`, `UnsafePointer`, etc.) now have a new
  `exclusive: Bool = False` parameter. Setting this parameter to true tells the
  compiler that the user knows this pointer and all those derived from it have
  exclusive access to the underlying memory allocation. The compiler is not
  guaranteed to do anything with this information.

- `Optional` values are now equality comparable with `==` and `!=` when their
  element type is equality comparable.

- Added a new [`Counter`](/mojo/stdlib/collections/counter/Counter)
  dictionary-like type, matching most of the features of the Python one.
  ([PR 2910#](https://github.com/modularml/mojo/pull/2910) by
  [@msaelices](https://github.com/msaelices))

- Mojo context managers used in regions of code that may raise no longer need to
  define a "conditional" exit function in the form of
  `fn __exit__(self, e: Error) -> Bool`. This function allows the context
  manager to conditionally intercept and handle the error and allow the function
  to continue executing. This is useful for some applications, but in many cases
  the conditional exit would delegate to the unconditional exit function
  `fn __exit__(self)`.

  Concretely, this enables defining `with` regions that unconditionally
  propagate inner errors, allowing code like:

  ```mojo
  def might_raise() -> Int:
      ...

  def foo() -> Int:
      with ContextMgr():
          return might_raise()
      # no longer complains about missing return

  def bar():
      var x: Int
      with ContextMgr():
          x = might_raise()
      print(x) # no longer complains about 'x' being uninitialized
  ```

- Now supports "conditional conformances" where some methods on a struct have
  additional trait requirements that the struct itself doesn't. This is
  expressed through an explicitly declared `self` type:

  ```mojo
  struct GenericThing[Type: AnyType]:  # Works with anything
    # Sugar for 'fn normal_method[Type: AnyType](self: GenericThing[Type]):'
    fn normal_method(self): ...

    # Just redeclare the requirements with more specific types:
    fn needs_move[Type: Movable](self: GenericThing[Type], owned val: Type):
      var tmp = val^  # Ok to move 'val' since it is Movable
      ...
  fn usage_example():
    var a = GenericThing[Int]()
    a.normal_method() # Ok, Int conforms to AnyType
    a.needs_move(42)  # Ok, Int is movable

    var b = GenericThing[NonMovable]()
    b.normal_method() # Ok, NonMovable conforms to AnyType

      # error: argument type 'NonMovable' does not conform to trait 'Movable'
    b.needs_move(NonMovable())
  ```

  Conditional conformance works with dunder methods and other things as well.

- `async` functions now support memory-only results (like `String`, `List`,
  etc.) and `raises`. Accordingly, both `Coroutine` and `RaisingCoroutine` have
  been changed to accept `AnyType` instead of `AnyTrivialRegType`. This means
  the result types of `async` functions do not need to be `Movable`.

  ```mojo
  async fn raise_or_string(c: Bool) raises -> String:
      if c:
          raise "whoops!"
      return "hello world!"
  ```

  Note that `async` functions do not yet support indirect calls, `ref` results,
  and constructors.

- As a specific form of "conditional conformances", initializers in a struct
  may indicate specific parameter bindings to use in the type of their `self`
  argument. For example:

  ```mojo
  @value
  struct MyStruct[size: Int]:
      fn __init__(inout self: MyStruct[0]): pass
      fn __init__(inout self: MyStruct[1], a: Int): pass
      fn __init__(inout self: MyStruct[2], a: Int, b: Int): pass

  def test(x: Int):
      a = MyStruct()      # Infers size=0 from 'self' type.
      b = MyStruct(x)     # Infers size=1 from 'self' type.
      c = MyStruct(x, x)  # Infers size=2 from 'self' type.
  ```

- The `Reference` type (and many iterators) now use "inferred" parameters to
  represent the mutability of their lifetime, simplifying the interface.

- `Dict` now implements `setdefault`, to get a value from the dictionary by
  key, or set it to a default if it doesn't exist
  ([PR #2803](https://github.com/modularml/mojo/pull/2803)
  by [@msaelices](https://github.com/msaelices))

- Added new `ExplicitlyCopyable` trait, to mark types that can be copied
  explicitly, but which might not be implicitly copyable.

  This supports work to transition the standard library collection types away
  from implicit copyability, which can lead to unintended expensive copies.

- Added `Identifiable` trait, used to describe types that implement the `__is__`
  and `__isnot__` trait methods.
  ([PR #2807](https://github.com/modularml/mojo/pull/2807))

  - Also added new `assert_is()` and `assert_is_not()` test utilities to the
    `testing` module.

- `Dict` now supports `popitem`, which removes and returns the last item in the `Dict`.
  ([PR #2701](https://github.com/modularml/mojo/pull/2701)
  by [@jayzhan211](https://github.com/jayzhan211))

- Added `unsafe_cstr_ptr()` method to `String` and `StringLiteral`, that
  returns an `UnsafePointer[C_char]` for convenient interoperability with C
  APIs.

- Added `C_char` type alias in `sys.ffi`.

- Added `StringSlice(..)` initializer from a `StringLiteral`.

- Added a `byte_length()` method to `String`, `StringSlice`, and `StringLiteral`
and deprecated their private `_byte_length()` methods. Added a warning to
`String.__len__` method that it will return length in Unicode codepoints in the
future and `StringSlice.__len__` now does return the Unicode codepoints length.
([PR #2960](https://github.com/modularml/mojo/pull/2960) by [@martinvuyk](https://github.com/martinvuyk))

- Added `String.from_unicode(values: List[Int]) -> String` and
`String.from_utf16(values: List[UInt16]) -> String` functions that return a
String containing the concatenated characters. If a Unicode codepoint
is invalid, the parsed String has a replacement character (�) in that index.
`fn chr(c: Int) -> String` function now returns a replacement character (�)
if the Unicode codepoint is invalid.
([PR #3239](https://github.com/modularml/mojo/pull/3239) by [@martinvuyk](https://github.com/martinvuyk))

- Added new `StaticString` type alias. This can be used in place of
  `StringLiteral` for runtime string arguments.

- Added `TemporaryDirectory` in module `tempfile`.
  ([PR 2743](https://github.com/modularml/mojo/pull/2743) by [@artemiogr97](https://github.com/artemiogr97))

- Added `NamedTemporaryFile` in module `tempfile`.
  ([PR 2762](https://github.com/modularml/mojo/pull/2762) by [@artemiogr97](https://github.com/artemiogr97))

- Added `oct(..)` function for formatting an integer in octal.
  ([PR #2914](https://github.com/modularml/mojo/pull/2914) by [@bgreni](https://github.com/bgreni))

- Added `String.format` method.
  ([PR #2771](https://github.com/modularml/mojo/pull/2771) by [@rd4com](https://github.com/rd4com))

  Support automatic and manual indexing of `*args`.

  Examples:

  ```mojo
  print(
    String("{1} Welcome to {0} {1}").format("mojo", "🔥")
  )
  # 🔥 Wecome to mojo 🔥
  ```

  ```mojo
  print(String("{} {} {}").format(True, 1.125, 2))
  #True 1.125 2
  ```

- Environment variable `MOJO_PYTHON` can be pointed to an executable to pin Mojo
  to a specific version:

  ```sh
  export MOJO_PYTHON="/usr/bin/python3.11"
  ```

  Or a virtual environment to always have access to those Python modules:

  ```sh
  export MOJO_PYTHON="~/venv/bin/python"
  ```

  `MOJO_PYTHON_LIBRARY` still exists for environments with a dynamic libpython,
  but no Python executable.

- The `math` package now includes the `pi`, `e`, and `tau` constants (Closes
  Issue [#2135](https://github.com/modularml/mojo/issues/2135)).

- Mojo now has a `UInt` type for modeling unsigned (scalar) integers with a
  paltform-dependent width. `UInt` implements most arithmetic operations that
  make sense for integers, with the notable exception of `__neg__`. Builtin
  functions such as `min`/`max`, as well as `math` functions like `ceildiv`,
  `align_down`, and `align_up` are also implemented for `UInt`.

- `os.path.expanduser()` and `pathlib.Path.exapanduser()` have been added to
  allow expanding a prefixed `~` in a `String` or `Path` with the users home
  path:

  ```mojo
  import os
  print(os.path.expanduser("~/.modular"))
  # /Users/username/.modular
  print(os.path.expanduser("~root/folder"))
  # /var/root/folder (on macos)
  # /root/folder     (on linux)
  ```

- `Path.home()` has been added to return a path of the users home directory.

- `os.path.split()` has been added for splitting a path into `head, tail`:

  ```mojo
  import os
  head, tail = os.path.split("/this/is/head/tail")
  print("head:", head)
  print("tail:", tail)
  # head: /this/is/head
  # tail: tail
  ```

- `os.path.makedirs()` and `os.path.removedirs()` have been added for creating
  and removing nested directories:

  ```mojo
  import os
  path = os.path.join("dir1", "dir2", "dir3")
  os.path.makedirs(path, exist_ok=True)
  os.path.removedirs(path)
  ```

- The `pwd` module has been added for accessing user information in
  `/etc/passwd` on POSIX systems. This follows the same logic as Python:

  ```mojo
  import pwd
  import os
  current_user = pwd.getpwuid(os.getuid())
  print(current_user)

  # pwd.struct_passwd(pw_name='jack', pw_passwd='********', pw_uid=501,
  # pw_gid=20, pw_gecos='Jack Clayton', pw_dir='/Users/jack',
  # pw_shell='/bin/zsh')

  print(current_user.pw_uid)

  # 501

  root = pwd.getpwnam("root")
  print(root)

  # pwd.struct_passwd(pw_name='root', pw_passwd='*', pw_uid=0, pw_gid=0,
  # pw_gecos='System Administrator', pw_dir='/var/root', pw_shell='/bin/zsh')
  ```

- Added `Dict.__init__` overload to specify initial capacity.
  ([PR #3171](https://github.com/modularml/mojo/pull/3171) by [@rd4com](https://github.com/rd4com))

  The capacity has to be a power of two and above or equal 8.

  It allows for faster initialization by skipping incremental growth steps.

  Example:

  ```mojo
  var dictionary = Dict[Int,Int](power_of_two_initial_capacity = 1024)
  # Insert (2/3 of 1024) entries
  ```

- `ListLiteral` now supports `__contains__`.
  ([PR #3251](https://github.com/modularml/mojo/pull/3251) by
  [@jjvraw](https://github.com/jjvraw))

- `bit` module now supports `bit_reverse()`, `byte_swap()` and `pop_count()` for
  `Int` type.
  ([PR #3150](https://github.com/modularml/mojo/pull/3150) by [@LJ-9801](https://github.com/LJ-9801))

### 🦋 Changed

- The pointer aliasing semantics of Mojo have changed. Initially, Mojo adopted a
  C-like set of semantics around pointer aliasing and derivation. However, the C
  semantics bring a lot of history and baggage that are not needed in Mojo and
  which complicate compiler optimizations. The language overall provides a
  stronger set of invariants around pointer aliasing with lifetimes and
  exclusive mutable references to values, etc.

  It is now forbidden to convert a non-pointer-typed value derived from a
  Mojo-allocated pointer, such as an integer address, to a pointer-typed value.
  "Derived" means there is overlap in the bits of the non-pointer-typed value
  with the original pointer value.

  It is still possible to make this conversion in certain cases where it is
  absolutely necessary, such as interoperating with other languages like Python.
  In this case, the compiler makes two assumptions: any pointer derived from a
  non-pointer-typed value does not alias any Mojo-derived pointer and that any
  external function calls have arbitrary memory effects.

- `await` on a coroutine now consumes it. This strengthens the invariant that
  coroutines can only be awaited once.

- Continued transition to `UnsafePointer` and unsigned byte type for strings:

  - `String.unsafe_ptr()` now returns an `UnsafePointer[UInt8]`
    (was `UnsafePointer[Int8]`)
  - `StringLiteral.unsafe_ptr()` now returns an `UnsafePointer[UInt8]`
    (was `UnsafePointer[Int8]`)

- `print()` now requires that its arguments conform to the `Formattable` trait.
  This enables efficient stream-based writing by default, avoiding unnecessary
  intermediate String heap allocations.

  Previously, `print()` required types conform to `Stringable`. This meant that
  to execute a call like `print(a, b, c)`, at least three separate String heap
  allocations were down, to hold the formatted values of `a`, `b`, and `c`
  respectively. The total number of allocations could be much higher if, for
  example, `a.__str__()` was implemented to concatenate together the fields of
  `a`, like in the following example:

  ```mojo
  struct Point(Stringable):
      var x: Float64
      var y: Float64

      fn __str__(self) -> String:
          # Performs 3 allocations: 1 each for str(..) of each of the fields,
          # and then the final returned `String` allocation.
          return "(" + str(self.x) + ", " + str(self.y) + ")"
  ```

  A type like the one above can transition to additionally implementing
  `Formattable` with the following changes:

  ```mojo
  struct Point(Stringable, Formattable):
      var x: Float64
      var y: Float64

      fn __str__(self) -> String:
          return String.format_sequence(self)

      fn format_to(self, inout writer: Formatter):
          writer.write("(", self.x, ", ", self.y, ")")
  ```

  In the example above, `String.format_sequence(<arg>)` is used to construct a
  `String` from a type that implements `Formattable`. This pattern of
  implementing a type's `Stringable` implementation in terms of its `Formattable`
  implementation minimizes boilerplate and duplicated code, while retaining
  backwards compatibility with the requirements of the commonly used `str(..)`
  function.

  <!-- TODO(MOCO-891): Remove this warning when error is improved. -->

  > [!WARNING]
  > The error shown when passing a type that does not implement `Formattable` to
  > `print()` is currently not entirely descriptive of the underlying cause:
  >
  > ```shell
  > error: invalid call to 'print': callee with non-empty variadic pack argument expects 0 positional operands, but 1 was specified
  >    print(point)
  >    ~~~~~^~~~~~~
  > ```
  >
  > If the above error is seen, ensure that all argument types implement
  > `Formattable`.

- The `StringRef` constructors from `DTypePointer.int8` have been changed to
  take a `UnsafePointer[C_char]`, reflecting their use for compatibility with
  C APIs.

- The global functions for working with `UnsafePointer` have transitioned to
  being methods through the use of conditional conformances:

  - `destroy_pointee(p)` => `p.destroy_pointee()`
  - `move_from_pointee(p)` => `p.take_pointee()`
  - `initialize_pointee_move(p, value)` => `p.init_pointee_move(value)`
  - `initialize_pointee_copy(p, value)` => `p.init_pointee_copy(value)`
  - `move_pointee(src=p1, dst=p2)` => `p.move_pointee_into(p2)`

- `DTypePointer.load/store/prefetch` has been now moved to `SIMD`. Instead of
  using `ptr.load[width=4](offset)` one should use `SIMD[size=4].load(ptr, offset)`.
  Note the default load width before was 1, but the default size of `SIMD` is
  the size of the SIMD type.
  The default store size is the size of the `SIMD` value to be stored.

- `Slice` now uses `OptionalReg[Int]` for `start` and `end` and implements
  a constructor which accepts optional values. `Slice._has_end()` has also been
  removed since a Slice with no end is now represented by an empty `Slice.end`
  option.
  ([PR #2495](https://github.com/modularml/mojo/pull/2495) by [@bgreni](https://github.com/bgreni))

  ```mojo
    var s = Slice(1, None, 2)
    print(s.start.value()) # must retrieve the value from the optional
  ```

- `NoneType` is now a normal standard library type, and not an alias for a raw
  MLIR type.

  Function signatures spelled as `fn(...) -> NoneType` should transition to
  being written as `fn(...) -> None`.

- Accessing local Python modules with `Python.add_to_path(".")` is no longer
  required, it now behaves the same as Python, you can access modules in the
  same folder as the target file:

  - `mojo run /tmp/main.mojo` can access `/tmp/mymodule.py`
  - `mojo build main.mojo -o ~/myexe && ~/myexe` can access `~/mymodule.py`

- The rank argument for `algorihtm.elementwise` is no longer required and is
  only inferred.

- The `ulp` function in `numerics` has been moved to the `math` module.

- The Mojo Language Server no longer sets `.` as a commit character for
  auto-completion.

- Types conforming to `Boolable` (i.e. those implementing `__bool__`) no longer
  implicitly convert to `Bool`. A new `ImplicitlyBoolable` trait is introduced
  for types where this behavior is desired.

- The `time.now()` function has been deprecated. Please use `time.perf_counter`
  or `time.perf_counter_ns` instead.

- `LegacyPointer.load/store` are now removed. It's use is replaced with
  `__getitem__` or `__setitem__`.

- `memcmp`, `memset` and `memset_zero` no longer take in `LegacyPointer`,
  instead, use `UnsafePointer`.

- A few bit functions have been renamed for clarity:
- `countl_zero` -> `count_leading_zeros`
- `countr_zero` -> `count_trailing_zeros`

- `sort` no longer takes `LegacyPointer`. The current API supports:
  - `sort(list)` just plain list
  - `sort[type, cmp_fn](list)` list with custom compare function
  - `sort(ptr, len)` a pointer and length (can change to Span in future)
  - `sort[type, cmp_fn](ptr, len)` above with custom compare

- `memcpy` with `LegacyPointer` has been removed. Please use the `UnsafePointer`
  overload instead.

- `LegacyPointer` and `Pointer` has been removed. Please use `UnsafePointer`
 instead.

- `UnsafePointer` now supports `simd_strided_load/store`, `gather`, and `scatter`
  when the underlying type is `Scalar[DType]`.

- `SIMD.load/store` now supports `UnsafePointer` overloads.

- Now that we have a `UInt` type, use this to represent the return type of hash.
  In general, hashes should be an unsigned integer, and can also lead to improved
  performance in certain cases.

- The `atol` function now correctly supports leading underscores,
  (e.g.`atol("0x_ff", 0)`), when the appropriate base is specified or inferred
  (base 0). non-base-10 integer literals as per Python's [Integer Literals](\
  <https://docs.python.org/3/reference/lexical_analysis.html#integers>).
  ([PR #3180](https://github.com/modularml/mojo/pull/3180)
  by [@jjvraw](https://github.com/jjvraw))

### ❌ Removed

- It is no longer possible to cast (implicitly or explicitly) from `Reference`
  to `UnsafePointer`. Instead of `UnsafePointer(someRef)` please use the
  `UnsafePointer.address_of(someRef[])` which makes the code explicit that the
  `UnsafePointer` gets the address of what the reference points to.

- Removed `String.unsafe_uint8_ptr()`. `String.unsafe_ptr()` now returns the
  same thing.

- Removed `StringLiteral.unsafe_uint8_ptr()` and `StringLiteral.as_uint8_ptr()`.

- Removed `UnsafePointer.offset(offset:Int)`.

- Removed `SIMD.splat(value: Scalar[type])`. Use the constructor for SIMD
  instead.

- The builtin `tensor` module has been removed. Identical functionality is
  available in `max.tensor`, but it is generally recommended to use `buffer`
  when possible instead.

- Removed the Mojo Language Server warnings for unused function arguments.

- Removed the `SIMD.{add,mul,sub}_with_overflow` methods.

- Removed the `SIMD.min` and `SIMD.max` methods. Identical functionality is
  available using the builting `min` and `max` functions.

### 🛠️ Fixed

- Fixed a crash in the Mojo Language Server when importing the current file.

- Fixed crash when specifying variadic keyword arguments without a type
  expression in `def` functions, e.g.:

  ```mojo
  def foo(**kwargs): ...  # now works
  ```

- [#3142](https://github.com/modularml/mojo/issues/3142) - [QoI] Confusing
  `__setitem__` method is failing with a "must be mutable" error.

- [#248](https://github.com/modularml/mojo/issues/248) - [Feature] Enable
  `__setitem__` to take variadic arguments

- [#3065]<https://github.com/modularml/mojo/issues/3065> - Fix incorrect behavior
  of `SIMD.__int__` on unsigned types

- [#3045]<https://github.com/modularml/mojo/issues/3045> - Disable implicit SIMD
  conversion routes through `Bool`
