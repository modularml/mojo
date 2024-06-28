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
  additional trait requirements that the struct itself doesn't.  This is
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
  argument.  For example:

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

- Added `PythonObject.__contains__`.
  ([PR #3101](https://github.com/modularml/mojo/pull/3101) by [@rd4com](https://github.com/rd4com))

  Example usage:

  ```mojo
  x = PythonObject([1,2,3])
  if 1 in x: 
     print("1 in x")
  ```

### 🦋 Changed

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
  implementing a types `Stringable` implementation in terms of its `Formattable`
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
  a constructor which accepts optional values. `Slice._has_end()` has also been removed
  since a Slice with no end is now represented by an empty `Slice.end` option.
  ([PR #2495](https://github.com/modularml/mojo/pull/2495) by [@bgreni](https://github.com/bgreni))

  ```mojo
    var s = Slice(1, None, 2)
    print(s.start.value()) # must retrieve the value from the optional
  ```

- Accessing local Python modules with `Python.add_to_path(".")` is no longer
  required, it now behaves the same as Python, you can access modules in the
  same folder as the target file:
  - `mojo run /tmp/main.mojo` can access `/tmp/mymodule.py`
  - `mojo build main.mojo -o ~/myexe && ~/myexe` can access `~/mymodule.py`

- The rank argument for `algorihtm.elementwise` is no longer required and is
  only inferred.

- The `ulp` function in `numerics` have been moved to the `math` module.

- The Mojo Language Server no longer sets `.` as a commit character for
  auto-completion.

### ❌ Removed

- It is no longer possible to cast (implicitly or explicitly) from `Reference`
  to `UnsafePointer`.  Instead of `UnsafePointer(someRef)` please use the
  `UnsafePointer.address_of(someRef[])` which makes the code explicit that the
  `UnsafePointer` gets the address of what the reference points to.

- Removed `String.unsafe_uint8_ptr()`. `String.unsafe_ptr()` now returns the
  same thing.

- Removed `StringLiteral.unsafe_uint8_ptr()` and `StringLiteral.as_uint8_ptr()`.

- Removed `UnsafePointer.offset(offset:Int)`.

- Removed `SIMD.splat(value: Scalar[type])`.  Use the constructor for SIMD instead.

- The builtin `tensor` module has been removed. Identical functionality is
  available in `max.tensor`, but it is generally recommended to use `buffer`
  when possible instead.

- Removed the Mojo Language Server warnings for unused function arguments.

### 🛠️ Fixed

- Fixed a crash in the Mojo Language Server when importing the current file.
