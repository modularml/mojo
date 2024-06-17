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

- The `Reference` type (and many iterators) now use "inferred" parameters to
  represent the mutability of their lifetime, simplifying the interface.

- Added new `ExplicitlyCopyable` trait, to mark types that can be copied
  explicitly, but which might not be implicitly copyable.

  This supports work to transition the standard library collection types away
  from implicit copyability, which can lead to unintended expensive copies.

- `Dict` now supports `popitem`, which removes and returns the last item in the `Dict`.
([PR #2701](https://github.com/modularml/mojo/pull/2701)
by [@jayzhan211](https://github.com/jayzhan211))

- Added `unsafe_cstr_ptr()` method to `String` and `StringLiteral`, that
  returns an `UnsafePointer[C_char]` for convenient interoperability with C
  APIs.

- Added `C_char` type alias in `sys.ffi`.

- Added `TemporaryDirectory` in module `tempfile`.
  ([PR 2743](https://github.com/modularml/mojo/pull/2743) by [@artemiogr97](https://github.com/artemiogr97))

- Added `NamedTemporaryFile` in module `tempfile`.
  ([PR 2762](https://github.com/modularml/mojo/pull/2762) by [@artemiogr97](https://github.com/artemiogr97))

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

### 🦋 Changed

- `await` on a coroutine now consumes it. This strengthens the invariant that
  coroutines can only be awaited once.

- Continued transition to `UnsafePointer` and unsigned byte type for strings:
  - `String.unsafe_ptr()` now returns an `UnsafePointer[UInt8]`
    (was `UnsafePointer[Int8]`)
  - `StringLiteral.unsafe_ptr()` now returns an `UnsafePointer[UInt8]`
    (was `UnsafePointer[Int8]`)

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

- Accessing local Python modules with `Python.add_to_path(".")` is no longer
  required, it now behaves the same as Python, you can access modules in the
  same folder as the target file:
  - `mojo run /tmp/main.mojo` can access `/tmp/mymodule.py`
  - `mojo build main.mojo -o ~/myexe && ~/myexe` can access `~/mymodule.py`

- Types conforming to `Boolable` (i.e. those implementing `__bool__`) no longer
  implicitly convert to `Bool`. A new `ImplicitlyBoolable` trait is introduced
  for types where this behavior is desired.

### ❌ Removed

- It is no longer possible to cast (implicitly or explicitly) from `Reference`
  to `UnsafePointer`.  Instead of `UnsafePointer(someRef)` please use the
  `UnsafePointer.address_of(someRef[])` which makes the code explicit that the
  `UnsafePointer` gets the address of what the reference points to.

- Removed `String.unsafe_uint8_ptr()`. `String.unsafe_ptr()` now returns the
  same thing.

- Removed `StringLiteral.unsafe_uint8_ptr()` and `StringLiteral.as_uint8_ptr()`.

- Removed `UnsafePointer.offset(offset:Int)`.

- Removed `SIMD.splat(value: Scalar[type])`.  Use the constructor for SIMD
  instead.

### 🛠️ Fixed
