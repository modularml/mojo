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

- Added temporary `SliceNew` type with corrected behaviour from `Slice` to facilitate
  an incremental internal migration due to reliance on the old, incorrect behaviour.
  ([PR #2894](https://github.com/modularml/mojo/pull/2894) by [@bgreni](https://github.com/bgreni))

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

### ❌ Removed

- It is no longer possible to cast (implicitly or explicitly) from `Reference`
  to `UnsafePointer`.  Instead of `UnsafePointer(someRef)` please use the
  `UnsafePointer.address_of(someRef[])` which makes the code explicit that the
  `UnsafePointer` gets the address of what the reference points to.

- Removed `String.unsafe_uint8_ptr()`. `String.unsafe_ptr()` now returns the
  same thing.

- Removed `StringLiteral.unsafe_uint8_ptr()` and `StringLiteral.as_uint8_ptr()`.

- Removed `UnsafePointer.offset(offset:Int)`.

### 🛠️ Fixed
