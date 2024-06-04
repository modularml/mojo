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

- The `Reference` type (and many iterators) now use "inferred" parameters to
  represent the mutability of their lifetime, simplifying the interface.

- Added new `ExplicitlyCopyable` trait, to mark types that can be copied
  explicitly, but which might not be implicitly copyable.

  This supports work to transition the standard library collection types away
  from implicit copyability, which can lead to unintended expensive copies.

- `Dict` now supports `popitem`, which removes and returns the last item in the `Dict`.
([PR #2701](https://github.com/modularml/mojo/pull/2701)
by [@jayzhan211](https://github.com/jayzhan211))

- Added `String.unsafe_cstr_ptr(self)` that returns an `UnsafePointer[C_char]`
  for convenient interoperability with C APIs.

- Added `C_char` type alias in `sys.ffi`.

- Added `TemporaryDirectory` in module `tempfile`.
  ([PR 2743](https://github.com/modularml/mojo/pull/2743) by [@artemiogr97](https://github.com/artemiogr97))

- Added temporary `SliceNew` type with corrected behaviour from `Slice` to facilitate
  an incremental internal migration due to reliance on the old, incorrect behaviour.
  ([PR #2894](https://github.com/modularml/mojo/pull/2894) by [@bgreni](https://github.com/bgreni))

### 🦋 Changed

- Continued transition to `UnsafePointer` and unsigned byte type for strings:
  - `String.unsafe_ptr()` now returns an `UnsafePointer[UInt8]`
    (was `UnsafePointer[Int8]`)

- The global functions for working with `UnsafePointer` have transitioned to
  being methods through the use of conditional conformances:

  - `destroy_pointee(p)` => `p.destroy_pointee()`
  - `move_from_pointee(p)` => `p.take_pointee()`
  - `initialize_pointee_move(p, value)` => `p.init_pointee_move(value)`
  - `initialize_pointee_copy(p, value)` => `p.init_pointee_copy(value)`
  - `move_pointee(src=p1, dst=p2)` => `p.move_pointee_into(p2)`

### ❌ Removed

- Removed `String.unsafe_uint8_ptr()`. `String.unsafe_ptr()` now returns the
  same thing.

- Removed `UnsafePointer.offset(offset:Int)`.

### 🛠️ Fixed
