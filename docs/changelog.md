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

- `Dict` now supports `popitem`, which removes and returns the last item in the `Dict`.
([PR #2701](https://github.com/modularml/mojo/pull/2701)
by [@jayzhan211](https://github.com/jayzhan211))

- Added `String.unsafe_cstr_ptr(self)` that returns an `UnsafePointer[C_char]`
  for convenient interoperability with C APIs.

- Added `C_char` type alias in `sys.ffi`.

### 🦋 Changed

- Continued transition to `UnsafePointer` and unsigned byte type for strings:
  - Rename `String._as_ptr()` to `String.unsafe_ptr()`
  - `String.unsafe_ptr()` now returns an `UnsafePointer[UInt8]`
    (was `DTypePointer[DType.int8]`)

### ❌ Removed

### 🛠️ Fixed
