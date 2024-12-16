# Mojo unreleased changelog

This is a list of UNRELEASED changes for the Mojo language and tools.

When we cut a release, these notes move to `changelog-released.md` and that's
what we publish.

[//]: # Here's the template to use when starting a new batch of notes:
[//]: ## UNRELEASED
[//]: ### ✨ Highlights
[//]: ### Language changes
[//]: ### Standard library changes
[//]: ### Tooling changes
[//]: ### ❌ Removed
[//]: ### 🛠️ Fixed

## UNRELEASED

### ✨ Highlights

### Language changes

### Standard library changes

- `UnsafePointer` is now parameterized on mutability. Previously,
  `UnsafePointer` could only represent mutable pointers.

  The new `mut` parameter can be used to restrict an `UnsafePointer` to a
  specific mutability: `UnsafePointer[T, mut=False]` represents a pointer to
  an immutable `T` value. This is analogous to a `const *` pointer in C++.

  - `UnsafePointer.address_of()` will now infer the origin and mutability
    of the resulting pointer from the argument. For example:

    ```mojo
    var local = 10
    # Constructs a mutable pointer, because `local` is a mutable memory location
    var ptr = UnsafePointer.address_of(local)
    ```

    To force the construction of an immutable pointer to an otherwise mutable
    memory location, use a cast:

    ```mojo
    var local = 10
    # Cast the mutable pointer to be immutable.
    var ptr = UnsafePointer.address_of(local).bitcast[mut=False]()
    ```

  - The `unsafe_ptr()` method on several standard library collection types have
    been updated to use parametric mutability: they will return an `UnsafePointer`
    whose mutability is inherited from the mutability of the `ref self` of the
    receiver at the call site. For example, `ptr1` will be immutable, while
    `ptr2` will be mutable:

    ```mojo
    fn take_lists(read list1: List[Int], mut list2: List[Int]):
        # Immutable pointer, since receiver is immutable `read` reference
        var ptr1 = list1.unsafe_ptr()

        # Mutable pointer, since receiver is mutable `mut` reference
        var ptr2 = list2.unsafe_ptr()
    ```

### Tooling changes

- mblack (aka `mojo format`) no longer formats non-mojo files. This prevents
  unexpected formatting of python files.

- Full struct signature information is now exposed in the documentation
  generator, and in the symbol outline and hover markdown via the Mojo Language
  Server.

### ❌ Removed

- `StringRef` is being deprecated. Use `StringSlice` instead.
  - removed `StringRef.startswith()` and `StringRef.endswith()`

### 🛠️ Fixed

- The Mojo Kernel for Jupyter Notebooks is working again on nightly releases.

- The command `mojo debug --vscode` now sets the current working directory
  properly.

- The Mojo Language Server doesn't crash anymore on empty **init**.mojo files.
  [Issue #3826](https://github.com/modularml/mojo/issues/3826).
