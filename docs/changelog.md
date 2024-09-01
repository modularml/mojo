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

- The VS Code Mojo Debugger now has a `buildArgs` JSON debug configuration
  setting that can be used in conjunction with `mojoFile` to define the build
  arguments when compiling the Mojo file.

- The VS Code extension now supports a `Configure Build and Run Args` command
  that helps set the build and run args for actions file `Run Mojo File` and
  `Debug Mojo File`. A corresponding button appears in `Run and Debug` selector
  in the top right corner of a Mojo File.

### 🦋 Changed

- A new `as_noalias_ptr` method as been added to `UnsafePointer`. This method
  specifies to the compiler that the resultant pointer is a distinct
  identifiable object that does not alias any other memory in the local scope.

### ❌ Removed

### 🛠️ Fixed

- Lifetime tracking is now fully field sensitive, which means that the
  uninitialized variable checker is more precise.
