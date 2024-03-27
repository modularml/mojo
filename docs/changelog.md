
This is a running list of significant UNRELEASED changes for the Mojo language
and tools. Please add any significant user-visible changes here.

[//]: # Here's the template to use when starting a new batch of notes:
[//]: ## UNRELEASED
[//]: ### ⭐️ New
[//]: ### 🦋 Changed
[//]: ### ❌ Removed
[//]: ### 🛠️ Fixed

## UNRELEASED

### 🔥 Legendary

### ⭐️ New

- The `sys` module now contains an `exit` function that would exit a Mojo
  program with the specified error code.

- The constructors for `tensor.Tensor` have been changed to be more consistent.
  As a result, one has to pass in the shape as first argument (instead of the
  second) when constructing a tensor with pointer data.

### 🦋 Changed

### ❌ Removed

### 🛠️ Fixed
