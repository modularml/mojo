
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

- The constructor for `tensor.Tensor` will now splat a scalar if its passed in.
  For example, `Tensor[DType.float32](TensorShape(2,2), 0)` will construct a
  `2x2` tensor which is initialized with all zeros. This provides an easy way
  to fill the data of a tensor.

- The `mojo build` and `mojo run` commands now support a `-g` option. This
  shorter alias is equivalent to writing `--debug-level full`. This option is
  also available in the `mojo debug` command, but is already the default.

### 🦋 Changed

- The behavior of `mojo build` when invoked without an output `-o` argument has
  changed slightly: `mojo build ./test-dir/program.mojo` now outputs an
  executable to the path `./program`, whereas before it would output to the path
  `./test-dir/program`.
- The REPL no longer allows type level variable declarations to be
  uninitialized, e.g. it will reject `var s: String`.  This is because it does
  not do proper lifetime tracking (yet!) across cells, and so such code would
  lead to a crash.  You can work around this by initializing to a dummy value
  and overwriting later.  This limitation only applies to top level variables,
  variables in functions work as they always have.

### ❌ Removed

### 🛠️ Fixed

- [#1987](https://github.com/modularml/mojo/issues/1987) Defining `main`
  in a Mojo package is an error, for now. This is not intended to work yet,
  erroring for now will help to prevent accidental undefined behavior.

- [#1215](https://github.com/modularml/mojo/issues/1215) and
  [#1949](https://github.com/modularml/mojo/issues/1949) The Mojo LSP server no
  longer cuts off hover previews for functions with functional arguments,
  parameters, or results.
