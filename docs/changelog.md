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

### 🔥 Legendary

- Tuple now works with memory-only element types like String.  Also, `Tuple.get`
  now supports a form that just takes an element index but does not
  require you to specify the result type.  Instead of `tup.get[1, Int]()` you
  can now just use `tup.get[1]()`.

### ⭐️ New

- Heterogenous variadic pack arguments now work reliably even with memory types,
  and have a more convenient API to use, as defined on the `VariadicPack` type.
  For example, a simplified version of `print` can be implemented as:

  ```mojo
  fn print[T: Stringable, *Ts: Stringable](first: T, *rest: *Ts):
      print_string(str(first))

      @parameter
      fn print_elt[T: Stringable](a: T):
          print_string(" ")
          print_string(a)
      rest.each[print_elt]()
  ```

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

- `PythonObject` now conforms to the `KeyElement` trait, meaning that it can be
  used as key type for `Dict`. This allows on to easily build and interact with
  Python dictionaries in mojo:

  ```mojo
  def main():
      d = PythonObject(Dict[PythonObject, PythonObject]())
      d["foo"] = 12
      d[7] = "bar"
      d["foo"] = [1, 2, "something else"]
      print(d)  # prints `{'foo': [1, 2, 'something else'], 7: 'bar'}`
  ```

- `List` now has several new methods:
  - `pop(index)` for removing an element at a particular index.  
    By default, `List.pop()` removes the last element in the list.
  - `resize(new_size)` for resizing the list without the need to
    specify an additional value.
  - `insert(index, value)` for inserting a value at a specified index
    into the `List`.
  - constructor from `(ptr, size, capacity)` to to avoid needing to do a deep
    copy of an existing contiguous memory allocation when constructing a new `List`.

- `Dict` now has a `update()` method to update keys/values from another `Dict`.

- `Set` now has named methods for set operations:
  - `Set.difference()` mapping to `-`
  - `Set.difference_update()` mapping to `-=`
  - `Set.intersection_update()` mapping to `&=`
  - `Set.update()` mapping to `|=`

- `String` now has `removeprefix()` and `removesuffix()` methods.

- `Optional` now implements `__is__` and `__isnot__` methods so that you can compare
  an `Optional` with `None`, e.g. `Optional(1) is not None` for example.

- The `ord` and `chr` functions have been improved to accept any Unicode character.

- `Atomic` is now movable.

- `Dict` and `List` are both `Boolable` now.

- `atol` now handles whitespaces so `int(String( " 10 "))` gives back `10`
  instead of raising an error.

- `SIMD` now implements `__rmod__`.

- `bool(None)` is now implemented.

- A low-level `__get_mvalue_as_litref(x)` builtin was added to give access to
  the underlying memory representation as a `!lit.ref` value without checking
  initialization status of the underlying value.  This is useful in very
  low-level logic but isn't designed for general usability and will likely
  change in the future.

- The `testing.assert_equal[SIMD]()` now raises if any of the elements
  mismatch in the two `SIMD` arguments being compared.

- The `testing.assert_almost_equal` and `math.isclose` functions now have an
  `equal_nan` flag. When set to True, then NaNs are considered equal.

- Mojo now supports declaring functions that have both optional and variadic
  arguments, both positional and keyword-only. E.g. this now works:

  ```mojo
  fn variadic_arg_after_default(
    a: Int, b: Int = 3, *args: Int, c: Int, d: Int = 1, **kwargs: Int
  ): ...
  ```

  Positional variadic parameters also work in the presence of optional
  parameters, i.e.:

  ```mojo
  fn variadic_param_after_default[e: Int, f: Int = 2, *params: Int]():
    pass
  ```

  Note that variadic keyword parameters are not supported yet.

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
- `AnyPointer` got renamed to `UnsafePointer` and is now Mojo's preferred unsafe
  pointer type.  It has several enhancements, including:
  1) The element type can now be `AnyType`: it doesn't require `Movable`.
  2) Because of this, the `take_value`, `emplace_value`, and `move_into` methods
     have been changed to be top-level functions, and were renamed to
     `move_from_pointee`, `initialize_pointee` and `move_pointee` respectively.
  3) A new `destroy_pointee` function runs the destructor on the pointee.
  4) `UnsafePointer` can be initialized directly from a `Reference` with
     `UnsafePointer(someRef)` and can convert to an immortal reference with
     `yourPointer[]`.  Both infer element type and address space.
- All of the pointers got a pass of cleanup to make them more consistent, for
  example the `unsafe.bitcast` global function is now a consistent `bitcast`
  method on the pointers, which can convert element type and address space.
- The `Reference` type has several changes, including:
  1) It is now located in `memory.reference` instead of `memory.unsafe`.
  2) `Reference` now has an unsafe `unsafe_bitcast` method like `UnsafePointer`.
  3) Several unsafe methods were removed, including `offset`,
     `destroy_element_unsafe` and `emplace_ref_unsafe`. This is because
     `Reference` is a safe type - use `UnsafePointer` to do unsafe operations.

- The `mojo package` command no longer supports the `-D` flag. All compilation
  environment flags should be provided at the point of package use (e.g.
  `mojo run` or `mojo build`).

- `parallel_memcpy` function has moved from the `buffer` package to the
  `algorithm` package. Please update your imports accordingly.

- `FileHandle.seek()` now has a whence argument that defaults to `os.SEEK_SET`
  to seek from the beginning of the file. You can now set to `os.SEEK_CUR` to
  offset by the current `FileHandle` seek position:

  ```mojo
  var f = open("/tmp/example.txt")
  # Skip 32 bytes
  f.seek(os.SEEK_CUR, 32)
  ```

  Or `os.SEEK_END` to offset from the end of file:

  ```mojo
  # Start from 32 bytes before the end of the file
  f.seek(os.SEEK_END, -32)
  ```

  - `FileHandle.read()` can now read straight into a `DTypePointer`:

    ```mojo
    var file = open("/tmp/example.txt", "r")

    # Allocate and load 8 elements
    var ptr = DTypePointer[DType.float32].alloc(8)
    var bytes = file.read(ptr, 8)
    print("bytes read", bytes)
    print(ptr.load[width=8]())
    ```

### ❌ Removed

- Support for "register only" variadic packs has been removed. Instead of
  `AnyRegType`, please upgrade your code to `AnyType` in examples like this:

  ```mojo
  fn your_function[*Types: AnyRegType](*args: *Ts): ...
  ```

  This move gives you access to nicer API and has the benefit of being memory
  safe and correct for non-trivial types.  If you need specific APIs on the
  types, please use the correct trait bound instead of `AnyType`.

- `List.pop_back()` has been removed.  Use `List.pop()` instead which defaults
  to popping the last element in the list.

- `SIMD.to_int(value)` has been removed.  Use `int(value)` instead.

- The `__get_lvalue_as_address(x)` magic function has been removed.  To get a
  reference to a value use `Reference(x)` and if you need an unsafe pointer, you
  can use `UnsafePointer.address_of(x)`.

### 🛠️ Fixed

- [#516](https://github.com/modularml/mojo/issues/516) and
  [#1817](https://github.com/modularml/mojo/issues/1817) and many others, e.g.
  "Can't create a function that returns two strings"

- [#1178](https://github.com/modularml/mojo/issues/1178) (os/kern) failure (5)

- [#1609](https://github.com/modularml/mojo/issues/1609) alias with
  `DynamicVector[Tuple[Int]]` fails.

- [#1987](https://github.com/modularml/mojo/issues/1987) Defining `main`
  in a Mojo package is an error, for now. This is not intended to work yet,
  erroring for now will help to prevent accidental undefined behavior.

- [#1215](https://github.com/modularml/mojo/issues/1215) and
  [#1949](https://github.com/modularml/mojo/issues/1949) The Mojo LSP server no
  longer cuts off hover previews for functions with functional arguments,
  parameters, or results.

- [#1901](https://github.com/modularml/mojo/issues/1901) Fixed Mojo LSP and
  documentation generation handling of inout arguments.

- [#1913](https://github.com/modularml/mojo/issues/1913) - `0__` no longer
  crashes the Mojo parser.

- [#1924](https://github.com/modularml/mojo/issues/1924) JIT debugging on Mac
  has been fixed.

- [#1941](https://github.com/modularml/mojo/issues/1941) Mojo variadics don't
  work with non-trivial register-only types.

- [#1963](https://github.com/modularml/mojo/issues/1963) `a!=0` is now parsed
  and formatted correctly by `mojo format`.

- [#1676](https://github.com/modularml/mojo/issues/1676) Fix a crash related to
  `@value` decorator and structs with empty body.

- [#1917](https://github.com/modularml/mojo/issues/1917) Fix a crash after
  syntax error during tuple creation

- [#2006](https://github.com/modularml/mojo/issues/2006) The Mojo LSP now
  properly supports signature types with named arguments and parameters.

- [#2007](https://github.com/modularml/mojo/issues/2007) and
  [#1997](https://github.com/modularml/mojo/issues/1997) The Mojo LSP no longer
  crashes on certain types of closures.

- [#1675](https://github.com/modularml/mojo/issues/1675) Ensure `@value`
  decorator fails gracefully after duplicate field error.
