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

- Tuple now works with memory-only element types like String and allows you to
  directly index into it with a parameter exprssion.  This means you can now
  simply use `x = tup[1]` like Python instead of `x = tup.get[1, Int]()`.  You
  can also assign into tuple elements now as well with `tup[1] = x`.

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
    ([PR #2041](https://github.com/modularml/mojo/pull/2041) by
    [@LJ-9801](https://github.com/LJ-9801), fixes
    [#2017](https://github.com/modularml/mojo/issues/2017)). By default,
    `List.pop()` removes the last element in the list.

  - `resize(new_size)` for resizing the list without the need to specify an
    additional value. ([PR #2140](https://github.com/modularml/mojo/pull/2140)
    by [@mikowals](https://github.com/mikowals), fixes
    [#2133](https://github.com/modularml/mojo/issues/2133))

  - `insert(index, value)` for inserting a value at a specified index into the
    `List`. ([PR #2148](https://github.com/modularml/mojo/pull/2148) by
    [@whym1here](https://github.com/whym1here), fixes
    [#2134](https://github.com/modularml/mojo/issues/2134))

  - constructor from `(ptr, size, capacity)` to to avoid needing to do a deep
    copy of an existing contiguous memory allocation when constructing a new
    `List`. ([PR #2182](https://github.com/modularml/mojo/pull/2182) by
    [@StandinKP](https://github.com/StandinKP), fixes
    [#2170](https://github.com/modularml/mojo/issues/2170))

- `Dict` now has a `update()` method to update keys/values from another `Dict`.
  ([PR #2085](https://github.com/modularml/mojo/pull/2085) by
  [@gabrieldemarmiesse](https://github.com/gabrieldemarmiesse)).

- `Set` now has named methods for set operations
  ([PR #2214](https://github.com/modularml/mojo/pull/2214) by
  [@arvindavoudi](https://github.com/arvindavoudi)):
  - `Set.difference()` mapping to `-`
  - `Set.difference_update()` mapping to `-=`
  - `Set.intersection_update()` mapping to `&=`
  - `Set.update()` mapping to `|=`

- `String` now has `removeprefix()` and `removesuffix()` methods.
  ([PR #2038](https://github.com/modularml/mojo/pull/2038) by
  [@gabrieldemarmiesse](https://github.com/gabrieldemarmiesse))

- `Optional` now implements `__is__` and `__isnot__` methods so that you can
  compare an `Optional` with `None`, e.g. `Optional(1) is not None` for example.
  ([PR #2082](https://github.com/modularml/mojo/pull/2082) by
  [@gabrieldemarmiesse](https://github.com/gabrieldemarmiesse))

- The `ord` and `chr` functions have been improved to accept any Unicode
  character. ([PR #2149](https://github.com/modularml/mojo/pull/2149) by
  [@mzaks](https://github.com/mzaks), contributes towards
  [#1616](https://github.com/modularml/mojo/issues/1616))

- `Atomic` is now movable.
  ([PR #2088](https://github.com/modularml/mojo/pull/2088) by
  [@StandinKP](https://github.com/StandinKP), fixes
  [#2026](https://github.com/modularml/mojo/issues/2026))

- `Dict` and `List` are both `Boolable` now.
  ([PR #2262](https://github.com/modularml/mojo/pull/2262) by
  [@gabrieldemarmiesse](https://github.com/gabrieldemarmiesse))

- `atol` now handles whitespaces so `int(String( " 10 "))` gives back `10`
  instead of raising an error.
  ([PR #2225](https://github.com/modularml/mojo/pull/2225) by
  [@artemiogr97](https://github.com/artemiogr97))

- `SIMD` now implements `__rmod__`.
  ([PR #2186](https://github.com/modularml/mojo/pull/2186) by
  [@bgreni](https://github.com/bgreni), fixes
  [#1482](https://github.com/modularml/mojo/issues/1482))

- `bool(None)` is now implemented.
  ([PR #2249](https://github.com/modularml/mojo/pull/2249) by
  [@zhoujingya](https://github.com/zhoujingya))

- The `DTypePointer` type now implements `gather` for gathering a `SIMD` vector
  from offsets of a current pointer. Similarly, support for `scatter` was added
  to scatter a `SIMD` vector into offsets of the current pointer.
  ([PR #2268](https://github.com/modularml/mojo/pull/2268) by
  [@leandrolcampos](https://github.com/leandrolcampos))

- The `len` function for unary `range` with negative end values has been fixed
  so that things like `len(range(-1))` work correctly now.
  ([PR #2204](https://github.com/modularml/mojo/pull/2204) by
  [@soraros](https://github.com/soraros))

- A low-level `__get_mvalue_as_litref(x)` builtin was added to give access to
  the underlying memory representation as a `!lit.ref` value without checking
  initialization status of the underlying value.  This is useful in very
  low-level logic but isn't designed for general usability and will likely
  change in the future.

- The `testing.assert_equal[SIMD]()` now raises if any of the elements mismatch
  in the two `SIMD` arguments being compared.
  ([PR #2279](https://github.com/modularml/mojo/pull/2279) by
  [@gabrieldemarmiesse](https://github.com/gabrieldemarmiesse))

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

- The `__getitem__`/`__getattr__` and related methods can now take indices as
  parameter values instead of argument values.  This is enabled when defining
  these as taking no arguments other than 'self' and the set value in a setter.
  This enables types that can only be subscript into with parameters, as well
  as things like:

  ```mojo
   struct RGB:
     fn __getattr__[name: StringLiteral](self) -> Int:
       @parameter
       if name == "r":   return ...
       elif name == "g": return ...
       else:
         constrained[name == "b", "can only access with r, g, or b members"]()
         return ...
    ```

- Added `reversed()` for creating reversed iterators. Several range types,
  `List`, and `Dict` now support iterating in reverse.
  ([PR #2215](https://github.com/modularml/mojo/pull/2215) by
  [@helehex](https://github.com/helehex),
  [PR #2327](https://github.com/modularml/mojo/pull/2327) by
  [@jayzhan211](https://github.com/jayzhan211), contributes towards
  [#2325](https://github.com/modularml/mojo/issues/2325))

- `object` now supports the division, modulo, and left and right shift
  operators. ([PR #2230](https://github.com/modularml/mojo/pull/2230),
  [PR #2247](https://github.com/modularml/mojo/pull/2247) by
  [@LJ-9801](https://github.com/LJ-9801), fixes
  [#2224](https://github.com/modularml/mojo/issues/2224))

  The following operator dunder methods were added:

  - `object.__mod__`
  - `object.__truediv__`
  - `object.__floordiv__`
  - `object.__lshift__`
  - `object.__rshift__`

  As well as the in-place and reverse variants.

- Added checked arithmetic operations.
  ([PR #2138](https://github.com/modularml/mojo/pull/2138) by
  [#lsh](https://github.com/lsh))

  SIMD integral types (including the sized integral scalars like `Int64`) can
  now perform checked additions, substractions, and multiplications using the
  following new methods:

  - `SIMD.add_with_overflow`
  - `SIMD.sub_with_overflow`
  - `SIMD.mul_with_overflow`

  Checked arithimetic allows the caller to determine if an operation exceeded
  the numeric limits of the type.

- Added `os.remove()` and `os.unlink()` for deleting files.
  ([PR #2310](https://github.com/modularml/mojo/pull/2310) by
  [@artemiogr97](https://github.com/artemiogr97), fixes
  [#2306](https://github.com/modularml/mojo/issues/2306))

- Properties can now be specified on inline mlir ops:

  ```mojo
  _ = __mlir_op.`kgen.source_loc`[
      _type = (
          __mlir_type.index, __mlir_type.index, __mlir_type.`!kgen.string`
      ),
      _properties = __mlir_attr.`{inlineCount = 1 : i64}`,
  ]()
  ```

  As the example shows above, the protected `_properties` attribute can be
  passed during op construction, with an MLIR `DictionaryAttr` value.

- Mojo now allows users to capture source location of code and call location of
  functions dynamically. For example:

  ```mojo
  @always_inline
  fn my_assert(cond: Bool, msg: String):
    if not cond:
      var call_loc = __call_location()
      print("In", call_loc.file_name, "on line", str(call_loc.line) + ":", msg)

  fn main():
    my_assert(False, "always fails")  # some_file.mojo, line 193
  ```

  will print `In /path/to/some_file.mojo on line 193: always fails`. Note that
  `__call_location` only works in `@always_inline("nodebug")` and
  `@always_inline` functions, as well as limitations on its use in parameter
  contexts (see the documentation for more details).

- `debug_assert` now prints its location (filename, line, and column where it
  was called) in its error message. Similarly, the `assert_*` helpers in the
  `testing` module now include location information in their messages.

- `int()` can now take a string and a specified base to parse an integer from a
  string: `int("ff", 16)` returns `255`.
  ([PR #2273](https://github.com/modularml/mojo/pull/2273) by
  [@artemiogr97](https://github.com/artemiogr97), fixes
  [#2274](https://github.com/modularml/mojo/issues/2274))

  Additionally, if a base of zero is specified, the string will be parsed as if
  it was an integer literal, with the base determined by whether the
  string contains the prefix `"0x"`, `"0o"`, or `"0b"`.

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
     `move_from_pointee`, `initialize_pointee_*` and `move_pointee` respectively.
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

- `Optional.value()` will now return a reference instead of a copy of the
  contained value. ([PR #2226](https://github.com/modularml/mojo/pull/2226) by
  [#lsh](https://github.com/lsh), fixes
  [#2179](https://github.com/modularml/mojo/issues/2179))

  To perform a copy manually, dereference the result:

  ```mojo
  var result = Optional(123)

  var value = result.value()[]
  ```

- Per the accepted community proposal
  [`proposals/byte-as-uint8.md`](https://github.com/modularml/mojo/blob/main/proposals/byte-as-uint8.md),
  began transition to using `UInt8` by changing the data pointer of `Error` to
  `DTypePointer[DType.uint8]`.
  ([PR #2318](https://github.com/modularml/mojo/pull/2318) by
  [@gabrieldemarmiesse](https://github.com/gabrieldemarmiesse), contributes
  towards [#2317](https://github.com/modularml/mojo/issues/2317))

- Continued transition to `UnsafePointer` away from the legacy `Pointer` type in
  various standard library APIs and internals.
  ([PR #2365](https://github.com/modularml/mojo/pull/2365),
  [PR #2367](https://github.com/modularml/mojo/pull/2367),
  [PR #2368](https://github.com/modularml/mojo/pull/2368),
  [PR #2370](https://github.com/modularml/mojo/pull/2370),
  [PR #2371](https://github.com/modularml/mojo/pull/2371) by
  [@gabrieldemarmiesse](https://github.com/gabrieldemarmiesse))

- `Bool` can now be implicitly converted from any type conforming to `Boolable`.
  This means that you no longer need to write things like this:

  ```mojo
  @value
  struct MyBoolable:
    fn __bool__(self) -> Bool: ...

  fn takes_boolable[T: Boolable](cond: T): ...

  takes_boolable(MyBoolable())
  ```

  Instead, you can simply have

  ```mojo
  fn takes_bool(cond: Bool): ...

  takes_bool(MyBoolable())
  ```

  Note that calls to `takes_bool` will perform the implicit conversion, so in
  some cases is it still better to explicitly declare type parameter, e.g.:

  ```mojo
  fn takes_two_boolables[T: Boolable](a: T, b: T):
    # Short circuit means `b.__bool__()` might not be evaluated.
    if a.__bool__() and b.__bool__():
      ...
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

- The method `object.print()` has been removed. Since now, `object` has the
  `Stringable` trait, you can use `print(my_object)` instead.

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

- [#2068](https://github.com/modularml/mojo/issues/2068) Fix simd.reduce for
  size_out == 2 ([PR #2102](https://github.com/modularml/mojo/pull/2102) by
  [@soraros](https://github.com/soraros))
