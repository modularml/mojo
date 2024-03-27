
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

- The Mojo standard library is now open source! Check out the
  [README](https://github.com/modularml/mojo/blob/nightly/stdlib/README.md)
  for everything you need to get started.

- Structs and other nominal types are now allowed to implicitly conform to
  traits. A struct implicitly conforms to a trait if it implements all the
  requirements for the trait. For example, any struct that implements `__str__`
  will implicitly conform to `Stringable`, and then is usable with `str`.

  ```mojo
  @value
  struct Foo:
      fn __str__(self) -> String:
          return "foo!"

  fn main():
      print(str(Foo())) # prints 'foo!'
  ```

  Explicit conformance is still strongly encouraged where possible, because it
  is useful for documentation and for communicating intentions. In the future,
  explicit conformance will still be useful for features like default methods
  and extensions.

- Mojo's Python interoperability now supports passing keyword arguments to
  Python callables:

  ```mojo
  from python import Python

  def main():
      plt = Python.import_module("matplotlib.pyplot")
      plt.plot((5, 10), (10, 15), color="red")
      plt.show()
  ```

### ⭐️ New

- String types all conform to the
  [`IntableRaising`](/mojo/stdlib/builtin/int#intableraising) trait. This means
  that you can now call `int("123")` to get the integer `123`. If the integer
  cannot be parsed from the string, then an error is raised.

- The `Tensor` type now has an `argmax` and `argmin` function to compute the
  position of the max or min value.

- The `FloatLiteral` type is now an infinite precision nonmaterializable type.
  When materialized, `FloatLiteral` is converted to `Float64`.

- The [`List`](/mojo/stdlib/collections/list.html#list) type now supports
  construction from a variadic number of values.  For example,
  `List[Int](1, 2, 3)` works now.

- The `print` function now takes a `sep` and `end` as keyword. This means that
  one can write `print("Hello", "Mojo", sep="/", end="!!!\n")` to print the
  message `Hello/Mojo!!!\n` to the screen.

- The Mojo LSP server, via the new `-I` argument, now allows specifying
  additional search paths to use when resolving imported modules in a document.
  A corresponding `mojo.lsp.includeDirs` setting was added to the VS Code
  extension as well.

- Mojo now has support for variadic keyword argument, often referred to as
  `**kwargs`. This means you can now declare and call functions like this:

  ```mojo
  fn print_nicely(**kwargs: Int) raises:
    for key in kwargs.keys():
        print(key[], "=", kwargs[key[]])

   # prints:
   # `a = 7`
   # `y = 8`
  print_nicely(a=7, y=8)
  ```

  There are currently a few limitations:
  - The ownership semantics of variadic keyword arguments are always `owned`.
    This is applied implicitly, and cannot be declared otherwise:

    ```mojo
    # Not supported yet.
    fn borrowed_var_kwargs(borrowed **kwargs: Int): ...
    ```

  - Functions with variadic keyword arguments cannot have default values for
    keyword-only arguments, e.g.

    ```mojo
    # Not allowed yet, because `b` is keyword-only with a default.
    fn not_yet(*, b: Int = 9, **kwargs: Int): ...

    # Okay, because `c` is positional-or-keyword, so it can have a default.
    fn still_works(c: Int = 5, **kwargs: Int): ...
    ```

  - Dictionary unpacking is not supported yet:

    ```mojo
    fn takes_dict(d: Dict[String, Int]):
      print_nicely(**d)  # Not supported yet.
    ```

  - Variadic keyword parameters are not supported yet:

    ```mojo
    # Not supported yet.
    fn var_kwparams[**kwparams: Int](): ...
    ```

- Added new `collections.OptionalReg` type, a register-passable alternative
  to `Optional`.

  - Doc string code blocks can now `%#` to hide lines of code from documentation
    generation.

    For example:

    ```mojo
    var value = 5
    %# print(value)
    ```

    will generate documentation of the form:

    ```mojo
    var value = 5
    ```

    Hidden lines are processed as if they were normal code lines during test
    execution. This allows for writing additional code within a doc string
    example that is only used to ensure the example is runnable/testable.

  - Doc string code blocks can now `%#` to hide lines of code from documentation
    generation.

    For example:

    ```mojo
    var value = 5
    %# print(value)
    ```

    will generate documentation of the form:

    ```mojo
    var value = 5
    ```

    Hidden lines are processed as if they were normal code lines during test
    execution. This allows for writing additional code within a doc string
    example that is only used to ensure the example is runnable/testable.

### 🦋 Changed

- Mojo now warns about unused values in both `def` and `fn` declarations,
  instead of completely disabling the warning in `def`s.  It never warns about
  unused `object` or `PythonObject` values, tying the warning to these types
  instead of the kind of function they are unused in.  This will help catch API
  usage bugs in `def`s and make imported Python APIs more ergonomic in `fn`s.

- The [`DynamicVector`](/mojo/stdlib/collections/list#list) and
  [`InlinedFixedVector`](/mojo/stdlib/collections/vector.html#inlinedfixedvector)
  types now support negative indexing. This means that you can write `vec[-1]`
  which is equivalent to `vec[len(vec)-1]`.

- The [`isinf()`](/mojo/stdlib/math/math#isinf) and
  [`isfinite()`](/mojo/stdlib/math/math#isfinite) methods have been moved from
  `math.limits` to the `math` module.

- The `ulp` function has been added to the `math` module. This allows one to get
  the units of least precision (or units of last place) of a floating point
  value.

- `EqualityComparable` trait now requires `__ne__` function for conformance in addition
  to the previously existing `__eq__` function.

- Many types now declare conformance to `EqualityComparable` trait.

- `DynamicVector` has been renamed to `List`.  It has also moved from the `collections.vector`
  module to `collections.list` module.

- `StaticTuple` parameter order has changed to `StaticTuple[type, size]` for
  consistency with `SIMD` and similar collection types.

- The signature of the elementwise function has been changed. The new order is
  is `function`, `simd_width`, and then `rank`. As a result, the rank parameter
  can now be inferred and one can call elementwise via:

  ```mojo
  elementwise[func, simd_width](shape)
  ```

- For the time being, dynamic type value will be disabled in the language, e.g.
  the following will now fail with an error:

  ```mojo
  var t = Int  # dynamic type values not allowed

  struct SomeType: ...

  takes_type(SomeType)  # dynamic type values not allowed
  ```

  We want to take a step back and (re)design type valued variables,
  existentials, and other dynamic features for more 🔥. This does not affect
  type valued parameters, so the following will work as before:

  ```mojo
  alias t = Int  # still 🔥

  struct SomeType: ...

  takes_type[SomeType]()  # already 🔥
  ```

- `PythonObject` is now register-passable.

- `PythonObject.__iter__` now works correctly on more types of iterable Python
  objects.  Attempting to iterate over non-iterable objects will now raise an
  exception instead of behaving as if iterating over an empty sequence.
  `__iter__` also now borrows `self` rather than requiring `inout`, allowing
  code like `for value in my_dict.values():`.

- `List.push_back` has been removed.  Please use the `append` function instead.

- We took the opportunity to rehome some modules into their correct package
  as we were going through the process of open-sourcing the Mojo Standard
  Library.  Specifically, the following are some breaking changes worth
  calling out.  Please update your import statements accordingly.
  - `utils.list` has moved to `buffer.list`.
  - `rand` and `randn` functions in the `random` package that return a `Tensor`
     have moved to the `tensor` package.
  - `Buffer`, `NDBuffer`, and friends have moved from the `memory` package
     into a new `buffer` package.
  - The `trap` function has been renamed to `abort`.  It also has moved from the
    `debug` module to the `os` module.
  - `parallel_memcpy` has moved from the `memory` package into
     the `buffer` package.

- The `*_` expression in parameter expressions is now required to occur at the
  end of a positional parameter list, instead of being allowed in the middle.
  This is no longer supported: `SomeStruct[*_, 42]` but `SomeStruct[42, *_]` is
  still allowed. We narrowed this because we want to encourage type designers
  to get the order of parameters right, and want to extend `*_` to support
  keyword parameters as well in the future.

### ❌ Removed

- `let` declarations now produce a compile time error instead of a warning,
  our next step in [removing let
  declarations](https://github.com/modularml/mojo/blob/main/proposals/remove-let-decls.md).
  The compiler still recognizes the `let` keyword for now in order to produce
  a good error message, but that will be removed in subsequent releases.

- The `__get_address_as_lvalue` magic function has been removed.  You can now
  get an LValue from a `Pointer` or `Reference` by using the `ptr[]` operator.

- The type parameter for the `memcpy` function is now automatically inferred.
  This means that calls to `memcpy` of the form `memcpy[Dtype.xyz](...)` would
  no longer work and the user would have to change the code to `memcpy(...)`.

- `print_no_newline` has been removed.  Please use `print(end="")` instead.

- `memcpy` on `Buffer` has been removed in favor of just overloads for `Pointer`
  and `DTypePointer`.

- The functions `max_or_inf`, `min_or_neginf` have been removed from
  `math.limit` and just inlined into their use in SIMD.

### 🛠️ Fixed

- [#1362](https://github.com/modularml/mojo/issues/1362) - Parameter inference
  now recursively matches function types.
- [#951](https://github.com/modularml/mojo/issues/951) - Functions that were
  both `async` and `@always_inline` incorrectly errored.
- [#1858](https://github.com/modularml/mojo/issues/1858) - Trait with parametric
  methods regression.
- [#1892](https://github.com/modularml/mojo/issues/1892) - Forbid unsupported
  decorators on traits.
- [#1735](https://github.com/modularml/mojo/issues/1735) - Trait-typed values
  are incorrectly considered equal.
- [#1909](https://github.com/modularml/mojo/issues/1909) - Crash due to nested
  import in unreachable block.
- [#1921](https://github.com/modularml/mojo/issues/1921) - Parser crashes
  binding Reference to lvalue with subtype lifetime.
- [#1945](https://github.com/modularml/mojo/issues/1945) - `Optional[T].or_else()`
  should return `T` instead of `Optional[T]`.
- [#1940](https://github.com/modularml/mojo/issues/1940) - Constrain `math.copysign`
  to floating point or integral types.
- [#1838](https://github.com/modularml/mojo/issues/1838) - Variadic `print`
  does not work when specifying `end=""`
- [#1826](https://github.com/modularml/mojo/issues/1826) - The `SIMD.reduce` methods
  correctly handle edge cases where `size_out >= size`.
