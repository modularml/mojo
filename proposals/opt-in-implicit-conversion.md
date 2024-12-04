# Opt-in Implicit Conversion

## Background

`Conversion`: Any mechanism which allows a value of type `S` to be treated as a
different type `T`, without manually calling some function `fn (S) -> T`.

For the purposes of this discussion I'm omitting conversions to interfaces or
supertypes, since these are conceptually polymorphism, ie. the function is
defined for type `S` but you need to pick the right function implementation
rather than convert to `T` (though this is not strictly true in C++).

Samples of conversion in a few languages:

C++

```cpp
struct Foo { Foo(int x) {} };
Foo foo = 10;

```

Rust

```rust
struct Foo;
impl From<i64> for Foo {
  fn from(i: i64) -> Foo { Foo(); }
}
let foo = Foo::from(10i64);
let foo2: Foo = (10i64).into();

```

Scala

```scala
implicit int2foo(int: Int): Foo = Foo
let foo: Foo = 10

```

Scala3

```scala
given Conversion[Foo, Int] = (_: Int) => Foo
let foo: Foo = 10

```

## Design Considerations

- Goal: Make implicit conversion constructors opt-in
- Non-goal: Design the full long-term conversion semantics for Mojo

## Detailed Design

### `@implicit_conversion` decorator for conversion constructors

- Constructors *not* decorated may not be used as conversions (this is an
    **opt-in** design)
- Since we know which conversion we’re inserting at a call-site at compile time,
    we can correctly mark it as `raises` or not. Implicit conversion constructors
    are therefore allowed to be `raises` .
- We continue with our same rules on 1-hop conversion and overload selection.

### Example

```python
struct Foo:
  @implicit_conversion
  fn __init__(out self, i: Int):
    pass

var foo: Foo = 10
```

## Alternatives Considered

Since this proposal is just about the specifics of conversion functions, the
design space looks like:

- Opt-in vs opt-out
  - Our existing experience is that we absolutely *do* want to be either opt-in
    or opt-out (there’s been pain with existing Stdlib constructors which
    shouldn’t be conversions).
  - Opt-in has the advantages of forcing explicitness in the call site as well
    as removing a vector of unexpected behavior from users coming from Python
    and defining plain constructors.
  - C++ is opt-out, and has developed a reputation for difficult to debug
    conversion semantics.
    [Google’s C++ style guide requires never using implicit conversion](https://google.github.io/styleguide/cppguide.html#Implicit_Conversions)
    and marking all such methods `explicit`.
- Constructor vs another method (say `__from__[T]`)
  - Constructor conversion has the advantage of already existing, and since
    Python is strongly typed there’s no real precedent here.
  - There doesn’t seem to be appetite for changing this from a constructor to
    another method.
- No implicit conversions
  - The Google C++ style guide [bans implicit conversions](https://google.github.io/styleguide/cppguide.html#Implicit_Conversions).
  - This would be a hard change for the current language. Literal types for
    example are only really usable through implicit conversion.
