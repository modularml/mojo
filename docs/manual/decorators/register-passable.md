---
title: '@register_passable'
description: Declares that a type should be passed in machine registers.
codeTitle: true

---

You can add the `@register_passable` decorator on a struct to tell Mojo that
the type should be passed in machine registers (such as a CPU register; subject
to the details of the underlying architecture). For tiny data types like an
integer or floating-point number, this is much more efficient than storing
values in stack memory. This means the type is always passed by value and
cannot be passed by reference.

The basic `@register_passable` decorator does not change the fundamental
behavior of a type: it still needs an `__init__()` and `__copyinit__()` method
to be copyable (and it may have a `__del__()` method, if necessary). For example:

```mojo
@register_passable
struct Pair:
    var a: Int
    var b: Int

    fn __init__(out self, one: Int, two: Int):
        self.a = one
        self.b = two

    fn __copyinit__(out self, existing: Self):
        self.a = existing.a
        self.b = existing.b

fn test_pair():
    var x = Pair(5, 10)
    var y = x

    print(y.a, y.b)
    y.a = 10
    y.b = 20
    print(y.a, y.b)
```

```mojo
test_pair()
```

```output
5 10
10 20
```

This behavior is what we expect from `Pair`, with or without the decorator.

You should be aware of a few other observable effects:

1. `@register_passable` types cannot hold instances of types
   that are not also `@register_passable`.

2. `@register_passable` types do not have a predictable identity,
   and so the `self` pointer is not stable/predictable (e.g. in hash tables).

3. `@register_passable` arguments and result are exposed to C and C++ directly,
   instead of being passed by-pointer.

4. `@register_passable` types cannot have a [`__moveinit__()`
   constructor](/mojo/manual/lifecycle/life#move-constructor), because
   values passed in a register cannot be passed by reference.

## `@register_passable("trivial")`

Most types that use `@register_passable` are just "bags of bits," which we call
"trivial" types. These trivial types are simple and should be copied, moved,
and destroyed without any custom constructors or a destructor. For these types,
you can add the `"trivial"` argument, and Mojo synthesizes all the lifecycle
methods as appropriate for a trivial register-passable type:

```mojo
@register_passable("trivial")
struct Pair:
    var a: Int
    var b: Int
```

This is similar to the [`@value`](/mojo/manual/decorators/value) decorator,
except when using `@register_passable("trivial")` the only lifecycle method
you're allowed to define is the `__init__()` constructor (but you don't have
to)—you *cannot* define any copy or move constructors or a destructor.

Examples of trivial types include:

- Arithmetic types such as `Int`, `Bool`, `Float64` etc.
- Pointers (the address value is trivial, not the data being pointed to).
- Arrays of other trivial types, including SIMD.

For more information about lifecycle methods (constructors and destructors)
see the section about [Value lifecycle](/mojo/manual/lifecycle/).

:::note TODO

This decorator is due for reconsideration. Lack of custom
copy/move/destroy logic and "passability in a register" are orthogonal concerns
and should be split. This former logic should be subsumed into a more general
decorator, which is orthogonal to `@register_passable`.

:::
