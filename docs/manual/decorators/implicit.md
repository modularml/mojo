---
title: '@implicit'
description: Marks a constructor as eligible for implicit conversion.
codeTitle: true

---

You can add the `@implicit` decorator on any single-argument constructor to
identify it as eligible for implicit conversion.

For example:

```mojo
struct MyInt:
    var value: Int

    @implicit
    fn __init__(out self, value: Int):
        self.value = value

    fn __init__(out self, value: Float64):
        self.value = int(value)


```

This implicit conversion constructor allows you to pass an `Int` to a function
that takes a `MyInt` argument, or assign an `Int` to a variable of type `MyInt`.
However, the constructor that takes a `Float64` value is **not** an implicit
conversion constructor, so it must be invoked explicitly:

```mojo
fn func(n: MyInt):
    print("MyInt value: ", n.value)

fn main():
    func(Int(42))             # Implicit conversion from Int: OK
    func(MyInt(Float64(4.2))) # Explicit conversion from Float64: OK
    func(Float64(4.2))        # Error: can't convert Float64 to MyInt
```
