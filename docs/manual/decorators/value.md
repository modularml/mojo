---
title: '@value'
description: Generates boilerplate lifecycle methods for a struct.
codeTitle: true

---

You can add the `@value` decorator on a struct to generate boilerplate
lifecycle methods, including the member-wise `__init__()` constructor,
`__copyinit__()` copy constructor, and `__moveinit__()` move constructor.

For example, consider a simple struct like this:

```mojo
@value
struct MyPet:
    var name: String
    var age: Int
```

Mojo sees the `@value` decorator and notices that you don't have any constructors
and it synthesizes them for you, the result being as if you had actually
written this:

```mojo
struct MyPet:
    var name: String
    var age: Int

    fn __init__(out self, owned name: String, age: Int):
        self.name = name^
        self.age = age

    fn __copyinit__(out self, existing: Self):
        self.name = existing.name
        self.age = existing.age

    fn __moveinit__(out self, owned existing: Self):
        self.name = existing.name^
        self.age = existing.age
```

Mojo synthesizes each lifecycle method only when it doesn't exist, so
you can use `@value` and still define your own versions to override the default
behavior. For example, it is fairly common to use the default member-wise and
move constructor, but create a custom copy constructor.

For more information about these lifecycle methods, read
[Life of a value](/mojo/manual/lifecycle/life).
