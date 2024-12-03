---
title: '@parameter'
description: Executes a function or if statement at compile time.
codeTitle: true

---

You can add the `@parameter` decorator on an `if` statement or on a nested
function to run that code at compile time.

## Parametric if statement

You can add `@parameter` to any `if` condition that's based on a valid
parameter expression (it's an expression that evaluates at compile time). This
ensures that only the live branch of the `if` statement is compiled into the
program, which can reduce your final binary size. For example:

```mojo
@parameter
if True:
    print("this will be included in the binary")
else:
    print("this will be eliminated at compile time")
```

```output
this will be included in the binary
```

## Parametric for statement

You can add the `@parameter` decorator to an `for` loop to create a loop that's
evaluated at compile time. The loop sequence and induction values must be
a valid parameter expressions (that is, an expressions that evaluate at compile
time).

This has the effect of "unrolling" the loop.

```mojo
fn parameter_for[max: Int]():
    @parameter
    for i in range(max)
        @parameter
        if i == 10:
            print("found 10!")
```

Currently, `@parameter for` requires the sequence's `__iter__` method to
return a `_StridedRangeIterator`, meaning the induction variables must be
`Int`. The intention is to lift these restrictions in the future.

### Compared to `unroll()`

The Mojo standard library also includes a function called
[`unroll()`](/mojo/stdlib/utils/loop/unroll) that unrolls a
given function that you want to call repeatedly, but has some important
differences when compared to the parametric `for` statement:

- The `@parameter` decorator operates on `for` loop expressions. The
  `unroll()` function is a higher-order function that takes a parametric closure
  (see below) and executes it a specified number of times.

- The parametric `for` statement is more versatile, since you can do anything
  you can do in a `for` statement: including using arbitrary sequences,
  early-exiting from the loop, skipping iterations with `continue` and so on.

  By contrast, `unroll()` simply takes a function and a count, and executes
  the function the specified number of times.

Both `unroll()` and `@parameter for` unroll at the beginning of compilation,
which might explode the size of the program that still needs to be compiled,
depending on the amount of code that's unrolled.

## Parametric closure

You can add `@parameter` on a nested function to create a “parametric”
capturing closure. This means you can create a closure function that captures
values from the outer scope (regardless of whether they are variables or
parameters), and then use that closure as a parameter. For example:

```mojo
fn use_closure[func: fn(Int) capturing [_] -> Int](num: Int) -> Int:
    return func(num)

fn create_closure():
    var x = 1

    @parameter
    fn add(i: Int) -> Int:
        return x + i

    var y = use_closure[add](2)
    print(y)

create_closure()
```

```output
3
```

Without the `@parameter` decorator, you'll get a compiler error that says you
"cannot use a dynamic value in call parameter"—referring to the
`use_closure[add](2)` call—because the `add()` closure would still be dynamic.

Note the `[_]` in the function type:

```mojo
fn use_closure[func: fn(Int) capturing [_] -> Int](num: Int) -> Int:
```

This origin specifier represents the set of origins for the values captured by
the parametric closure. This allows the compiler to correctly extend the
lifetimes of those values. For more information on lifetimes and origins, see
[Lifetimes, origins and references](/mojo/manual/values/lifetimes).
