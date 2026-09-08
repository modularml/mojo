# Mojo: Mutable `def` arguments - removed

June 2025. Status: Implemented in Mojo 25.4.

Note: This document describes historical rationale for an old behavior where
Mojo allowed `def` arguments to be directly mutated, the original rationale for
that, and the problems this introduces. This behavior has since been removed,
simplifying the language and eliminating bugs.

## Introduction

`read` arguments in a `def` live a dual life: on the one hand we want them to
work like a normal `read` argument (passing in an reference for efficiency) on
the other hand, they are mutable values for compatibility with Python. Why is
this? Python allows code that mutates arguments:

```python
def foo(a):
    print(a)
    a = 0  # Note that this affects 'a' in this function, not the caller
    print(a)
```

We retains the goal of Mojo growing into a superset of Python over time, so we’d
like to preserve this behavior for when people are migrating code to Mojo. Mojo
requires type annotations (for now, we can consider relaxing that in the
future), so the equivalent Mojo code is:

```python
def foo(a: PythonObject):
    print(a)
    a = 0  # Ok!
    print(a)
```

This works today, but the implementation details are pretty ugly and surprising,
and the model cannot work in generality.

This doc proposes eliminating the known-broken model we have today and making it
the future-Python-translator’s problem. The goal is to delete a lot of compiler
bugs, making parameter inference simpler, and making Mojo more predictable and
consistent.

## How the existing design attempts to work

The current design implements `def` arguments of any type with the
`DefArgumentWrapperDLValue` class. This class attempts to provide special
behavior: if the argument is never mutated, it doesn’t generate a copy. If the
argument is mutated, a copy into a local temporary is created on entry to the
function, and all uses are (supposed to) use the local copy instead. Consider:

```python
def example(a: String, b: List[Int], c: List[Int]):
   print(a)

   b.append(4)
   print(b)

   ref r = c[0]
   c[0] += 1
```

This ends up working like this:

```python
**def** example(a: String, b: List[Int], c: List[Int]):
   var b2 = b

   print(a)

   b2.append(4)
   print(b2)

   ref r = c[i]
   c[0] += 1    # error: expression must be mutable
```

The compiler has a set of heuristics about when to force the copy. In the case
of `ref` binding, it binds to an immutable reference, so some things don’t work
right, like the `+=`

## Some of the many problems

As you can start to see above, there are problems with this. Here are a few
more problems:

### Origin inconsistency

Consider something like this:

```python
def origin_broken(x: String) -> ref[x] String:
    mutate(x)
    return x
```

This can never work: when type checking the function signature, this gets the
origin of the input argument (that’s all the result type contains) but when type
checking the body, we see that we need a mutation, so we end up getting
something like this, which breaks:

```python
def origin_broken(x: String) -> ref[x] String:
    var x2 = x
    mutate(x2)
    return x2 # Origin mismatch, with horrible error
```

### Performance Risks

There are lots of reasons people might want to use `def` functions even in
systems code, and we have types that are copyable with non-trivial cost, for
example `List`. We want to improve this over time, but it is concerning that we
allow this to compile:

```python
def perf(list: List[Int]):
    # Implicit and invisible copy of a list made!
    list.append(4)
```

This is inconsistent with Mojo’s goals of driving performance and making the
programming experience predictable.

### Compiler complexity

There are a bunch of hooks used in parameter inference, origin analysis and
other places that attempt to hold this mess together. Given the model above
cannot actually work, they are holding together a specific set of unit tests in
a weird way that doesn’t compose. It would be great to shed this tech debt so
we can move faster.

## What are our actual goals here?

The alternate approach is to go back to our original goals and principles. We
want:

1. To provide a migration path for Python code and make mechanical translation
   from Python to Mojo work.
2. We want something simple and predictable and explainable
3. We want to break down differences between `def` and `fn` .
4. We are ok with Mojo being different than Python, particularly in corner
   cases, if we think it is better.
5. We don’t want the body of a function to affect its signature, like this
   approach did.
6. We also aren’t focused on “superset of Python” in the foreseeable future!

What does “mechanical translation” mean? We haven’t formally defined this, but
I think it is something like a Python script (e.g. black) that reads in the
.python files and writes out .mojo files. For example if you use the `fn`
identifier in your Python file, we might turn it into `_fn` to avoid the Mojo
keyword.

## Alternate approach

The alternate approach is simple: just remove all this code and complexity from
Mojo. Let all `def` arguments be just like `fn` `read` arguments. But wait,
how do we provide compatibility with Python? Simple, we have the translator
turn all arguments into `owned` arguments unless it can prove they are not
mutated in the function body:

```python
# Python
def foo(a):
  print(a)
  a = 0       # Note that this affects 'a' in this function, not the caller
  print(a)

# Generated Mojo
def foo(owned a: PythonObject):
  print(a)
  a = 0       # Ok!
  print(a)
```

The `owned` argument convention gives the right semantics here: it forces a copy
of the `PythonObject` (just the pointer itself, bumping a reference count) it
does not do a deep copy. This provides exactly Python semantics. This makes
the behavior simple and explicit in the Mojo source code, and gives the
programmer pulling code form Python into Mojo the chance to understand what is
happening and tune/tweak the code as they see fit.

This also has the advantage of allowing us to delete a bunch of stuff from the
Mojo compiler, allowing us to move faster.
