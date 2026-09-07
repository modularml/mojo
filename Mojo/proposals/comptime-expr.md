# `comptime` Expression Syntax

**Status**: Implemented.

## Introduction

Mojo has a hard divide between compile time and runtime evaluation. This exists
for a number of reasons, including that function calls can have side effects
(e.g. print something) and the side effect may happen at comptime and runtime.

Mojo decides whether something is comptime based on context, for example, type
positions and parameters are evaluated at comptime, but local statements default
to being runtime:

```mojo
def example1(arg: Int):     # types like Int are comptime expressions
   someFunction[foo()+5]() # parameter expressions are comptime
   comptime c = foo()+5    # comptime assignments are comptime

   otherFunction(foo()+5)  # arguments are runtime
   var abc = foo()+5       # Normal expressions are runtime.
```

This all works predictably and has served us well, but some have pointed out
that there are times where you might want to evaluate a subexpression of a
dynamic value at comptime, one example is (assuming `Layout` becomes
non-implicitly-materializable in the near future, because it can malloc):

```mojo
def example2[layout: Layout]():
   # Error, materializes "layout" to runtime, not its size.
   use(layout.size())

   # This is ok, get the size at comptime
   comptime layout_size = layout.size()
   use(layout_size) # materialize the Int, not the layout
```

This works, but is awkward.

## Proposal

With the rename of `alias` to `comptime`, we now have a more general lexical
concept to play with. We can introduce the notion of “comptime expressions”,
allowing an subexpression to be forced to evaluate at comptime. This would
allow writing the code above as:

```mojo
def example3[layout: Layout]():
   # Ok, evaluate .size() at comptime and materialize the resulting Int.
   use(comptime(layout.size()), layout.size())
```

Such a statement subsumes and generalize the existing "comptime assignment"
support, and aligns with a likely future direction to rename `@parameter if` to
`comptime if`.

### Detailed design

The implementation of this involves parsing `comptime(x)` as a primary
expression, in addition to a statement modifier.

## Polish

- We should reject `comptime` in an already-comptime expression.

- When emitting a function call like `foo(x, y, z)` when in a runtime context,
  if all of the arguments are comptime-PValues and any argument
  is , we should emit a nice error + fixit hint to rewrite to
  `comptime foo(x, y, z)`

- When combined with the fixes to remove `ImplicitlyCopyable` from GPU types
  that malloc, we should get great QoI on the externally reported issue.

## Alternatives to consider

We could use a standard-library-only declaration like this:

```mojo
comptime comptime_eval[T: AnyType, //, val: T] = val
```

which would allow writing:

```mojo
def example4[layout: Layout]():
   # Ok, evaluate .size() at comptime and materialize the Int.
   use(comptime_eval[layout.size()])
```

This is nice because it doesn’t require any language changes or support (it
works today!) but the flip side of that is that we can’t build the same tooling
support in and turns into a bit of punctuation soup.

## Future Directions

We're likely to re-syntax `@parameter if` to `comptime if`, which this aligns
with.

We can also consider removing the parentheses around the subexpressions to align
with the Python grammar for `not` and `yield` and other expressions, but should
consider `type_of` and `origin_of` at the same time.
