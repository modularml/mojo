# Auto def-Reference for Mojo

Chris Lattner, May 6, 2024

**TL;DR:** Add an ‘eager decay’ model for references to improve developer
ergonomics and work more like C++ References.  Delete `__refitem__` as well.

# Motivation

The safe `Reference` type in Mojo is slowly making progress in terms of its
implementation and capabilities, but one of the major problems with it is
usability.  This whitepaper tackles one specific part of the problem, that you
have to explicitly dereference references.  

The lack of automatic dereferencing leads to confusing user problems, e.g. when
iterating over a `List`, the iterator is returning references, so you have to
write:

```mojo
for name in names:
    print(name[])
```

Yuck.🤮  Furthermore, other Mojo code end up being piles of punctuation:

```mojo
fn offset_momentum(inout bodies: List[Planet]):
    var p = SIMD[DType.float64, 4]()

    for body in bodies:
        # What is this?
        p += body[].velocity * body[].mass

    var body = bodies[0]
    body.velocity = -p / SOLAR_MASS

    bodies[0] = body
```

Worse yet, we’re using subscript to mean two different things - in the case of a
reference, this is an explicit “dereference” operation that we want people to
think of as transparent.  In the case of array (and pointers) we want an
explicit “index and subscript” operation, which is a “proper” use of subscript.

Ok, so let’s eliminate the need for `foo[]` when `foo` is a `Reference`!

# Background

One very important thing to get out of the way: this document is **only**
talking about the `Reference` type, this is not talking about “things with
reference semantics”, and is definitely not changing `PythonObject` or anything
related to Python interop or compatibility.  Python doesn’t have a notion like
`Reference`.

### Current API of `Reference`

For reference (haha, zing 🔥), the entire public API of `Reference` is currently:

```mojo
@value
struct Reference[...]:
    fn __init__(inout self, value: Self._mlir_type):
        """Constructs a Reference from the MLIR reference."""

    fn __refitem__(self) -> Self._mlir_type:
        """Enable subscript syntax `ref[]` to access the element."""

    fn __mlir_ref__(self) -> Self._mlir_type:
        """Enable the Mojo compiler to see into `Reference`."""
        return self.value
```

It took quite a lot of work to get here, but these three methods are the key to
all the magic:

1) The first is used by the compiler to form a `Reference` from an xValue.  

2) The second enables `foo[]` syntax

3) The third allows the compiler to implement `__refitem__`.

As you can tell, we don’t want typical users to interact directly with
`Reference`.  It is something you can declare as part of a type signature, but
you don’t want (and shouldn’t have to) interact with it directly.

For a point of intuition, compare `Reference` to a C++ reference like `int&`:
there is no API on the reference itself, it is just a part of the type system
that affects overloading and argument passing, but  is fully transparent to
method resolution etc.

### The difference between pointers and references

It might be surprising to consider these two cases differently: why is `myPtr[]`
a good thing but `myRef[]` is not?  Several reasons:

 1) users generally expect and want references to be transparent.

 2) pointers can be indexed with `myPtr[i]` : `myPtr[]` is just sugar for when
    `i` is zero.

 3) `Reference` is a safe type and “dereferencing” it is always safe and
    correct, whereas `UnsafePointer` (and friends) are unsafe and dereferencing
    should be explicit in source.

 4) Dereferencing a `Reference` doesn’t actually do anything - it converts an
    RValue of Reference type to an LValue or BValue in the compiler - it is only
    a load or store to the resultant xValue that does something!

### Not competing with `getattr`

Mojo already has the ability to overload `x.y` syntax on a type both dynamically
and statically through `__getattr__` and `__setattr__`.  This is a different
part of the design space here, because reference promotion needs to work even
when not accessing a member, e.g. in `var x = List(someRef)`.

# Two possible designs

There are two fundamental designs that we could take to make references
transparent, I call them the “*eager reference decay”* model and the
“*persistent reference*” model.  Both models solve the most important problems
and allow these examples to work:

```mojo
# Things common to both models.
fn show_commonality(a: Reference[String, _, _]):
    # Gets the length of the string, not the reference.
    var length = len(a)
  
    # call __add__ on the string, deref'ing the reference twice.
    var doubled = a+a
  
    # Methods look through the reference.
    var joined = a.join("x", "y")
  
    # r is Reference[String]: this copies the reference not string.
    var r : Reference[...] = a
  
    # This gives a List of references in both models.
    var reflist = List[Reference[...]](a, a, a)
```

Note the lack of any `a[]`’s in these examples!

I think we should use the former, but this section outlines both approaches
and makes the case.

## The “Eager Reference Decay” model

One approach is to follow C++’s approach and make it so `Reference` decays to an
LValue or BValue of the underlying type. This decay specifically happens when
the `Reference` is produced as an RValue (e.g. as the result of a function call)
and during LValue to RValue conversion.  This means that LValues of `Reference`
type are allowed, but RValue’s of `Reference` type will never be observable to a
Mojo programmer.

Let’s look at an example:

```mojo
fn show_differences(a: Reference[String, _, _]):
    # 'T' equals String, not Reference[String]
    alias T = __type_of(a)
   
    # 'v1' has type String, so this copies the string
    var v1 = a
   
    # List of strings, not a list of references of strings
    var list = List(a, a, a)
   
    # Slices the string, not a dereference!
    var strslice = a[]
```

The intuition here is the compiler internally has a way to reason about
“[LValue](https://en.wikipedia.org/wiki/Value_(computer_science)#lrvalue)”s, and
the formal type of a variable doesn’t include how it can be accessed: whether it
is mutable or just readable etc.  With this as a guide, it makes sense that
`Reference` immediately decays to this internal representation - the utility of
Reference is that it allows one to declare the lifetime and indirection as part
of a function signature or in a struct member in a safe way.  Beyond that, we
want it to go away.

This approach is proven to work (i.e. be teachable and usable) at scale because
it is used in the C++ programming language.  It also has a number of other
advantages, which we explore after introducing the second model.

## The “Persistent Reference” model

An alternative approach is to try to maintain the `Reference` in the type system
and eliminate it *only when necessary* to type check the program, e.g.
auto-dereference `[x.join](http://x.foo)` because `Reference` doesn’t have a
`join` member, but the underlying `String` does.  This model maintains the
reference as much as possible, which leads to  differences in the examples shown
above:

```mojo
fn show_differences(a: Reference[String, _, _]):
    # 'T' equals Reference[String], not String
    alias T = __type_of(a)
   
    # 'v1' has type Reference[String] so this doesn't copy the string
    var v1 = a
   
    # List of references, not a list of strings.
    var list = List(a, a, a)
   
    # Dereferences the reference(??): doesn't slice the string!
    var strslice = a[]
```

This model seemed initially appealing to me, and is similar to Rust’s approach.
After exploring it, I noticed that it has two major downsides:

1) Because references still exist as user-defined values, you need a way to
   dereference them, and this leads to confusion when referring to things that
   can be sliced or subscripted.

2) This violates the general design point of Mojo that references shouldn’t
   “infect” code that isn’t aware of them.

3) This approach is also significantly more complicated to implement and teach
   than the eager decay model.

The first issue is perhaps tolerable - we could introduce new syntax to
dereference a reference, or come up with some other way to get that out of the
way.  As we mentioned before, `Reference` intentionally has a minimal API, so
the probability of conflicts is low.

That said, **the second issue is a showstopper, and this is a major intentional
difference between Rust and Mojo**.  It is already very common for functions to
return references (e.g. this is why we have to solve the `[]` problem!), and we
don’t want lifetimes to invade types unexpectedly through type inference, e.g.
`var list = List(self.get_thing())` .

Mojo is a language with value semantics and strong copying and moving support,
aimed at the Python community.  We are embracing safe references and lifetimes,
but want to progress on the usability challenges that the Rust community is
still working through. The goal of Mojo is to be familiar and learnable by
Python programmers, and as such, we want management of references to be opt-in,
not implicit because you’re calling a function that happens to return one.

Furthermore, both models are equally expressible - they both permit references
in structures, propagating references around when needed, and other fancy
tricks.  The two models differ on the defaults observed and whether reference
propagation is opt-in or opt-out.

For all these reasons, I recommend going with the “Eager Reference Decay” model.

# Implementation approach

There are three steps to implementing this:

## Implement auto-deref itself

The implementation approach is straightforward - all expressions that yield a
`Reference` will automatically decay to an LValue or BValue when the expression
is formed.

We don’t want to hard code knowledge of `Reference` in the compiler, instead we
should key this off of the presence of the existing `__mlir_ref__` member.

## Remove `Reference.__refitem__`

Now that there are no persistent expressions of `Reference` type, we can further
narrow the interface by dropping the newly pointless refitem implementation.
You can never have a value of Reference type, so there is no way to apply `[]`
to it.

## Remove `__refitem__` / `__refattr__` from the language

Now that references automatically dereference, we don’t need `__refitem__` and
`__refattr__` anymore - types that use them can just implement `__getitem__` and
have it return a `Reference`.  This eliminates a pile of complexity and magic
from the compiler.

# Future directions

This approach is simple, but could enable extensions to future things that need implicit dereference abilities, e.g. existentials.
