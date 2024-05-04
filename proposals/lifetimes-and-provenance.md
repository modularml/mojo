# Provenance Tracking and Lifetimes in Mojo

As of mid-May 2023, Mojo has full support for ownership (including move
semantics, borrows and transfers, mutability, ASAP destruction of values, and
member synthesis). This provides more expressiveness than many languages, but does
not meet the expectations of Rust and C++ programmers because it is impossible
to **return references** and **put references in structs**.

This makes Mojo unable to express common patterns like `StringRef` in the LLVM
APIs because it is a struct containing a reference, and this makes our `Pointer`
type a massively unsafe API.

## Goals of this document

This document explores the first step in adding lifetimes to Mojo: what changes
we’ll have to introduce, some thinking on syntax we may want to use, and how
this may want us to reconsider existing design decisions.  This is written in
the style of the "[Value ownership design for Mojo](value-ownership.md)"
document from January.

This document is really just the “first step” of lifetimes.  Rust includes a few
more exotic features, including [subtyping constraints between
lifetimes](https://discourse.llvm.org/t/rfc-lifetime-annotations-for-c/61377#no-subtyping-constraints-between-lifetimes-15),
[equality constraints between lifetime
parameters](https://discourse.llvm.org/t/rfc-lifetime-annotations-for-c/61377#no-equality-constraints-between-lifetime-parameters-16),
[unbounded
lifetimes](https://doc.rust-lang.org/nomicon/unbounded-lifetimes.html) and
perhaps other features.  We don't have all the mechanics of a generic system and
trait system yet to tie into - and it makes sense to lazily add complexity based
on need - so these are not included in this initial design.

## Context

Mojo already has much of the required infrastructure in place to support
lifetimes: we now have references, we just need to be able to return them.
Similarly, the Mojo parameter system provides a powerful way to model and
propagate lifetime information around in our type system. We have a
CheckLifetime compiler pass which infers lifetimes, diagnosing use of
uninitialized values and inserting destructor calls.

Similarly, we can learn a lot from Rust’s design for lifetimes. That said, the
ownership system in Mojo is quite different than the one in Rust. In Rust,
scopes define the implicit lifetimes of values and references, and lifetimes are
used to verify that uses of the value happen when the value is still alive. Mojo
flips this on its head: values start their life when defined, but end their life
after their last use. This means that in Mojo, a lifetime reference **extends
the liveness of a value** for as long as derived references is used, which is a
bit different than Rust.

For example, we expect this to behave like the comments indicate:

```mojo
    var some_str = String("hello")

    # The StringRef contains a reference to the some_str value
    var some_str_ref = StringRef(some_str)

    # Last use of some_str, but don't destroy it because there is a use of
    # the reference below!
    use(some_str)

    # References must be propagatable through methods.
    some_str_ref = some_str_ref.drop_front().drop_back()
    print(some_str_ref)                                   # Prints "ell"
    # some_str destroyed here after last reference to it
```

The major thing that Mojo (and Rust) need from lifetimes is what is called
“local reasoning”: We need to be able to reason about the memory behavior of a
call just by looking at its signature. We cannot have to look into the body of a
function to understand its effects, and a function cannot know about its callers
to reason about the behavior of its arguments.  Similarly, when accessing a
member `a.ref` that is a reference, we need to know what lifetime is being used
in the context of `a`.

Because of this, the lifetimes in a function signature are something of a
"transfer function" that expresses mappings from input lifetimes to returned
lifetimes, and that allows reasoning about the lifetimes of field references.
Mojo already has a powerful parameter system that allows it to express these
sorts of relationships, so this all plugs together.

### What to name / how to explain this set of functionality?

Rust has a few things going on that are tightly bound and somewhat confusing: it
has scoping, lifetimes, and lifetime holes.  It has a borrow checker that
enforces the rules, all together this is its ownership system.

When it comes to the “lifetimes” part of the puzzle, it seems better to clarify
two very different concepts: on the one hand, stored **values** in memory each
have a conceptual “lifetime” that starts when the value is defined and ends at
the last use - this is where the destructor call is inserted.  Because each
declared variable has an independently tracked lifetime, each also needs an
implicitly declared “lifetime parameter” that is tracked in the Mojo type
system.

On the other hand, when reasoning about parametric functions, the type signature
defines a transfer function that expresses the “provenance” of the result values
from the function and how they relate to input values.  This relationship is a
transfer function from the input lifetime parameters to output lifetime
parameters, and can be somewhat more complicated.  The framing of “provenance
tracking” may be more general conceptually than “lifetime tracking” which seems
specific to the lifetime parameters.

## Mojo Reference + Lifetime Design

Lifetimes enable us to use references as first class types. This means they
can occur in nested positions: You can now have a reference to a reference, you
can have an array of references, etc.  At the machine/execution level, a
reference is identical to a pointer, but the reference type system enables an
associated lifetime, tracking of mutability etc.

### Writing a lifetime bound reference

The first question is how to spell this.  Rust uses the `'a` syntax which is
pretty unconventional and (weirdly but) distinctly Rust.  For example, here are
some simple Rust functions:

```rust
// This is Rust
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {..}
fn longest2<'a>(x: &'a mut str, y: &'a mut str) -> &'a mut str {..}
```

Mojo already have a general set of values in our generic signature list: we just
need to "parameterize" references and make them explicit.  For now, we recommend
using new `ref` and `mutref` keywords parameterized on a lifetime, and a new
`Lifetime` type.  For example:

```mojo
# Proposed Mojo syntax.
fn longest[a: Lifetime](x: ref[a] String,
                        y: ref[a] String) -> ref[a] String:
    return x if len(x) >= len(y) else y

fn longest2[a: Lifetime](x: mutref[a] String,
                         y: mutref[a] String) -> mutref[a] String: ...
```

There are many other options for syntax that we can consider, but this is simple
and will get us going until we have more experience.  For example, we can make
any of these work:

```mojo
fn f(x: ref[a] String, y: mutref[a] String): ... # Homage to Rust!
fn f(x: ref(a) String, y: mutref(a) String): ... # Homage to Rust!
fn f(x: ref a String,  y: mutref a String): ... # Homage to Rust!
fn f(x: ref 'a String, y: mutref 'a String): ... # Homage to Rust!
fn f(x: 'a String,     y: mut 'a String): ... # Homage to Rust!
```

The argument for square brackets vs parens is if we like the explanation that
we’re
"parameterizing the reference with a lifetime".  However, remember types can
also be parametric, and those are spelled with square brackets **after** the
type name, so parens may be better to make these more syntactically distinct.

The spelling in structs should flow naturally from this:

```mojo
struct StringRef[life: Lifetime]:
    var data : Pointer[UInt8, life]
    var len : Int
```

Being first class types, we will naturally allow local references: these should
also allow late initialization and explicitly declared lifetimes as well:

```mojo
fn example[life: Lifetime](cond: Bool,
                           x: ref[life] String,
                           y: ref[life] String):
    # Late initialized local borrow with explicit lifetime
    let str_ref: ref[life] String

    if cond:
        str_ref = x
    else:
        str_ref = y
    print(str_ref)
```

### Mojo Argument Conventions vs References

One non-obvious thing is that function argument conventions and references are
different things: keeping them different allows argument conventions to be a
convenient sugar that avoids most users from having to know about references and
lifetimes.  For example, the `borrowed` (which is usually implicit) and `inout`
argument conventions are sugar that avoids having to explicitly declare
lifetimes:

```mojo
// Mojo today
fn example(inout a: Int, borrowed b: Float32): …

struct S:
  fn method(inout self): …

// Written long-hand with explicit lifetimes.
fn example[a_life: Lifetime, b_life: Lifetime]
          (a: mutref[a_life] Int, b: ref[b_life] Float32): …
struct S:
  fn method[self_life: Lifetime](self: mutref[self_life]): …
```

This is very nice - every memory-only type passed into or returned from a
function must have a lifetime specified for it, but you really don't want to
have to deal with this in the normal case.  In the normal case, you can just
specify that you're taking something by borrow (the default anyway) with an
implicit lifetime, or taking it `inout` if you want to mutate it but don't care
about the lifetime.

NOTE: `inout` arguments also have one additional feature: function calls with
`inout` arguments know how to work with getter/setter pairs.  For example
something like `mutate(a[i])` will call the getter for the element before
calling the function, then call the setter afterward.  We cannot support this
for general mutable reference binding (because we wouldn't know where to perform
the writeback) so `inout` will have a bit more magic than just providing an
implicit lifetime.

NOTE: Internally to the compiler, references (which are always pointer sized)
are treated as a register-passable types.  This means they compose correctly
with implicit borrow semantics, and you can even pass a reference `inout` if you
want to.  Such a thing is a "mutable reference to a reference", allowing the
callee to mutate the callers reference.

### Keyword (?) for static lifetime

I think we can have a useful feature set without requiring the ability to
specify a static lifetime - the uses in Rust appear to be more about
constraining input lifetimes than it is about the core propagation of lifetimes,
that said, we can definitely dream up a spelling when it is needed.

Similarly, unsafe pointer tricks (e.g. when working with C) may want to use the
static lifetime marker to disable all tracking.  We can start with a stub like
`__static_lifetime` and then re-syntax it later.

### Implicitly declared lifetime parameter names and other sugar

One common use of named lifetime parameters is to tie the lifetime of the result
of a function back to the lifetime of one of the function arguments.  One
refinement over Rust we could permit is for arguments to implicitly declare
lifetimes on their first use.  For example, we don’t need to require a
declaration of `life` in this example:

```mojo
fn longest(x: ref[life] String,
           y: ref[life] String) -> ref[life] String:

# Alternatively follow Rust's lead.
fn longest(x: 'life String, y: 'life String) -> 'life String:
```

Let's **NOT** add this in the near future.  Explicit lifetimes will be much less
common in Mojo than they are in Rust, and it is better for learnability to not
have magic like this.

### Lifetime of `Self`

The `Self` keyword (upper case) produces an elaborated type name for the current
struct, but that does not include the lifetime of `self` (lower case) which is
generally a reference. In a method you can name the lifetime of `self` by
declaring it explicitly like this:

```mojo
    struct MyExample:
        fn method[self_life: Lifetime](self: mutref[self_life] Self)
                -> Pointer[Int, self_life]:
            ...

    fn callMethod(x: mutref[life1] MyExample):
        use(x.method())

        var y = MyExample()
        use(y.method())
```

`self_life` will bind to the lifetime of whatever lvalue the method is called
on, which is the `life1` lifetime in the first example, and the implicit
lifetime of y in the second example.  This all composes nicely.

One problem though - this won’t work for var definitions inside the struct,
because they don’t have a self available to them, and may need to reason about
it:

```mojo
    struct IntArray:
        var ptr : Pointer[Int, Self_lifetime]
```

It isn’t clear to me how the compiler will remap this though.  We’d have to pass
in the pointer/reference instead of the struct type.  An alternative is to not
allow expressing this and require casts.  We can start with that model and
explore adding this as the basic design comes up.

### Extended `getitem`/`getattr` Model

Once we have references, we’ll want to add support for them in the property
reference and subscripting logic.  For example, many types store their enclosed
values in memory: instead of having `Pointer` and `Array` types implement both
`__getitem__` and `__setitem__` (therefore being a "computed LValue") we'd much
rather them to expose a reference to the value already in memory (therefore
being more efficient).  We can do this by allowing:

```mojo
    struct Pointer[type: AnyType, life: Lifetime]:
        # This __getref__ returns a reference, so no setitem needed.
        fn __getref__(self, offset: Int) -> mutref[life] type:
            return __get_address_as_lvalue[life](...)
```

We will also need to extend the magic `__get_address_as_lvalue` style functions
to take a lifetime.

## Examples using Lifetimes

This section attempts to build a few example data structures that are important
to express with lifetimes.  They obviously haven’t been tested.

### Pointer / UnsafePointer / Reference

This is the bottom of the stack and needs to interface with other unsafe
features.  Suggested model is to make Pointer be parameterized on the lifetime
that it needs to work with as well as element type:

```mojo
    @value
    @register_passable("trivial")
    struct MutablePointer[type: AnyType, life: Lifetime]:
        alias pointer_type = __mlir_type[...]
        var address: pointer_type

           fn __init__(inout self): ...
        fn __init__(inout self, address: pointer_type): ...

        # Should this be an __init__ to allow implicit conversions?
        @static_method
        fn address_of(arg: mutref[life] type) -> Self:
            ...

        fn __getitem__(self, offset: Int) -> inout[life] type:
               ...

        @staticmethod
        fn alloc(count: Int) -> Self: ...
        fn free(self): ...

    fn exercise_pointer():
        # Allocated untracked data with static/immortal lifetime.
        let ptr = MutablePointer[Int, __static_lifetime].alloc(42)

        # Use extended getitem through reference to support setter.
        ptr[4] = 7

        var localInt = 19
        let ptr2 = MutablePointer.address_of(localInt)
        ptr2[0] += 1  # increment localInt

        # ERROR: Cannot mutate localInt while ptr2 lifetime is live
        localInt += 1
        use(ptr2)
```

It’s not clear to me if we need to have a split between `Pointer` and
`MutablePointer` like Swift does.  It will depend on details of how the
CheckLifetimes pass works - I’m hoping/expecting that the borrow checker will
allow mutable references to overlap with other references iff that reference is
only loaded and not mutated. NOTE: We probably won't be able to do this with the
proposed model, because generics can't be variant over mutability of the
reference.

Another aspect of the model we should consider is whether we should have an
`UnsafePointer` that allows unchecked address arithmetic, but have a safe
`Reference` type that just allows dereferencing.  This `Reference` type would be
completely safe when constructed from language references, which is pretty cool.
We may also want to wire up the prefix star operator into a dunder method.

### ArraySlice

`ArraySlice` (aka `ArrayRef` in LLVM) should compose on top of this:

```mojo
    @value
    @register_passable("trivial")
    struct MutableArraySlice[type: AnyType, life: Lifetime]:
        var ptr: MutablePointer[type, life]
        var size: Int

        fn __init__(inout self):
        fn __init__(inout self, ptr: MutablePointer[type, life], size: Int):

        # All the normal slicing operations etc, with bounds checks.
        fn __getitem__(self, offset: Int) -> inout[life] type:
            assert(offset < size)
            return ptr[offset]
```

As with `UnsafePointer`, this has to be parameterized based on the underlying
element type.  `ArraySlice` is just a bound checked pointer, but because of
lifetimes, it is safe once constructed: the references it produces are bound to
the lifetime specified so can’t dangle.

### Array / ValueSemanticArray

Given these low level types, we can start to build higher level abstractions.
One example of that is an `Array` type.  I’d suggest that our default array type
be value semantic with lazy copy-on-write 🐄, but a simpler example can be
implemented with `std::vector` style eager copying:

```mojo
    # Doesn't require a lifetime param because it owns its data.
    struct Array[type: AnyType]:
        var ptr: MutablePointer[type, Self_lifetime]
        var size: Int
        var capacity: Int

        fn __getitem__[life: Lifetime](self: inout[life], start: Int,
                            stop: Int) -> MutableArraySlice[type, life]:
            return MutableArraySlice(ptr, size)
```

By tying the lifetime of the produced slice to the lifetime of the Array `self`,
the borrow checker will prevent use/mutation of the `Array` itself while a
mutable slice is produced.
