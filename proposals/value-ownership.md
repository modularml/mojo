
# Value ownership design for Mojo

**Written**: Jan 2, 2023

**Status**: Implemented but the design has been refined, kept for historical
*interest.

This document explores a design for ownership support in Mojo.  This learns from
other contemporary languages including C++, Rust, Swift and Val, providing a
novel blend that should integrate well with our base Python syntax, and be
powerful enough to express a wide range of kernel and systems programming
applications.

Rust is the language that most people will naturally think of in this space, and
when compared to it, I expect that we will provide support for a wider range of
types than Rust, yet provide a more familiar programming model and friendly API
than it.  While I assume that we will extend this to support lifetime types for
better generality and safety with reference semantic types, that design is not
included in this proposal.

# Motivation and Background

Modern systems languages aspire to provide memory safety, good low-level
performance, and allow advanced library developers to build expressive
high-level APIs that are easy to use by less experienced API users.

In the case of Mojo, we have two specific “now” problems to solve:

1. We need to provide a way to allocate memory for a Tensor-like type, and given
our existing support for raising exceptions, we need for them to be cleaned up.
Thus we need destructors.  We also need to disable copying of this sort of type.

2. We also need to implement transparent interop with CPython with an “object”
struct.  This requires us to have copy constructors and destructors so we can
maintain the CPython reference count with an ergonomic Python-like model.

Over time, we want Mojo to be a full replacement for the C/C++ system
programming use cases in Python, unifying the “two world problem” that Python
has with C.  This is important because we want to have a unifying technology,
and because CPUs will always be an important accelerator (and are fully
general), and because our bet is that accelerators will get more and more
programmable over time.

## Related Work

I am not going to summarize the related work fully here, but I recommend folks
interested in this topic to read up on relevant work in the industry, including:

1. C++’11 r-value references, move semantics, and its general modern programming
model.  It is yucky and has lots of problems, but is table-stakes knowledge and
powers a tremendous amount of the industry.

2. Have a programmer-level understanding of [Rust’s Memory Ownership
model](https://doc.rust-lang.org/book/ch04-00-understanding-ownership.html), and
read [the Rustonomicon](https://doc.rust-lang.org/nomicon/) end-to-end for bonus
points.

3. Swift has a quite different approach which made some mistakes (Swift suffers
from pervasive implicit copying) but has nice things in its [initialization
design](https://docs.swift.org/swift-book/documentation/the-swift-programming-language/initialization),
[ARC
design](https://docs.swift.org/swift-book/LanguageGuide/AutomaticReferenceCounting.html),
[exclusivity enforcement](https://www.swift.org/blog/swift-5-exclusivity/)
[[more
details](https://github.com/apple/swift-evolution/blob/main/proposals/0176-enforce-exclusive-access-to-memory.md)],
etc.

4. The [Val programming language](https://www.val-lang.dev/) is an early phase
research system that is learning from Swift and trying to provide a much simpler
programming model than Rust does with other advantages.  It isn’t at all clear
if it will be expressive enough to be useful at this point though.

C++ and Rust are the most widely known in this space and they make different
tradeoffs in defaults and what it means for a type author (ignoring lifetimes,
which C++ lacks and is therefore generally unsafe). A few random observations
that are related to the commentary below:

- Rust in particular defaults to move’ing values but allows types to opt out of
that by implementing the Copy trait.

- Rust’s type system doesn’t appear to support values like `std::atomic` which
require a pinned address in memory (Rust assumes it can transport things around
at will), nor does it support things like `llvm::SmallVector` which is movable,
but has an interior pointer so needs custom move constructors.

- Because Rust defaults to moving everything around, it puts a huge amount of
pressure on memcpy and LLVM memcpy optimization (e.g. see Patrick Walton’s
recent work to improve this).  In my opinion, this is a self imposed mistake
that we can correct structurally.

- Rust could support C++-like moves from values that leave them
inert-and-to-be-destroyed, but does not do that.  For lack of this, there is a
lot of complexity and unsafe APIs required (e.g.
[Drain](https://doc.rust-lang.org/stable/std/vec/struct.Vec.html#method.drain)
and other things) when you want to construct an array progressively or take
elements out of an array.

## Relevant Existing Mojo Type System Features

The Mojo design currently has a few basic features that are precursors to proper
ownership.  The design below makes significant extensions and some changes to
it.

### L-Values and R-Values

Lvalues and Rvalues follow a conventional design found in many programming
languages: an Rvalue is an immutable value that can be passed around by copying
its bits by-value in an SSA register.  An  Lvalue is mutable, represented by its
address, and can be promoted to an Rvalue with a “load”.

### Argument Conventions

Functions can be declared to take any of their arguments by-reference, with an &
sigil in the function definition:

```mojo
    fn globalFn(a: Int, b&: Int):
      b = a  # ok
      a = 4  # error

    struct Vec:
     ...
     fn push_back(self&, item: Int): ...  # mutable self
     fn size(self): ...                   # immutable self

    fn workWithVecs(a: Vec, b&: Vec):
     use(a.size()) # ok
     use(b.size()) # ok

     a.push_back(4) # Error, a isn't mutable
     b.push_back(4) # ok
```

As shown, by-ref arguments are Lvalues and thus may be mutated, otherwise
arguments are passed by-value as immutable Rvalues.  There is another
similar-but-different thing going on with “def” functions.  By-value arguments
are passed by copy but are allowed to be mutable for better compatibility with
Python semantics:

```mojo
    def exampleDef(a: Int, b&: Int):
      b = 4 # mutable as before.

      # passed by value into mutable copy.
      # This change isn't visible in the caller.
      a = 4
```

The ‘def’ semantics are implemented by taking a value in by copy and introducing
a local ‘var’ that shadows the argument.  It is therefore a purely local
transformation that we’ll ignore in the rest of this document.

### Type initializers

Mojo supports Python type initializers and currently allows them to participate
in implicit conversions:

```mojo
    struct Int:
      var value : __mlir_type.index

    fn __init__(value: __mlir_type.index) -> Int:
      return Int { value: value }
```

# Proposed Mojo Additions / Changes

This proposal introduces a number of new concepts to cover the space of value
ownership and parameter passing conventions that Rust, Swift and C++ provide,
but in a different way.  We layer on the features in individual groups.  Let’s
explore them one piece at a time.

## Extended parameter conventions

We should extend the existing type system to support owning (aka moving, aka
consuming) parameter convention, immutable borrow semantics (“`&T"` in Rust,
“`const T&"` in C++) and mutable borrow semantics (“`&mut T`” in Rust, and
“`T&`” in C++).

### Change & to mean mutable borrow or “inout”

Right now, & is a C++-like reference semantic sigil, we should keep it, but
change it to mean a “mutable borrow” in Rust terminology or “inout” using Swift
terminology.  This won’t require Mojo source changes, but will be a terminology
change inside of the compiler.  The definitive initialization pass (below) will
need to enforce its correct semantics.

Tangential to this proposal, we could require the use of `&` on the caller side
of arguments passing a mutable borrow to make it more clear in the caller code.
If we do this, I propose that it be a postfix operator, and use it for methods
for consistency:

```mojo
  swap(x.first&, y.first&)      # Obvious what is being mutated
  b&.push_back(42)              # Obvious that b is mutated.
```

This proposal will not use this syntax below, and this is probably way too
onerous for methods, this is probably a bad idea.

### Introduce “owned” argument conventions

We need a way to specify that ownership of a value is being passed into the
function:

```mojo
  # This takes ownership of a unique vector, including its resources.
  fn someFunction(owned v: Vec):
      print(v)
      v.push_back(42)  # Ok: v is mutable because we own it
```

In the code above, we show “owned v” takes and owns a value from the caller.
Just like normal arguments, we would need a copy to get them as mutable:

```mojo
  fn anotherFunction(a: Int, owned v: Vec):
      a += 1           # Error: a is immutable
      var a2 = a       # Copy the argument to make it mutable
      a2 += 1          # Ok, a2 is mutable.

      v.push_back(a)   # Ok, owned values are mutable
      var v2 = v^      # Transfer v argument into a new var binding.
      v2.push_back(a)  # Also ok
```

This should be an owned _reference_ and thus lower to LLVM pointers, just like a
mutable reference does, unless the value is `@register_passable` (see below).
Not doing this would significantly impact our ability to model things like
`std::atomic` and `SmallVector` whose address is significant.  See the
“extensions” at the bottom for why this will be efficient for trivial types like
integers.

### Introduce “borrowed” argument conventions and change default

Similarly we need the ability to specify that we are passing and returning an
immutable borrow, which is like a `'const &`’ in C++.  The spelling of this
isn’t particularly important because it will frequently be defaulted, but we
need something concrete to explain the examples.

```mojo
  # This takes a borrow and return it.
  # Returned references need lifetimes for safety of course.
  fn printSizeAndReturn(borrowed v: Vec) -> borrowed Vec
      print(v.size())
      return v
```

At the LLVM level, immutable references are passed by pointer, just like mutable
references.

Note that C++ and Rust entered into a strange battle with programmers about how
to pass values by default.  Templates generally use “`const&`” to avoid copies,
but this is an inefficient way to pass trivial types like “`int`”.  We propose a
type level annotation to directly address, which allows us to use borrow far
more pervasively for arguments in the language.  See the ‘extensions’ section
below.

### Change the default argument and result conventions

Now we have the ability to express ownership clearly, but we don’t want all code
everywhere to have words like `borrowed` on it: we want more progressive
disclosure of complexity and better defaults.

The first piece of this is the default return value convention.  The right
default convention for return values is to be owned:

```mojo
    # Result defaults to being an owned reference.
    fn getVec() -> Vec:   # Equivalent to "...) -> Vec^"
        ...
```

because we otherwise have no way to return newly created values. Code can
override the return convention by using another sigil, e.g. `inout Vec` if a
mutable reference is required.

We also need to decide what to do with arguments.  We don’t want to copy
arguments by default (a mistake Swift made), because this requires types to be
copyable, and depends on unpredictable copy elision that makes performance and
COW optimization sad. I don’t think we want to depend on pass-by-move like Rust
did because Rust forces tons of things to be marked “&” pervasively, this
introduces a ton of `memcpy` operations, and Python programmers won’t think that
passing a value to a function makes it unavailable for use by other things by
default.

In my opinion, the C++ got this almost right: `const&` is the right default
argument convention (and is optimized to register value passing in an opt-in
way, described below).  This is good for  both self and value arguments and is
what Swift semantically did with its +0 ownership convention in a bunch of
places.  Consider this example:

```mojo
    struct Dictionary:
      fn size(self) -> Int: ...
```

You don’t want to **copy the dictionary**, when calling size!  This worked for
Swift because of ARC optimizations and that its Dictionary type was a small
thing implemented with COW optimizations, but this is very unpredictable.

Passing arguments-by-borrow by default is also great because it eliminates the
pervasive need to do a load operation when converting Lvalues to Rvalues.  This
is a huge improvement in the model, because “loading” a value is extremely
expensive when the value is large or cannot be loaded (e.g. variable sized
types, e.g. some languages representation for existential types and Rust’s DSTs,
which we may or may not want to support anyway).

It also means that we can support types like `std::atomic` which need a
guaranteed ‘self’ address - natively and with no trouble - since we’re never
trying to implicitly load the value as a whole.

## Adding support for value destruction

Now that we have a notion of ownership, we can complete it by destroying values
when their lifetime has ended.  This requires introducing the ability to declare
a destructor, and the machinery to determine when to invoke the destructor.

### User defined destructors

We should embrace the existing Python convention of implementing the `__del__`
method on a type.  This takes ownership of the self value, so it should be
defined as taking `owned self`.  Here’s a reasonable implementation of Vec’s
ctors and dtor, but without the push_back and associated methods (which are
obvious):

```mojo
    struct Vec:
      var data: Pointer<Int>   # I just made this type up.
      var capacity: Int
      var size: Int
      fn __init__(inout self, capacity: Int):
         self.data = Pointer<Int>.malloc(capacity)
         self.capacity = capacity
         self.size = 0

      # default args will be nice some day
      fn __new__(inout self): return Vec(1)
      fn __del__(owned self):  # owning reference to self.
        # Any int values don't need to be destroyed.
        self.data.free()
```

There is some nuance here and a special case that we need to handle in the
`__del__` method.  Ideally, we should track the field sensitive liveness of the
‘self’ member that comes into del.  This will allow us to handle sub-elements
that are individually consumed, safely handle exceptions that early-exit from
the destructor etc.  This is something that Swift gets right that Rust
apparently doesn’t.

With respect to the simple definition of Vec above, it is enough to define a
safe vector of integers which is creatable, destroyable, can be passed by
borrowed and mutable reference, but isn’t enough to support movability or
copyability.  We’ll add those later.

### When do we invoke destructors for value bindings?

Now that we have a way to define a destructor for a value, we need to invoke it
automatically.  Where do we invoke the destructor for a local value binding?
Two major choices exist:

1. End of scope, ala C++ (and I think Rust).
2. After last use, ala Swift and Val (but Val has a better model).

The difference can be seen in cases like this:

```mojo
  fn bindingLifetimeExample():
      var vec = Vec()
      vec.push_back(12)
      use(vec)
      # Val/Swift destroys 'vec' here.
      do_lots_of_other_stuff_unrelated_to_vec()
      # C++/Rust destroy vec here.
```

I would advocate for following the Swift model.  It reduces memory use, and I
haven’t seen it cause problems in practice - it seems like the right default.
Furthermore, this dovetails well with ownership, because you want (e.g.)
immutable borrows to die early so you can form mutable borrows in other
statements.  It also makes the “form references within a statement” special case
in C++ go away.

The tradeoff on this is that this could be surprising to C++ programmers,
something that Swift faced as well.  The balance to that is that GC languages
with finalizers are not used for RAII patterns, and Python has first-class
language support for RAII things (the `with` statement).

There are specific cases like RAII that want predictable end-of-scope
destruction, so you end up needing a `@preciseLifetime` decorator on the struct
or use closures - [both work
fine](https://developer.apple.com/documentation/swift/withextendedlifetime(_:_:)-31nv4).

NOTE: This was pushed forward during implementation to the "ASAP" model that
Mojo uses.

### Taking a value from a binding

The other thing you may want to do is to intentionally end a binding early,
transferring ownership of the bound value out as an owned rvalue.  Swift and
Rust both support mutable value lifetimes with holes in them, and ending
immutable bindings early (Rust with the `drop(x)` operator or by moving out of
the binding, Swift with the recently proposed
[consume/take/move](https://github.com/apple/swift-evolution/blob/main/proposals/0366-move-function.md)
operation).

I propose supporting this with the `^` postfix operator, e.g.:

```mojo
    fn takeVec(owned v: Vec): ...

    fn showEarlyBindingEnd():
      var vec = Vec()
      vec.push_back(12)
      takeVec(vec^)   # Ownership of vec is transferred to takeVec.
      do_lots_of_other_stuff_unrelated_to_vec()

      var vec2 = Vec()
      vec2.push_back(12)
      ...
      _ = vec2^ # force drop vec2.
```

This is postfix so it composes better in expressions, e.g.
“`someValue^.someConsumingMethod()`”.

### Supporting “taking” a value, with a convention (+ eventually a Trait)

I believe it is important for common types to support a “destructive take” to
support use-cases where you want to std::move an element out of the middle of a
`std::vector`.  C++ has `std::move` and move constructors for this, and Rust has
a ton of complexity to work around the lack of this.  Swift doesn’t appear to
have a story for this yet.  I think we just use a method convention (eventually
formalized as a trait) where types who want it define a `take()` method:

```mojo
  struct Vec:
      ...
      fn __moveinit__(inout self, inout existing):
          # Steal the contents of 'existing'
          self.data = existing.data
          self.capacity = existing.capacity
          self.size = existing.size

          # Make sure 'existing's dtor doesn't do bad things.
          existing.data = None
          existing.size = 0
          existing.capacity = 0
```

This is analogous to defining a move constructor in C++.  Note that you only
need to define this if you want to support this operation, and we eventually
should be able to synthesize this as a default implementation of the “Takable”
trait when we build out traits and metaprogramming features.

## Value Lifetime Enforcement

Now that we have all the mechanics in place, we actually have to check and
enforce lifetime in the compiler.  This entails a few bits and pieces.

### Implement a “Definitive Initialization” Like Pass: CheckLifetimes

The first thing we need is a dataflow pass that tracks the initialization status
of local bindings.  The basic mechanics needed here are implemented in the Swift
Definitive Initialization pass, and a lot of the mechanics are well described in
the [Drop Flags section of the
Rustonomicon](https://doc.rust-lang.org/nomicon/drop-flags.html) and [slide 135+
in this
talk](https://www.llvm.org/devmtg/2015-10/slides/GroffLattner-SILHighLevelIR.pdf).
This is a combination of static analysis and dynamically generated booleans.

When building this, we have the choice to implement this in a field sensitive
way.  I believe that this is a good thing to do in the medium term, because that
will allow making `__new__` and `__del__` methods much easier to work with in
common cases, and will compose better when we get to classes.  That said, we
should start with simple non-field-sensitive cases and extend it over time.
This is what we did when bringing up Swift and it worked fine.

### Implement support for variable exclusivity checking

While it isn’t a high priority in the immediate future, we should also add
support for variable exclusivity checking to detect dynamic situations where
aliases are formed.  See the [Swift
proposal](https://github.com/apple/swift-evolution/blob/main/proposals/0176-enforce-exclusive-access-to-memory.md)
for details on the issues involved here.  Mojo will be working primarily with
local variables in the immediate future, so we can get by with something very
simple for the immediate future.

### Synthesize destructors for structs

The other thing necessary in the basic model is for the destructors of field
members to be run as part of `__del__` methods on structs.  We don’t want people
to have to write this manually, we should synthesize a `__del__` when needed.
For example in:

```mojo
    struct X:
        var a: T1
        var b: T2

        fn __del__(owned self):  # This should be synthesized.
            _ = self.a^
            _ = self.b^
```

## Extensions to make the programming model nicer

With the above support, we should have a system that is workable to cover the
C++/Rust use-cases (incl `std::atomic` and `SmallVector`), handle the motivating
Tensor type and Python interop, as well as provide a safe programming model
(modulo dangling reference types which need lifetimes to support).  That said,
we still won’t have a super nice programming model, this includes some “table
stakes” things we should include even though they are not strictly necessary.

In particular, the support above completely eliminated the need to copy and move
values, which is super pure, but it would be impractically painful to work with
simple types that have trivial copy constructors.  For example:

```mojo
  fn useIntegers(a: Int):
      var b = a+4        # ok, b gets owned value returned by plus operator.
      let c = b          # error, cannot take ownership from an lvalue.
      let c2 = b.copy()  # Ok, but laughable for Int.
```

It is worth saying that the “c = b” case is something that we explicitly want to
prohibit for non-trivial types like vectors and dictionaries: Swift implicitly
copies the values and relies on COW optimization and compiler heroics (which are
not amazingly great in practice) to make it “work”.  Rust handles it by moving
the value away, which breaks value semantics and a programmer model that people
expect from Python.

It is better for vectors and Dictionary’s (IMO) to make this a compile time
error, and say “this non-copyable type cannot be copied”.  We can then
standardize a `b.copy()` method to make the expense explicit in source code.

### Copy constructors for implicitly copyable types

The solution to both of these problems is to allow types to opt-in to
copyability (as in the Rust `Copy` trait). The obvious signature for this in
Mojo seems to be a `__copyinit__` implementation (eventually formalized as a
trait):

```mojo
    struct Int:
      var value: __mlir_type.index
      fn __copyinit__(inout self, borrowed existing: Int):
        self.value = existing.value
```

Given this new initializer, a type that implements this is opt-ing into being
implicitly copied by the compiler.  This (re)enables lvalue-to-rvalue conversion
with a “load” operation, but makes it explicit in user source code.  It allows
integers to work like trivial values, and allows the library designer to take
control of what types support implicit copies like this.  This is also required
for the Python interop “object” type, since we obviously want `x = y` to work
for Python objects!

One nice-to-have thing that we should get to eventually (as we build out support
for traits and metaprogramming) is `Copyable` trait with a default
implementation.  This would allow us to manually provide a copy constructor if
we want above, or get a default synthesized one just by saying that our struct
is `Copyable`.  See the appendix at the end of the document for more explanation
of how this composes together.

All together, I believe this will provide a simple and clean programming model
that is much more predictable than the C++ style, and is more powerful than the
Swift or Rust designs, which don’t allow custom logic.

### Opting into pass-by-register parameter convention

The final problem we face is the inefficiency of passing small values
by-reference everywhere.  This is a problem that is internalized by C++
programmers through common wisdom (“pass complex values like `std::vector` as
`const& or rvalue-ref` but trivial values like `int` by value!”), but ends up
being a problem for some generic templated code - e.g. `std::vector` needs to
declare many things as being passed by `const&` or rvalue reference, which
becomes inefficient when instantiated for trivial types. There are ways to deal
with this in C++, but it causes tons of boilerplate and complexity.

The key insight I see here is that the decision is specific to an individual
type, and should therefore be the decision of the type author.  I think the
simple way to handle this is to add a struct decorator that opts the struct into
being passed by owning copy, equivalent to the Rust Copy convention:

```mojo
    @register_passable
    struct Int:
        ...
```

This decorator would require that the type have a copy constructor declared, and
it uses that copy constructor in the callee side of an API to pass arguments
by-register and return by-register.  This would lead to an efficient ABIs for
small values.

This decorator should only be used on small values that makes sense to pass in
registers or on the stack (e.g. 1-3 machine registers), and cannot be used on
types like `llvm::SmallVector` that have interior pointers (such a type doesn’t
make sense to pass by-register anyway!).

# Conclusion

This proposal attempts to synthesize ideas from a number of well known systems
into something that will fit well with Mojo, be easy to use, and easy to teach -
building on the Python ideology of “reducing magic”.  It provides equivalent
expressive power to C++, while being a building block to provide the full power
for Rust-style lifetimes.

## Parameter Conventions Summary

This is the TL;DR: summary of what I think we end up with:

```mojo
    fn borrow_it(a: X)           # takes X as borrow (sugar).
    fn borrow_it(borrowed a: X)  # takes X as borrow.
    fn take_it(owned a: X)            # takes owned X
    fn ref_it(inout a: X)             # takes mutable reference to X
    fn register_it(a: Int)       # by copy when Int is register-passable.

    fn ret_owned(self) -> X:           # Return an owned X (sugar).
    fn ret_owned(self) -> owned X:     # Return an owned X.
    fn ret_borrow(self) -> borrowed X: # Return an X as a borrow
    fn ret_ref(self) -> inout X:       # Return an X as a mutable ref
    fn ret_register(self) -> Int:      # Return by copy when Int is register-passable
```

## Extension for Lifetime types

Lifetimes are necessary to support memory safety with non-trivial correlated
lifetimes, and have been pretty well proven in the Rust world.  They will
require their own significant design process (particularly to get the defaulting
rules) and will benefit from getting all of the above implemented.

That said, when we get to them, I believe they will fit naturally into the Mojo
and MLIR design.  For example, you could imagine things like this:

```mojo
    # Take a non-copyable SomeTy as a borrow and return owned copy
    fn life_ex1['a: origin](value: 'a SomeTy) -> SomeTy:
      return value.copy()

    # Take a non-copyable SomeTy and return the reference
    fn life_ex2['a: origin](value: 'a SomeTy) -> borrowed 'a SomeTy:
      return value
```

This is not a concrete syntax proposal, just a sketch.  A full design is outside
the scope of this proposal though, it should be a subsequent one.

# Appendix: Decorators and Type Traits

Above we talk loosely about decorators and type traits.  A decorator in Python
is a modifier for a type or function definition.  A Trait in Rust (aka protocol
in Swift, aka typeclass in Haskell) is a set of common behavior that unifies
types - sort of like an extended Java interface.

Let’s see how these two concepts can come together in the future assuming we get
Swift/Rust-style traits and extend Python’s decorator concept with
metaprogramming features enabled by Mojo.

Traits include “requirements”: signatures that conforming types are required to
have, and may also include default implementations for those.  The type can
implement it manually if they want, but can also just inherit the default
implementation if not.  Let’s consider copy-ability.  This isn’t a standard
library proposal, but we could implement some copy traits like this:

```mojo
  trait Copyable:
      # Just a signature, no body.
      fn copy(self) -> Self: ...

  trait ImplicitlyCopyable(Copyable):     # Could use a better name :)
      # A __copyinit__ is required, and this is the default implementation.
      fn __copyinit__(inout self, borrowed existing: Self):
          self = existing.copy()
```

Type may conform to the `Copyable` trait, which allows generic algorithms to
know it has a `copy()` method.  Similarly, they may conform to
`ImplicitlyCopyable` to know it is implicitly copable (supports “`let a = b`”).
`ImplicitlyCopyable` requires the type to have a `copy()` method (because
`ImplicitlyCopyable` refines `Copyable`) and a `__copyinit__` method with the
specified signatures, and also provides a default implementation of the
`__copyinit__` method.

This allows types to use it like this:

```mojo
    struct Int(ImplicitlyCopyable):
        var value: __mlir_type.index
        fn copy(self: Self) -> Self:
            return Int{value: self.value}

      # Don't have to write this, we get a default impl from ImplicitlyCopyable
      # fn __copyinit__(inout self, borrowed existing: Self):
      #     self = existing.copy()
```

This allows clients to implement a simple `copy()` method, but get the internal
machinery for free.

The decorator is a different thing that layers on top of it.  Decorators in
Python are functions that use metaprogramming to change the declaration they are
attached to.  Python does this with dynamic metaprogramming, but we’ll use the
interpreter + built-ins operations to also enable static metaprogramming in
Mojo.

I'm imagining that this will allow someone to write just:

```mojo
    @implicitlyCopyable
    struct Int:
      var value : __mlir_type.index
```

And the `implicitlyCopyable` function (which implements the decorator) would be
implemented to do two things:

1. When it understands all the stored properties of a copy, because they are
built-in MLIR types like index, or because they themselves conform to at least
`Copyable`, it synthesizes an implementation of a `copy()` method that builds a
new instance of the type by invoking the `copy()` member for each element.

2. It adds conformance to the `ImplicitlyCopyable` trait, which provides the
`__copyinit__` method above.

This is all precedented in languages like Swift and Rust, but they both lack the
metaprogramming support to make the decorator synthesis logic implementable in
the standard library.  Swift does this for things like the Hashable and
`Equatable` protocols.  I believe that Mojo will be able to support much nicer
and more extensible designs.

NOTE: The `@value` decorator provides some of this now.
