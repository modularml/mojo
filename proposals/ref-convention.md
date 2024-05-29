# Mojo references, an alternate take

Nick Smith + Chris Lattner; May 28, 2024

**TL;DR**: This white paper proposes several changes that rethink Mojo’s general
UI around references. It does not change the formal semantic model, but has a
profound impact on explainability and auto-dereferencing.

## Motivation

Mojo’s safe references have evolved and iterated a lot. From the initial
compiler re-plumbing that made memory-only types possible, to threading
lifetimes through everything with the introduction of `!lit.ref` to the
development of a “user-space” Reference type, to the recent discussions about
[adding automatic dereference to Reference](https://github.com/modularml/mojo/discussions/2594),
we’ve been iteratively improving the model with a goal of ending up with
something powerful and explainable.

Along the way, we’ve had a number of challenges to address:

1. How to make references work, how they dovetail with ASAP destruction
   insertion, with exclusivity checking, and how lifetimes and mutability are
   represented: These topics are **NOT** touched on in this paper.

2. The introduction of parametric mutability, which is important to representing
   things like a “refitem” whose result is a reference with the same mutability
   as ‘self’. We currently support this, but require a weird “define self as
   Reference type instead of Self type” approach which is inconsistent with all
   the other argument conventions in Mojo.

3. We need “automatically dereferenced” references for ergonomics, and require
   an explainable and predictable model. The recently proposed “automatic
   dereference” model follows C++ precedent by building this in as a new form of
   LValue to RValue conversion, but this makes the language significantly more
   complicated by breaking an invariant: the type loaded from an LValue (an
   RValue) has different type than the type stored into it.

4. We have persistent confusion about the word “reference”: Python considers all
   “PythonObject”s to be “object references”, which is completely different than
   what the `Reference` type provides. Furthermore, there are also “reference
   semantic” types which are more similar to the Python notion and less similar
   to `Reference`. It would be awesome to clarify this.

5. We still need to
   [reconsider which keywords](https://github.com/modularml/mojo/blob/main/proposals/lifetimes-keyword-renaming.md)
   to use for argument conventions. The `inout` keyword, for example, is
   problematic because it works with types that are not movable or copyable. The
   callee doesn’t actually move things in and out, it takes a mutable reference.
   It is the caller that does the inout magic when needed, so it seems
   inappropriate to put this on the callee declaration.

We believe that this proposed model is also significantly simpler than the
previous approaches.

## Proposal #1: Introduce a new `ref` argument convention

The existing `inout` and `borrowed` conventions are argument conventions that
are syntax sugar for an underlying MLIR type `!lit.ref` (with a bit of
additional semantics on top). These references are always “auto dereferenced” in
the body of the function. Let’s introduce a new `ref` convention that allows
specifying an expected lifetime and that is auto-dereferenced in the body like
`inout` is. For the moment, we'll use the syntax `ref [<lifetime>]`. Here's a
basic example:

```mojo
fn take_int_ref(a: Int, ref [_] b: Int) -> Int:
    return a+b  # b, not b[]
```

Alternative syntaxes are proposed later in this document.

Given this feature, we can remove the ability to define `self` as a `Reference`
and just use this new argument convention. Instead of:

```mojo
fn __refitem__(self: Reference[Self, _, _], index: Int) -> Reference[
        Self.ElementType, self.is_mutable, self.lifetime
    ]:
```

You would now use an argument convention:

```mojo
fn __refitem__(ref [_] self, index: Int) -> Reference[
        # This is a bit yuck, but is simplified further below.
        Self.ElementType, __lifetime_of(self).is_mutable, __lifetime_of(self)
    ]:

# Alternatively, name the Lifetime:
fn __refitem__[life: Lifetime](ref [life] self, index: Int) -> Reference[
        # This is a bit yuck, see below.
        Self.ElementType, life.is_mutable, life
    ]:
```

### Supporting address spaces

`!lit.ref` currently has three parameters: a type, a lifetime, and an address
space. The address space is a relatively niche but important feature used for
GPUs and other accelerators. We specify the address space as follows:

```mojo
fn foo(...) -> ref [lifetime, addr_space] T: ...
```

This parameter would be optional, just as it is currently optional for
`Reference`.

In the future, the lifetime and address space could potentially be combined into
one value.

### How do we explain this?

If we implement this, we would advise Mojo users to use `ref` with an explicit
lifetime in a few situations:

1. If you want to bind an argument to something of parametric mutability.

2. When you want to tie the lifetime of one argument to the lifetime of another
   argument.

3. When you want an argument that is guaranteed to be passed in memory: this can
   be important and useful for generic arguments that need an identity,
   irrespective of whether the concrete type is register passable.

This also aligns with the C++ notion of “passing by reference”.

## Proposal #2: Allow the use of `ref` in result positions

The other feature we need is the ability to _return_ an "automatically
dereferenced" reference. For the moment, we'll use the same `ref [<lifetime>]`
syntax for this. This feature can replace the use of `Reference` in the previous
example:

```mojo
fn __refitem__(ref [_] self, index: Int)
    -> ref [__lifetime_of(self)] Self.ElementType:

# Hopefully someday we'll have a Lifetime type:
fn __refitem__(ref [_] self, index: Int)
    -> ref [Lifetime(self)] Self.ElementType:
```

This is much more clear. As with the `ref` argument convention, the `ref` result
convention automatically dereferences itself... but this happens on the caller
side.

This means that auto-deref behavior happens in exactly one place in the
compiler: in the result of function calls whose result convention is a `ref`
result convention. This makes it far less invasive than the previously proposed
auto-deref behavior and is much more predictable for users.

Note that this gives the developer control over auto-deref, because both of
these are valid:

```mojo
fn yes_auto_deref(...) -> ref [lifetime] Int: ...
fn no_auto_deref(...) -> Reference[Int, lifetime]: ...
```

We would expect `ref` to be the default choice. It's the most versatile: if a
function call returns a `ref` but you want a `Reference`, you can just wrap the
call in the `Reference` constructor. The purpose of `Reference` is to allow
references to be _stored_ in data structures. Furthermore, it supports nesting
such as `Reference[Reference[Reference[Int], ...], ...], ...]` without implicit
promotions interfering.

## Proposal #3: Remove `__refitem__`

Now that general functions can return an auto-dereferenced reference, we can get
rid of the `__refitem__` special case and just use `__getitem__`. This
simplifies the language and improves consistency with Python. Our example above
becomes:

```mojo
fn __getitem__(ref [_] self, index: Int)
    -> ref [__lifetime_of(self)] Self.ElementType:
```

Note: Neither `inout` nor `borrowed` are allowed in a result position. The
result requires a lifetime so that the caller knows what it refers to, and how
long it is valid for.

## Proposal #4: Rename `Reference` to `Pointer` (or `SafePointer` )

As of Mojo 24.3, our status is:

1. The `Reference` type doesn’t have support for automatic dereferencing.

2. The old `Pointer` type got renamed to `LegacyPointer`, and we aim to replace
   it entirely with `UnsafePointer`.

3. The `UnsafePointer` API has been cleaned up and improved, and interacts well
   with `Reference`.

Because there is no automatic dereference, the existing `Reference` type
currently behaves like a pointer, not a reference. You need to explicitly
dereference it with `ptr[]` just like an unsafe pointer… but it adds safety
through lifetime management and semantic checking.

Let’s just rename it to `Pointer`: The consequence of this is that explicit
dereference syntax is a feature, not a bug. Furthermore, this entire feature
area becomes more explainable and separate from the existing Python reference
types. This allows a consistent story about how Mojo supports both `Pointer`
(for use when building data structures) and `UnsafePointer` (for interacting
with C code).

## Summary

We believe that the proposed `ref` conventions will lead to a simpler and more
consistent UX, and a more predictable and simpler automatic dereferencing
feature than previously proposed. Furthermore, we believe that renaming
`Reference` to `Pointer` will clarify terminology and allow library developers
to continue working with explicit safe pointers when building data structures
and other libraries.

## Discussion + Future direction

We believe that the proposed model would simplify a lot of things, but there are
some consequences worth mentioning:

### Argument/Result conventions are not first class types

The proposal above is consistent with Mojo’s current use of argument
conventions, but is inconsistent with Rust references. Notably, it is NOT
possible to define a local “ref”, and it isn’t possible to define a “ref to a
ref”:

```mojo
fn weird_tests[life1: Lifetime, life2: Lifetime](
           ref [life1] arg1: Int,  ## ok

           # error: 'ref' is an argument convention, not a type.
           ref [life1] arg2: ref [life2] Int,

           # This is ok
           ref [life1] arg3: Pointer[Int, life2]):

    # error: 'ref' is an argument convention, not a type.
    var local1: ref [life1] Int
    var local2: Pointer[Int, life1] # ok!
```

While this is _different_ than Rust, we feel this is actually a good thing - it
clarifies where automatic dereference happens (on arguments and results of ‘ref’
functions). It is also clear that `typeof(arg3) == Pointer[Int, life2]` and that
`typeof(arg3[]) == Int` through composition.

### Multiple results wouldn’t get automatic dereference

One downside of this proposal (compared to the C++-style autoderef proposal) is
that it wouldn’t be possible to write a function that returns multiple
auto-dereferenced values in a tuple:

```mojo
# Simple tuple result type is ok of course:
fn g1(...) -> (Int, Int): ...

# Returning safe pointers is fine: each element needs explicit derefs
fn g2(...) -> (Pointer[Int, life1], Pointer[Int, life2])

# error: 'ref' is an argument convention, not a type.
fn g3(...) -> (ref [life1] Int, ref [life2] Int)
```

This is unfortunate, and really doesn’t fit with this model, but it isn’t a
common occurrence.

#### Aside: Mojo may need native support for multiple results anyway

If this were important to solve for, we could consider extending Mojo to
natively support multiple return values directly. The compiler internally
already supports this, but it is not exposed to Mojo source code. One example of
something that could be supported with native multiple return values are
“unpacking” for non-movable results, e.g.:

```mojo
fn do_stuff() -> result: NonMovable:
  var a : NonMovable
  a, result = f()
```

… this isn’t possible to support with “multiple results are returned as tuples”.
This doesn’t seem like a priority to solve in the short term, but is a plausible
long term path.

### What would this simplify in the Mojo compiler?

This generally makes the type checker simpler, predictable, and more composable,
because it doesn’t cause reference decay to affect literally everything in the
expression type checker. Concretely, we can revert a lot of the stuff Chris has
been working on lately:

- No need for the new `@automatically_dereference` decorator.
- IRValue::getRValueType is no longer context sensitive.
- Removes the support for allowing ‘self’ to be declared as Reference.
- Removes the `!lit.ref` special case hack in call argument parsing and
  parameter inference.

It also eliminates the need for all the parameter inference improvements that
went in, but they are also not harmful, so they’ll stay in.

### Reference patterns are still needed

We will need to build out Mojo's support for
[patterns](https://docs.python.org/3/reference/compound_stmts.html#grammar-token-python-grammar-patterns)
and the
[`match`](https://docs.python.org/3/tutorial/controlflow.html#match-statements)
statement, which are currently completely missing. As part of this, we'll need
to investigate adding a `ref` pattern. Such a thing would allow a foreach loop
that mutates the elements from within the loop:

```mojo
  for (ref elt) in mutableList:
     elt = 42
```

Mojo needs patterns across the language in general: they should work in `match`
statements, `var` declarations, and a few other places. Adding `ref` patterns
would allow having an autodereferenced `var` reference.

## Syntax alternatives

We've relegated syntax discussions to this appendix. Let the bikeshedding
commence! 👏

We have proposed two new conventions: an argument convention and a result
convention. For both of these conventions, we have used the keyword `ref`. This
terminology has precedent in C++, C#, and many other languages. However, Mojo
aims to be a superset of Python, and in Python, a "reference" is understood to
be a pointer to a heap-allocated, garbage-collected instance of a class. The
argument conventions that we've proposed are entirely unrelated to that feature.
Therefore, it seems wise to consider other syntaxes for the conventions. We
encountered a similar issue with the `Reference` type, and this motivated our
proposal to rename `Reference` to `Pointer`.

Our use of square brackets in the `ref [<lifetime>]` syntax is also worth
interrogating. This syntax is motivated by the fact that the lifetime is being
provided as a parameter to a `!lit.ref` type. However, this type is an
implementation detail — Mojo users are not supposed to know about this. In this
light, the square brackets are quite "mysterious". There is no function being
invoked, and (apparently) no type being instantiated. This usage of square
brackets has no precedent.

### Syntaxes for the argument convention

Here are some alternative syntaxes that avoid both of the issues mentioned
above:

```mojo
# We could repurpose Python's `from` keyword:
fn foo(x from <lifetime>: Int):

# Or its `in` keyword:
fn foo(x in <lifetime>: Int):

# Or introduce a new `at` keyword:
fn foo(x at <lifetime>: Int):
```

All of these keywords are suggestive of a lifetime being a "location" associated
with a variable. This is a helpful way to think about Mojo's `Lifetime` type —
one of the major purposes of this type (which is still in-development) is to
keep track of where a variable is located (e.g. on the stack), to ensure that
the variable is not used after it is freed. For this reason, "lifetimes" will
likely be renamed at some point, perhaps to something like "regions" or
"origins". In this light, the syntax `x from <origin>: Int` makes a lot of
sense.

That said, it's too early to settle on a keyword for this. If `Lifetime` ends up
being called something weird like `AccessProtocol`, none of these keywords would
make sense! The main thing to take away from this discussion is that
"bracketless" syntaxes are an option.

### Syntaxes for the result convention

We can consider a similar syntax for the result convention. However, results are
usually not named, so a naïve translation of the above syntax doesn't look very
clean:

```mojo
fn foo() -> from <origin>: Int:
```

Using the `ref` keyword here would avoid this problem:

```mojo
fn foo() -> ref from <origin>: Int:
```

But given that we're trying to avoid the term "reference", we should consider
other options. A noun would make the most sense, because we'd like to be able to
say "this function returns a ...". But what word could we use here?

Traditionally, a function returns a **value**. For example, `len` returns a
value of type `Int`. Our new result convention works differently. It returns a
"reference" to a variable, but this reference is not a value — it can't be
stored in a collection, for example. (At least, not without wrapping it in a
`Pointer`.)

A reasonable path forward would be say that some functions return **values**,
and other functions return **variables**. This deftly dodges the word
"reference", while still providing intuition about how the result can be used.

A variable _holds_ a value. Thus, it should be clear that if a function returns
a variable, you can read its value, and if the variable is mutable, you can
overwrite its value. These are exactly the affordances provided by the new
result convention.

Given all of this, the following syntax might be reasonable:

```mojo
fn foo() -> var from <origin>: Int:

# The use of `var` here is independent of the `from` syntax.
# We could combine it with the square bracket syntax instead:
fn foo() -> var [<origin>] Int:
```

The expression `foo()` behaves just like a variable does. If it is mutable, you
can reassign it, or mutate it:

```mojo
foo() = 0
foo() += 1
```

The biggest downside of using the `var` keyword for the new result convention is
that as of today, `var` is exclusively used to declare _new_ variables, whereas
in the above syntax, `var` is being used to declare that a function returns a
reference to an _existing_ variable. That said, it's unlikely that users will
misinterpret `-> var from <origin>` as declaring a new variable, given the
context in which it appears.

As mentioned, `var` is the only noun that seems suitable. However, we could
consider a verb or an adjective:

```mojo
fn foo() -> select from <origin>: Int:

fn foo() -> select [<origin>] Int:

fn foo() -> shared from <origin>: Int:

fn foo() -> shared [<origin>] Int:
```

Unfortunately, if `->` is read as "returns", then a verb doesn't make
grammatical sense. An adjective _might_ make sense, but it's not clear what a
good adjective would be. The word `shared` is problematic, because it has
connotations concerning synchronization etc.

Finally, using the `from`/`in`/`at` keywords in the result convention leads to
an issue with the `:` character. We are using it to specify both the type of the
returned variable, and to mark the end of the function signature:

```mojo
fn foo() -> var from <origin>: Int:
```

This is clunky, and may complicate parsing. We can fix this by adding
parentheses:

```mojo
fn foo() -> (var from <origin>: Int):
```

Alternatively, if (for some reason) the `Lifetime`/`Origin` type ends up being
parameterized by the type of the variables that it contains, we could drop the
type annotation entirely:

```mojo
fn foo() -> var from <origin>:

fn foo() -> var at <origin>:
```

Another option would be to revert to the square bracket syntax, since this
allows us to omit the first colon:

```mojo
fn foo() -> var [<origin>] Int:
```

In summary: there are a lot of alternatives worth considering!

### Keyword alternatives for `inout`, `borrowed`, and `owned`

Along with contemplating syntaxes for the _new_ conventions, it makes sense to
revisit the syntax of Mojo's _existing_ conventions. The conventions are all
related, and we need a consistent way to talk about them.

Let’s look at a few examples in 24.3 Mojo:

```mojo
## Today in mojo:
struct MyInt:
   # Note that inout is a lie here, the input is uninitialized.
   fn __init__(inout self, value: Int): ...

   # borrowed is the default (e.g. on 'rhs') so not usually written.
   fn __add__(borrowed self, rhs: MyInt) -> MyInt: ...

   # This function actually just takes a mutable reference, the
   # caller may do some copy in/out depending on its situation though.
   fn __iadd__(inout self, rhs: MyInt): ...

   # An owned argument must be initialized when the function is called,
   # and it will be deinitialized by the time the function returns.
   # (Unless the caller provides a copy.)
   fn __del__(owned self): ...
```

A common question a programmer has in their mind when browsing unfamiliar code
is: "what does this function do?" A well-designed syntax for arguments and
results will help make it clear what a function does to them. The current
keywords don't achieve this goal:

1. Using `inout` for the `self` argument of a constructor call is semantically
   misleading. `inout` is meant to be used to declare that an _existing_ value
   is mutated, but `self` does not have a value at the beginning of the function
   call. The role of a constructor is to _initialize_ `self`, so it makes sense
   to introduce a new keyword (such as `init`) for this purpose. Note: this
   convention is also the convention associated with a standard return type,
   e.g. `-> T`. If we had a syntax for naming the result, we could write this as
   `-> (init result: T)`.
2. More generally, the keyword `inout` is misleading no matter where it is used.
   It suggests that a value will be copied (or moved) into the function, and
   then copied back out again. In actuality, an `inout` value is often (but not
   always) passed by address. The true purpose of this keyword is to declare
   that the argument will be **mutated**, so it makes sense to rename this
   keyword to `mut`, or `mutate`. The former is more concise and is consistent
   with Rust, so it might be preferable.
3. The keyword `owned` isn't very informative. The purpose of this convention is
   to allow a function to "use up" the value of a variable. The variable still
   belongs to its original owner, the callee just _deinitializes_ it. There's a
   notable caveat: if the caller doesn't use the `^` sigil, they will provide a
   _copy_ of their variable, and therefore they will not witness this
   deinitialization. But on the callee's side, the variable must always end up
   deinitialized, for example by calling its destructor, or its move
   constructor. To emphasise this fact, it would make sense to rename `owned` to
   `consume`.

Notice that all of these words are verbs, or abbreviations thereof:

- `init` (i.e. "initialize")
- `mut` (i.e. "mutate")
- `consume`

Verbs are nice, because they communicate what a function _does_ to an argument.
In comparison, adjectives such as `owned` are not as clear.

We also have `borrowed`, which we can rename to `borrow`, thus ensuring that all
of our conventions are verbs. We could consider renaming this to `read` instead
(as a
[soft keyword](https://docs.python.org/3/reference/lexical_analysis.html#soft-keywords)).
But we might not actually need a keyword at all, since this is now the default
convention for both `def` and `fn`.

In summary, a function will either:

- `init` an argument
- `borrow` or `read` an argument
- `mut` or `mutate` an argument
- `consume` an argument

These are just suggestions. Various synonyms of these words might also work
well. Regardless, we should postpone any final decisions until argument
conventions and lifetimes have fully settled.

If we repaint the earlier example with the new keywords, we get:

```mojo
## Proposed
struct MyInt:
   fn __init__(init self, value: Int): ...

   fn __add__(borrow self, rhs: MyInt) -> MyInt: ...

   fn __iadd__(mut self, rhs: MyInt): ...

   fn __del__(consume self): ...
```

On top of this, we would have `ref`, which would be used whenever you need to
associate an argument or a result with an explicit lifetime.

Note that argument conventions are not types - they are modifiers to the
behavior of arguments that are local to the function declaration syntax. This
means that all of these keywords can be soft keywords, if we want.

### Terminology for the `^` sigil

Along with renaming `owned` to `consume`, it would make sense to simultaneously
rename the `^` sigil, which is currently known as the "transfer operator".

The current name is problematic for two reasons:

1. In everyday English, "transfer" and "move" are synonyms. However,
   "transferring" and "moving" are completely different concepts in Mojo.
   Appending the `^` sigil to an expression does not imply that its move
   constructor will be invoked. In fact, it's possible (and common) to transfer
   a value whose type is **immovable**.
2. The `^` sigil isn't actually an operator. Or rather, it doesn't correspond to
   any _particular_ operation. Instead, it is merely used to indicate that a
   value should be consumed, i.e. passed to a function using the `consume`
   convention. In the case of `y = x^`, `x` is (typically) passed to
   `__moveinit__`, which consumes it, and in the case of a call such as
   `foo(x^)`, `x` is being passed to `foo`, which consumes it. Hence, `^` is not
   really an operator, it's a way of indicating and/or influencing what
   operation gets invoked. The chosen operation (if any) is determined by the
   surrounding context.

So we have problems with both the words "transfer" and "operator". To resolve
this, we suggest that the `^` sigil be referred to as the "consumption sigil".

- The word "consumption" establishes that each use of `^` is associated with a
  function call that uses the `consume` convention. As a fun bonus: the `^`
  symbol is usually pronounced "carrot", and carrots 🥕 are consumable. Whatever
  name we end up using, we should make sure that it matches the associated
  keyword.
- The word "sigil" has precedent in a few languages. It's usually used to refer
  to a character that influences how an identifier or expression is interpreted.
  In the case of Mojo, we can just quote the dictionary definition:
  > _sigil_: an inscribed symbol considered to have magical power

That's exactly what `^` is. If you inscribe an expression with `^`, its value
will be "magically" consumed. That's the hand-wavey definition. To provide a
formal definition, you would need to talk about move constructors, copy
constructors, temporary variables, etc.
