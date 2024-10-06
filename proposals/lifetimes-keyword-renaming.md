# Resyntaxing argument conventions and References

Date: October 2024
Previous revision: [[June 2023](https://github.com/modularml/mojo/blob/f8d7cb8ba4c21ec3fbc87e21609b3fd56cab695f/proposals/lifetimes-keyword-renaming.md)]

The design of the Mojo references subsystem is starting to come together.  To
finalize the major points, it helps to come back and re-evaluate several early
decisions in Mojo to make the design more self consistent.  This is a proposal
to gain consensus and alignment on naming topics, etc without diving deep into
the overall design (which is settling out).

I anticipate that this is going to drive significant bikeshed'ing, so let's get
one thing out of the way first: we're not trying to make Mojo similar to Rust.
Our implementation model and the semantics are significantly different (and
better IMO) than Rust's references and lifetimes, so while it is a good idea to
be inspired by Rust and bring in good ideas that fit into Mojo based on first
principle analysis, the argument that "it is similar to Rust" is not seen as
itself a good reason to do something.

Similarly, the general design of argument conventions in Mojo fits very well
into the ownership model and is scaling very effectively.  Changing the
architecture of how things work isn't in scope for this proposal.

## Rename `Reference` to `Pointer`

The `Reference` type is an explicitly dereferenced "safe pointer" type that
offers a memory safe subset of the `UnsafePointer` API.  This is an important
and useful type for some purposes, but is rarely used in practice, and it has
nothing to do with the behavior of `ref` elsewhere in Mojo.  As such, we have
already renamed it to avoid confusion and clarify the model.

## Renaming Mojo argument conventions

Let's take a survey of the Mojo language today.  As of Oct 2024, we have the
following argument conventions:

1) `borrowed`: This is the implicit convention that provides an immutable
   reference to another value with an inferred lifetime.
2) `inout`: This argument convention is a mutable reference to a value from a
   caller with an inferred lifetime.
3) `ref [lifetime]`: this argument convention allows either a mutable or
   immutable reference with a specified lifetime.  It can be used with `ref [_]`
   to infer an arbitrary lifetime.
4) `owned`: This argument convention provides a mutable reference to value that
   the callee may need to destroy.  I'd like to ignore this convention for the
   purposes of this document to keep it focused.
5) `fn __init__(inout self)`: Mojo has a special hack that allows (and requires)
   one to write the `self` argument on init functions as `inout`.  This doesn't
   make sense because the value isn't live-in, and indeed you will see poor
   error messages with code like `var x: fn (inout Foo) -> None = Foo.__init__`.

In addition, Mojo functions have the following return syntax:

1) `fn f():` means either `-> None` or `-> object` for a `fn` or `def`.
2) `fn f() -> T:` returns an owned instance of T.
3) `fn f() -> ref [life] T:` is a returned reference of some specified lifetime.
4) `fn f() -> T as name:` allows a named return value.  The intention of this syntax
   was to follow Python pattern syntax, but it is weird and we can't allow other
   pattern syntax here.

I suggest we rename `borrowed` to `ref` (without square brackets), rename
`inout` to `mutref` and introduce a new `out` convention.  Such a change will
give us:

1) `ref`: This is the implicit convention that provides an immutable
   reference to another value with an inferred lifetime.
2) `mutref`: This argument convention is a mutable reference to a value from a
   callee with an inferred lifetime.
3) `ref [lifetime]`: **No change:** this works as it does today.
4) `owned`: **No change:** unrelated to this proposal, let's stay focused.
5) `fn __init__(out self)`: The `__init__` function takes `self` as
   uninitialized memory and returns it initialized (when an error isn't thrown)
   which means it's a named output.  Let's call it `out`, which will allow one
   to write `var x: fn (out Foo) -> None = Foo.__init__` as you'd expect.
6) `mutref [lifetime]`: If there is a good reason, we could allow `mutref` to
   be used as a constraint that the lifetime is mutable.  This is actually nice
   for things like `mutref [_]` which would only infer a mutable lifetime (not
   a parametricly mutable lifetime) but is mostly a consistency thing.

Finally, let's align the result syntax:

1) `fn f():` **No change**.
2) `fn f() -> T:` **No change**.
3) `fn f() -> ref [life] T:`:  **No change**.
4) `fn f() -> (out name: T):` and maybe eventually `fn f(out name: T):`.  The
   former is a bit ugly because you'd want parens (or something else) to
   disambiguate (for humans, not the compiler) the `:` as a terminator of the
   function definition from the type specification.  The later is very pretty,
   would provide a path to "real" multiple return values, and would make the
   model consistent with `__init__` but has implementation risk that we'd have
   to explore.

As a benefit of these changes, we'd get rid of the `borrowed` terminology, which
is loaded with Rust semantics that we don't carry, and get rid of the `inout`
keyword which is confusing and technically incorrect: the callee never does copy
in/out.

### Alternatives considered

I expect the biggest bikeshed to be around the `mutref` keyword, we might
consider instead:

- `mut`: This is shorter, and the same length as `ref` but isn't clear to
  readers that it is a reference (losing important readability and information)
  and I don't see why it is useful for mutable references to be specifically
  compact.
- `mut ref`: we could use a space, but this seems like unnecessary verbosity
  and I don't see an advantage over `mutref`.

## Rename "Lifetime" (the type, but also conceptually)

Mojo uses parameters of types `Lifetime`, `MutableLifetime`,
`ImmutableLifetime` etc to represent and propagate lifetimes into functions.
For example, the standard library `StringSlice` type is declared like this:

```mojo
struct StringSlice[
    is_mutable: Bool, //,
    lifetime: Lifetime[is_mutable].type,
]

# ASIDE: We have a path to be able to write this as:
#    struct StringSlice[lifetime: Lifetime[_]]:
# but are missing a few things unrelated to this proposal.
```

This is saying that it takes a lifetime of parametric mutability.  We chose
the word "Lifetime" for this concept following Rust's lead when the feature
was being prototyped.

We have a problem though: the common notion of "the lifetime of a value" is
the region of code where a value may be accessed, and this lifetime may even
have holes:

```mojo
  var s : String
  # Not alive here.
  s = "hello"
  # alive here
  use(s)
  # destroyed here.
  unrelated_stuff()
  # ...
  s = "goodbye"
  # alive here
  use(s)
  # destroyed here
```

Mojo's design is also completely different in Mojo and Rust: Mojo has early
"ASAP" destruction of values instead of being scope driven.  Furthermore,
the use of references that might access a value affects where destructor
calls are inserted, changing the actual "lifetime" of the value.  These are
all point to the word "lifetime" as being the wrong thing.

So what is the right thing?  These parameters indicate the "provenance" of a
reference, where the reference might be pointing.  We also want a word that
is specific and ideally not overloaded to mean many other things in common
domains.  I would recommend we use the word **`Origin`** because this is a
short and specific word that we can define in a uniquely-Mojo way.  There is
some overlap in other domains (e.g. the origin of a coordinate space in
graphics) but I don't expect any overlap in the standard library.

To be specific, this would lead to the following changes:

- `Lifetime` => `Origin`
- `MutableLifetime` => `MutableOrigin`
- `ImmutableLifetime` => `ImmutableOrigin`

etc.  More importantly, this would affect the Mojo manual and how these
concepts are explained.

### Alternatives considered

There are several other words we might choose, here are some thoughts on
them:

- `Memory`: too generic and not tied to provenance, the standard library
  already has other things that work on "memory" generically.
- `Region`: this can work, but (to me at least) has connotations of nesting
  which aren't appropriate.  It is also a very generic word.
- `Target`: very generic.

## Rename the "transfer operator" (`x^`)

This is a minor thing, but the "transfer operator" could use a rebranding
in the documentation.  First, unlike other operators (like `+`) it isn't
tied into a method (like `__add__`): perhaps renaming it to a "sigil"
instead of an "operator" would be warranted.

More importantly though, this operator *makes the specified value able to
be transferred/consumed*, it does not itself transfer the value.  I'm not
sure what to call this.

## Implementation

All of these changes can be done in a way that is effectively additive: we need
to take `mutref` as a new keyword, but otherwise we can accept the new and the
old syntax in the 24.6 release and then remove the old syntax in subsequent
releases.

## Future directions

It is important to consider how this proposal intersects with likely future
work.  The most relevant upcoming feature work will be to introduce "patterns"
to Mojo.  Patterns are a [core part of Python](https://docs.python.org/3/reference/compound_stmts.html#patterns)
which we need for compatibility with the Python ecosystem, and are the basis
for `match` statements and other things.  When we implement them, we will allow
`var` statements to contain irrefutable patterns, and we will introduce a new
`ref` and `mutref` pattern to bind a reference to a value instead of copying
it.

This will allow many nice things for example it will eliminate the need for
`elt[]` when foreaching over a list:

```mojo
  # Copy the elements of the list into elt (required for Python compat), like
  # "for (auto elt : list)" in C++.
  for elt in list:
      elt += "foo"

  # Bind a mutable reference to elements in the list, like
  # `for (auto &elt : list)` in C++.
  for mutref elt in list:
      elt += "foo"

  # Bind an immutable reference to elements in the list, like
  # `for (const auto &elt : list)` in C++.
  for ref elt in list:
      elt += "foo"
```

Furthermore, we will allow patterns in `var` statements, so you'll be able to
declare local references on the stack:

```mojo
  # Copy an element out of the list, like "auto x = list[i]" in C++.
  var a = list[i]

  # Mutref to an element in the list, like "auto &x = list[i]" in C++.
  var (mutref b) = list[i]
  b += "foo"

  # Mutref to an element in the list, better syntax
  mutref c = list[i]
  print(len(c))

  # Immutable ref to an element in the list, like
  # "const auto &x = list[i]" in C++.
  var (ref d) = list[i]
  d += "foo"

  # Immutable ref to an element in the list, better syntax
  ref e = list[i]
  print(len(e))
```

We believe that this will provide a nice and familiar model to a wide variety
of programmers with ergonomic syntax and full safety.  We're excited for what
this means for Mojo.
