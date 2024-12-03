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
3) `ref [lifetime]`: this argument convention allows a reference to something of
   the specified lifetime, the lifetime specifies the mutability requirements.
4) `ref [_]`: this is a shorthand for binding a reference to an anonymous
   lifetime with any mutability.
5) `owned`: This argument convention provides a mutable reference to value that
   the callee may need to destroy.  I'd like to ignore this convention for the
   purposes of this document to keep it focused.
6) `fn __init__(inout self)`: Mojo has a special hack that allows (and requires)
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

I suggest we rename `borrowed` to `immref` (without square brackets), rename
`inout` to `mutref` and introduce a new `out` convention.  Such a change will
give us:

1) `immref`: This is convention provides an immutable reference to another value
   with an inferred lifetime.  As with `borrowed` it is allowed, but will never
   be written in practice.
2) `mutref`: This argument convention is a mutable reference to a value from a
   callee with an inferred lifetime.
3) `ref [lifetime]`: **No change:** this works as it does today.
4) `ref`: Bind to an arbitrary reference with inferred lifetime and mutability,
   this is the same as `ref [_]`.
5) `owned`: **No change:** unrelated to this proposal, let's stay focused.
6) `fn __init__(out self)`: The `__init__` function takes `self` as
   uninitialized memory and returns it initialized (when an error isn't thrown)
   which means it's a named output.  Let's call it `out`, which will allow one
   to write `var x: fn (out Foo) -> None = Foo.__init__` as you'd expect.

I don't see a reason to allow explicit lifetime specifications on `immref` and
`mutref`, e.g. `mutref [lifetime]`. The only use-case would be if
you'd want to explicitly declare lifetime as a parameter, but I'm not sure why
that is useful (vs inferring it).  We can evaluate adding it if there is a
compelling reason to over time.

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
   to explore.  In any case, this isn't really core to the rest of the proposal.

As a benefit of these changes, we'd get rid of the `borrowed` terminology, which
is loaded with Rust semantics that we don't carry, and get rid of the `inout`
keyword which is confusing and technically incorrect: the callee never does copy
in/out.

### Alternatives considered

I expect the biggest bikeshed to be around the `mutref` keyword.  The benefits
of its name is that it is clear that this is a reference, keeps in aligned with
`ref` in other parts of the language, and is short.

We might consider instead:

- `mut`: This is shorter, but loses that this is a reference.  Also the name
  `imm` for the borrowed replacement would be a bit odd.
- `mut ref`: we could use a space, but this seems like unnecessary verbosity
  and I don't see an advantage over `mutref`.

I'd love to see discussion, new ideas, and additional positions and rationale:
I'll update the proposal with those changes.

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
- `Provenance`: verbose and technical.

Furthermore if we're taking "mut" and "imm" as the root word for references, we
should decide if we're enshrining that as a [term of art](https://www.swift.org/documentation/api-design-guidelines/#use-terminology-well)
in Mojo.  If so, we should use names like `MutOrigin` and `ImmOrigin`.

## Implementation

All of these changes can be done in a way that is effectively additive: we need
to take `mutref` as a new keyword, but otherwise we can accept the new and the
old syntax in the 24.6 release and then remove the old syntax in subsequent
releases.

Lifetimes have been in heavy development, so I don't think we need to "phase in"
the changes to `Lifetime` etc, but we can keep around compatibility aliases for
a release if helpful.

## Future directions

It is important to consider how this proposal intersects with likely future
work.  This section includes a few of them for framing and perspective, but with
the goal of locking in the details above before we tackle these.

### Patterns + Local reference bindings

The most relevant upcoming feature work will be to introduce "patterns"
to Mojo.  Python supports both [Targets](https://docs.python.org/3/reference/simple_stmts.html#grammar-token-python-grammar-target)
and [Patterns](https://docs.python.org/3/reference/compound_stmts.html#patterns)
(closely related)
which we need for compatibility with the Python ecosystem.  These are the basis
for `match` statements, unpack assignment syntax `(a,b) = foo()` and other
things.  

Mojo currently has support for targets, but not patterns.  When we implement
patterns, we will extend `var` and `for` statements to work with them and we
will introduce a new
`ref` pattern to bind a reference to a value instead of copying it.  Because
there is always an initializer value, we can allow `ref` to always infer the
mutability of the initialized value like the `ref [_]` argument convention does.

This will allow many nice things for example it will eliminate the need for
`elt[]` when for-each-ing over a list:

```mojo
  # Copy the elements of the list into elt (required for Python compat), like
  # "for (auto elt : list)" in C++.
  for elt in list:
      elt += "foo"

  # Bind a mutable reference to elements in the list, like
  # `for (auto &elt : mutlist)` in C++.
  # This happens when `mutlist` yields mutable references.
  for ref elt in mutlist:
      elt += "foo"

  # Bind an immutable reference to elements in the list, like
  # `for (const auto &elt : immlist)` in C++
  # This happens when `mutlist` yields immutable references.
  for ref elt in immlist:
      use(elt.field) # no need to copy elt to access a subfield.
      #elt += foo # This would be an error, can't mutate immutable reference.
```

Furthermore, we will allow patterns in `var` statements, so you'll be able to
declare local references on the stack.

```mojo
    # Copy an element out of the list, like "auto x = list[i]" in C++.
    var a = list[i]

    # Unpack tuple elements with tuple patterns:
    var (a1, a2) = list[i]

    # Bind a ref to an element in the list, like "auto &x = mutlist[i]" in C++.
    var (ref b) = mutlist[i]
    b += "foo"

    # I don't see a reason not to allow `ref` in "target" syntax, so let's do
    # that too: 
    ref c = mutlist[i]
    c += "foo"

    # Parametric mutability automatically infers immutable vs mutable reference
    # from the initializer.
    ref d = immlist[i]
    print(len(d))
```

This should fully dovetail with parametric mutability at the function signature
level as well, an advanced example:

```mojo
struct Aggregate:
    var field: String

fn complicated(ref [_] agg: Aggregate) -> ref [agg.field] String:
    ref field = agg.field  # automatically propagates mutability
    print(field)
    return field
```

The nice thing about this is that it propagates parametric mutability, so the
result of a call to `complicated` will return a mutable reference if provided a
mutable reference, or return an immutable (or parametric!) reference if provided
that instead.

We believe that this will provide a nice and familiar model to a wide variety
of programmers with ergonomic syntax and full safety.  We're excited for what
this means for Mojo.

### Making `ref` a first class type

Right now `ref` is not a first class type - it is an argument and result
specifier as part of argument conventions.  After adding pattern support, we
should look to extend ref to conform to specific traits (e.g. `AnyType` and
`Movable` and `Copyable`), which would allow it to be used in collections, and
enable things like:

```mojo
struct Dict[K: KeyElement, V: CollectionElement]:
    ...
    fn __getitem__(self, key: K) -> Optional[ref [...] Self.V]:
        ...
```

Today this isn't possible because `ref` isn't a first class type, so you can't
return an optional reference, or have an array of ref's.  The workaround for
today is to use `Pointer` which handles this with a level of abstraction, or
use `raises` in the specific case of `Dict.__getitem__`.

### Consider renaming `owned` argument convention

This dovetails into questions about what we should do with the `owned` keyword,
and whether we should rename it.

One suggestion is to rename it to `var` because a `var` declaration is an owned
mutable value just like an `owned` argument.  The major problem that I see with
this approach is that the argument is "like a var" to the *callee*, but the
important part of method signatures are how they communicate to the *caller*
(e.g. when shown in API docs).

Because of this, I think the use of `var` keyword would be very confusing in
practice, e.g. consider the API docs for `List.append`:

```mojo
struct List[...]:
   ...
   fn append(out self, var value: T): # doesn't send a "consuming" signal.
      ...
```

Furthermore, I don't think it would convey the right intention in `__del__` or
`__moveinit__`:

```mojo
struct YourType:
   fn __del__(var self):
      ...
   fn __moveinit__(out self, var existing: Self):
      ...
```

The problem here is that we need some keyword that conveys that the function
"consumes an owned value", which is is the important thing from the caller
perspective.

Other potential names are something like `consuming` (which is what Swift uses)
or `consumes` or `takes`.  I would love suggestions that take into consideration
the above problems.

### Rename the "transfer operator" (`x^`)

This is a minor thing, but the "transfer operator" could use a rebranding
in the documentation.  First, unlike other operators (like `+`) it isn't
tied into a method (like `__add__`): perhaps renaming it to a "sigil"
instead of an "operator" would be warranted.

More importantly though, this operator *makes the specified value able to
be transferred/consumed*, it does not itself transfer the value.  It seems
highly tied into the discussion about what to do with `owned`, but I'm not
sure what to call this.  Perhaps one will flow from the other.
