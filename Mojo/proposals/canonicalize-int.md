# Canonicalize APIs around `Int`

**Status**: Implemented.

Author: Laszlo Kindrat

Date: December 15, 2025

## Background

The standard library and our kernels use an incoherent mix of `Int`, `UInt`,
`UInt32`, `SIMD`, `Indexer` (and perhaps other types) in their APIs. In some
cases, the APIs are meant to enforce invariants (e.g. some index passed by the
user is not-negative), improve performance (e.g. unsigned division), or they
merely carry the vestiges of past decisions that we have since reconsidered
(e.g. negative indexing). The inconsistencies between different APIs have
resulted in an abundance of (until recently implicit) conversion between `Int`
and `UInt`, unintuitive behavior, bugs, and decreased usability. This proposal
aims to radically simplify these APIs and the standard library types.

## Proposed design

Before we discuss specific design points, we would like to establish the guiding
philosophy for dealing with the design of `Int`, `UInt`, and indexing.

This is best summarized with a quote by Chris Lattner:

> “Mojo is a performance oriented language, we want the defaults to be fast - so
> long as that doesn’t sacrifice our goal to get to eventual memory safety”.

The purpose of `Int`, `UInt`, and more generally `SIMD` is allowing programmers
to extract maximum performance from their code, therefore Mojo cannot afford
excess runtime overhead associated with safety or ergonomics in these types.
Note that this is not to say that we will not try to make these as safe or
ergonomic as possible; we merely posit that the constraints on ergonomics and
safety are strict and dictated by potential performance. It is also not to say
that all Mojo types must follow this philosophy, but more on that later.

This philosophy forces our implementation to be as close to hardware as
possible, and yields two immediate consequences:

- Mojo’s `Int` will not be an arbitrary precision integer, and therefore it is
  not an equivalent of Python’s `int`.
- `Int` and `UInt` operations don’t perform overflow/underflow checks; `UInt`
  will wrap around, while `Int` follows 2’s complement overflow/underflow
  semantics. These checks would be impractical for core numeric types (`Int`,
  `UInt`, or `SIMD`) in performance sensitive code.

Now let us enumerate what this means for the affected types and use cases.

### Indexing / treatment of negative indexes

Simple things should be simple, and what’s simpler than wanting to index into a
list? The following needs to work as expected:

```mojo
var idx = 1
var my_list = [“a”, “foo”, “bar”]
print(my_list[idx])
```

While it’s is possible that in the future Mojo will have special syntax for
different literal types, we feel strongly that the only reasonable default for
`idx` above is `Int`, and therefore core types must be indexable with `Int`.

So, we must now decide what the exact semantics of the indexing (e.g.
`__getitem__`) operation should be. To do that, we need to be explicit about our
scope: this document is concerned with low-level, performance-sensitive types,
such as `InlineArray` and `List`, and as Mojo grows, the Libraries team will
need to decide exactly which types fall into this category. (Ultimately,
indexing semantics are up to the type’s designer, and we understand that
numpy-like array types might make different decisions here.) Our philosophy
above dictates the following consequences:

- Since Mojo’s `Int` is not equivalent to Python’s `int`, we see no reason to
  try to retain negative indexing.
  - Negative indexing would also necessitate runtime checks and therefore result
    in overhead. Explicitly vectorized loads are important, and negative
    indexing would be a significant performance problem for them (as well as for
    auto-vectorization, eventually).
- We want Mojo to be a memory safe language in the future, so we need dynamic
  array bounds checks (which compiler passes can eliminate). We aren’t sure if
  we want them enabled in fully optimized mode at this time, but they should be
  enabled at least in debug mode. We may continue to change the policy on this
  in the future, but these checks need to be efficient and implemented “just so”
  to ensure that the llvm backend can optimize them downstream; performance is
  important even in debug builds.

The consequence of these points is that we will not support negative indexing
from the back of the array with the standard `__getitem__` function. If negative
indexing is found to be important over time, we can develop explicit subscript
operations that enable it.

While this guidance applies to core types like `List`, there are other types
(e.g. `LayoutTensor`) that are address-space aware. These types may want to
enable smaller indices (e.g. 32-bit indices on 64-bit targets) because of their
specialized use-case. `List` does not have this consideration.

### What should `len(x)` return?

Since `Int` is the default integer literal type, and indexing on the core types
will use `Int`, it is natural that `len` should return `Int`. The motivating
simple example is

```mojo
var b = my_list[len(my_list) - 1]
```

Unlike Python, the implementation of `__len__` and `len` will not check if their
result is negative, but it seems reasonable to do a simple check in debug
builds.

We highlight again that user defined types (e.g. types outside the standard
library) can perform runtime checks in their implementation of `__len__`, if
they wish.

### But what about 32bit/16bit/8bit systems?

First, let’s clarify that support of severely constrained systems like
16bit/8bit MCUs is not on any roadmap. Second, there is nothing in the
*language* itself that would prevent supporting these. Third, the standard
library’s job is to be… standard. We don’t have to cater to all applications and
all systems and deliver maximum performance and/or safety.

The only situation where 32bit systems would run into problems using `Int` as an
index type is when a collection is larger than ~2GB, and we want to index into
it (and even then, only if it’s a collection of bytes). This can only happen if
an app has a single array that covers more than half of the available address
space of the process. Such an extreme use-case can use other specialized types
(e.g. specific adapters of `List`) if it arises.

### How does this relate to `SIMD` unification?

`SIMD` unification is a longer term project that aims to remove the dedicated
`Int`, `UInt` and `Bool` types in favor of aliases to corresponding flavors of
`Scalar`. As such, the process of deprecating `Int` and `UInt` would probably
benefit from reduced fragmentation in the libraries. Once our numerical types
are unified in `SIMD`, it will be possible to design clean APIs (for math
helpers, indexing, etc.), that are generic over the `SIMD` types. At this point,
we are not ready to commit to what these would look like or if we even want
them, though.

### What are the consequences on kernels?

However, it is already clear that implicit conversions between various numeric
types need to be carefully designed to avoid the situation we’ve seen with
`Int`/`UInt`. It is likely that signed ↔ unsigned conversions will never be
allowed to happen silently.

To put it concisely, kernel APIs should almost never use `UInt`, and all thread
and block indexing primitives (e.g. `thread_idx.x`) will also be `Int`. It seems
the main reason `UInt` was chosen in many kernels to model indices is to ensure
division and modulo operations happen using unsigned integers. To make this
easier, we can provide wrappers for performing unsigned div/mod on `Int`
arguments.

### Improved safety in compile-time metaprogramming

While we can’t detect overflow and wrapping at runtime in a performant way, we
could provide some additional safety for compile-time computations. This carries
the risk that compile-time and runtime code won’t exactly do the same thing,
because the latter might warn/error on integer overflow. Whether this is a good
tradeoff or not is hard to decide without also understanding the implications of
compilation time.
