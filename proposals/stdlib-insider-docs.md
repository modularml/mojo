# Stdlib Insider Docs

Owen Hilyard, Created November 17, 2024

**Status**: Accepted

## Motivation

For most languages, people who work on the standard library have the ability
to look inside of the compiler to clarify questions they have about the
semantics of particular operations or compiler builtins. For Mojo, that is
not the case for many people working on the standard library. As a result,
the exact semantics of some important compiler builtins, such as
`lit.ownership.mark_destroyed`, are not fully known to a large number of
people working on the standard library. For example, the fact that
`lit.ownership.mark_destroyed` still runs the destructors of fields was a
surprise to many at the stdlib meeting. This creates issues where Modular
employees have to catch misuses. These language-internal dialects are, like
Mojo itself, subject to enhancements, breaking changes, and even complete
removal. This presents a problem since the stdlib is correctness-critical code,
and when people who don't understand the API contract of a construct use it
in correctness-critical code, issues are bound to happen.

## Proposal

In order to help address this, I propose the creation of a "stdlib insider"
document, which contains information on MLIR operations/types, WIP features and
other parts of the language which are either intended to only be used in the
standard library/MAX drivers or language features which are still subject to
change and evolution. This can be substantially less polished than the Mojo
manual, including "X is like Y in C++ but ...", links to academic
papers, links to LLVM docs, pseudocode, and other ways one might explain a
concept to a colleague. This document, likely maintained as a markdown file in
/docs, is intended to be internally facing to core Mojo developers. This means
that a large "everything in here is subject to being totally rewritten in a
bugfix release or security update, do not use outside of the standard library
or MAX" warning, which will hopefully dissuade people from using what is
documented there in ways that will get them stuck on old Mojo versions.

For MLIR operations, I'd like the following information documented
for each operation used in the standard library; Operation name,
arguments (potentially as Mojo function syntax), a description of the
operation, pre-conditions, post-conditions, and clear hazards. Clear hazards
would ways in which the operation can cause UB (ex: is what happens with a
null pointer well defined?), conditions under which the operation will force
the program to abort or other like "`lit.ownership.mark_destroyed`
still runs the destructors of fields" which may be surprising behavior.

For MLIR types, information about what a type is intended to do, what
parameters the type has (and their types), the size of the instantiated type
(for alignment), and any non-trivialities in the type (is it ok to copy/move
it?). MLIR attributes should have similar information.

For features, a short description of what the feature is intended for, and
then syntax examples that show the capabilities of the feature. Ideally, some
differentiation between the design and the current implementation should be
present, slowly moving parts of the documentation over as they are available
on nightly. Documenting known sharp edges or limitations is also helpful, for
instance if trait objects could only represent a single trait (ex: no Movable
\+ Copyable + Formattable trait) or if some part of the implementation has a
high time or space complexity (ex: O(N^2) compile time overhead in the number
of traits in a trait object).

## Current State

At present, the majority of MLIR operations are things which I think
are reasonable to explain with a link to LLVM or C++ docs. For example, `pop.max`
is mostly self-explanatory, so unless there are extra semantics I don't
think it really needs more of an explanation than "see C++ std::max".

What I consider the important things to document:

### `pop.external_call`

Lots of people want to call into OpenSSL to get basic HTTPS working, and that
needs to be done correctly. There's also a lot of ABI issues around this,
for instance whether Mojo structs are C layout (for now) or whether we need
a mechanism to force that behavior.

### `co.*`

This area is WIP, but some community discussion around the direction would be
helpful. Information about the API of each of the types would also be nice
since it looks like we would need to use MLIR to implement future combinators
like Rust's `FuturesUnordered`. Documentation around synchronization
requirements is also important for correctness as people move towards async io.

### `#kgen.param.expr<target_get_field`/`#kgen.param.expr<target_has_feature`

Filling out the full target introspection capabilities that are present inside
of the compiler will be fairly labor intensive, but also very useful. If we can
get better documentation on this, the community can help fill out the soup of
x86 CPU among other things. If there is a way to query the data layout
information, that would save us a lot of effort in parts of std that interact
with libc.

### `lit.ownership.mark_destroyed`/`lit.ownership.mark_initialized`

There was already confusion about how this works, and it's used all over the
place in the stdlib, so I consider this critical to document.

### `lit.ref.from_pointer`

Documenting the soundness preconditions would help anyone writing smart
pointers or certain types of collections to use this correctly.

### `pop.call_llvm_intrinsic`/`sys.intrinsics.llvm_intrinsic`

There's a lot of places where this is used, but how to use it is not really
documented. In particular specifying legal Mojo type <-> LLVM type conversions,
Documenting limitations is also helpful, for instance, can is `@call` ok to
use? Can I directly write phi nodes? I had a lot of difficulty interacting
with anything that returns a `!llvm.struct` or `!llvm.vec`.

### `pop.inline_asm`

A lot of CPU functionality doesn't have LLVM intrinsics and we'll need to use
assembly (CPU time stamp counters, poking MSRs for runtime capability
discovery, etc). I personally ran into difficulties doing multiple return
(ex: x86 `cpuid` returns results in 4 registers). Information on the asm
dialect, how to create clobbers and scratch registers, and rules for control
flow (ex: can I implement computed gotos, jump into the middle of a function or
return from the current function?).

### `!lit.origin.set`

This is used in Coroutines (presumably to store capture origins), but it looks
like it may be useful for storing collections of references.

### The backing allocator(s) for `pop.aligned_alloc` and `pop.global_alloc`

Time spent in GDB has led me to believe this is tcmalloc, but knowing for
sure means we have information like the minimum alignment the allocator will
provide (storing information in the lower bits), what kind of caching it does,
information about how it handles memory overcommit being off (db servers), and
what kind of instrumentation we might have access to.
