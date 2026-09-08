# `@always_inline("builtin")`

**Status**: Implemented.

Author: Chris Lattner

Date: January 31, 2025

## Background and Motivation

As Mojo has evolved, we’ve continued to push for a design where builtin
primitives are represented with user-defined types, instead of raw MLIR types
directly. This started back in the day by wrapping `__mlirtype.i1` with `Bool`
and `__mlirtype.index` with `Int` and more recently by wrapping
`__mlirtype.origin<..>` with `Origin`. This approach has a number of
advantages:

1. This allows us to use methods on the library defined types to define
   operators that apply to it, e.g. `__and__` for `Bool` and `Int`.
2. This is much easier to write and works with name lookup properly.
3. This moves MLIR types into being “implementation details of the standard
   library” rather than user-exposed types.

Overall, this has been a great result, part of the “Mojo is syntax sugar for
MLIR”.

However, recent engineering discussions have shown a weakness with this
approach, Stephen recently pointed out:

> One high level Q, I think a lot of the parameter pain is derived from
> primitive structures like Int and Bool being custom types. It means they
> become function calls in the IR and its a struggle to simplify. It seems
> like maybe we should revisit these being in the library and just have a
> small set of compiler builtin types for things like Bool.

It would be very sad to have another parallel universe for “compiler builtin”
types that are neither Int nor the existing MLIR types! That said, he’s got a
really great point, and we’re accumulating a small collection of hacks to work
around the problems.

There are several challenges that occur when these types are used in symbolic
parameter expressions. What is an example? Consider the result type of:

```mojo
def widen[dt: DType, n: Int](a: SIMD[dt, n], b: SIMD[dt, n]) -> SIMD[dt, n+n]:
```

The result type ends up being represented as `SIMD[dt, Int.__add__(n, n)]`.
This expression cannot be folded or parameter inlined at parse time (because `n`
is not a simple constant), so we end up representing the call to `Int.__add__`
symbolicly through elaboration time.

This is nice and flexible and general, a core part of the full-general comptime
model that Mojo provides, but it poises a number of problems for the core types
like `Int` and `Bool` and `Origin` that Stephen is pointing out. A few example
problems:

1. ✅ This causes significant IR bloat that annoys compiler developers because a
   call to `Int.__add__` is far more verbose than a `POC::Add` with its sugared
   form.
2. ✅ Underlying dialects like KGEN have symbolic optimizations for core types.
   In this example, it would canonicalization “n+n” to “2*n” and these canonical
   forms are important to reduce rebinds.
3. ✅ We have complicated and fragile logic to work around this in the case when
   the operands are “simple constants” (see
   `inlineFunctionCallIntoPValueIfPossible` and transitive code it calls,
   *shudder*) which has lots of problems, e.g. it completely fails on
   parametric function calls.
4. ✅ Because that doesn’t work on parametric function calls, we get further
   hacks like `tryOriginInitFold` which special cases `Origin.__init__` because
   Origin is a parametric struct, and we need low-level canonical `!lit.origin`
   values for internal reasons.
5. ✅ We have other hacks like `refineResultValue` . This was added long ago
   when we were first bringing up dependent types and it shouldn’t be needed.
   It serves to handle things like the example below. We want `x` to have type
   `SIMD[dt2, 8]` instead of `SIMD[dt, Int.add(4, 4)]`. The formal type
   maintains the apply expression, so it goes through and does a rebind to get
   it out of the way, reflecting the additional information we have in the call
   site. If we modeled this correctly, this would all happen automatically and
   no rebind would be needed:

   ```mojo
   def example[dt2: DType](a: SIMD[dt2, 4]):
   var x = widen(a, a)
   ```

6. ✅ Generally anything that uses the comptime interpreter at parse time is
   wrong because the IR hasn't been lowered through CF lowering and
   CheckLifetimes. We need to get off of this for dependent type support.

To summarize, Mojo has worked this way for a very long time, but there are too
many design smells adding up to there being a problem. I would love to resolve
this once and for all and I think one simple solution will resolve all this
mess.

## Proposal: `@always_inline("builtin")`

The proposed solution to this is to introduce another level to `@always_inline`
that is “like `"nodebug"` but harder”. The observation is that these types
really are special, the methods that apply to them are trivial and generally
known to the compiler, and the operations within these methods have well-known
magic representations (e.g. `POC::Add` attribute instead of `index.add`
operation). We don’t want to special case the methods themselves into the
compiler, but we do necessarily have a tight coupling and want type checking.

This new level of `always_inline` would have the same behavior as `"nodebug"`
(e.g. get the same LLVM representation, still disable debug info generation etc)
but add two more behaviors.

### Function definition body-resolution checking

The magic behavior we’re looking for has specific limited use-cases and can only
handle specific limited forms. After the function body has been parsed, we need
to validate that it doesn’t use anything that `@always_inline("builtin")` can’t
handle - this checks that there is no control flow, no function calls to
functions that are not themselves `@always_inline("builtin")`, no use of
unsupported MLIR operations, etc. The forms we will be able to accept are
very limited, but that seems like it should be ok given that these methods are
all just wrappers around singular MLIR operations anyway.

### Change CallEmission to do the inlining

These change should be very simple and localized - if `emitCallInParamContext`
returns an apply parameter expression, check to see if the callee is an
`@always_inline("builtin")` function. If so, unconditionally inline the body
into the parameter representation (doing the remapping of operations like
`index.add` to `POC::Add` and form `StructAttr` and `StructExtractAttr` instead
of `StructExtractOp` which will all fold and cancel out).

That’s it.

## Thoughts and Implications

This is something that has been haunting me for quite some time. I think that
this relatively simple extension will subsume and allow us to remove a bunch of
fragile complexity I mentioned before. I believe it should be straight-forward
to support parametric functions, because we’re already in the parameter domain
and we can specifically “just not support” hard cases if they came up.

## Alternative Considered: Try inlining *all* “nodebug” param calls

Weiwei points out that we could avoid adding the syntax for this: we could just
make the call emission logic notice that the call is to a “nodebug” function,
and scan it to see if it can be “parameter inlined” using the rules above.
These are the pros and cons I see of this approach:

- Pro: No new syntax, nothing to explain or document.
- Con: It would be less predictable, you wouldn’t get a compile time error in a
  case that you (i.e. a graph compiler engineer) wants inlined but isn’t getting
  inlined.
- Con: Compile time would be much worse for every non-inlinable parameter call
  (most of comptime).

The compile time cost is induced because the parser would have to “body resolve”
the callee, which forces parsing the callee, type checking and IR generating it.
The vastly most common case will be that something cannot be inlined. The
benefit of the new syntax is we get a decoupling between these two things: we
just need to signature resolve to know the callee is “parameter”, and only then
do we body resolve it to do the inlining.
