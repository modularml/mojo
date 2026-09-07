# Generative Kernel Compiler Task List (OLD)

Modular Confidential (obviously)

This document outlines tasks for the implementation work to bring up the
[Generative Kernel Compiler + Language](https://docs.google.com/document/u/1/d/12J0o1z4NgJvsWsi6LsHuGhZUBeRYPrJ9WdLHJCO0nYk/edit).
This document describes the implementation effort in granular chunks. It is
intended to be a working document that we evolve over time.

## Tasks to be implemented

This section contains tasks that have basic scoping and decomposition but that
haven't been fully implemented. When the broad area is complete they should be
moved to the end of the document, in the “Tasks Completed” section.

### ❌ Define a model for the target machine

One nice thing about our approach is that generators are allowed to fail for a
variety of reasons. One key one is that we want them to fail when they are not
relevant to the target hardware, e.g. they’re using a specific VNNI feature but
codegen’ing for a chip that doesn’t have it. We need to have a model that
controls this, including less exotic things like vector length. IMO we should
not allow LLVM to legalize unsupported vector lengths for us, even though it
can.

### ❌ Define more interesting generators

Actual generators for memset and erf: this means we need to implement math, and
the “decision tree found by search” that memset needs, dynamic switches over
dtype, things like scf.for, buffer partitioning operations, etc.

### ❌ Dynamic programming to cache things

As we build things up we will start caring about the execution time of search.
We’ll want to cache kernels and take advantage of hierarchy. This will drive
the need to be able to evaluate microkernels in isolation from the greater
kernel. We may be able to “push down” the input data constraints (e.g.
histogram of lengths) to microkernels but will also want to be able to just
allow users to define their own metrics (e.g. FLOPS for expected dimensions).

## To be scoped

This doc doesn’t include a lot of things that we’ll eventually want to add.

### ✅ Multidimensional tensors

I hear that these are a thing, and so are a lot of other data types with
implicit parallelism (e.g. tables, trees, etc), they will be added later. The
key observation is that the “buffer” type we need for 1D operations isn’t hard
coded into the system. It is a type and set of operations that allow kernel
authors to express algorithms. Once this is proven we can add new types and
operations for more complicated algorithms.

### ⚠️Quantization types + generators support

One nice thing about having an extensible platform is that we can re-discuss
old topics like quantization types and how they get codegen’d, and can play
with models for them. Reimplementing stuff like the Ruy/XNNPACK kernels with
something simple is a good starting point, but is not covered in the milestones
above.

### ⚠️Fancy search algorithms

With good infra, we can explore a wide range of different search and search
space pruning, and black box optimization, and ML and other ways of exploring
search spaces. For now, let’s brute force it to prioritize other infra, and
come back to this when things get more established. We will want to eventually
support multiple pluggable/modular search algorithms.

### ⚠️Higher order generators

Eventually we’ll want generators that can take regions, e.g. a “vectorize” op
that takes a region and width and attempts to vectorize it (e.g. using the
[ISPC approach](https://ispc.github.io/) to SPMD’ify arbitrary code). This will
make it possible to define an algorithm (e.g. erf) in scalar code and product
the simd versions directly from the scalar spec.

This raises some interesting representational questions but also a huge amount
of power and flexibility. Even lowerings like scf.for could be done this way in
principle.

## Tasks completed - for historical reference

This section gives an overview of the major technology components we need to
build, in roughly bottom-up order.

### ✅ A new library/framework in the Modular repo

We need to define a new top-level directory with the usual
include/lib/tools/test structure of any module in the Modular repo. This stuff
generally depends on MLIR and other compiler’y things, but most libraries here
won’t depend on runtime stuff - it generates machine code which integrates with
runtime, and will surely eventually have libraries that depend on runtime
stuff.

The hardest part of this is to decide on a name for the project, something I
deftly dodged in the design doc. Some options:

- One name I considered but didn’t like is “corn” or “popcorn” given it has
  kernels, pro: there is an emoji, con: it sounds dumb.
- Abdul suggest “kgen”, pronounced 🍤 Cajun, contraction of kernel generator.
- Tatiana suggests “kir” kernel intermediate representation

✅ We are going with kgen for now, we can rename it when marketing comes up with
a better name.

This will also end up being a name for a dialect that has container things.

### ✅ Metaprogram parameter infrastructure

One of the key things we need to do is describe both a program and a metaprogram
in the same IR. The “kgen” dialect will therefore need to define the containers
and enough to describe the metaprogram, see e.g.
[some examples in the design doc](https://docs.google.com/document/d/12J0o1z4NgJvsWsi6LsHuGhZUBeRYPrJ9WdLHJCO0nYk/edit#heading=h.9yy1dcksqpx).
CIRCT proved a reasonable implementation approach for this, so I refer to it
below but am open to other better models if they exist
([example CIRCT mlir file](https://github.com/llvm/circt/blob/main/test/Dialect/HW/parameters.mlir)).
This includes things like:

1. ✅A `kgen.generator` operation which is “func like” but which allows
   parameters on it. Relevant prior art is the
   [hw.module op in CIRCT](https://github.com/llvm/circt/blob/main/include/circt/Dialect/HW/HWStructure.td#L52)
   and its ParamDeclArrayAttr abstraction.
2. ✅We need to be able to describe parameter expressions, so it makes sense to
   start with some simple binary expressions and references to parameter
   declarations. Relevant art is
   [ParamDeclAttr](https://github.com/llvm/circt/blob/main/include/circt/Dialect/HW/HWAttributes.td#L130)
   and
   [ParamExprAttr](https://github.com/llvm/circt/blob/main/include/circt/Dialect/HW/HWAttributes.td#L208).
   We need to implement algebraic canonicalization of these (I’m happy with what
   [CIRCT does](https://github.com/llvm/circt/blob/main/lib/Dialect/HW/HWAttributes.cpp#L656),
   it is far more principled than the AffineExpr stuff in MLIR).
3. ✅I am not happy with the printed/parsed syntax of CIRCT param attrs. I never
   got around to sugaring
   “`#hw.param.expr.add<#hw.param.expr.mul<#hw.param.decl.ref<"p1">, 2>, 4>`”
   as something like “`#hw.param.expr<p1*2+4>`”. This is super important to
   get right before we create lots of test cases that use the wrong syntax.
4. ✅We then need the `kgen.param.constant` op in the design doc (the equivalent
   of
   [hw.param.value](https://github.com/llvm/circt/blob/main/include/circt/Dialect/HW/HWMiscOps.td#L71))
   which projects a parameter into an SSA value. This allows using values from
   the metaprogram in the program
5. ✅We need the ability to return parameters, e.g. like `panelDotInner` in the
   whitepaper. CIRCT doesn’t have an analog of this, but it should just be a
   terminator like hw.output but that allows a parameter list as attributes.
   This does bring up a significant representational issue that I’m not sure
   about: in CIRCT all parameters are resolved by looking at the hw.module, but
   now parameters will be defined by other operations in the kgen.generator op.
   This raises some annoying implementation concerns that we’ll have to wrestle
   with.
6. ✅[Issue #983](https://github.com/modularml/modular/issues/983) We need to
   decide what to do with the type system for parameters: I recommend allowing
   parameters with any type, but parameters with no specified types should
   default to being index types with signed interpretation. Use of index
   allows projections into the SSA domain to be architecture independent, and
   dovetails with things like the SCF dialect better. Syntactically, this means
   that these are equivalent:
    1. `kgen.generator @foo<p1, p2, p3>(`... (
    2. `kgen.generator @foo<p1: index, p2: index, p3: index>(`... (
7. ✅[Issue #960](https://github.com/modularml/modular/issues/960) We need a
   `kgen.call` op + verification so generators can invoke other generators.
8. ✅[Issue #966](https://github.com/modularml/modular/issues/966) We need a
   way to declare local parameters and bind values to them, to make the IR
   more expressive and readable. This is relatively nice to have but will
   make kernels much easier to read and maintain over time as we scale.

This infrastructure has a lot of moving pieces and a bunch of subtleties to it,
e.g. the canonicalization of expressions and the pretty printing/parsing logic
needs to be built. Verification logic needs to be put in place early, it will
make all the subsequent work easier.

### ✅ Type system for ‘DType’ parameters

The parameter type system allows parameters of arbitrary MLIR types, but we
need to decide how to handle dtypes, which are very common in parameter lists.
One simple (but yucky) way to handle them is as an `i8` parameter with magic
constants corresponding to `DType`, but this will make reading the IR really
annoying and difficult.

For our own sanity as compiler engineers I think it makes sense to parse and
print them nicely (e.g. `scalar<f32>` instead of `scalar<42>`). To do this, we
need to do the following things:

1. ✅Introduce a new `!kgen.dtype` type.
2. ✅Introduce a new attribute that allows integer initializers in parameter
   lists from 0..255 corresponding directly to their `DType` value.
3. ✅Modify `printParamValue`’s handling of `ParamDeclRefAttr` to notice
   parameters named things like `f32` and print them with double quotes around
   them: “`f32`”. The parser is already set up to handle this. This makes
   them illegal as barewords just like MLIR keywords already are.
4. ✅Introduce special “keywords” for the common types like `f32`, parsing
   them (in `parseParamValue`) like their corresponding `DType` value.

✅ This should allow us to use things like:

```mojo
kgen.generator @foo<t1: !kgen.dtype>(...
```

and:

```mojo
kgen.generate @foo<t1: !kgen.dtype = f32>(...
```

✅[Issue #980](https://github.com/modularml/modular/issues/980) We can further
sugar this by knowing about kgen type contexts, printing this as:

```mojo
kgen.generator @foo<t1: dtype>(...
```

and:

```mojo
kgen.generate @foo<t1: dtype = f32>(...
```

### ✅ New Parameterized Types

Given a parameterization system, we need to build parameterized types,
specifically these to start (take a look at `hw.Int` and `hw.Array` in CIRCT,
which have parameterized widths):

1. ✅[Issue #978](https://github.com/modularml/modular/issues/978)
   `meta.scalar<x>` where x is a `dtype`: represents a concrete scalar type
   like `scalar<f32>` but also things like `scalar<someparameter>` in
   parameterized type contexts.
2. ✅[Issue #1009](https://github.com/modularml/modular/issues/1009)
   `meta.buffer<size, x>` here size is an `index` and x is a dtype, an analog
   of “memref” (I still regret the name “memref” btw :))
3. ✅`meta.simd<size, ty>` where size is an `index` and `ty` is a `dtype`. The
   integer can be an arbitrary parameter expression of course.
4. ✅[Issue #1663](https://github.com/modularml/modular/issues/1663)
   `pop.pointer<T>` for boundless pointer arithmetic.
5. Not initially, but we’ll need other types for nD arrays etc.

In addition to the types themselves, we need supporting infrastructure for
working with them, including:

- ✅[Issue #979](https://github.com/modularml/modular/issues/979) In addition,
  we need some verification infrastructure, e.g. making sure that parameters are
  defined in scope in the kernel etc.
- ✅[Issue #1221](https://github.com/modularml/modular/issues/1221) - Operations
  to cast these to/from concrete types.
- ✅[Issue #1249](https://github.com/modularml/modular/issues/1249) - Need a
  meta.buffer.address operation

Note that we **do not** generally want to define operations that work on
meta.scalar and meta.simd types for arithmetic. These operations should
themselves be kernels.

### ✅ Support for dynamic shapes + dtypes in buffers

✅[Issue #1082](https://github.com/modularml/modular/issues/1082) In addition to
parametric types with known-at-meta-programming time, the meta.buffer type also
needs to support for dynamic size and element type. These can be modeled in
MLIR as null parameters for where they are unknown, for example, we should be
able to write:

```mlir
kgen.generator @fillWithOnes(%dest: !buffer<?, ?>) {}
kgen.generator @knownFloatAlgo(%dest: !buffer<?, f32>) {}
kgen.generator @fixedSizeUnknownAlgo(%dest: !buffer<1024, ?>) {}
```

This should only apply to the `meta.buffer` type. We do not need to (or want
to) support dynamic dtype in scalar or simd, and do not want to support dynamic
sizes in simd.

✅[Issue #1083](https://github.com/modularml/modular/issues/1083) (type and
size), ✅[Issue #1084](https://github.com/modularml/modular/issues/1084) (cast)

We also need to support runtime operations that extract these parameters as SSA
values, and cast one buffer type to another. E.g. something along the lines of:

```mlir
kgen.generator @algo(%dest: !buffer<?, ?>, %other : !buffer<1024, f32>) {
  %dtype = meta.buffer.dtype %dest: !buffer<?,?> // returns !kgen.dtype
  %size = meta.buffer.size %dest: !buffer<?,?>   // returns index
  %buf1 = meta.buffer.cast %dest: !buffer<?,?> to !buffer<?, f32>
  %buf2 = meta.buffer.cast %other: !buffer<1024xf32> to !buffer<?, f32>
}
```

The first one returns an i8 value corresponding to the enums in `DType`. The
latter should return the “index” type, which corresponds to a size_t. These
should all get `fold()`ers for when the parameter value is actually a known
constant integer value. Note that we should not add support for dynamic SIMD
length or dynamic SIMD datatypes. See
[this for rationale](https://github.com/modularml/modular/blob/main/Mojo/docs/compiler/manual/Rationale.md#support-for-dynamic-shapes-in-zapbuffer-et-al).

### ✅ Generator interface declarations and instances

One key idea in the design is that we can have multiple implementations of a
generator interface. This means that we need to be able to separate the
declaration of a generator interface from the implementations of it. We need,
for example:

1. ✅[Issue #1086](https://github.com/modularml/modular/issues/1086) The ability
   to declare a generator interface. This is the `kgen.generator.interface`
   thing mentioned in the whitepaper. Given this, we need `generator`
   declarations to be able to say which ones they are implementing, and we
   need type checking to make sure the declaration and implementation are
   compatible. There isn’t a perfect analog for this CIRCT, the closest is
   [hw.module.generated](https://github.com/llvm/circt/blob/main/include/circt/Dialect/HW/HWStructure.td#L287)
   which refers to a generator declaration, but this is a conceptually different
   thing.
2. ✅Add support for remapping concrete information at call sites through to
   generic things in interface definitions by binding types at the call site to
   generic parameters on the declaration side. This requires being able to pass
   down parameters, requires reifiying now-concrete types with the expectations
   of the generator:

    ```mlir
    kgen.generator.interface @thing<dtype=?>(%dest: !meta.buffer<?xdtype>)
          ...
    %dstCast = meta.buffer.cast %dest to buffer<?xi32>
    meta.call @thing<i32>(%dstCast)
    ```

### ✅ Constraints for Generators + Interfaces

An individual generator is a parameterized function that produces a kernel
based on the inputs, but it may not be a total function: some inputs may be
invalid for the generator. Generator constraints allow defining boolean
conditions that control the validity of the generator w.r.t. its input
parameters. This allows the kernel elaborator to prune a generator from
expansion early, without wasting time exploring it or the recursive expansions
it may kick off.

Generator interfaces should have the same functionality to allow constraining
all implementations for the interface, factoring constraint checking for all
generators.

1. ✅[Issue #1087](https://github.com/modularml/modular/issues/1087) We need to
   add a “constraints” property to generators and generator interfaces that hold
   an array of boolean expressions. Syntactically it can look like this:

    ```mojo
    constraints <in(vecLen, 2,4,8,16,32), notequals(type, bf16)>
    ```

   on kernels in the main design doc. Eventually expression syntax in general
   can [move to infix syntax](https://github.com/modularml/modular/issues/1395)
   (e.g.: `constraints <vecLen` ∈ `{2,4,8,16,32}, type != bf16>`).

2. ✅[Issue #1396](https://github.com/modularml/modular/issues/1396) When we
   have an IR representation for constraints, the elaborator needs to start
   using them to prune generation.
3. ✅Generators failing leads to a new set of challenges with error reporting:
   when we fail to generate /any/ variant of a kernel, we need to report an
   elaboration “stack trace” of why expansion failed. This means we need to
   revamp error diagnoses and tracking.
4. ✅We should support kgen.param.assert as a generalized assertion that works
   against arbitrary parameter expressions, even those that aren’t direct
   generator parameters.

### ✅ Build compiler elaboration algorithm to run the generator

Given the ability to describe things, and given multiple implementations and
basic parameters, we need the compiler infrastructure that walks the tree of
expansions. In time this will be done with search, but initially we should just
**generate all** of the possible implementations of the kernels exhaustively.

This will require handling the order of generator logic, diagnosing cyclic
parameters, being able to clone and walk the tree of generators, etc. Here’s a
sketch of some steps:

1. ✅Implement a new kgen-generate tool that takes an MLIR file containing
   generative kernels and a library and loads them.
2. ✅Check that kernel interfaces in the source mlir file and library file agree
   in signature and other details.
3. ✅Process calls and other operations within the body of a kernel, folding
   away the parameter expressions.
4. ✅Walk the call tree of the generators in SCC order (bottom up), diagnosing
   cycles as errors. This should consider implementations in the library file
   as part of the same graph of kernel generators.
5. ✅Walk the call tree bottom-up generating fully specialized implementations
   of the kernels, dropping them into the target MLIR file (leaving the library
   unmodified). The result of this should be fully specialized and have all
   parameters eliminated. This will generate all possible implementations of
   the kernels.
6. ✅Track bindings for each kernel to keep track of which direction a multiway
   expansion goes for an interface site, to make sure it expands consistently
   within any given kernel.

### ✅ Infra to run a kernel and measure the time

✅[Issue #1610](https://github.com/modularml/modular/issues/1610) (JIT and
execute)

✅[Issue #1611](https://github.com/modularml/modular/issues/1611) (compile to
obj and execute)

We need to wire up full support for invoking LLVM to generate a .o file or JIT
into a buffer, execute the code, and run it. To run it, we need input
generation infrastructure and metadata to know about the expected dtypes.

From there we’ll need the ability to collect realistic data to compare against,
e.g. the mmperf
[matmul dimension list](https://github.com/mmperf/mmperf/blob/main/benchmark_sizes/benchmark_all_sizes.txt)
or the memset/memcpy “histogram of lengths” dataset.

At this point the system will be able to decide which is a GOOD generated
kernel.

### ✅ Design/define/implement various UX and tooling things

✅[Issue #2125](https://github.com/modularml/modular/issues/2125) (Refactor
Elaborator)

✅[Issue #2126](https://github.com/modularml/modular/issues/2126) (Build
über-tool)

✅[Issue #2127](https://github.com/modularml/modular/issues/2127) (Add kernel +
test to Faux)

We will want a live and responsive system which is playful, eventually (long
term) building up to interactive notebook-like experiences for building and
evaluating kernels etc. In the short term, we need to deal with more pedestrian
things like “what happens if there are no expansions for a kernel that work”,
“how do I reason about / chart the performance of all the possible expansions”
etc.

Similarly, while everything can start out as one massive .mlir file, that will
eventually stop being ok.

### ✅ Define library of basic constructs

✅[Issue 1088](https://github.com/modularml/modular/issues/1088) (scalar and
SIMD add), ✅[Issue 1089](https://github.com/modularml/modular/issues/1089)
(buffer add op)

❌[Issue 1090](https://github.com/modularml/modular/issues/1090) (fillWithOnes)

One thing we should do is make sure the entire system is defined in itself, and
reduce the amount of privileged technology. As such, I think that basic scalar
and SIMD operators (like “fadd”) should be themselves defined as kernels in the
language. As Tatiana points out, this is important because our primitives
aren’t just simple scalar things like “add”, they will also be target-specific
aggregate operations like “tile load” and “tile multiply”.

This requires being able to define something like:

```mlir
kgen.generator @fadd<dtype=f32>(%lhs: scalar<f32>, %rhs: scalar<f32>) -> scalar<f32> {
  %lhsc = meta.cast_to_builtin %lhs: scalar<f32> to f32
  %rhsc = meta.cast_to_builtin %lhs: scalar<f32> to f32
  %resc = llvm.fadd %lhsc, %rhsc : f32
  %res = meta.cast_from_builtin %resc : f32 to scalar<f32>
  kgen.return %res
}
```

Building on top of the LLVM dialect is convenient because we can talk to target
specific intrinsics as well as inline assembly as needed. There is some
integration work needed to do this, and we’ll need to evaluate whether arith or
other higher level dialects add value for us.

Pounding this out shouldn’t be too hard given our narrow kernel goals for the
first milestone, but this is effectively “defining the standard library” which
will take some discussion. As a trivial example, do we separate fadd and iadd
given that we’re working with signful dtypes?

### ✅ Add higher level “lit” Dialect

As we start writing more kernels, doing so at a high level will become painful
for a variety of reasons. We want the kgen level to be simple to analyze and
transform - it should support things like the elaborator and static analysis
tools that work on a type checked metaprogram. Achieving this means defining a
higher level dialect that gets desugared down to kgen.

✅[Issue #1410](https://github.com/modularml/modular/issues/1410) When a
generator implements an interface, it should be able do so with a more
specialized signature than the interface has, and use that to infer
constraints. This requires defining a new “`lit.fn`” operation.

✅We need some sort of module system and an “`kgen.include`” operation. This
will raise questions like “how does separate compilation work”, “is the
imported thing kgen or lit level of abstraction, etc.

### ✅ Add a “parametric operations” Dialect “pop”

✅[Issue #1250](https://github.com/modularml/modular/issues/1250) One issue we
have is that these dialects do not support parametric types, and defining
overloads (like the fadd example above) for every integer width will be a huge
pain for us humans, and not be great for compile time. The reason we need to do
this is that the LLVM dialect (for example) doesn’t support parametric types.
We could solve this by adding a “parametric operations” (pop) dialect that
allows things like this:

```mlir
kgen.generator @fadd<dt: dtype>(%lhs: !kgen.scalar<dt>,
                                %rhs: !kgen.scalar<dt>) -> !kgen.scalar<dt>
 constraints type = f32, f64 {
  %res = pop.add %lhs, %rhs : !kgen.scalar<dt>
  kgen.return %res
}
```

The annoyance here is that this requires a significant amount of work for each
dialect we want to use - we don’t want a “parith” or other dialects, so the
general specialization approach above is important to allow us to talk to
anything in the MLIR ecosystem we might want to experiment with. The LLVM
dialect is special and important enough to be worth investing into IMO.

While doing this, we should also reconsider the signless design: Our new pop
dialect could adopt signful types, which could make writing type generic
kernels much easier (e.g. don’t have to write a divs and divu version of the
same kernel).

### ✅ Add support for full type parametricity

✅ [Issue #2504](https://github.com/modularml/modular/issues/2504) Add full
support for type parameters: With the support above, we have the ability to
make scalar and vector operations that are generic over their element type (and
vector length), but we don't have the ability to write kernels that are generic
over both scalar and vector. This is a different form of parametricity which
requires MLIR types as parameter values.
