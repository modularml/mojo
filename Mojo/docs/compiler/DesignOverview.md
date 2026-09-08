# Generative Kernel Compiler Design Overview

May 14, 2022, Chris Lattner

NOTE: This is a very early design doc written before Mojo was even conceived.
It is retained unchanged for historical significance and inspiration, but
shouldn't be considered a design doc for the current system.

## Introduction

"Kernel libraries" provide high performance implementations of numeric and data
processing algorithms, optimized to take advantage of vectors, multi-core,
dedicated hardware blocks, and dedicated accelerators available on modern
computers. The creation of these kernels is a well studied area in computer
science, and modern machine learning frameworks are typically collections of
hundreds-to-thousands of these kernels - amalgamating kernels from many
different sources into a consistent API that is supplemented with gradient
calculation and other higher level features. These kernel libraries are
responsible for the early success of frameworks like TensorFlow and PyTorch,
and provide a reasonably "hackable" interface to extend these frameworks: this
extensibility has successfully enabled novel research as well as integration
into existing legacy applications (important for data loading).

Unfortunately, kernel libraries have well known scalability challenges:
generating a high performance numeric kernel is difficult, and takes rare
experts significant time to build. Furthermore, these libraries quickly grow to
include thousands of operators - this makes it difficult to bring up new
hardware. Indeed, these libraries typically do not get re-tuned for new
generations of hardware: kernel libraries grow in scope, but who has time to
retune all the operators for an AMD GPU or the new ARM datacenter CPU that just
came out?

Compiler engineers have stepped up to try to solve this problem, notably with
systems like XLA and TVM. XLA focuses on a small subset of the problem (dense
linear algebra operators with static shapes) with a closed operator set with
human intensive "intelligent design" of compiler heuristics. That said, XLA has
successfully shown that aggressive kernel fusion, data layout optimizations,
and extremely challenging accelerators are within reach of modern ML compilers.
TVM extends ideas from Halide to provide a more flexible and extensible
architecture, replacing many of these heuristics with search, providing a
different design point between the kernel author and computer.

Unfortunately, neither of these systems have stepped up to solve the whole
problem of kernel authoring: XLA is intentionally limited to the expressivity
of HLO and its architecture prevents it from expanding to a broader operator
set. TVM is "research quality", with very slow compile times, unpredictable
performance, a somewhat expert-only design aesthetic, and poor support for
modern accelerators with spatial (2D/3D) hardware. Further, neither effort has
generalized to non-tensor data types, even though there are many
parallelization opportunities available on tree based and tabular data
structures.

We propose building a next-generation system that breaks down these barriers,
establishing a new point in the design space. One that celebrates a human
expert’s ability to reason about kernel architecture and numeric precision,
while benefiting from a computer’s powerful attention to detail, and benefiting
from a large world-wide team of collaborators that we hope emerges to build on
our platform. We’d like to embrace the reality that successful kernel libraries
grow organically into huge libraries over time, and do so with scalability,
modularity, user experience, and the goal of "working in the real world" as
center-points to our approach.

One note: our goal is not "_academic novelty_". We are looking to design and
engineer a practical and useful system: reuse of well known techniques is to be
celebrated, not scorned.

### High Level Goals

This is a long-term project that is looking to make progress on a well-studied
problem. This effort will consume significant resources over time, so we should
make sure to keep a North Star in mind that will make it worthwhile. In no
particular order, we would like to demonstrate:

- **Replace existing tech**: There are a wide range of existing ML frameworks
and kernel libraries out there, which come with their own oddities, historical
mistakes, and non-orthogonal behavior. We need to provide drop-in replacements
for their behavior despite this. While we expect to define a cleaner world in
the future, we should embrace existing systems first.

- **Performance**: Generated kernels shouldn’t be used until they meet or
exceed existing expert tuned kernels in apples-to-apples comparisons. We should
not depend on novel kernel fusion or specialization to achieve performance,
they should enhance performance beyond the baseline.

- **HW generality**: While we may focus on CPUs to ground the initial bring-up
work, dedicated HW blocks and accelerators are inherent to the ML domain, we
need to be able to express arbitrary accelerator features and utilize them
effectively.

- **Modularity + reuse**: Supporting the cross product between a wide range of
hardware and a wide range of kernels is only possible with a massive amount of
reuse across macro architectures. For example, many architectures benefit from
the
[im2col transform](https://towardsdatascience.com/how-are-convolutions-actually-performed-under-the-hood-226523ce7fbf):
it should be expressed in one place, not in the implementation of convolution
for many architectures.

- **Velocity**: The time to bring up and tune a new architecture or variant
should be proportional to the difference between the new architecture and
already-supported ones, not proportional to the size of the kernel library.

- **Generality**: Tensors are not enough! We should grow to support many
parallel data structures beyond dense tensors, e.g. trees and tabular data, and
of course sparse and ragged tensors.

- **Open extensibility**: It is important that "users" can build on and extend
our system without hacking the compiler’s source code. We don’t expect the
framework and all the kernels to be open source, but users will want to invent
new algorithms, experiment with new numerics, and access low-level hardware
features.

- **Progressive disclosure of complexity**: we want the kernel authoring system
to be reasonably easy to learn for high-level kernel design, but we want future
experts at HW partners to be able to fully exploit their architecture.

- **Embrace assembly blobs and C++**: we are building a pragmatic system and
want people to be able to "get stuff done" even when the underlying system
isn’t cooperative. We also need to integrate with legacy systems and suboptimal
backend compilers. We will surely need to support hacks for benchmarks, and
handwritten microkernels for inner loops and weird accelerators.

- **AOT + JIT**: JIT compilers are great for generality and flexibility, but
not all targets (e.g. iOS) allow JIT compilers, and others (e.g. embedded
platforms) cannot support them due to lack of runtime or other considerations.
Both AOT and JIT compilation are important.

- **Fast compiles**: While some scenarios may allow extreme compile times, the
normal usage scenarios should be real time. We should develop the system with
the assumption that we need to compile a model in seconds, not hours or days.
While search is important for what we are doing, our goals imply that we cannot
use it for everything.

- **Sub-settable**: the operator set should not be monolithic, we should be
able to deploy different subsets of kernels for different products. We should
also eventually be able to slice out unnecessary `dtypes`, e.g. complex number
support, f64, etc.

There is a lot here, and it will take time to achieve all of this, but we
believe a clean and extensible architecture will allow us to scale to this over
time.

### High Level Architecture

We expect to build the following major subsystems from the bottom up. This is
listed in rough chronological order in terms of when we should kick off the
various pieces. While the focus of this document is on the first point, we
lightly touch on the other pieces in this document as well:

1. **Kernel generation**: The first phase is to build a framework to code
generate kernels that operate on memory buffers, starting with scalar memory
operations, then 1D memory arrays, building up to tensor buffers (bringing in
layout considerations), and eventually generalizing to more exotic user defined
data structures. Kernels at this level of abstraction can directly use C/C++,
assembly, and intrinsics for exotic hardware features.

2. **Library of kernel components**: Given the ability to describe buffer-level
kernel generators, we will start building a large number of these, proving the
design and replacing the legacy kernels we have inherited from the systems we
work with. These components will be modular and reusable, including core
algorithms like memory fills, reductions, element wise operators, etc in
addition to more specialized primitives used in quantized kernels and other
domains.

3. **Tools and features**: Given a new framework that allows describing a wide
range of kernels, we will grow a correspondingly wide range of tools and
utilities that work with this to ensure we maintain a single source of truth.
For example, we can derive shape functions from slicing the value-semantic
operator descriptions, and backwards versions of operators can be generated
from buffer level abstractions in many cases. Many other simpler tables will
also be useful, e.g. determination of whether an op has side effects, which
dtypes it supports, etc. Good infra will allow many other tools, e.g. using
formal methods to compare equivalence between multiple implementations of the
same operators.

4. **Value semantic op aggregations**: Many operators in machine learning
frameworks are most naturally defined as a fusion of several other lower level
operators, e.g. broadcast, activation functions, and sometimes even larger
fused amalgams like LSTM operations. Describing operations at this level of
abstraction has a lot of precedent, and simplifies high level optimizations,
extraction of shape functions, and generation of operator gradients. This is
only touched on later in this document, it isn’t explored in depth.

5. **New kernel language**: Our design is based on a declarative model with
explicit search-enabled meta programming features. Initially it will be
important to anchor the design work on determining the low level semantics,
type system, and features we need to build into the system: we can handle this
by writing MLIR directly and with a C++ EDSL. In time, we are likely to develop
a full-fledged programming language (or extend an existing one if that can
achieve our goals) that exposes this model in a way that can be grok’d by
people outside the Modular team. This is a lot of work, but will dramatically
differentiate our work from Halide/TVM and even CUDA, and provides a clear
"user supportable API" separate from our core compiler data structures.

There is a lot of ground to cover here, so we will discuss each of these in
their own macro section below, and intersperse discussion about tooling. As is
typical, we need to focus on building things the "right way" from the bottom
up—we shouldn’t race to build out 5% of each subsystem ahead of its natural
time.

## "Buffer Level" Kernel generation

The first major technology component we need to build is a system that allows
high productivity (e.g. high reuse) when defining kernel libraries. There is a
lot of prior art in this field, and many different angles on which we could
approach this. Let’s start by defining an approach and set of goals that we can
use as a North Star.

### Approach: Synthesize existing kernel libraries

While other frameworks have focused on targeting the "HLO subset of ML" or
started with fusion of element-wise tensor operators, we propose a different
approach: build a system that can **generate** implementations of existing
operators in existing ML frameworks (TFLite, TF, ONNX, PyTorch, etc). Operators
in existing ML frameworks have a lot of weird things that element wise
operators and HLO don’t, including:

1. Broadcasting and type promotion support.

2. "Hand fused" operators chosen by "experts" that are known to be important to
certain classes of models, e..g activation functions fused into element wise
operators like "add".

3. Support for quantized algorithms that depend on architecture-specific DSP
operations.

4. Layouts assumed by existing frameworks.

5. Support for dynamic shapes and dynamic dtypes.`

Why would we start with this? The most important thing is that it keeps us
grounded: we can always A/B compare our codegen against the existing
hand-written code. We won’t be able to declare success without meeting and
beating the existing hand-written kernel libraries. This approach also forces
us to design for a certain amount of generality, which will help make sure we
generalize to more complex situations in the future.

One nice thing to observe here is that we have a great carrot to encourage us
to do this work: success means that we can delete the legacy code that we
started with when we can show that we’re consistently better than it. As we
implement this, we will build in the ability to specialize on one or more
aspects of the design, including target architecture (+ µarch), desired
operator set, supported data types, shapes, or anything else in the future.

### Defining Terminology

It became too hard to write this paper without forward references to concepts,
so here is a high level definition of key terms and concepts, which are
explained in more detail in subsequent sections:

- **Kernel** vs **Microkernel** vs **Function**: these are an implementation of
an algorithm that does computation against memory objects like memory buffers
of a certain layout. These terms may be used interchangeably (and shouldn’t
have an implementation difference in our system), but "microkernel" tends to
connote a small operation (e.g. memset, dot product or reduction) within a
larger operator kernel implementation. Algorithmically
interchangeable/equivalent/replaceable kernels are sometimes referred to as
"codelets" in literature.

- Function **Generator**: a program (usually expressed in MLIR form) that is
(in general) parameterized and is executed to generate a non-parametric
implementation of a function. Fixed implementations (e.g. a panel dot product
implemented in assembly) are just a degenerate case of a generator with no
parameters.

- Generator **Interface Declaration**: microkernels are (in general)
implemented multiple times in multiple different ways. An interface declaration
can stand alone from the implementations, allowing clients and implementations
to be type checked.

- Generator **Parameter Arguments**: the generator is (in general) a meta
program embedded in MLIR that generates a function, and "parameters" are the
values that this meta program is allowed to act on. These parameters are not
SSA values in MLIR, they are encoded into MLIR attributes.

- Generator **Parameter Results** - These values are returned by a generator to
its invoker as parameters, allowing them to adapt to behavior in the generated
sub-kernel. For example, a panel dot product generator could return "I
processed a 3x5 panel of memory", which causes the invoking for loop to step by
3 and 5 on each dimension.

- Generator **Constraints** - Generators are allowed to be _[partial
functions](https://en.wikipedia.org/wiki/Partial_function)_ from the interface
declaration to a concrete implementation. Constraints indicate limitations on
their parameters, e.g. "this implementation only works with dtype=float32", or
"this only works on machines with X86 VNNI extension", this "works for sizes
modulo 128" etc. Constraints will eventually be upward propagated from kernel
implementations out to the operator graph (XLA-style).

- **Generator/Function Arguments** - These are SSA argument values in MLIR,
used for three things: 1) Buffers and other user defined types for structured
abstractions over memory, like linear memory, N-dimensional tensors with
layouts, and eventually other higher level data types like trees and tables. 2)
The values corresponding to op attributes in the tensor graph level. While they
may be modeled as constants there, they are dynamic values for the runtime
implementation of the kernel. 3) Very small micro kernels at the bottom of the
stack (e.g. add two integers) use arguments for their inputs.

- **Generator/Function Results**- These are SSA result values in MLIR, used for
two things: 1) dynamically allocated result buffers, e.g. those that have data
dependent shapes. 2) Very small micro kernels at the bottom of the stack (e.g.
add two integers) use results for their outputs.

These concepts are explored more in later sections.

### Existing Operator Kernels have Multi-Level abstractions

Operator kernels in a machine learning framework are surprisingly complicated -
they aren’t just a few "for" loops over an array. Let’s look at a simple
operator like binary element wise "mul" op as implemented by typical ML
frameworks. They typically need to support:

1. **Dynamic shapes!**: Effectively all kernel libraries support dynamic
shapes, though they sometimes have limitations, e.g. TF only supports kernels
up to rank 5 in most cases.

2. **Broadcasting, type promotion**: "mul" is a binary operator, and the two
operands can have different shapes and dtypes. ML frameworks often improve
usability by providing implicit promotion to a common element type, and support
broadcasting of elements. Efficiently implementing these operations is
non-trivial: should this be done into a new buffer before invoking the core
kernel, or done inline in the operation? The answer depends on the op algorithm
in question.

3. **Layout munging**: some frameworks support multiple different layouts, e.g.
row-major and col-major, tiled layouts, etc. When the inputs are in different
formats a conversion may be needed. Some libraries use strides to provide a
common implementation that can work with many different layouts, but strides
aren’t general to [tiled layouts](https://www.tensorflow.org/xla/tiled_layout).

4. **Type dispatch**: standard kernel libraries work on multiple dtypes, which
is only known dynamically at kernel invocation time. This requires the kernel
to dynamically dispatch over the dtype and dispatch to kernels specialized for
many different dtypes. Some dtypes may have special cases, e.g. "complex add"
can be handled by the same code path as "scalar add" (since complex addition is
element wise), but "complex mul" is a completely different algorithm than
"scalar mul".

5. **Thread Tiling**: At the outer level of the type-specific kernel algorithm,
the computation is carved into blocks that can be executed in parallel by
multiple threads. The size of each subunit needs to be determined, and is
generally best evaluated based on hardware characteristics and size of input
data (not based on # available threads).

6. **Cache Tiling**: Within the per-thread computation, the computation is
typically cache blocked, e.g. at the L2 level. The size of the L2 is target
specific. This is particularly important for algorithms that make multiple
passes over the data, less important for element wise operations that have
little reuse.

7. **Per Tile Algorithms**: Within the per-L2 tiles, there are many ways to
implement the core algorithm, including with scalars, vectors, using
prefetches, etc. There are also special cases that are interesting to handle
when broadcasting is handled internally to the kernel, e.g. when the fastest
varying dimension of one operand is broadcasted.

8. **Many microkernels**: Algorithms like matrix multiplication depend on
lower-level operations like memset to clear buffers, panel dot products,
reductions, etc. These "microkernels" are themselves implementable in many
different ways. See
[XNNPack](https://github.com/google/XNNPACK/tree/master/src) or
[BLIS](https://github.com/flame/blis/tree/master/kernels) for many examples.

9. **Macro algorithms**: Many operators have multiple completely different
algorithms for computing the result, e.g. in convolution we see the
[im2col approach](https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/),
direct convolution, [Winograd](https://arxiv.org/abs/1509.09308). Matmul has
many implementations (particularly when quantization and accelerators force
weird data layouts), also including
[Strassen's algorithm](https://proceedings.mlsys.org/paper/2020/hash/8f14e45fceea167a5a36dedd4bea2543-Abstract.html),
etc.

10. **Hardware** targets now frequently have spatial operations (like [Apple
AMX](https://medium.com/swlh/apples-m1-secret-coprocessor-6599492fc1e1) or
[Intel AMX](https://en.wikipedia.org/wiki/Advanced_Matrix_Extensions)) that can
speed up multiple loop nests at a time, e.g. for matrix multiplication and
large element wise blocks. They also have many architectural families that will
want things register-blocked, pipelined, and unrolled differently.

We aspire to build a pile of modular and reusable components that allows us to
describe these levels of abstraction and remix them into other operator
implementations. Getting massive reuse across operator families and across
hardware is the only way we can scale kernel libraries (including implicitly
defined ones with tensor-level operator fusion) to a wide range of hardware.

To do this, we need to up-level from fixed kernels to something flexible and
reusable that the compiler can manipulate.

### Parametric Kernel "Generators"

It isn’t possible for a human to create and maintain all permutations of a
kernel by hand (e.g. for all dtypes, all target machines etc), so they
pervasively turn to meta programming. This meta programming comes in a variety
of forms, for example
[C macros and ifdefs](https://github.com/google/ruy/blob/master/ruy/pack_arm.cc),
XNNPack uses a
[Python generator framework](https://github.com/google/XNNPACK/blob/master/src/f32-gemm/sse-dup.c.in),
XLA uses "[emitters](https://www.tensorflow.org/mlir/xla_gpu_codegen)" written
in C++ against "IRBuilder" compiler APIs, but the most widely used are C++
templates.

Many frameworks (e.g. TFLite, CUTLASS, Eigen and many others) define kernels as
C++ templates, using them to generate many instances of a given kernel, e.g.
specialized for dtype. The fancier of these use partial template specialization
and have proven that many interesting cases can be handled this way. However,
these meta-programs are difficult to write and the meta program is not captured
in the IR (beyond the C++ AST), we only get the _output_ of the meta program in
IR form. This is a key problem that makes search more challenging: **_we must
capture the meta program in our IR!_**

To do this, our kernels are defined as declarative "**kernel generators**"
(written initially in MLIR form, eventually an EDSL or perhaps a kernel
language) that take **kernel parameters** and have arbitrary imperative logic
coded against them that are "burned into" the generated code for a kernel. This
can be used to specialize on things like the dtype, unroll factors, vector
lengths, cache sizes etc. Most parameters have integer type and are bounded by
range (e.g. unroll <= 8 times), a list of valid values (e.g. vector length = 2,
4, 8, 16, 32), and should support enums (e.g. consider dtype), which makes them
searchable. Note that "**Kernel generators**" as a core concept still permit us
to use concrete kernels (e.g. a fixed blob of assembly) since they are a valid
generator with no parameters (or, equally, fully constrained parameters).

Let’s look at a couple simple examples: a kernel may have parameters bound at
its invocation site, e.g. after a dynamic switch on dtype, the next-level down
microkernel is invoked with a dtype parameter bound to a constant value:

```mlir
// This fills a 1D buffer with unknown length but known dtype with ones.
kgen.generator.interface
   @fillWithOnesFixedDType<type: dtype>(%dest: !meta.buffer<?xtype>)

// Fills a 1D buffer with unknown length and unknown dtype with ones.
kgen.generator @fillWithOnes(%dest: !meta.buffer<?x?>) {
  %dtype = meta.buffer.dtype %dest : !meta.buffer<?x?>
  scf.switch %dtype { // dynamic switch
  case f32:
    %dstCast = meta.buffer.cast %dest : !meta.buffer<?x?> to buffer<?xf32>
    kgen.call @fillWithOnesFixedDType<type: dtype = f32>(%dstCast)
  case i32:
    %dstCast = meta.buffer.cast %dest : !meta.buffer<?x?> to buffer<?xi32>
    kgen.call @fillWithOnesFixedDType<type: dtype = i32>(%dstCast)
  case i8:
    %dstCast = meta.buffer.cast %dest : !meta.buffer<?x?> to buffer<?xi8>
    kgen.call @fillWithOnesFixedDType<type: dtype = i8>(%dstCast)
  // ...
  }
}
```

Given a uniform representation for dynamic values as well, we can layer in
value specialization when we statically know things about input arguments to
the kernel generator. For example, if we are generating a specialized version
of this kernel for f32 type, we can constant propagate the `meta.buffer.dtype`
operation and the `scf.switch` op using existing MLIR machinery. We can use
fancier value propagation to propagate sets (e.g. specialize on f32 and i8,
removing other dtypes) if there is a reason to.

A key aspect of parameterized generators is that we will need MLIR types to be
parameterized based on expressions derived from generator parameters. This
isn’t just for types like ‘buffer’, but also for things like SIMD vectors
length/dtype and scalars with parametric type:

```mlir
kgen.generator @fillWithOnes<type: dtype, vecLen>
                            (%dest: !meta.buffer<?xtype>)
  // Better handled as a "simd length" type to avoid repeating in each kernel.
  constraints <vecLen ∈ [2,4,8,16,32]>
{
  %bufferLen = meta.buffer.size %dest : !meta.buffer<?xtype>
  // Parameters are not SSA values, but can be projected into them explicitly.
  %vecLen = kgen.param.constant: index = <vecLen>

  %ones = kgen.call @simd_splat<veclen, type> 1 : !simd<veclen x type>
  scf.for i = 0 ... %bufferLen step %vecLen {
    %simdPtr = meta.buffer.cast %dest[%i] to simd<...>
    kgen.call @simd.store<...>(... %ones -> %simdPtr)
  }

  // Cleanup loop
  %one = kgen.call @scalar_constant<type>(... 1 ...)
  scf.for i = (%bufferLen/%vecLen)*%vecLen ... %bufferLen {
    kgen.call @scalar.store<...>(... %ones -> %dest[%i]...)
  }
}
```

The need to have a parametric IR drives a number of other decisions in the
implementation. For example, we aren’t likely to be able to use the `llvm`
dialect directly. We will likely have to define a "parametric llvm operations"
(`pop`) dialect that is lowered to the `llvm` dialect when the generator is
run.

Kernel generators are partial functions, and they are allowed to fail during
generation time. This just removes a candidate from the set of implementations
that is explored by search. If no implementations are available for an operator
for a given target, then that needs to be solved at a higher level, e.g. by
graph partitioning the accelerator vs host computation.

We will also need **parameter results**, to pass meta-programmed values back up
to the invoker. For example, a panel dot product microkernel is a very common
ingredient used in matrix multiplication implementations (e.g.
[see this blog post](https://ai.facebook.com/blog/qnnpack-open-source-library-for-optimized-mobile-deep-learning/)
for an intro to panel dot product). Panel dot may be implemented in fancy
target-specific ways using low-level vector register blocking, DSP instructions,
and target-specific instructions like AMX:

```mlir
// Return up the height/width of the memory block processed by the kernel.
kgen.generator.interface @panelDotInner<() -> height, width>
 (%src1: !meta.buffer<...>, %src2, %dest /*some buffers*/)

kgen.generator @panelDotFullBuffer(%src1: !meta.buffer<..>, %src2, %dest /*some buffers*/) {
  // Note we refer to parameter results "before they are defined".  See
  // "order of generator evaluation" subsection below.
  %x_step = meta.param.value<tileWidth> : i32
  %y_step = meta.param.value<tileHeight> : i32
  scf.for %i = 0 ... %width step %x_step {
    scf.for %j = 0 ... %height step %y_step {
      kgen.call @panelDotInner<() -> tileHeight, tileWidth>(...)
    }
  }
  // cleanup code.
}
```

The implementation of parameter logic needs some simple type checking logic,
e.g. parameter arguments and parameter results cannot be cyclic.

Obscure technical point: the right way to express these is with MLIR attribute
expressions. This allows types to refer to attribute expressions (necessary for
parametric types) and cleanly partitions them out of the SSA namespace,
providing clarity about what happens at kernel runtime vs kernel generation
time. This design point was explored in the CIRCT project for parametric verilog
(e.g.
[circt/test/Dialect/HW/parameters.mlir](https://github.com/llvm/circt/blob/main/test/Dialect/HW/parameters.mlir))—our
needs are more general but the same basic approach should suffice.

#### Order of generator evaluation

Note in the example above that we’re using the parameter results of the
`panelDotInner` generator invocation lexically before the invocation itself.
These aren’t SSA values, so it isn’t a problem for MLIR given we will implement
this with MLIR attributes... but you may be curious how this works.

The thing to remember here is that the _generated kernel_ is a traditional
imperative program that is (probably) eventually shoved into LLVM for code
generation, but the metaprogram is not. The metaprogram is interpreted by the
compiler framework at kernel generation time, and it does not need to execute
things in the lexical order specified by the kernel. Instead, the position in
the kernel is used to indicate where the generated code (or a call to it) is
inserted related to the other code in the kernel: **it’s an insertion point for
a builder**.

This makes the order of evaluation of the generators quite flexible: the
generator is valid if there is a valid topological ordering to the generator
invocation (thus, cycles are invalid). Consider this example:

```mlir
kgen.generator @something<inParam -> result>(%data: !meta.buffer<128xf32>) {
  kgen.call @subKernel1<intermediate -> result>(%data: buffer<128xf32>)
  kgen.call @subKernel2<inParam -> intermediate>(%data: buffer<128xf32>)
}
```

The eventual IR generated by `@subKernel1` will be executed before the eventual
IR generated by `@subKernel2`, but the generator for `@subKernel2` will be run
before the generator for `@subKernel1` because of the dependence on the
`intermediate` parameter that needs to be computed. Furthermore, in this
example:

```mlir
kgen.generator @something<inParam -> result>(%data: buffer<128xf32>) {
  kgen.call @subKernel1<inParam -> result>(%data: buffer<128xf32>)
  kgen.call @subKernel2<inParam -> result2>(%data: buffer<128xf32>)
  %t1 = meta.param.value : i32 = <result>
  %t2 = meta.param.value : i32 = <result2>
  ...
}
```

There is no dependence between the two generators, so our compiler can go off
and generate them in parallel. This structure (along with the general
tree/forest/DAG structure of our computation) contributes to our compilation
process for kernels having an incredible amount of parallelism that may be
exploited to speed up kernel generation on multicore machines.

### Interface Declarations + Multiple Implementations

The key observation behind this work is that kernels are defined with
**domain-** and **target-specific** **abstractions**. One simple example is the
"panel dot product" microkernel above. This is domain specific (to matrix
multiplications) with lots of details specific to how it is used, and also its
_implementations_ are often widely target specific - you can use parametric
LLVM IR generator to produce them, but will also want to use inline assembly
and implementations using target specific intrinsics. As we discussed above, ML
operators are multilevel with lots of interesting problems at many levels of
abstraction.

It simply doesn’t make sense to hardcode these concepts into the compiler
framework: we should allow kernel authors to declare their own abstractions,
just like you can invent your own abstractions in C++. To support this, our
system allows **declaring interfaces to (micro) kernels**, and supports having
**many different implementations** for each micro-kernel - each of which
implements the common interface. Each kernel may be defined recursively based
on simpler smaller kernels, which can themselves have multiple different
implementations.

The interface declaration stands alone from the implementations, allowing
clients to call into them and type check that the implementations obey the
intended API. This provides type checking, but also a framework in which we can
reason about many different implementations of the same algorithm (typically
with different tradeoffs/constraints, specialized to an architecture etc).

Another nice thing about this approach is that it provides natural ways to
abstract the runtime interfaces and other concerns. For example, we can express
a "parallel for loop" kernel and provide implementations defined in terms of
different runtimes we might want to target, e.g. if we hypothetically want to
target OpenMP instead of AsyncRT.

### Cost Model Directed Search

Given multiple available implementations of each kernel and micro-kernel, we
need to decide which one is "best" for a given target and scenario (dtype, size
class etc). We propose addressing this by allowing (micro)kernel interface
declarations to define a **cost model** that is **optimized by search** (e.g.
find the implementation with the "best achieved FLOPS").

For example, we could have an implementation of a microkernel using scalar
operations, implemented with SIMD operators of multiple different lengths, a
few implemented in inline assembly, and maybe one implemented with Apple AMX.
We expect the system to pick the implementation with the highest throughput for
the current hardware, _empirically_, by measuring it (implementations for
incompatible systems are ignored as infinite cost). We should enable this at
the level of microkernels, not just for whole operator kernels.

This is a simple and well understood thing, but defines away a ton of
complexity in the kernel library, where the human experts typically do the work
and encodes it into the source base for the architectures they care about. That
approach prevents someone from bringing up new hardware or a new
microarchitecture without re-doing that work. That effort is proportional in
scope to the size of the kernel library, and is extremely specialized
knowledge.

There are a few non-trivial sub-projects that will be necessary to enable
search. Notably, we’ll want to build up a large collection of models (Modular
needs to do this anyway of course), which use the operators we care about in
realistic ways. This allows us to collect data about the right tensor input
sizes to measure against (using realistic input dimensions instead of random
ones) similar to the [mmperf](https://github.com/mmperf/mmperf#readme)
"[benchmark sizes](https://github.com/mmperf/mmperf/tree/main/benchmark_sizes)"
lists. We can also collect a profile or weight certain dimensions more heavily
to achieve goals like "prioritize MLPerf performance" or "generate best
possible code for one model", depending on any particular product’s goal.

Given dimensions weighting for top-level operator kernels, we can propagate
them down the tree of expansions into microkernels - for example, a microkernel
that does broadcasting of tensor data into a buffer can be generated knowing
all the most common dimensions being input from the kernel that uses it. If we
choose to emit a kernel like this out-of-line (to reduce code size vs inlining
it) then we can aggregate expected input dimensions from all the different
kernels that call into it.

### Unbound Generator Parameters

Above we describe generator parameters and how they can be bound at invocation
sites, e.g. as we show above with the dynamic dtype switch. However, it should
also be possible to **leave parameters _unspecified_.** These parameters will
be explored and determined by search. I don’t want to have to hard-code into my
kernel the number of iterations that will fit in the L2 cache: the system
should find that for me, and it should be returned as a **parameter result**,
allowing the enclosing generator to tile or parallelize around that.

For example, we may have an element-wise multiply microkernel implemented in
terms of vectors over a 1D block of memory. A loop utilizing one of these low
level operators will increase in FLOPS until the L2 cache is exceeded, at which
point a cache blocked algorithm above will typically be more efficient.
Allowing the kernel to define the metric (e.g. FLOPS) will allow the use of
search to find the right implementation. Top level operator kernels can use
latency as their metric.

Refinement: While most generator parameters (e.g. dtype) will be defined on the
generator _interface_ (and thus common to all implementations) we can also
allow _implementations_ of a kernel generator to have additional parameters as
well (e.g. an ARM implementation of a kernel providing three implementations of
the same function for different microarchitectures). This would be sugar for
"flattening" these parameters as individual different implementations of the
same microkernel.

### The "Tree (or Forest) of Generator Expansions"

We are describing a system with multiple implementations of each micro-kernel,
which are then implemented in terms of other interfaces which may have many
implementations. These expansions form a **tree** of possible expansions, and
when you consider that there are many top level operators in the framework, we
have a **"forest"** of expansions to work with at many levels of abstraction
(Note: there are similarities to generating text from BNF grammars and
[attribute grammars](https://en.wikipedia.org/wiki/Attribute_grammar)).

For example, one can implement a matrix multiplication microkernel with a
three-level for loop, with cache blocking, and with internal L2 tiling. You can
implement it to use target specific dot product operations, and with 2D
operators like in AMX and common accelerators. Each of these can (and should)
be implemented in our system independently of each other, all implementing the
same interface (but typically with constraints, explained below).

This design is inherent to how operators work, but if each level of expansion
has a dozen or more expansions, we will quickly find that this "tree of
expansions" has an exponential number of expansions possible for a single
framework operator. This makes it impractical to search the entire space for a
single kernel, and even more challenging to support an entire ML framework –
particularly when a single framework may have hundreds/thousands of individual
kernels.

There are three major ways to address this explosion:

1. A simple approach is to define human-authored constraints on the kernels to
cut off the search space or guide the exploration. This is what the basic
bounds provide in the parameter declarations: we can go further by having
conditional constraints, e.g. target specific ones.

2. We can use [fancy black box optimization
techniques](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/46180.pdf)
to explore subsets of the search space.

3. We can exploit redundancy in the tree-based structure with dynamic
programming techniques.

We should build all of these in time, but the most important one to get
architecturally right when building the system is to get the structure of the
computation right, which means we should prioritize dynamic programming.

### Dynamic Programming / Caching

[Dynamic programming](https://en.wikipedia.org/wiki/Dynamic_programming) uses
memoization/caching of subproblems to algorithmically improve the performance
of hierarchical tree-based algorithms. In our case, each tree of expansions
will have a lot of common leaves, and a forest will have many many shared
leaves, subtrees, and potentially entire kernels. By allowing our cost model to
be defined at many levels (not just at the top level framework operator), we
get modularity for our searches and can cache the results. The use of dynamic
programming collapses the "expansion tree" into an "**DAG of kernel generator
expansions"**.

Caching has profound repercussions for our system and we need to design and
build the system assuming we will use it. In addition to improving our compile
time, we should be able to own, manage and distribute the cache in various
ways. Some ideas:

1. Host the cache on a cloud service, providing an oracle for Modular AI
customers so they get _offline_ search. This is good for them in that they
don’t need to do full search algorithms on their device. It is good for us
because we get a reason to have analytics on what people are using our stack
for (opt in vs opt out policy would be determined later).

2. This allows us to make the install size for an Mobile ML framework very
small: instead of shipping a typical kernel library with lots of bloated
kernels, ship a JIT compiler that can generate the kernels. You might not want
to do search on the device, so you can either bundle a binary blob with the app
or add logic to download the right kernel parameters for the target hardware
and generate/cache machine code for the kernels at app install time. This is
using the compiler as a "compression scheme" to reduce the download size impact
of the kernel library.

3. We can also take the "most frequently used" results and compile them into a
binary blob, shipping it with our framework. This ensures the most common
things (e.g. all the BERTs) are always a cache hit. Modular would have peak
benchmark performance out of the box.

4. We should be able to extend this system to higher level problems like
operator fusions etc.

The key thing to ensure when we’re building the system is that each level of
kernel generator tree expansion is properly functional (side-effect free), and
the "key" used to look up the computation is encodable in a way we can hash and
lookup (e.g. the key is a blob of serialized MLIR). This is important anyway
for parallelizing the tree compilation (tree/DAGs have a lot of parallelism).

### Region Parameter Arguments

The description above supports adding search to typical kernel constructs like
you’d see in systems like CUTLASS and Eigen, and supports things like the
TFLite approach of "you can have any fused activation as long as it is one of
these five" design points. However, we’d like to go further: enabling kernel
fusion of _arbitrary_ element-wise computation into matrix multiplication, for
example. We can support this by allowing kernel generators to be parameterized
by regions. Regions are just a different form of parameter argument, where a
body of code is passed down and is accessible to meta programming constructs.

Exposing regions as a general feature in our system will have very powerful
effects. It means that things like "switch on dtype" and "statically unroll the
loop based on this parametric expression" can be defined in the system
itself(See, for example, how regions are used in GraphRT to allow us to define
new control flow operators without hacking the runtime), rather than being hard
coded into the system. This is what enables the stack to be user-extensible in
a very powerful way: nothing in the stack is specific to dense linear algebra,
users can build their own library of generators that partition work against
tables of data or trees, talk to their own foreign storage (e.g. databases,
etc).

Parameterized generators also lead to a natural expansion in the expressivity
of the ML operator graph abstractions. Instead of mtfl.conv2d having an enum of
activations, conv can take a region that does elementwise computation on
scalars, allowing arbitrary elementwise operators to be fused in - at the graph
level. Taken to an extreme, this may allow us to implement XLA-style kernel
fusion through graph rewrites which get lowered to generators in a predictable
way.

Some related work includes "[algorithmic
skeletons](https://homepages.inf.ed.ac.uk/mic/Pubs/skeletonbook.pdf)" which
allow describing higher order transformations that enable encoding parallel
patterns in a reusable way: "The implementation task is simplified by the fact
that each skeleton may be considered independently, in contrast to the
monolithic programming interfaces of existing systems at a similar level of
abstraction".

### Generator Constraints

FIXME(clattner) TOWRITE: Kernel **Generator** **Constraints** - Kernel
generators are allowed to be _[partial
functions](https://en.wikipedia.org/wiki/Partial_function)_ from the interface
declaration to a concrete implementation. Constraints indicate limitations on
their parameters, e.g. "this implementation only works with dtype=float32", or
"this only works on machines with X86 VNNI extension", this "works for sizes
modulo 128" etc. Constraints will eventually be upward propagated from kernel
implementations out to the operator graph (XLA-style).

### Using ML within the Compiler

As described above, we can make a lot of progress with brute force search, and
dynamic programming will help us scale. However, in some scenarios we won’t be
able to use search effectively (e.g. because the latency is critical, the
system isn’t quiet enough to measure performance, or the target hardware isn’t
easily available). It would be useful to have an efficient oracle that makes
"good enough" decisions for us in a generalized way without actually running
the code.

Fortunately by this point we’ve built a system that generates and captures a
tremendous amount of data and can even have "importance weights" on the data.
Given this data, we can build some models for kernels that generalize from data
we have seen to handle unknown situations we haven’t. We can supplement the
captured data we collect with randomly synthesized kernels (e.g. novel fusions)
for directed learning. This should allow our system to be extremely efficient
and nice for things we know are important, while also generalizing to new
hardware in a faster and more scalable way than hiring epic numbers of
XLA-style experts.

Note that we don’t necessarily need fancy transformers for these oracles. We
should start by looking at simple things like linear regression, random
forests, etc.

### Implementation Plan

**Note:** these were early thoughts that have since been subsumed into the more
detailed [Generative Kernel Compiler Task List](attic/TaskList.md).

There is a lot to build here, it will likely take awhile to build out the
majority of the functionality we need. That said, simple things will be useful
much sooner than that. I recommend we start by building the basic kernel
generator infrastructure including parameters / parameterized type support,
defining the key data types we need (focusing on dense linear algebra), and
start building some concrete kernels we can measure.

There is a logical path to build the stack out, starting boring but getting
incrementally more interesting over time:

1. We need to support value-level micro-micro-kernel implementations of scalar
operations like "add", "multiply", "horizontal vector dot product" etc. It
makes sense to start implementing this in terms of LLVM dialect and the
intrinsics embedded into it given our CPU focus. This will require us to build
out some support for reasoning about the current target, and infrastructure to
represent and drive kernel generators. This will deliver interesting technical
underpinnings, but will not be interesting to the product team in any way.

2. From there, we can move on to supporting 1D memory operations, e.g. memset,
memcpy, 1D reductions, and element wise ops. These are the building blocks of
more complex algorithms, and will show that "multiple implementations" + search
is working. We should be able to show that our system works on many different
architectures and microarchitectures without much manual effort. This will
allow us to replace element-wise operations in TFLite and could be the subject
of a blog post talking about architectural generality.

3. From there we move up to tensors with fixed memory layout, e.g. what TF and
TFLite (mostly) use. This brings in another level of complexity in terms of
address calculation, multi-dimensional tiling etc, opens the door for
broadcasting support etc. This will allow us to replace many key operators
including matmul and conv, per-axis reductions, transposes, and many others.
This will suddenly be very interesting.

4. The next logical step is to generalize the memory layout approach to support
many different layouts, e.g. column major, strided, XLA/MKL tiled memory
layout, etc. This is important when you start wanting to target aggressive
accelerators with weird memory layouts.

5. We eventually reach the point where we go beyond dense tensors to other data
types, including tabular data, tree based data, sparse operators etc. These
should fit in as a generalization of our existing parameterized tensor, they
just require lots of domain specific kernels and microkernels to be built out
in their own universe of domain abstractions.

Coming back to our need/desire to be "grounded", we can make a lot of progress
by directly porting existing algorithms from existing kernel libraries to our
framework (which will just constantly learn new tricks as it grows). We should
be able to replicate Ruy, XNNPACK, and CUTLASS in a much cleaner framework,
eliminating the
[Python metaprogramming](https://github.com/google/XNNPACK/blob/master/src/f16-dwconv2d-chw/3x3p1-neonfp16arith.c.in)
and C++ template metaprogramming.

### Related work and background reading

There are far too many things in the space of kernel generation to link to.
Instead of linking to usual suspects, here are a few less traditional things to
read up on:

**[Attribute grammars](https://en.wikipedia.org/wiki/Attribute_grammar)** allow
defining recursively expanded structures with "attributes" that are inherited
from parents to children, as well as back up from children to parents and even
among siblings. This is familiar to our tree-based generator approach with
parameters being passed up and down the expansion hierarchy. This work has rich
academic fundamentals and a lot of formalism that we may find useful some day.

**[Algorithmic
skeleton](https://homepages.inf.ed.ac.uk/mic/Pubs/skeletonbook.pdf)**’s are a
little-known body of work that build on alternative implementations of generic
algorithms parameterized by higher order functions to tackle parallel
programming. There are analogies to the kernels we implement, which end up
having similar transformation and manipulation power. [More recent
work](https://www.lift-project.org/publications/2015/steuwer15phdthesis.pdf)
applies the ideas to GPUs for all-pair algorithm and stencil computations.

**Memset/memcpy generation**:
[Nadav Rotem](https://twitter.com/nadavrot/status/1458886590659399680?s=20)
adapted work in a
[blog post by Joe Bialek](https://msrc-blog.microsoft.com/2021/01/11/building-faster-amd64-memset-routines/),
to reimplement memset/memcpy in C, finding optimal branch cuts given a histogram
of length distributions. His
[code is here](https://github.com/nadavrot/memset_benchmark), the output was
eventually incorporated into the
[Meta folly framework](https://github.com/facebook/folly/blob/main/folly/memset.S)
(amusingly as a blob of AVX2 assembly instead of in generic C form). Memset and
memcpy are the smallest interesting 1D kernels, we should totally use this as an
example to show how the same results can be achieved with a more general
framework + search.

**[The Design and Implementation of
FFTW3](https://www.fftw.org/fftw-paper-ieee.pdf)**: Early classical generator
for FFT algorithms that uses search and multiple implementations of kernels to
find a good FFT for a given platform. We should be able to express this sort of
hard-coded algorithm in our framework as a proof point. FFTW also shows that
the generator approach provides generality: efficient codegen for strided and
high dimensionality kernels. It also shows recursive kernel decomposition. It
uses the terminology "planner" to talk about "kernel generator" and "plan" to
talk about a generated "kernel".

## Going beyond Buffer Kernel Code Generation

In addition to code generation of high performance kernels, there is a lot we
can do with kernel descriptions in IR form, a machine analyzable/transformable
format. For example:

- We can extract shape functions for operators by using code slicing to extract
the computation from the kernel description. This ensures we have a single
source of truth for our kernels + shape functions.

- We can derive the "what ops+dtypes are supported by this target" set from the
kernel library statically, and encode that data into a table that is used by
the device graph partitioner (using standard algorithms). This keeps a single
source of truth, instead of redundantly encoding this in the graph partitioner.
This is a small thing, but is the bit of magic that allows you to progressively
implement a few micro kernels for a new target and have the operator set start
lighting up incrementally.

- We can detect "invocation independent computation", e.g. a lookup table that
only depends on known-constant-at-the-graph-level operator attributes. This
computation can be automatically sliced out of the main kernel computation into
a "prepare-like" function that computes the lookup table into a custom struct
at initialization time, rather than computing it every invocation of the
kernel.

- We can implement "kernels generators" with MLIR compiler APIs to provide
things that are fancier than parameterized expansions. We should be able to
encode many halide-esque operators (e.g. vectorize, parallelize, etc) as
compiler transformations and provide a really interesting and flexible
programming model to kernel authors. These are just "generators" that take a
region of IR as a parameter and produce a new one. I’m particularly interested
in exposing an [ISPC-style](https://ispc.github.io/) SPMD transformation as a
generator.

- We can generate backwards versions of kernels automatically, using widely
understood techniques used in Tensor Comprehensions and other systems.

- We can extract metadata about the operations, e.g. whether they are
associative, side effectful, etc.

- We can synthesize versions of the kernels for other considerations, e.g. code
size. This can be useful for constant folding operators within the compiler.

- Given our multiple theoretically identical implementations of the same
algorithms, we can do a lot of testing and correctness checking. The formal
method people would love to crawl all over this to prove various things, find
bugs in implementations, and more.

- We can rapidly experiment with novel numeric techniques, new quantization
approaches, etc.

- Specialization of kernels and operators when static data is known about the
target model is very easy. For example, if a model only uses float32 or int8,
we can strip away all the support for other dtypes, producing a much thinner
kernel library. This can be useful for deployment considerations as well as
reducing instruction cache pressure (improving performance). We can specialize
when shapes are statically known as well.

There is so much tooling that can be built over time, which will be much
simpler than before given an integrated framework.

### Value Semantic Operators

Once the buffer-level of abstraction is mastered, it makes sense to build out
and tackle the value-semantic "tensor operator graph" level of abstraction
using a similar approach and philosophy. I won’t go into this in depth in this
document, but when we tackle this, we will have strong fundamental footing
underneath us that will allow us to re-open the "unifying operator" discussion,
but with a lot more experience handling quantization, experience with
parameterized kernel generators, and with new data types.

There are many things we can explore and tackle:

1. Canonicalizing complex framework-specific operators into simpler framework
agnostic region-parameters operators, ala PyTorch primitives.

2. While our buffer-level kernels will generally be taking output buffers as
arguments, that won’t be exposed into the graph. It may make sense to have a
"buffer exposed" graph-level representation that allows memory planning,
in-place optimizations for concat, etc. We will also need the ability for
kernels to return new tensors as well in time (this is needed in generality for
operators that have data dependent shapes).

3. We can take all the metadata we know about buffer-level operator
implementations and reflect it back up to the operator graph level.

There is a lot that can be said here, more when we get further down the road.

## New Kernel Language

The system proposed above should support broad-base extensibility in a large
number of directions:

1. It is target independent and should scale to CPUs and many accelerators.
2. It is ML framework independent, separating all integration issues out and
   focusing on kernel generation only.
3. It isn’t specific to one memory layout or other narrow set of assumptions.
4. It isn’t ML or dense linear algebra specific, it supports a wide range of
   data types and problem domains. You can use it to build high performance
   audio or data kernels.

That said, it also isn’t magic. It assumes that expert kernel programmers will
help design and define the architecture of the kernels that are generated.
These folks have deep domain expertise including information about numerics and
the target hardware, but they are not necessarily compiler engineers. We would
like these sorts of experts to be able to extend the system without
understanding how the compiler internals work (e.g. they shouldn’t have to
write .mlir files).

Furthermore, we as Modular have another goal: we want our system to be
extensible by customers, partners, and academics without having access to our
compiler source code. This implies that we want a user-supportable surface
area: compiler frameworks have notoriously fragile APIs, and we will want to
continuously evolve our design over time.

Finally, we also care about usability and understandability of the kernel
components in the libraries we generate. This implies that we want to be able
to **holistically design** our system in a vertically integrated way, making
the components feel native with each other.

### Extend Existing Systems or Build New?

The first question we’ll have to address is whether to extend an existing
language implementation or to build a new one. Designing and building a new
language is a significant undertaking, and I virtually always recommend
_against_ new languages, because there are many (many) benefits to reusing or
extending an existing system, including:

1. Embracing knowledge within an existing community, reducing adoption/learning
costs.

2. Leveraging existing training materials, books, blog posts, etc.

3. Reusing existing tooling, including compilers, debuggers, IDEs, source
formatters, etc

4. You get a fully general base language with a proven type system, etc.

5. Integration with legacy code bases (e.g. in the data layer) are easier if
you can just call them.

That said, you need to have a reasonable language to start from. The only
plausible base language to use in our case is C++, which is widely known by us
and the high performance kernel authoring community, and is the basis for CUDA
and others. Unfortunately, C++ has a number of problems:

1. Our language isn’t a traditional imperative design: our core model is
declarative, and that is extended with a specifically designed model for meta
programming. This is fundamentally different from how C++ works.

2. C++’s existing meta programming feature is based around templates and
constexpr. While modern C++ has made some improvements, it is still very user
unfriendly, and compile times for massive Eigen metaprograms are problematic.

3. C++ doesn’t support declaring meta programmed parameters via search.

4. There are no C++ compilers that generate MLIR (sigh).

5. We as Modular AI do benefit from having "our thing" in terms of brand
attachment, ability to move fast, and not being saddled with the decisions of
other committees.

In the immediate term, we can start by building an embedded DSL in C++ (e.g.
like Halide has) or Python - this provides a convenient way to get things off
the ground, allowing us to develop other parts of the stack. If/when we get to
the point where this isn’t working well for us, we can decide what to do about
it. Lazily evaluating this allows us to make the decision with more information
and experience, particularly knowledge of the primary programming models we
need to enable and what pain points need to be solved.

If the negatives of inventing a new language are overwhelmed by the benefits
(thus having proper business justification), then it will be no problem to
bring up something simple quickly, and enable demos. If the language is a big
usability win, it could be useful to help us scale our team and allow us to
move more quickly. A language could be useful to engender excitement about what
we’re doing (differentiating our work from other designs), as nothing causes
more furor in the programming community and tech press than programming
languages. :-)

### Looking ahead to ML modeling APIs

This document focuses on the core compiler, kernel abstractions, and technical
features that allow building and synthesizing operator libraries in a scalable
way. The proposed conclusion of this is a user-extensible hybrid declarative /
imperative programming language that allows expressing arbitrary MLIR operator
graphs in a usable way. It is very important for us to stay focused while
building this technology and meeting users where they are: doing so will allow
us to unlock the potential of a lot of hardware, and unify a fragmented
low-level ML infrastructure world.

However, it is also impossible not to notice that this technology is also
useful when tackling other abstraction levels in the stack - a good way to
define the models themselves is also an unsolved problem. ML frameworks are
also mostly-declarative with imperative constructs, but instead of staging out
CPU instructions, they stage out ML operators into the resultant graph. When
building out our technology stack and eventual language, we should keep an eye
on the programming model side of the equation.

Many people are unhappy with Python for deployability, tooling, and performance
-- an untold amount of money has been spent on various approaches that have
tried to solve these problems for ML, and it is reasonable to conclude that
there isn’t a good solution in reach. When building out our ultimate language
strategy we should be mindful that we want to "scale down" in complexity, e.g.
embracing progressive typing approaches or other things that simplify common
cases in model architecture. Having a single system that scales from low-level
bit-banging to high-level model architecture will radically simplify the stack
and expand the capabilities of many ML developers.

The challenge with entering the ML modeling space will be that a new language
will be "new" (always a bad thing), proprietary, and unknown. We will be served
well by getting broad industry adoption at the kernel authoring layer,
appealing to more technical users who understand hardware and numeric
algorithms. Allowing these experts (many of whom will eventually work for
Modular) to scale up into deployable model authoring will be a natural
progression.

In any case, I don’t think we should ever assume Python in ML will go away. We
need to continue to meet people where they are while providing an optional path
to increased happiness and productivity on our stack.
