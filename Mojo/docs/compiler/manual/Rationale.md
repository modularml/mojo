# KGEN ⚜️: Design Rationale

This file contains design notes and other details about parts of the KGEN, along
with rationale for their design. This is an evolving document that may be
turned into better structured documentation at some point.

## KGEN Parameter design

Note: This describes the MLIR parameter design in the KGEN dialect, not the
source-level Mojo language.

### Parameter syntax notes

Parameters work differently than SSA values in a variety of ways and have their
own little mini-language. To delineate they are special and different, we keep
them in the `<...>` syntax, which gives them a corner of the lexical world that
we know is theirs. This section describes a bit of how they work and why.
Nothing is precious here, we can change this, this just reflects the current
approach.

Individual parameters:

0) First, it is important to understand that MLIR doesn't allow us to do
   contextual lookups to determine the type of a name. Parameters can be
   declared after they are used, and we have a one pass parser.

1) Parameters can have many different MLIR types for future proofness (we might
   want to have string parameters etc) but there will be a high bias towards
   simple integer values (secondarily dtypes will occur, then there will be a
   longer tail). We use the builtin MLIR `index` type as a convenient
   `ssize_t` type for math.

2) We want to reduce syntactic verbosity where reasonably possible, because
   syntactic noise makes it more difficult to write and read IR dumps. In some
   cases, we "know" the type of a parameter expression, for example, in a buffer
   type like `!xyz.buffer<a, b>` we "know" the type of `a` is `index` and the
   type of `b` is `!kgen.dtype`, as such, we don't require their type specifiers
   at all.

3) In cases with take arbitrary types (for example the input list to
   `kgen.call`, the parameter expression in `kgen.param.constant` etc) we allow
   specifying a type with `: type = value` syntax which provides full generality
   for dtypes, strings etc. However, because almost all parameters are of type
   'index', we allow omitting a type with `= value` syntax. Note that an omitted
   type defaults to type `index` - it is not inferred from the initializer value
   (we can't do this for parameter references because of the forward reference
   issue mentioned above).

4) We will eventually have an expression evaluator that does constant folding
   etc, and that will need to have an integer width for the `index`
   computations. We should use the width of the target's pointer size for this
   math, and overflows should be trapped as errors.

Parameter list syntax:

1) In practice, we expect almost all generator parameters to be input
   parameters, not result parameters. As such, it is nice to have ceremony
   free syntax like `<height, width, p1, cacheSize>` for this common case. We
   shouldn't require a result parameter specifier for no reason. We do *allow*
   you to write `<height, width, p1, cacheSize -> ()>` for generality, but the
   IR printer won't generate it.

2) Return parameters follow the argument list and are separated from it with an
   arrow. Like with arguments, we don't need to have parens in the normal case,
   we just use `<vecLen, unrollFactor -> outTileWidth, outTileHeight>` syntax.
   NOTE: We'd like to entirely remove result parameters some day.

3) We need a way to specify cases that use return parameters without arguments,
   and it "looks weird" to have an empty argument list (like
   `< -> outTileWidth, outTileHeight>`). To solve this, we specify an empty
   argument list with empty parentheses, ala
   `<() -> outTileWidth, outTileHeight>`.

This design is a consequence of why you only see parens for empty argument
lists, and why (if you're working on the compiler parser itself) we should
support parens in the result type parser.

## Structure of parameter definitions and uses

The kgen dialect and system is defined in a way that makes it moderately open
for extension, but for that to work, operations need to follow some conventions
for their parameter declarations and uses.

Any operation is allowed to declare new parameters with a `ParamDeclAttr`. This
node contains the `StringAttr` name for the parameter as well as its type. The
key requirement is that `ParamDeclAttr`s may only occur in one place on an
operation: the operation must have them in a `paramDecls` attribute: if present,
that attribute must be an `ArrayAttr` of `ParamDeclAttr`s. This means the
`paramDecls` attribute name is reserved for this purpose in kgen compatible
dialects.

Parameter uses, on the other hand, are far more flexible. Parameters
expressions may occur anywhere in an operation -- including in types of values
referred to or returned by an operation. This allows parameterized types,
allows an open and expressive set of operators that use parameters (for example
to pass to invoked generators, to materialize as SSA values, to return from the
function) etc. There are no limitations on where they occur.

Parameter definitions and uses do not follow the standard dominance structure of
SSA or the MLIR region tree. Instead, their requirement is that operations
that define and use parameter must have *some DAG ordering* that respects the
parameters definitions and uses within a kernel or kernel generator context. By
convention, the location of the operation in the MLIR graph typically
represents an insertion point, not the order of execution of the metaprogram.

## Meta dialect types

### Support for dynamic shapes in `!zap.buffer` et al

NOTE: The details here are obsolete, as these MLIR types were moved into
Mojo library-defined types, but the ideas are still possibly useful.

The kgen infrastructure natively supports kernels that work with dynamic shapes
and dynamic dtypes, currently with the `!zap.buffer<?, ?>` type. This allows
extracting the size/dtype as SSA values, which can then be switched over, or
have other calculations done at runtime. When kgen supports Nd-arrays (tensors)
we will have the equivalent for that. In order to work with dynamic shapes,
we need to be able to extract the only-known-at-runtime values with some
operations that produce SSA values. These are:

```mlir
kgen.generator @algo(%dest: !zap.buffer<?, ?>) {
  // This returns a SSA value of type `!kgen.dtype`.
  %dtype = zap.buffer.dtype %dest: !zap.buffer<?, ?>

  // This returns a SSA value of type `index`.
  %size = zap.buffer.size %dest: !zap.buffer<?, ?>
  ...
}
```

Note that we do *not* support dynamic shapes or dtypes for the `!kgen.scalar` or
`!kgen.simd` types. These may be *parameterized* with arithmetic that determines
the vector length or element, but it may not be dynamic (that is, there is no
`?` allowed) - parameters are always resolved to static values as part of the
code generation process. This is because these are register-equivalent types,
not memory-equivalent types. In the case of the runtime representation of a
buffer, the size and dtype doesn't affect how the buffer value itself is
codegen'd: it is always a tuple of `{void*, numElements, dtype}` at runtime.

Because the SIMD/scalar types do not support dynamic shapes or dtypes, they also
do not need operations like `pop.simd.size`. For any SIMD type, you either have
an integer constant in the IR or a parameter expression. You can materialize
either of these into an SSA value with `kgen.param.constant`:

```mlir
kgen.generator @algo<veclen, dt: dtype>(%src: !kgen.simd<mul(veclen,veclen), dt>) {
  // These do not need to exist!
  %dtypeSSAValue = pop.simd.dtype %src: !kgen.simd<mul(veclen,veclen), dt>
  %veclenSSAValue = pop.simd.size %src: !kgen.simd<mul(veclen,veclen), dt>

  // Use this instead:
  %dtypeSSAValue = kgen.param.constant: dtype = <dt>
  %veclenSSAValue = kgen.param.constant = <mul(veclen,veclen)>
}
```

## "pop" dialect design

The `pop` dialect solves two problems for KGEN:

1) It enables the definition of parametric operations (pre-elaboration) that can
   be generated by a front end parser. The elaborator then resolves these to
   concrete values that exist post-elaboration.
2) The post-elaboration IR is serialized to IR and can be used as a distributed
   code IR (e.g. sent over a wire and executed remotely) and to enable tooling.
   This requires it to be sufficiently high level that doesn't expose target
   specific information (e.g. ABIs) unnecessarily, and also means that we want
   to capture information needed by tooling (e.g. header file printing wants to
   know whether integers are signed or not).

This section captures other specific design points that may be surprising about
its design and why.

## Naming Convention

### Avoid Identical Names

For concepts that need to be represented in multiple layers due to nuances in
each layer (e.g. functions), the goal is to avoid giving them identical names to
avoid the ambiguity in code and documentation (without needing to refer to
namespaces each time).

When possible, use different names to better reflect the semantic nuances of the
layers they belong to. If the same name must be used, the adopted convention is
to use shorter names (more abbreviation) for higher level representations, and
longer names (less abbreviation) for lower level representations when possible.
This is analogous to how concepts get more concrete as they are lowered.

For example, a "function type" needs to exist in both LIT & KGEN. Following this
convention, we assign:

- LIT: "fn" (e.g. `FnType`, `lit.fn`)
- KGEN: "func" (e.g. `FuncType`, `kgen.func`)

This also avoids running into "FunctionType", which MLIR already took as a
builtin.

These names also show up in other contexts, e.g. `FnOp` is a function op in LIT,
while `FuncOp` is a function op in KGEN.

### Abbreviations for Core Concepts

For core concepts of a dialect, which are likely to appear in many places across
code & IR,adopting abbreviated names is recommended when unambiguous, and when
it improves code clarity and IR readability.
The abbreviation should be used consistently everywhere in code & IR to avoid
any confusion between the abbreviated version and the non-abbreviated version.

For example, a "generator" is a central concept of KGEN, thus the guideline is
to use the abbreviated "gen" term everywhere in code & IR.

Once an abbreviation is adopted, it must only stand for one singular concept.
For example, "param" is used in place of "parameter". This means "TypeParam"
stands for "type parameter", and "ParamType" stands for "parameter type".
A "*parameterized* type" should therefore *not* be abbreviated to "ParamType"
(instead, it is named "generator type", or "GenType").

## Why `DType`?

KGEN uses `DType` to represent fundamental primitive types: integers and floats.
But `DType` is from the ML domain and can represent other data types as well:
complex, strings, ragged tensors, tables, etc. This presents KGEN with a
problem: `DType` can represent data types that are beyond the understanding of
the compiler. It is not clear what `scalar<string>` would mean, for example.

We could replace `DType` with MLIR builtin types, but this makes a mess out of
the parameter system, because then everything would have to be parameterized
with `!kgen.anytype`, which can be any MLIR type! We can introduce a new enum
that consists only of supported element types for scalars and vectors, but this
adds unnecessary friction with an enum conversion.

More fundamentally, it's not clear that KGEN should prescribe a set of
"supported" data types. `TF32` and `TF64` are not supported on all systems (e.g.
CPUs), for example, so even a more restricted enum would not solve this problem;
there will always be unsupported data types.

This means that no IR can guarantee that it is valid on all platforms. Some
functions will only discover this during lowering, and that's OK. The elaborator
can handle lowering failures, for example, when compiling an interface
implementation that uses AVX instructions, and reject the candidate. In that
sense, `scalar<invalid>` or `simd<8, invalid>` make sense, in that these are
types that are unsupported on all platforms.

## `kgen.call` Syntax

Syntax of `kgen.call` might look intimidating. Let's break it down to small
pieces to make it easier to comprehend.

On the high level, a `kgen.call` has the following syntax:

```mlir
  kgen.call @FUNCTION_NAME <PARAMETERS..> (ARGUMENTS..) : FUNCTION_SIGNATURE [REGIONS..]
```

Let's now dive deep into each of these components.

### Meta Parameter Bindings

There are can be multiple meta parameter bindings separated by comma, and each
meta parameter binding has the following syntax:

```mlir
  PARAMETER := NAME : TYPE = VALUE
```

For example,

```mlir
  ty : type = !kgen.simd<8, f32>
```

or

```mlir
  fn : (!kgen.simd<4, si32>) -> !kgen.simd<4, f32> = @my_function
```

The type can be omitted, in which case it will default to `index`.

### Arguments

Arguments are input SSA values passed to the call.

### Function Signature

Function signature specifies types of arguments and the return value and has
the following syntax:

```mlir
  (!kgen.simd<4, si32>) -> !kgen.simd<4, f32>
```

### Regions

Lastly, the call instruction might have regions (sort of lambda functions)
attached to it. Definitions of regions look similarly to definitions of usual
functions, except:

- we do not specify return value type for regions,
- the name of the region should match the name of the corresponding parameter.

For example, a region might look like the following:

```mlir
    fn(%arg0: !kgen.simd<4, si32>) {
      ...
      kgen.return %res : !kgen.simd<4, si32>
    }
```

### Putting It All Together

In the end, when all these components are combined, `kgen.call` syntax look
like the following:

```mlir
kgen.generator @foo
    <size, dt: dtype, fn: <size, dt: dtype>(!kgen.simd<size, dt>) -> !kgen.simd<size, dt>>
    (%arg: !kgen.simd<size, dt>) -> !kgen.simd<size, dt> {
  %x = kgen.call_param[<size, dt: dtype>(!kgen.simd<size, dt>) -> !kgen.simd<size, dt>: fn]<size = size, dt: dtype = dt>(%arg)
  kgen.return %x : !kgen.simd<size, dt>
}

kgen.generator @bar() {
  %a = kgen.param.constant: simd<4, f32> = <<"2.0", "2.0", "2.0", "2.0">>

  // Call example 1
  kgen.call @foo
    // Parameters:
    <size = 4, dt : dtype = f32, fn : <size, dt: dtype>(!kgen.simd<size, dt>) -> !kgen.simd<size, dt> = region>
    // Input arguments:
    (%a) :
    // Function signature:
    (!kgen.simd<4, f32>) -> !kgen.simd<4, f32>
    // Regions:
    fn<size, dt: dtype>(%arg: !kgen.simd<size, dt>) {
      %res = pop.mul %arg, %arg : !kgen.simd<size, dt>
      kgen.return %res : !kgen.simd<size, dt>
    }

  %b = kgen.param.constant: simd<8, si32> = <<3, 3, ...>>

  // Call example 2
  kgen.call @foo
    // Parameters:
    <size = 8, dt : dtype = si32, fn : <size, dt: dtype>(!kgen.simd<size, dt>) -> !kgen.simd<size, dt> = region>
    // Input arguments:
    (%b) :
    // Function signature:
    (!kgen.simd<8, si32>) -> !kgen.simd<8, si32>
    // Regions:
    fn<size, dt: dtype>(%arg: !kgen.simd<size, dt>) {
      %res = pop.add %arg, %arg : !kgen.simd<size, dt>
      kgen.return %res : !kgen.simd<size, dt>
    }
  kgen.return
}
```

If we elaborate this IR, we would get the following output:

```sh
kgen-opt test.mlir -elaborate-generators="search-path=/Path/to/kgen-elaborate"
```

```mlir
module {
  kgen.func @"foo,size=4,dt=f32,fn=bar_concrete_region_0"(%arg0: !kgen.simd<4, f32>) -> !kgen.simd<4, f32> {
    %0 = pop.mul %arg0, %arg0 : !kgen.simd<4, f32>
    kgen.return %0 : !kgen.simd<4, f32>
  }
  kgen.func @"foo,size=8,dt=si32,fn=bar_concrete_region_1"(%arg0: !kgen.simd<8, si32>) -> !kgen.simd<8, si32> {
    %0 = pop.add %arg0, %arg0 : !kgen.simd<8, si32>
    kgen.return %0 : !kgen.simd<8, si32>
  }
  kgen.func @bar() {
    %cst = kgen.param.constant: simd<4, f32> = <<"2.0", "2.0", ...>>
    %0 = kgen.call @"foo,size=4,dt=f32,fn=bar_concrete_region_0"(%cst) : (!kgen.simd<4, f32>) -> !kgen.simd<4, f32>
    %cst_0 = kgen.param.constant: simd<8, si32> = <<3, 3, ...>>
    %1 = kgen.call @"foo,size=8,dt=si32,fn=bar_concrete_region_1"(%cst_0) : (!kgen.simd<8, si32>) -> !kgen.simd<8, si32>
    kgen.return
  }
}
```
