---
markdown-notebook-data-directory: mdnb-data/manual-passes-ir/
---

# Passes and Intermediate Representations

The best way to start understanding a compiler is to understand the various
lowering stages, the differences between them,
and which compiler passes make those transformations happen.

The Mojo compiler transforms Mojo code into various intermediate
representations, such as LIT, KGEN, LLVM, and many others.

This doc covers the basics of what our IR looks like, and how the passes
transform from Mojo to all of those.

## MLIR

The Mojo compiler is built on top of MLIR framework where we define
different MLIR dialects to capture IRs at different abstraction level for Mojo.

To see it, put this snippet into a
`main.mojo` file:

```mojo
def foo(arg: Int):
  pass

def main():
  foo(5)
```

and run this command to run the parser/type-checker:

`br //Mojo/tools/kgen-translate -- -import-mojo main.mojo`

The output contains this for the `main` function:

```mlir
lit.fn @”main()”() -> !kgen.none attributes {sourceName = “main”, specialFnKind = 0 : i8} {
  %0 = kgen.param.constant: !Int = <{5}>
  %1 = lit.call @main::@”foo(::Int)”(%0) : !lit.generator<(“arg”: !Int) -> !kgen.none>
  %none = kgen.param.constant: none = <#kgen.none>
  lit.return %none : !kgen.none
  lit.end_fn
}
```

This is in Mojo's MLIR `lit` dialect.
One could think of it as a higher-level abstraction form of IR that reflects
close to logical program flow which will be lowered by the compiler to LLVM IR.

Useful background on MLIR:

- [MLIR Documentation Overview](https://mlir.llvm.org/docs/)
- [MLIR Language Reference](https://mlir.llvm.org/docs/LangRef/)
- [Builtin Dialect](https://mlir.llvm.org/docs/Dialects/Builtin)

## Mojo Dialects

Above, we saw Mojo IR can **contain multiple “dialects”.** in one
compilation unit (MLIR module) at the same time.

Here it is again:

```mlir
lit.fn @”main()”() -> !kgen.none attributes {sourceName = “main”, specialFnKind = 0 : i8} {
  %0 = kgen.param.constant: !Int = <{5}>
  %1 = lit.call @main::@”foo(::Int)”(%0) : !lit.generator<(“arg”: !Int) -> !kgen.none>
  %none = kgen.param.constant: none = <#kgen.none>
  lit.return %none : !kgen.none
  lit.end_fn
}
```

Here we see operations from two dialects (`lit` and `kgen`) working together.

Some things from the `lit` dialect:

- `lit.call` - A mojo function call. (This is an mlir operation).
- `!lit.ref` - A mojo reference type (`!` indicates that this is an mlir type).
- `!lit.trait` - A mojo trait (`!` indicates that this is an mlir type).

Some things from the `kgen` dialect:

- `!kgen.none` - The kgen "none" type (`!` indicates that this is an mlir type).
- `kgen.param.constant` - defines an mlir operation that represents
a constant value.

(We'll cover these, the rest of the `lit`/`kgen` dialects, and other
dialects further below.)

Mojo has several dialects:
[`kgen`](../../../Mojo/include/Mojo/KGENDialect),
[`lit`](../../../Mojo/include/Mojo/LITDialect),
[`pop`](../../../Mojo/include/Mojo/POPDialect),
[`hlcf`](../../../Mojo/include/Mojo/HLCFDialect),
[`interp`](../../../Mojo/include/Mojo/Interpreter/InterpreterDialect.td) and
[`co`](../../../Mojo/include/Mojo/CODialect).

Mojo compiler also uses upstream dialects:
[`index`](https://github.com/llvm/llvm-project/tree/main/mlir/include/mlir/Dialect/Index),
[`llvm`](https://github.com/llvm/llvm-project/tree/main/mlir/include/mlir/Dialect/LLVMIR),
[`nvvm`](https://mlir.llvm.org/docs/Dialects/NVVMDialect), and
[`rocdl`](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/).

[`lit`](../../../Mojo/include/Mojo/LITDialect) is a high-level
dialect to reflect what's most close to
a logical mojo program. It is the IR the parser and
type-checker (semantics check) are working on.
It is quickly lowered to `kgen` once semantics checks pass.
(The `lit` dialect should more properly be named `mojo` perhaps
but currently reflects how “lit” Mojo is 🔥.
Historical reason is because Mojo parser level IR used to be
called lightning (LIT). Mojo doesn't have an AST).

[`kgen`](../../../Mojo/include/Mojo/KGENDialect) defines the
canonical IR for mojo program after semantics check
which can describe both parametric and
concretized (parameters substituted with concrete values) IR.
The dialect defines the mojo parameters as mlir attributes as well as
types, and operations to represent the whole program.

[`pop`](../../../Mojo/include/Mojo/POPDialect), which stands
for “parametric operations”, are parameterized,
target-independent dialect used to represent more algorithm level operations,
attributes and types,
such as `pop.variadic.xxx`, `!pop.array`, `!kgen.simd`, etc.

[`hlcf`](../../../Mojo/include/Mojo/HLCFDialect) is the dialect to describe
higher-level control flow in the form of IR instead of something the compiler
has to extract from the IR (e.g. CFG is an analysis in LLVM). `hlcf` is
non-parametric and target independent, and it exists in both pre-elaboration and
post-elaboration IR.

[`interp`](../../../Mojo/include/Mojo/Interpreter/InterpreterDialect.td)
dialect implements attributes and utilities for communicating with the
interpreter in MLIR. The interpreter is a significant part of the elaborator
where we substitute parameters with concrete values. `interp` is mostly
a dialect for Mojo compiler internal implementation.
Currently `interp` only has attributes, but no operations.

[`co`](https://github.com/modularml/modular/tree/main/Mojo/include/Mojo/CODialect).
dialect defines a parametric operation set for working with async functions
implemented as coroutines with arbitrary suspension points. `co` dialect is also
mostly used for Mojo compiler internal implementation for **co**routines.

[`index`](https://mlir.llvm.org/docs/Dialects/IndexOps/) contains operations
for manipulating values of the builtin `index` type.

[`llvm`](https://github.com/llvm/llvm-project/tree/main/mlir/include/mlir/Dialect/LLVMIR)
dialect is the code-gen target of the MLIR level of the Mojo compiler, i.e. all
mojo dialects (`kgen`, `pop`, `hlcf`, etc.) are lowered to `llvm` at the end of
mojo compiler's MLIR pipeline so that we can then go to the LLVM pipeline. It
serves as a translation layer for mojo compiler to cross from MLIR to LLVM land.

[`nvvm`](https://mlir.llvm.org/docs/Dialects/NVVMDialect),
and [`rocdl`](https://mlir.llvm.org/docs/Dialects/ROCDLDialect) are dialects
modeling public GPU ISAs: `nvvm` for Nvidia GPUs and `rocdl` for AMD GPUs.
Now we only use these two to get GPU specific address spaces at the MLIR level.

In summary:

- `lit` exist pre-elaboration. They are lowered to `kgen` and `pop` before
  elaboration.
- `kgen` contains all the instructions that must be monomorphized (concretized)
  by the elaborator. It is lowered to `llvm` for final code-gen
  into runtime executables.
- `pop` exists pre and post elaboration. Operations in the dialect become
  non-parametric post-elaboration. It is lowered to `llvm` for final code-gen
  into runtime executables.
- `hlcf` exist pre and post elaboration. It is lowered to `llvm` for final
  code-gen into runtime executables.
- `index` exists pre and post elaboration. It is not parametric. It is used to
  to represent operations for builtin `index` type.
- `llvm` mostly exists at the bottom of Mojo MLIR compilation pipeline. It
  serves as the bridge between MLIR and LLVM. Mojo library can create `llvm` ops
  using `__mlir_op` syntax, but these operations will not be processed by Mojo
  MLIR passes.
- `nvvm` and `rocdl` are dialects to model public GPU ISAs. They are both only
  used to specify GPU specific address space. They are not parametric.

## MLIR Guide

Ground truth for MLIR:
[MLIR language reference](https://mlir.llvm.org/docs/LangRef)

- `%name` — A mojo run-time value; an MLIR SSA value; the result of an MLIR
  operation.
- `@name` — A
  [SymbolRefAttr](https://mlir.llvm.org/docs/Dialects/Builtin/#symbolrefattr)
- `!type` — An MLIR type.
- `#expr` or `{expr}` — an MLIR attribute.
Mojo parameters are represented as mlir attributes in the IR.
MLIR attributes can also be used to represent other compile time values that are
otherwise not operations or types, such as some metadata
for the operations and types.

- Anything else is an MLIR **operation.**

It's important to know these, because `!kgen.none` and `#kgen.none` are very
different things.

All of them are explained in the next sections.

For now, forget about parameters (`#expr`), and pretend they only come
into play with generics are involved.
Let's start with a non-parametric program that just uses
run-time values, operations, and concrete types.

### Operations, values, types and attributes

MLIR is fundamentally based on a graph-like
data structure of nodes, called Operations,
and edges, called Values.
Each Value is the result of exactly one Operation or Block Argument,
and has a Value Type defined by the type system.

For example, this program:

```mojo
def main():
    var x = 42
```

...when run through the parser, gives this MLIR:

```mlir
lit.fn @"main()"() -> !kgen.none attributes {sourceName = "main", specialFnKind = 0 : i8} {
  %x = lit.var.decl "x" var : !lit.ref<!Int, mut *"x`">
  %0 = kgen.param.constant: !Int = <{42}>
  lit.ref.store %0, %x : <!Int, mut *"x`">
  %none = kgen.param.constant: none = <#kgen.none>
  lit.return %none : !kgen.none
  lit.end_fn
}
```

Here's what each of those lines means.

` %x = lit.var.decl "x" var : !lit.ref<!Int, mut *"x``"> `

This is declaring a **run-time value** in Mojo and is represented as an
**SSA value** in MLIR named `%x`.

It is the result of `lit.var.decl "x" var` which is an
**[MLIR operation](https://mlir.llvm.org/docs/LangRef/#operations)**. Every MLIR
operation has zero or more **operands**, like `var` here (operands are inputs to
the operations). The `"x"` here is a
property/[attribute](https://mlir.llvm.org/docs/LangRef/#attributes) of the
operation, specifically it indicates the
[string name](https://github.com/modularml/modular/blob/9a86eb41be85aba5f1203247dfb4e8a83645ec65/Mojo/include/Mojo/LITDialect/LITOps.td#L883)
of the `lit.var.decl`.

After that and the `:`, we specify the operation's resulting **MLIR type**,
i.e. `!lit.ref<!Int, mut \*"x``">`.

Note that the type directly describes the operation's result. In this case,
`lit.var.decl` has only one result, hence there is only one type. MLIR
operations can have multiple results, so operations with multiple results will
have types defined for each result respectively, e.g
[`kgen.cost_of`](https://github.com/modularml/modular/blob/4ccab6c575bcb24a6cdf4a3776fccdbe8e667aa0/Mojo/include/Mojo/KGENDialect/KGENOps.td#L859-L866).
has 8 results, all are of index type.

` %x = ( lit.var.decl "x" var : !lit.ref<!Int, mut *"x``"> ) `

In our MLIR, the `%x =` is always the lowest precedence.

Now the next line:

`%0 = kgen.param.constant: !Int = <{42}>`

The `%0 =` is always the lowest precedence, so we first look at the
`kgen.param.constant: !Int = <{42}>` part.

That `=` is not an assignment like the first `=`. One should interpret this like
a (hypothetical) `kgen.param.constant <{42}> : !Int`.

Since there's no `%`/`@`/`!`/`#`/`{` symbol in front of `kgen.param.constant`,
it's an MLIR operation.

That operation has an attribute which is `<{42}>`. Here, it represents the value
of the parameter input to this operation.
This is how we write constants (and
parameter expressions in general, but we'll get there later).

Now the next line:

` lit.ref.store %0, %x : <!Int, mut *"x``"> `

This follows the same rules; The `lit.ref.store` operation takes operands `%0`
and `%x`, and the operation's result type is ` <!Int, mut *"x``"> `. Let's
explore that type a little more.

For lit.ref.store specifically, the `<..., ...>` is actually shorthand for
`!lit.ref<..., ...>`. So that line is more like:
` lit.ref.store %0, %x : !lit.ref<!Int, mut *"x``"> `

MLIR allows dialect to define custom printers and parsers to serialize
and deserialize the IR for ease of readability.

As you can see, there's a lot of context-dependent sugar in our MLIR. If you
don't know what something means, ask in slack (and then add the answer to this
guide!). Or, if you're feeling brave, you can try and trace the printing logic
(usually in a `_.td` and its corresponding `_.cpp` file).

### Symbols

Anything with a `@` in front is an **MLIR symbol ref.**

For example, this program:

```mojo
def my_func(x: Int):
    pass

def main():
    my_func(42)
```

parses `main` to this MLIR:

```mlir
lit.fn @"main()"() -> !kgen.none attributes {sourceName = "main", specialFnKind = 0 : i8} {
  %0 = kgen.param.constant: !Int = <{42}>
  %1 = lit.call @mymain::@"my_func(::Int)"(%0) : !lit.generator<("x": !Int) -> !kgen.none>
  %none = kgen.param.constant: none = <#kgen.none>
  lit.return %none : !kgen.none
  lit.end_fn
}
```

Let's talk about this line:

`%1 = lit.call @mymain::@"my_func(::Int)"(%0) : !lit.generator<("x": !Int) -> !kgen.none>`

The `@mymain::@"my_func(::Int)"` is an **MLIR symbol ref**. It refers to
something defined somewhere else.

In the above `lit.call` line, the type after the `:` doesn't describe the
operation's type, it describes the type of the symbol ref.
In other words, that's `my_func`'s type which includes both the list of
types for function argument(s) and the result type,
not just the `lit.call`'s result type.

### Attributes (not a field)

Anything with a `#` in front of it (`#Thing`), or surrounded with curly braces
like `{Thing}` is an MLIR
**[attribute](https://mlir.llvm.org/docs/LangRef/#attributes)**,
often referred to as a compile-time "value", "constant",
or "parameter" in the context of Mojo compilation.
Remember, 'attribute' doesn't mean 'field', attribute is data that only exists
at compile time. If it represents a parameter, it only exists pre-elaboration.

Attributes are the MLIR mechanism for specifying constant data on
operations in places where a variable is never allowed.
This is also the foundation on how Mojo parameters for generic programming
is represented in the IR.

Let's see some attributes. This program:

```mojo
def my_func[N: Int](x: Int):
    pass

def main():
    my_func[73](42)
```

...parses `main` to this MLIR:

```mlir
lit.fn @"main()"() -> !kgen.none attributes {sourceName = "main", specialFnKind = 0 : i8} {
  %0 = kgen.param.constant: !Int = <{42}>
  %1 = lit.call @mymain::@"my_func[::Int](::Int)"<:!Int {73}>(%0) : !lit.generator<("x": !Int) -> !kgen.none>
  %none = kgen.param.constant: none = <#kgen.none>
  lit.return %none : !kgen.none
  lit.end_fn
}
```

Notice the `lit.call` line's new part: `<:!Int {73}>`. That `{73}`
is an attribute of type `!Int` that is associated with the `lit.call` operation.

We've also seen this before; `%0 = kgen.param.constant: !Int = <{42}>` had a
`<{42}>` which was an attribute of the `kgen.param.constant` op,
though that one didn't have the type (`:!Int`) in front.

Most attributes used in Mojo have types. We'll talk about that more further
below.

### Terminologies (from Mojo's perspective): Parameters, Constants, Values

Every compilation stage up till the elaborator contains IR that represents Mojo
generic programming. It is similar to C++ template meta-programming before
concretizing. "**parameters**" in Mojo are generic representation that can be
concretized into compile time constants (similar as C++'s
[template parameters](https://en.cppreference.com/w/cpp/language/template_parameters.html)).
This is to be distinguished from "function parameter" which is a named variable
in a function's definition, serving as a placeholder for values that will be
passed into the function when it is called. To avoid confusion, we call
"function parameter" **argument** in Mojo for both the named variable and the
passed value for a function.

“parameter” can means one of three things in Mojo.

For example,

```mojo
struct Foo[T: Stringable]:
    var field: T

def main():
  var f = Foo[Int]()
```

- A "parameter declaration" (or "param decl" or "input param"), is like the
  `T: Stringable` in that first line.
- A "parameter reference" (or "param ref"), is like the mention of `T` in
  `var field: T`. It refers to a param decl.
- A "parameter value" (or "param value"), is the `Int`.

All of them in the same sentence: The param value `Int` is fed into `Foo`'s
param decl `T: Stringable` and makes its way to the param ref `T` in `field: T`.

We often say “in parameter-space” or "in the parameter domain".
For a Mojo programmer, it means “at compile time”.
For a Mojo compiler engineer, it means before elaboration.

There can be subtle differences between the various terms:

- "Attribute" refers to MLIR attributes. There are typed attributes and untyped
  attributes. All parameters are typed attributes, and most (but not all) typed
  attributes are parameters.
- "Value" can mean both "parameter value" (which should be a constant), or
  an MLIR SSA value which represents a run-time value in Mojo (confusing!!).
- "Constant" can be "parameter value", probably most constants we care about
  for Mojo compilation are for parameters. It can also be a hard-coded
  constant value in the Mojo program that is never represented by a parameter.

### Does Parameter Value Have Types?

Yes. Like C++, Mojo's parameter values have types.

In C++, the template parameter `N` has type `int`:

```c++
#include <iostream>
template<int N>
void zork() {
  std::cout << N << std::endl;
}
int main() {
  zork<42>();
}
```

Same in this Mojo snippet, `N` is an `Int`:

```mojo
def zork[N: Int]():
      print(N)
def main():
      zork[42]()
```

Our LIT dialect also remembers parameters' types. Let's see the LIT IR, by
feeding that to
`kgen-translate -import-mojo main.mojo | kgen-opt -lower-semantic-cf -check-lifetimes`:

```mlir
lit.fn @"zork[::Int]()"<N: !Int>() -> !kgen.none attributes {sourceName = "zork", specialFnKind = 0 : i8} {
  ...
}
lit.fn @"main()"() -> !kgen.none attributes {sourceName = "main", specialFnKind = 0 : i8} {
  %0 = lit.call @main::@"zork[::Int]()"<:!Int {42}>() : !lit.generator<() -> !kgen.none>
  ...
}
```

- The `N: !Int` on the `lit.fn` line makes a parameter-decl named `N` of type
  `Int`.
- The `:!Int {42}` on the `lit.call` line makes a parameter-value of type `Int`
  with value `42`.

KGEN, however, doesn't have types for its parameters. Let's see the KGEN IR, by
feeding the LIT IR to `kgen-opt -lower-semantic-cf -check-lifetimes -lower-lit`:

```mlir
kgen.generator @"main::zork[::Int]()"<N>() -> !kgen.none {
  ...
}
kgen.generator @"main::main()"() -> !kgen.none {
  %0 = kgen.call @"main::zork[::Int]()"<42>() : () -> !kgen.none
  ...
}
```

However, whereas C++ only supports basic types (`int`, `bool`, etc.), Mojo can
take anything, even memory types,
like in this program that takes an entire `List[Int]`:

```mojo
def zork[L: List[Int]]():
    for x in L:
        print(x[])

def main():
    zork[[1, 2, 3, 4]]()
```

In this, the LIT contains this `lit.call` line:

```mlir
      %0 = lit.call @main::@"zork[::List[::Int, ::Bool(False)]]()"<:@std::@collections::@list::@List<:!Copyable_Movable #Int1, :!Bool {:i1 0}> apply_result_slot(...)>() : !lit.generator<() -> !kgen.none>

```

(The `apply_result_slot` is LIT-speak for "call at compile-time".)

C++ and Mojo are also different in how they handle types. In C++, a template can
expect a type as a template parameter by saying `typename`, like:

```c++
template<int N, typename T>
class Vec { ... };
```

In Mojo (and the LIT dialect), a parameter can't just be a "type", we must
specify the rough shape of `T`, by specifying a trait.

`template<int N, typename T> class Vec { ...` in C++ would therefore be
equivalent to

`struct Vec[N: Int, T: Copyable]: ...` in Mojo.

In that `Vec`, we can say two things:

- `N`'s type is `Int`.
- `T`'s type is `Copyable`.

"Type" is a relative term. `N`'s type is `Int`, and Int's type is something
else, and that has a type, and so on. Everything has a type.

### Types as Parameter Values, Parameter Values as Types

When we instantiate a generic type, like the `Vec` above, we feed it a
parameter-value for each of its parameter-decls. For example, we might say
`Vec[3, Float32]`.

However, `3` and `Float32` are not parameter-values, they're an integer and a
type.

To resolve this, the compiler automatically converts/wraps those into the proper
parameter-values.

The `3` int literal will be wrapped in a `pop.int_literal` parameter-value, like
`:!pop.int_literal 2`.

The `Float32` type will be wrapped in a type-param (a.k.a. `kgen.type` or
"type-value" or `KGEN::TypeParamAttr` from KGENAttrs.td), like
`#kgen.type<Float32, ...>`.

Rule of thumb: **to convert a type to a parameter-value, use a type-param.**

Some examples of type-params:

- `#kgen.type<@blork::@MyStruct> : !lit.anystruct<@blork::@MyStruct>`
- `:!MyTrait #MyStruct1`, means a parameter-value of type `MyTrait` with value
  `#MyStruct1` (which is defined elsewhere as
  `#MyStruct1 = #kgen.type<!MyStruct, {"bork" : ...}> : !MyTrait`).

To see that last one, you can run this program through
`kgen-translate -import-mojo main.mojo`:

```mojo
@explicit_destroy("Can't destroy a MyTrait")
trait MyTrait(TrivialRegisterPassable):
    def bork(self):
        ...


@fieldwise_init
struct MyStruct(MyTrait):
    def bork(self):
        print("hello")


def my_func[T: MyTrait](x: T):
    x.bork()


def zork[N: Int]():
    print(N)


def main():
    zork[42]()
    var x = MyStruct()
    my_func(x)
```

To do the opposite, **to turn a type-param back into a type,** we use
`kgen.param` (a.k.a. `KGEN::ParamType`).

In the above snippet, you can see it in `my_func`'s argument:

```mlir
lit.fn @"my_func[main::MyTrait]($0)"<T: !MyTrait>(%x: !kgen.param<:!MyTrait T>) -> !kgen.none attributes {sourceName = "my_func", specialFnKind = 0 : i8} {
  %0 = lit.call [!lit.generator<("self": !kgen.param<:!MyTrait T>) -> !kgen.none>: get_witness(:!MyTrait T, "MyTrait", "bork")](%x)
  %none = kgen.param.constant: none = <#kgen.none>
  lit.return %none : !kgen.none
  lit.end_fn
}
```

...because arguments must be types, not parameter-values.

### Generators

Whenever you see a `lit.generator`, that's a signature.

If it has a number in it like the `2` in `lit.generator<[2](`, the [2] isn’t the
number of arguments, it’s the number of implicit origins.

### Miscellaneous Dialect Oddities

When you see `#kgen<`, like in this:

`#kgen<param.decl callee : !kgen.generator<!lit.generator<...>>>`

we’re not actually “instantiating a `kgen`”.

Rather that’s a `#kgen` prefix followed by another thing.

It’s similar to this (hypothetical) syntax with a `.` instead and with the `<`
moved:

`#kgen.param.decl<callee : !kgen.generator<!lit.generator<...>>>`

Supposedly this happens because `def KGEN_ParamDeclAttr`'s `assemblyFormat`
didn’t specify that the *parameters* should start with `<`, so it assumed the
*entire thing* starts with `<`

## Passes

The Mojo compiler has a lot of passes. Some of the big ones are:

- Parsing, which does lexing, parsing, and type-checking.
- Elaborating, which instantiates generics, for example `def add[x: Int](...)`
  into `def add[3](...)`, `def add[42](...)`, `def add[1337](...)` etc.
- Post-elaboration optimizations on concretized IR before lowering to LLVM.
- Lowering all Mojo MLIR dialects to LLVM.

...but there are a lot more.

You can learn about all of them in Weiwei's excellent
[Mojo Compilation Model](https://www.notion.so/modularai/Mojo-Compilation-Model-Now-and-Future-6028a58015034f38b037e520ee2e2d78)
doc.

You can see all the passes that run for a particular program by running
`kgen --mlir-print-ir-before-all -elaborate main.mojo 2>&1 | grep 'IR Dump Before'`.
For example when run on a simple `def main(): pass` it mentions these passes
coming after the parser:

- DebugInfoStrip
- LowerSemanticCF
- VerifyParameters
- CheckLifetimes
- AnnotateKernels
- VerifyKernels
- LowerLIT
- MOGGPreElabPipeline
- RemoveUnusedParams
- EliminateDeadSymbols
- SROA
- Mem2Reg
- Canonicalizer
- InlineParametric
- SCCP
- ApplyInliner
- OutlineClosures
- CSE
- LiftAndFoldApply
- ElaborateGenerators
- EliminateDuplicateFunctions
- ResolveCompilerPromises
- LowerArgConventions
- LowerCallingConventions
- EnsureNoParameters
- AutomaticInline
- RaiseForLoops
- LoopUnrolling
- ArgPromotion
- SimplifyCF
- LowerLoops
- LowerClosures
- LowerAsyncFunctions
- DeadArgumentElimination

...and many of these passes are run multiple times.

The `mojo` command will run the entire pipeline from beginning to end, but you
can use `kgen` to run specific passes, `kgen-translate` to run only the parser,
and `kgen-opt` to run a customized pipeline of passes. For more details on
those, and other commands, see
[Mojo Dev Tools](https://www.notion.so/modularai/Mojo-Dev-Tools-027879ef5e4d480ea6f8f73b1cbc2ad3).
