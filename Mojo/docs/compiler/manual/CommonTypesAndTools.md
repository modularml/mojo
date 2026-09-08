---
markdown-notebook-data-directory: mdnb-data/manual-common-types-tools/
---

# Common Types and Tools

One must be prepared before diving into an unfamiliar compiler.

![TakeThisHelpfulExprEmitter](mdnb-data/manual-common-types-tools/takethis.jpeg)

By the end of this page, you should have most of the tools you need to add an
interesting feature to the Mojo compiler.

We recommend reading [Passes and Intermediate Representations](PassesAndIR.md)
and [Terminology](Terminology.md) before this.

## Nodes, Data Wrappers, and Helpers

There are three kinds of types you'll see in the codebase:

- **MLIR nodes**, like attributes and operations. If you're looking for Mojo's
  AST or IR, you're looking for these. We define these to MLIR, and then MLIR
  generates the C++ for us. For example:
  - The `def LIT_StructType { ... }` block in our `LITTypes.td` tells MLIR how
    to generate `LIT::StructType`.
- **Data Wrappers** around those MLIR things. For example:
  - Our `ASTType` is a wrapper around `mlir::Type`.
  - Our `CValue`, `LValue`, `MLValue`, `DLValue`, `SBValue`, `MBValue`,
    `MBPValue`, `PValue`, `SRValue`, and `MRValue` are wrappers around the
    various MLIR attributes we defined.
- **Helpers** like builders, transformers, and algorithms, which help us
  manipulate the above two things. For example:
  - Our `IREmitter` wraps a `mlir::OpBuilder` and helps us output expressions.
  - `OverloadSet` represents a call's valid candidates and gives us methods to
    help narrow them down and call them.
  - `ParameterInferenceState` holds all of the tentative conclusions/bindings
    when doing parameter inference. When matching `Spaceship[42]` against
    `struct Spaceship[N: Int]`, `ParameterInferenceState` is the one that
    figures out `N` = `42`.

All of those are explained more below. Read on!

## MLIR Nodes

This section will talk about all the MLIR nodes.

Recall how running `br //Mojo/tools/kgen-translate -- -import-mojo main.mojo` on
this `main.mojo`...

```mojo
def foo(arg: Int):
  pass

def main():
  foo(5)
```

...produced this MLIR:

```mlir
lit.fn @”main()”() -> !kgen.none attributes {sourceName = “main”, specialFnKind = 0 : i8} {
  %0 = kgen.param.constant: !Int = <{5}>
  %1 = lit.call @main::@”foo(::Int)”(%0) : !lit.generator<(“arg”: !Int) -> !kgen.none>
  %none = kgen.param.constant: none = <#kgen.none>
  lit.return %none : !kgen.none
  lit.end_fn
}
```

All of those things are MLIR nodes. This section will show you how to learn
about those.

Rule of thumb: every MLIR node is defined in a `.td` file. For example:

- `lit.return` is defined by the `def LIT_ReturnOp` in `LITOps.td`.
- `lit.call` is defined by the `def LIT_CallOp` in `LITOps.td`.
- `lit.fn` is defined by the `def LIT_FnOp` in `LITOps.td`.
- `kgen.param.constant` is defined by `def KGEN_ParamConstantOp` in
  `KGENOps.td`.

You can generally find the `.td` that something was defined in by doing a global
search for its `def` like this:

- `^def.*lit.*return`
- `^def.*lit.*call`
- `^def.*lit.*fn`
- `^def.*kgen.*param.*constant`

These were all MLIR operations, and we tend to put operations in the pass's
operations definitions, like `LITOps.td`, `KGENOps.td`, `POPOps.td`,
`HLCFOps.td`, etc.

So what about things other than ops, like values, types and compile-time data?

Recall from [Passes and IR](PassesAndIR.md) the basic MLIR notation rules:

- `%name` — A run-time value; an MLIR SSA value; the result of an MLIR
  operation.
- `@name` — A
  [SymbolRefAttr](https://mlir.llvm.org/docs/Dialects/Builtin/#symbolrefattr)
- `!name` — An MLIR type.
- `#expr` or `{expr}` — Compile-time data / attributes.
- Anything else is an MLIR operation.

You'll find types defined in `LITTypes.td`, `KGENTypes.td`, `POPTypes.td`, etc.

You'll find compile-time data / attributes defined in `LITAttrs.td`,
`KGENAttrs.td`, `POPAttrs.td`, etc.

For example, the snippet's `!kgen.none` is defined by the `def KGEN_NoneType` in
`KGENTypes.td`.

And of course, the snippets `#kgen.none` is defined by the `def KGEN_NoneAttr`
in `KGENAttrs.td`.

Let's see one of those definitions, from `LITOps.td`:

```td
def LIT_ReturnOp : LITOp<"return", []> {
  let summary = "Lexical return statement.";
  ...
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict (`:` type($operands)^)?";
  let hasVerifier = 1;
}
```

This is a
[Operation Definition Specification](https://mlir.llvm.org/docs/DefiningDialects/Operations/),
or more commonly, a "tablegen file".

The most important parts are:

- The `"return"` at the top. That's why it shows up as `lit.return` in MLIR.
- `let arguments =` lists the properties/fields of the operation. `return` only
  has one field, named `operands`, which represent the `42` and `True` in Mojo
  code `return 42, True`.
- `let assemblyFormat =` describes how this operation's MLIR should be printed
  and parsed. This is why it shows up as `lit.return %none : !kgen.none` in the
  MLIR; you can see the `$operands`, the `:`, and the `type($operands)` in the
  `assemblyFormat`.

The compiler's own build system will transform these `.td` files into C++. For
example, this tablegen code:

```td
def LIT_CallOp : LITOp<"call", [
    DeclareOpInterfaceMethods<KGEN_CallOpInterface>]> {
  let summary = "call a function";
  ...
  let arguments = (ins
    AttrOfType<LIT_FnTypeGeneratorType>:$callee,
    KGEN_ParameterExprArrayAttr:$implicitOrigins,
    Variadic<AnyType>:$operands
  );
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;

  let assemblyFormat = [{
    `` custom<CallOp>($callee, $implicitOrigins, $operands,
                      type($operands), type($results)) attr-dict
  }];

  let extraClassDeclaration = [{
    /// Get the direct callee symbol if this is a direct call.
    SymbolRefAttr getDirectCallee();
    /// Get the callee signature type.
    FnTypeGeneratorType getCalleeType() {
      return cast<FnTypeGeneratorType>(getCallee().getType());
    }
  }];
}
```

...will be transformed into some C++ code in the generated `LIT.h.inc` like
this:

```c++
class CallOp : public ::mlir::Op<CallOp, ::mlir::OpTrait::ZeroRegions, ...> {
public:
  ...
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("lit.call");
  }
  ...
  ::mlir::TypedAttr getCallee();
  ::llvm::ArrayRef<TypedAttr> getImplicitOrigins();
  ...
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::mlir::TypedAttr callee, ::M::KGEN::ParameterExprArrayAttr implicitOrigins, ::mlir::ValueRange operands);
  ...
  ::llvm::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  ...
  /// Get the direct callee symbol if this is a direct call.
  SymbolRefAttr getDirectCallee();
  /// Get the callee signature type.
  FnTypeGeneratorType getCalleeType() {
    return cast<FnTypeGeneratorType>(getCallee().getType());
  }
};
```

Notice how they correspond:

- The various `arguments`, like `callee`, `implicitOrigins`, and `operands`,
  each get their own helper methods, like the `getCallee` method there.
- The `extraClassDeclaration` text, containing `getDirectCallee` and
  `getCalleeType`, appears in the final C++ too.

And MLIR also gave us some other useful helper methods, like `verify`, `parse`,
and `print`.

**Our own C++ can then interact with this.** For example, `SharedState.cpp`
includes `LITOps.h` which includes the generated `LIT.h.inc`, and
`SharedState.cpp` has this snippet which uses `call.getCallee()`:

```c++
if (auto call = dyn_cast<LIT::CallOp>(op)) {
  SmallVector<TypedAttr> calleeOperands;
  calleeOperands.push_back(evaluator.getReboundAttribute(call.getCallee()));
  ...
```

We can add our own methods onto an MLIR node as well. For example, the above
`LIT_CallOp` defines these `extraClassDeclarations`:

```td
  let extraClassDeclaration = [{
    /// Get the direct callee symbol if this is a direct call.
    SymbolRefAttr getDirectCallee();
    /// Get the callee signature type.
    FnTypeGeneratorType getCalleeType() {
      return cast<FnTypeGeneratorType>(getCallee().getType());
    }
  }];
```

and we define that `getDirectCallee` in our own `LITOps.cpp`:

```c++
SymbolRefAttr LIT::CallOp::getDirectCallee() {
  if (auto symbolCst = dyn_cast<SymbolConstantAttr>(getCallee()))
    return symbolCst.getSymbol();
  return {};
}
```

To clarify the files involved:

- We write `LITOps.td`.
- `LIT.h.inc` is generated for us, from `LITOps.td`, `LITAttrs.td`, etc.
- We write `LITOps.h` (and #include `LIT.h.inc` from it).
- We write `LITOps.cpp`.
- We write the rest of the compiler (like `SharedState.cpp`) and include
  `LITOps.h` to use all the above.

### Parameter Operator Code

Parameter Operator Code’s are named operations that are variants of the `POC`
enum. POC defines the names of operations supported by the
`#kgen.param.expr<op, args...>` MLIR attribute. This mechanism provides a way
for Mojo code to query values from the compiler at compile time.

This is used for a variety of reasons, from “simple” operations like `sizeof()`,
to innovative use-cases like `compile_assembly` (a way to compile a Mojo
function to assembly that is then embedded in the resulting binary).

<wolfram-cell ctext="Input07.wl" />

```td
def KGEN_POCAttr : I32EnumAttr<"POC", "Parameter Operator Code", [
  /// Fully associative variadic expressions.
  I32EnumAttrCase<"Add", 0, "add">,
  I32EnumAttrCase<"Mul", 1, "mul">,
  I32EnumAttrCase<"MulNoWrap", 2, "mul_no_wrap">,
  I32EnumAttrCase<"And", 3, "and">,
  I32EnumAttrCase<"Or",  4, "or">,
  I32EnumAttrCase<"Xor", 5, "xor">,
  I32EnumAttrCase<"Max", 6, "max">,
  I32EnumAttrCase<"Min", 7, "min">,
```

## Data Wrappers

### Values

In the parser, almost every operation (run-time expression) and parameter
expression will produce a value. In LLVM-based compilers, you can reference that
expression with a `llvm::ValueRef`, but in our MLIR-based compiler, we'll
reference it with a `mlir::Value`.

...except we try not to, if we can avoid it. Our code is a bit more solid when
the parser knows what kind of data it's dealing with. For example, a
`LIT::CallOp`'s operands must be run-time values, and its callee's parameters
must be compile-time values.

For example, let's look at the code handling stores, like `x = y`. We want to
make sure that:

- `x` is an l-value (in other words, a `var`, not an `alias`)
- `y` is a concrete value (in other words, has a known size).

If we don't check these things, our compiler could easily produce crashing code.

So, let's do this:

- Make a wrapper class `LValue` which only holds an `mlir::Value` that we know
  is an l-value.
- Make a wrapper class `CValue` which only holds an `mlir::Value` that we know
  has a known size.
- Make it so our code that handles this, `IREmitter::emitStoreToLValue`, only
  takes in an `LValue` and a `CValue`, like this:

```c++
CValue IREmitter::emitStoreToLValue(ASTExprAnd<CValue> value, LValue destLV,
                                    ExprContext context) {
```

Now, the caller is forced to ensure that the source is a concrete value, and the
destination is an l-value.

`LValue` and `CValue` aren't the only kinds of values we have.

`SRValue` only holds register-passable types: primitives like `int64`,
`float32`, and any struct marked `@register_passable` (deprecated, use
`RegisterPassable` trait instead) or `@register_passable("trivial")`
(deprecated, use `TrivialRegisterPassable` trait instead) (see
[Life of Mojo reg-passable arguments](../overviews/LifeOfMojoRegPassableArgs.md))

`MValue` only holds memory types (non-register passable things, like most
structs).

In fact, there's a whole taxonomy. Here are all the various kinds of values
you’ll encounter (from `IRValues.h`):

<wolfram-cell cexpr="ImportIRValuesSnippet.wl" />

```wolfram,cell:Output
AnyValue       <- Expr emitted to MLIR...
  UValue         <- unresolved value that cannot be materialized
    OverloadSetUValue  <- with an unresolved overload set
    InitializerUValue  <- constructor operands for an unknown type
  CValue        <- Concrete value: something with a known type.
    LValue         <- mutable reference to storage
      MLValue        <- value is in memory with a mutable reference
      DLValue        <- with dynamic get/set accessors
    BValue         <- with a borrowed value
      SBValue        <- value is register-passable and in an SSA \
register
      MBValue        <- value is in memory with a reference (may be \
mutable)
      MBPValue       <- reference with parametric mutability
      PValue         <- value is a parameter expression.
    RValue         <- with an owned value
      SRValue        <- with a register-passable value in an SSA \
register
      MRValue        <- value is in memory with a mutable reference
      PValue         <- with a parameter value"
```

For example, in this mojo snippet:

```mojo
struct Spaceship:
    var hp: Int64

def launch(imm ship: Spaceship):
    var x: Int64 = ship.hp
```

Our parser will see that `ship` as a `MBValue` because it's a memory type, and
we're borrowing it as a `imm` reference.

All the above mostly applies to the parser. In later passes, we more often
interact directly with our MLIR nodes and the built-in MLIR classes like
`mlir::Value` and `mlir::Type`.

### ASTDecl

The Mojo parser largely doesn't think in terms of ASTs, it handles IR; the
parser consumes text directly (lexing it lazily on the fly), and produces
semi-flattened IR: a tree of structured control flow (`if`, `loop`, `try`, etc)
containing lists of SSA statements (`lit.call`, etc).

However, even though the parser doesn't _produce_ an AST, it does
create a temporary one. For every scope (`fn`,
`struct`, `if`, `loop`, `try`, anything inheriting `ASTDeclInterface`) we have
an `ASTDecl` that does some bookkeeping for it.

Its main purpose is to track members: directly owned children, children
inherited from parent traits, and anything else that might be visible in some
way from inside the scope.

Then, various places in the parser can use `ASTDecl::lookupInCurrentScope` to
find any of those matching a certain name.

It also holds a cursor for lazily, gradually parsing itself (see
`ASTDecl::getCursor`).

An `ASTDecl` will usually be backed by an operation, but can also be backed by a
`CValue`. One time, we observed a `struct MyStruct<T: AnyType>` containing a
child "T" that was an `ASTDecl` that was backed by a `ParamDeclRefAttr("T")`
`PValue`. So our best theory is that `ASTDecl`s can be `CValue`s when we want to
store a precomputed reference (`ParamDeclRefAttr`) instead of a pointer to the
actual declaration (`ParamDeclAttr`), presumably for performance reasons.

### Types

The parser generally doesn't interact with `mlir::Type` directly, we instead use
`ASTType` which has a lot of helpful methods.

## Helpers: Builders, Transformers, Algorithms

There are a lot of useful utilities that you'll see all over the place in our
compiler.

### Walkers/Replacers

- `mlir::AttrTypeWalker`: Lets us walk a attribute (parameter value) and all of
  the attributes it indirectly contains.
- `mlir::AttrTypeReplacer`: An `AttrTypeWalker` that lets us do replacements as
  we go.
- `IndexParameterReplacer` and `ParameterReplacer`: More advanced
  `AttrTypeReplacer`s that are "depth-aware" (they know when the current
  parameter-reference node is referring to something from the parent scope or
  something else).
- `ParameterEvaluator`: A `ParameterReplacer` that has a list of replacements.
- `ParserParameterEvaluator`: A `ParameterEvaluator` with some convenience
  constructors.

### Function Callers

- `OverloadSet`: A class that helps narrow down the exact function you want to
  call. Try not to use it directly if you can avoid it.
- `emitGetterSetterAccess`: Useful for doing field accesses (like `ship.hp`) or
  subscripts (like `my_list[42]`).

### Witness Tables and Conformance

"Conformance" is when we check whether a struct (or a trait) correctly meets all
the requirements for a parent trait.

This is done via `doesNominalTypeConformTo`, which also has the side effect of
first filling in the particular witness table (a.k.a. vtable a.k.a.
`ConformanceOp`) for the struct+trait pair (or trait+trait pair), see CALROC.

Major players involved: `doesNominalTypeConformTo` calls
`DeclResolver::resolveBody(ConformanceOp, ASTDecl &)` which calls
`verifyConformance`.

See [Conformance.md](../arcana/Conformance.md) for more on how this
all works.

### Allocation

We rarely use `new` or `make_unique` or `make_shared`.

Rules of thumb:

- When allocating an MLIR op, create it via `OpBuilder::create` or
  `ImplicitLocOpBuilder::create`, for example
  `b.create<LIT::ReturnOp>(results)`.
- When allocating an attribute, we often use its `::get` static method, for
  example `BoolAttr::get(context, value)`.
- When allocating a `mlir::Type`, we either:
  - Call its `::get` static method, for example
    `LIT::StructType::get(symbol, unbound, sig)`.

`::get` methods are often declared in our tablegen `.td` files, via the
`let builders = [` directive. Their implementations are defined in the
corresponding `.cpp` file.

If you're not sure how a Something is created, you can search for:
`(alloc\w*<|create<)Something|Something::get`

Parser-specific:

- When creating an `ASTType`:
  - It's a lightweight wrapper, a value type, you can directly construct one
    given an `mlir::Type`.
  - Some you can get from SharedState, for example
    `shared.lookupBuiltinType("Bool", declScope, loc)`.
  - `StructDeclOp::bindReference` creates one from a `StructDeclOp`.
- When allocating memory that shouldn't escape the parser invocation, use
  `SharedState::allocPersistent` or `ExprParser::alloc` (which calls
  `allocPersistent` for you). We do this for expression nodes (e.g.
  `alloc<BoolLiteralNode>(startTok.getLoc(), false)`) and `ASTDecl`s.

<!--
rule of thumb: if we use it when parsing statements/expressions, feel free to
factor it out into a common helper.
-->

<!--
TODO: add https://modular-ai.slack.com/archives/C03GM7S2VMZ/p1747608165446719
-->
