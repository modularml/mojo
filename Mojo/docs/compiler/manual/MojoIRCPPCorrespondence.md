---
markdown-notebook-data-directory: mdnb-data/manual-correspondence/
---

# Mojo ↔ IR ↔ C++ Correspondence

For any given piece of language semantics, there are three different domains in
which you will view things:

- Mojo source code (`def main(): ...`)
- MLIR code (`lit.fn @”main()”() -> !kgen.none { ... }`)
- The compiler C++ code that produces that MLIR.

The goal of this section is to give you an intuition for how the same “thing” is
modeled in each of those domains. As a very basic example, consider the question
of how a named function call is represented.

## Setting the stage: The `sprongle` statement

This page will show you how to generate various operations, types, attrs, etc.

We'll generate those in `main`.

The easiest way to do that is to add a `sprongle` statement that we can call
from `main`, like so:

```mojo
def main():
    sprongle
```

Creating a new statement is pretty straightforward, just copy the `var`
statement. Check out [this PR](https://github.com/modularml/modular/pull/62701)
for an example.

## Function Calls

```mojo
def zork(i: Int):
  pass

def main():
  zork(42)
```

<wolfram-cell ctext="Input17.wl" />

`$ kgen-translate --import-mojo example.mojo`

```mlir
    lit.fn @"main()"() -> !kgen.none attributes {sourceName = "main", specialFnKind = 0 : i8} {
      %0 = kgen.param.constant: !Int = <{42}>
      %1 = lit.call @main::@"zork(::Int)"(%0) : !lit.generator<("i": !Int) -> !kgen.none>
      %none = kgen.param.constant: none = <#kgen.none>
      lit.return %none : !kgen.none
      lit.end_fn
    }
```

Let's see the C++ to generate that `zork(42)` call!

There's two ways to do this: the manual way, and the easy way.

The manual way:

- Lookup "zork" from the current scope.
- Assemble an OverloadSet containing the lookup results.
- Ask the OverloadSet to emit a call.

For example, here's how we would parse the `sprongle` statement in...

```mojo
def zork(i: Int):
  pass

def main():
  sprongle
```

...to turn it into a `zork(42)`:

```C++
SyntheticNode synthNode(smLoc);

ValueDest dest(EC_Sprongle);
std::string spelling = "zork";
LookupResult lookup = emitter.shared.lookupAndResolveDecl(
    spelling, smLoc, emitter.declScope, /*searchParentScopes=*/true);
if (lookup.isFailure()) {
  emitter.emitError(smLoc, "couldn't find function '") << spelling << "'";
  return failure();
}
ArrayRef<ASTDecl *> decls = lookup.getIfSuccess();
auto firstDecl = dyn_cast_or_null<FnOp>(decls[0]->getIfOperation());
if (!firstDecl) {
  emitter.emitError(smLoc, "found a '")
      << spelling << "' but it wasn't a function";
  return failure();
}
auto result = OverloadSetUValue::create(
    spelling, decls, ParamBindings(emitter.getDeclScope()), &synthNode,
    CallSyntax::kDirectCall);
IntLiteralNode int42("42");
auto operands =
    CallOperands(std::vector<ASTExprAnd<AnyValue>>{ASTExprAnd<AnyValue>{
        emitter.emitExprRValue(&int42, EC_Sprongle),
        &int42,
    }});
result->emitCall(std::move(operands), dest, emitter);
```

(You can't use OverloadSet::lookup because there is no `ASTType` around.)

See this code in context
[here](https://github.com/modularml/modular/pull/62701/files#diff-42ea563c834abefb6324d1bd2762b5646f776d563179fb4ad98e7318cd90fb96R2743).

If you want to specify parameters, we have another section for that, see
Function Calls With Parameters.

The easier way to do all that is to conjure up some expression nodes, pretend
they came from the user, and emit them:

```c++
ValueDest dest(EC_Sprongle);
IntLiteralNode int42("42");
DeclRefNode zorkDeclRef("zork");
std::vector<Operand> operands = {
    Operand(&int42, smLoc, Operand::kPositional)
};
CallNode callNode(&zorkDeclRef, smLoc, ArrayRef<Operand>(operands), smLoc);
callNode.emitIR(dest, emitter);
```

See this code in context
[here](https://github.com/modularml/modular/pull/62701/files#diff-42ea563c834abefb6324d1bd2762b5646f776d563179fb4ad98e7318cd90fb96R2776).

<!--

TODO: automatically extract this from the basics PR

TODO: don't inline code like that, extract it from the PR or preferably
from main

-->

## Method Calls

The previous section called a normal top-level function, so let's see a method
call:

```mojo
struct Person:
    var name: String
    var age: Int

    def __init__(inout self, owned name: String, age: Int):
        self.name = name^
        self.age = age

        self.greet()

    def greet(self):
        pass


def main():
    var me = Person("Connor", 25)
```

<wolfram-cell ctext="Input20.wl" />

`$ kgen-translate --import-mojo example.mojo`

```mlir
module {
  lit.file_module @example {
    lit.struct.decl @Person(!AnyType) attributes {sourceName = #Person_name}
      destructor :!lit.signature<[1]("self": !lit.ref<!Person, mut *[0,0]> owned_in_mem, |) -> !kgen.none> @example::@Person::@"__del__(example::Person)" {
      lit.struct.field name : !String
      lit.struct.field age : !Int
      lit.func @"__init__(example::Person=&,std::collections::string::String,::Int)"[mut *"self`2x", mut *"name`2x1"](%self: !lit.ref<!Person, mut *"self`2x"> init_self, %name: !lit.ref<!String, mut *"name`2x1"> owned_in_mem, %age: !Int) -> !kgen.none attributes {sourceName = "__init__", specialFnKind = 2 : i8} {
        lit.ownership.use %name : !lit.ref<!String, mut *"name`2x1">
        %0 = lit.ref.struct.ger %self[name] : <!Person, mut *"self`2x"> -> !String
        %1 = lit.call @std::@collections::@string::@String::@"__moveinit__(std::collections::string::String=&,std::collections::string::String)"[mut *"self`2x"->name, mut *"name`2x1"](%0, %name) : !lit.signature<[2]("self": !lit.ref<!String, mut *[0,0]> init_self, "other": !lit.ref<!String, mut *[0,1]> owned_in_mem, |) -> !kgen.none>
        %2 = lit.ref.struct.ger %self[age] : <!Person, mut *"self`2x"> -> !Int
        lit.ref.store %age, %2 : <!Int, mut *"self`2x"->age>
        %3 = lit.ref.immut %self : <!Person, mut *"self`2x">
        %4 = lit.call @example::@Person::@"greet(example::Person)"[muttoimm *"self`2x"](%3) : !lit.signature<[1]("self": !lit.ref<!Person, imm *[0,0]> borrow_in_mem) -> !kgen.none>
        %none = kgen.param.constant: none = <#kgen.none>
        lit.return %none : !kgen.none
        lit.end_func
      }
      lit.func @"greet(example::Person)"[imm *"self`2x"](%self: !lit.ref<!Person, imm *"self`2x"> borrow_in_mem) -> !kgen.none attributes {sourceName = "greet", specialFnKind = 0 : i8} {
        %none = kgen.param.constant: none = <#kgen.none>
        lit.return %none : !kgen.none
        lit.end_func
      }
      lit.func @"__del__(example::Person)"[mut *"self`"](%self: !lit.ref<!Person, mut *"self`"> owned_in_mem, |) -> !kgen.none always_inline_no_debug attributes {isSynthetic, sourceName = "__del__", specialFnKind = 5 : i8} {
        %none = kgen.param.constant: none = <#kgen.none>
        lit.ownership.mark_destroyed %self : <!Person, mut *"self`">
        lit.return %none : !kgen.none
        lit.end_func
      }
    }
    lit.func @"main()"() -> !kgen.none attributes {sourceName = "main", specialFnKind = 0 : i8} {
      %me = lit.var.decl "me" var : !lit.ref<!Person, mut *"me`">
      %anonymous2A = lit.var.decl "anonymous*" synth : !lit.ref<!String, mut *"anonymous*`1">
      %0 = kgen.param.constant: !StringLiteral = <{:string "Connor"}>
      %1 = lit.call @std::@collections::@string::@String::@"__init__(std::collections::string::String=&,::StringLiteral)"[mut *"anonymous*`1"](%anonymous2A, %0) : !lit.signature<[1]("self": !lit.ref<!String, mut *[0,0]> init_self, "literal": !StringLiteral) -> !kgen.none>
      %2 = kgen.param.constant: !Int = <{25}>
      %3 = lit.call @example::@Person::@"__init__(example::Person=&,std::collections::string::String,::Int)"[mut *"me`", mut *"anonymous*`1"](%me, %anonymous2A, %2) : !lit.signature<[2]("self": !lit.ref<!Person, mut *[0,0]> init_self, "name": !lit.ref<!String, mut *[0,1]> owned_in_mem, "age": !Int) -> !kgen.none>
      %none = kgen.param.constant: none = <#kgen.none>
      lit.return %none : !kgen.none
      lit.end_func
    }
    lit.func export @main(%argc: !lit.struct<#SIMD <:!DType {:dtype si32}, :!Int {1}>>, %argv: !kgen.pointer<pointer<scalar<ui8>>>) cabi -> !lit.struct<#SIMD <:!DType {:dtype si32}, :!Int {1}>> attributes {linkageName = "main", sourceName = "__mojo_main_prototype", specialFnKind = 0 : i8} {
      %0 = lit.call @std::@builtin::@_startup::@"__wrap_and_execute_main[fn() -> None](::SIMD[{int32}, {1}],__mlir_type.!kgen.pointer<pointer<scalar<ui8>>>)"<:!lit.signature<() -> !kgen.none> @example::@"main()">(%argc, %argv) : !lit.signature<("argc": !lit.struct<#SIMD <:!DType {:dtype si32}, :!Int {1}>>, "argv": !kgen.pointer<pointer<scalar<ui8>>>) -> !lit.struct<#SIMD <:!DType {:dtype si32}, :!Int {1}>>>
      lit.return %0 : !lit.struct<#SIMD <:!DType {:dtype si32}, :!Int {1}>>
      lit.end_func
    }
  }
  lit.package @std { }
}
```

There are a few helpers for generating a method call via C++:

- If calling `__getitem__`, `__setitem__`, `__getattr__`, `__setattr__`, use
  `emitGetterSetterAccess`.
- If calling a constructor, use `IREmitter::emitConstructorCall`
- If calling a method, used `IREmitter::emitNamedMethodCall`

For other cases, use the approach in the above Function Calls section.

([source](https://modular-ai.slack.com/archives/C03GM7S2VMZ/p1748355046451549))

## Declaring a Variable

```mojo
def foo():
  var x: Int = 5
```

<wolfram-cell ctext="Input16.wl" />

`$ kgen-translate --import-mojo example.mojo`

```mlir
module {
  lit.file_module @example {
    lit.func @"foo()"() -> !kgen.none attributes {sourceName = "foo", specialFnKind = 0 : i8} {
      %x = lit.var.decl "x" var : !lit.ref<!Int, mut *"x`">
      %0 = kgen.param.constant: !Int = <{5}>
      lit.ref.store %0, %x : <!Int, mut *"x`">
      %none = kgen.param.constant: none = <#kgen.none>
      lit.return %none : !kgen.none
      lit.end_func
    }
  }
  lit.package @std { }
}
```

To do this in the parser, just use `IREmitter::emitVarDecl`:

```c++
ASTType intType = shared.lookupNamedType("Int", *curDeclScope, smLoc);
VarDeclOp varDeclOp =
    emitter.emitVarDecl("x", intType.mlirType, loc, VarDeclKind::Var);
ValueDest dest(MLValue(varDeclOp), EC_VarInit);
IntLiteralNode int5Node("5");
if (!int5Node.emitIR(dest, emitter)) {
  emitError(varDeclOp.getLoc(), "failed emitting var decl");
  return failure();
}
// Optional, but required if you want `DeclRefNode`s to reference it.
getDeclResolver().addFullyResolvedDecl(DeclIRValue(xVarDeclOp), "x", smLoc,
                                        curDeclScope);
```

See this code in context
[here](https://github.com/modularml/modular/pull/62701/files#diff-42ea563c834abefb6324d1bd2762b5646f776d563179fb4ad98e7318cd90fb96R2787).

In our real code, this is done very circuitously through our powerful pattern
matching subsystem. But in the end, it really is just a call to `emitVarDecl`.

## Reassigning a Local Variable

```mojo
def main():
  var x = 5
  x = 7
```

<!-- TODO: add some IR -->

To do this in the parser, create a `BinOpNode` and call `emitIR` on it.

```c++
DeclRefNode xRefNode("x");
IntLiteralNode int7Node("7");
BinOpNode assignNode(ExprNode::Kind::kAssign, &xRefNode, smLoc, &int7Node);
ValueDest ignoredDest(EC_Assignment);
AnyValue result = assignNode.emitIR(ignoredDest, emitter);
if (!result) {
  emitError(smLoc, "failed assigning variable");
  return failure();
}
```

See this code in context
[here](https://github.com/modularml/modular/pull/62701/files#diff-42ea563c834abefb6324d1bd2762b5646f776d563179fb4ad98e7318cd90fb96R2808).

Or, if you want to be lower-level than that, you can use
`IREmitter::emitResult`:

```c++
SyntheticNode locNode(smLoc);
auto int7PValue =
    PValue(IntegerAttr::get(IndexType::get(shared.getContext()), 7));
ValueDest dest(MLValue(xVarDeclOp), EC_VarInit);
emitter.emitResult(int7PValue, &locNode, dest);
```

See this code in context
[here](https://github.com/modularml/modular/pull/62701/files#diff-42ea563c834abefb6324d1bd2762b5646f776d563179fb4ad98e7318cd90fb96R2822).

## If Statement

```mojo
def bork() -> Int:
    if True:
        return 5
    else:
        return 7
```

```mlir
lit.fn @"bork()"() -> !Int attributes {sourceName = "bork", specialFnKind = 0 : i8} {
  hlcf.elif {
    %0 = kgen.param.constant: i1 = <1>
    hlcf.elif.yield %0
  } then {
    %0 = kgen.param.constant: !Int = <{5}>
    lit.return %0 : !Int
    hlcf.yield
  } else {
    kgen.unreachable
  }
  lit.end_fn
}
```

To do this in the parser, use ParserStmts.cpp's `emitIfClause` or follow its
example (use an `IREmitter` to create an `HLCF::ElifOp` and then insert
into its various regions).

## Operators

To do an operator, like a less than or greater than, use a `BinOpNode` like we
saw in Reassigning a Local Variable.

## Loops

```mojo
def hello():
    print("hello")


def main():
    var i = 0
    while i < 5:
        hello()
        i = i + 1
```

```mlir
lit.fn @"main()"() -> !kgen.none attributes {sourceName = "main", specialFnKind = 0 : i8} {
  %i = lit.var.decl "i" var : !lit.ref<!Int, mut *"i`">
  %0 = kgen.param.constant: !Int = <{0}>
  lit.ref.store %0, %i : <!Int, mut *"i`">
  lit.loop cond {
    %1 = lit.ref.load %i : <!Int, mut *"i`">
    %2 = kgen.param.constant: !Int = <{5}>
    %3 = lit.call @std::@builtin::@stubs::@Int::@"__lt__(::Int,::Int)"(%1, %2) : !lit.generator<("lhs": !Int, "rhs": !Int) -> !Bool>
    %4 = lit.call @std::@builtin::@stubs::@Bool::@"__mlir_i1__(::Bool)"(%3) : !lit.generator<("self": !Bool) -> i1>
    lit.loop.condition %4 : i1
  } body {
    %1 = lit.call @main::@"hello()"() : !lit.generator<() -> !kgen.none>
    %2 = lit.ref.load %i : <!Int, mut *"i`">
    %3 = kgen.param.constant: !Int = <{1}>
    %4 = lit.call @std::@builtin::@stubs::@Int::@"__add__(::Int,::Int)"(%2, %3) : !lit.generator<("lhs": !Int, "rhs": !Int) -> !Int>
    lit.ref.store %4, %i : <!Int, mut *"i`">
    lit.loop.continue
  } else {
    lit.loop.yield
  }
  %none = kgen.param.constant: none = <#kgen.none>
  lit.return %none : !kgen.none
  lit.end_fn
}
```

To do this in the parser, do the same thing as `StmtParser::parseWhileStmt`;
create a `LIT::LoopOp` and insert into the various blocks.

## Bonus challenge: combining it all

A challenge, if you're following along... make it so this program:

```mojo
def printStr(x: String):
    print(x)

def main():
    sprongle
```

Does the same thing as this program:

```mojo
def printStr(x: String):
    print(x)

def main():
    var player_row = 1
    var player_col = 2
    var row = 0
    var col = 0
    while row < 18:
        while col < 80:
            if row == player_row and col == player_col:
                printStr("@")
            else:
                printStr(".")
            col = col + 1
        printStr("\n")
        row = row + 1
```

You'll need variables, assignments, loops, operators, and if-else statements.
Good luck!

## Coming in V2

This doc is lazily populated, so let Evan know if you want to know about
anything, especially:

- Printing
- Aliases
- Defining a function
- Struct declaration
- Using a struct; struct type symbol reference
- Modifying a field
- Argument conventions: borrowed, inout, owned, register passability
- Overloads
