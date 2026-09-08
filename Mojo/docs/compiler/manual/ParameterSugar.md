# Mojo’s Type & Parameter Sugar System

## A Guided Tour for Compiler Engineers

Mojo has a powerful and expressive type system, including dependent types and
sophisticated type/parameter inference. These features allow Mojo to scale far
beyond traditional C++-style type systems. But this increased expressiveness
comes with a major responsibility:

> When you raise the level of abstraction, you must raise the quality of
> diagnostics to match. Errors become more abstract, and programmers need the
> compiler to clearly explain relationships between types, parameters, symbolic
> expressions, and constraints.

To support this, Mojo implements a unified system for representing both semantic
types and their syntactic sugar. This document provides an introduction to that
system, with the goal of helping compiler engineers understand:

- Why we preserve “sugared” forms of types and expressions
- How types and compile-time parameters are represented in MLIR:
- How canonicalization works
- What you need to know when working on the Mojo parser and LIT-level passes.

We think that quality of error/warning messages and other code diagnostic are
one of the best ways to improve the quality of life of C++ programmers coming
to Mojo as well as engineers that have to understand the output of the Mojo
compiler's error messages. If we make a significant step forward here, we can
help both humans (and AI coding systems!) move forward with a more productive
coding experience.

## Why Typedef Sugar Matters — Lessons From C++

Before we dive into how Mojo handles type and parameter sugar, it's helpful to
understand **why this problem exists in the first place**, and why languages
like C++ and Mojo benefit from preserving typedef/alias sugar.

A classic example in C++ is:

```cpp
// This is how string is defined in the C++ standard library.
// See https://en.cppreference.com/w/cpp/string/basic_string.html
namespace std {
  template<class CharT, class Traits = std::char_traits<CharT>,
          class Allocator = std::allocator<CharT>
  class basic_string;

  // Everyone uses std::string though!
  typedef std::basic_string<char> string;
}
```

From the compiler’s point of view, `std::string` and
`std::basic_string<char, std::char_traits<char>, std::allocator<char>>` are
**exactly the same type**. However from the programmer’s point of view, the are
not interchangeable in diagnostics, readability, or intent. Most C++ programmers
don't need or want to know about the fact that `string` is a typedef.

Consider an error like:

```cpp
std::string s = get_value();
```

If the compiler prints an error message like:

```text
error: cannot convert from 'MyType' to 'std::basic_string<char, std::char_traits<char>, std::allocator<char>>'
```

this is *correct* but is very surprising and difficult for the
programmer to understand. The user never wrote that type. They wrote
`std::string` but got a ton of library details that have nothing to do with
their problem. The entire point of the alias is to *hide* the complexity of the
underlying type.

### Type "sugar" to the rescue

As a consequence of this, compilers like Clang preserve typedef "sugar" so it
can print:

```text
error: no viable conversion from 'MyType' to 'std::string' (aka 'basic_string<char>')
```

Notice that it preserved the typedef `std::string` - it even omitted printing
the defaulted template parameters derived from the element type, Mojo does this
as well, but this isn't related to type sugar. Clang implements this at the AST
level with it's Type class and "canonical type" system as
[described in its internals manual](https://clang.llvm.org/docs/InternalsManual.html#the-type-class-and-its-subclasses).

This is a good thing!

### Mojo has the same problem, but arguably even worse

Mojo goes beyond C++ in a number of ways, including the addition of powerful
comptime metaprogramming. We want people to build powerful and expressive
libraries, and we want clients of those libraries to not be exposed to the
internal implementation details unless needed. Poor error messages were a
common source of complaints, particularly from GPU programmers that are building
into high level abstractions like `LayoutTensor`.

Mojo now supports type sugar, so let's look at a simple example:

```mojo
comptime ideal_width = some_complex_calculation()*4
def get_data() -> SIMD[DType.int8, ideal_width]: ...

def simple_example():
  var x = get_data()  # Ok
  var y : SIMD[DType.int8, 4]

  y = get_data() # This is an error.

  var z = x.join(x) # z is a dependent type twice the width of x

  y = z  # This is an error.
```

With type sugar, we can generate nice error messages like:

```text
test.mojo:15:15: error: cannot implicitly convert 'SIMD[DType.int8, ideal_width]' value to 'SIMD[DType.int8, 4]'
  y = get_data()
              ^
test.mojo:15:15: note: 'ideal_width' is aka '(some_complex_calculation() * 4)'

test.mojo:19:7: error: cannot implicitly convert 'SIMD[DType.int8, (2 * ideal_width)]' value to 'SIMD[DType.int8, 4]'
  y = z
      ^
test.mojo:19:7: note: .size of left value is '(2 * ideal_width)' but the right value is '4'
```

Note that by default Mojo retains the sugar for `ideal_width` in the type
of `x` and `z` to keep the initial errors short, but then can selectively unwrap
it when Mojo thinks you might want to know what it is.

This document explains how Mojo implements support for type "sugar" in MLIR
and what compiler engineers need to know about it.

## Sugar in the Mojo parameter system

The core challenge to implementing support for Sugar in a type system is that
you must maintain two different forms for a type: the "user visible" form (known
as the "sugared" version) along with the semantic version - used for type
comparisons and the other things the language cares about.

Note: In Mojo, types are comptime-values, and can exist in parameter
expressions, everything applies equally well to types and parameter expressions
in this document.

This section describes how we do that, and introduces terminology.

### Introduction to `SugarAttr`

Mojo implements types and parameters with MLIR attributes, and type sugar is
no different - it uses an attribute named `KGEN::SugarAttr`. `SugarAttr` looks
like this (but check `KGENAttrs.td` in case this drifts, and update this if so):

```tblgen
def KGEN_SugarAttr : KGENAttr<"Sugar", "sugar", [TypedAttrInterface]> {

  // You build a SugarAttr with a kind, the sugared form, and the expanded form.
  AttrBuilderWithInferredContext<(ins "SugarKind":$kind,
                                      "TypedAttr":$sugared,
                                      "TypedAttr":$expanded), "", "TypedAttr">,

  // SugarAttr stores those fields plus a 'canonical' form, described later.
  let parameters = (ins
    "SugarKind":$kind,
    "TypedAttr":$sugared,  // Fully sugared form.
    "TypedAttr":$expanded, // One level expanded form, other sugar may exist.
    "TypedAttr":$canonical // Recursively desugared form.
  );

  let extraClassDeclaration = [{
    /// Remove any top-level sugar nodes from this type, but don't fully
    /// canonicalize it.
    static TypedAttr strip(TypedAttr value);
    static Type strip(Type value);
  }];
```

`SugarAttr` directly reflects the duality of a potentially sugared type system,
by representing both the "user-visible" form of the expression (the "sugared"
form) as well as the "original" form of the expression. To understand this,
let's look at an simple example:

```mojo
comptime size = 4
```

To understand how a lookup of `size` is processed, let's look at the logic in
`DeclRefNode::emitUnqualLookup`. A slightly simplified (error handling removed)
version looks like:

```C++
    PValue result = resolveAliasReference(param, spelling, ...);

    // Maintain alias sugar, e.g. print "UInt8" as UInt8 instead of SIMD[..].
    auto sugared = ParamDeclRefAttr::get(param.getParamDecl());
    result = SugarAttr::getAlias(sugared, result);
    return result;
```

If we go back to our example, when looking up `size`, we resolve the
alias value, giving us an integer literal of value `4`. This is the non-sugared
value returned by `resolveAliasReference`. We could use this value directly
in the parser, but we would lose the symbolic name `size`.

To address this, we create the other form of the expression - the textual form
which is a reference to the parameter declaration (in `sugared`). We then pass
both of these to `SugarAttr` so it has both the user-visible sugared form as
well as the underlying expansion.

Given this, we've now captured the information we need in the type system -
error messages can use the "Sugared" form, and semantic analysis can use the
"original" (or "canonical") forms for type checking. This also allows the
diagnostic printing logic to selectively unpack sugared forms based on its
heuristics.

### `AlwaysInlineBuiltin` sugar

`SugarAttr` has a "kind" enum which is typically `Alias`, but it also has an
`AlwaysInlineBuiltin` form. This is used when resolving parameter references to
`@always_inline("builtin")` calls, for example:

```mojo
def example[a: Int]():
    comptime b = a+4  # + is Int.__add__ which is builtin.
```

When resolving the call to `Int.__add__`, the compiler notices that it is an
`@always_inline("builtin")` function and goes ahead and inlines the body of
add into the parameter expression, generating something like
`(struct_attr Int, (POC::add (struct_extract A, mlirvalue), 4))`. This is
important for being able to do symbolic
analysis and simplification of integer expressions, but is not something we want
to expose to Mojo programmers. The solution to this, of course, is to use
`SugarAttr` - the sugared form is the call to `Int.__add__` and the expanded
form has the inline value.

So why do we keep track of this difference from other sugar forms, why not use
`Alias` for this? The rationale is that we want the compiler to unpack simple
alias sugar in error messages to help the programmer understand what is going
on, but we *never* want to do this for builtin functions.

### Type/Parameter equality in the presence of sugar

Type equality is a key part of type checking - if a function takes an `Int`, you
need to determine whether the value passed in is an `Int`, and if not, whether
you can convert to `Int`. Before type sugar, we had a very helpful invariant
built on MLIR type and attribute uniquing: we could test to see if types and
parameters were "the same" just by testing them for pointer equivalence.

Unfortunately, in the presence of sugar, this isn't possible. Type sugar can
exist at any level, and so - worst case - we need to recursively walk two type
trees to see if they are equal to each other. This is expensive in compile time
and error prone.

To solve for this, Mojo (like Clang) introduces the notion of a "canonical" form
of a parameter expression. This canonical form is the recursively desugared
form of a type. This is calculated by the `Canonicalizer` class in
KGENAttrs.cpp, and exposed to compiler authors through these these global
functions:

```c++
/// Given an attribute or type, return the "canonical" version of the attribute
/// with all type sugar removed.
Attribute getCanonicalAttr(Attribute src);
TypedAttr getCanonicalAttr(TypedAttr src);
Type getCanonicalType(Type type);

/// Return true if the specified types are canonically equal.
bool isEqualCanon(Type t1, Type t2);
bool isEqualCanon(TypedAttr ta1, TypedAttr ta2);
```

There are other helpers as well, including the `ASTType::isEqualCanon` method
which simply forwards to these.

### Canonical parameter caching for compile time efficiency

Now, you might wonder - isn't this slow to recursively walk types and attribute
trees to get canonical forms? Yes, it can be. Mojo types and parameter
expression can get to be very large and we want these operations to be close to
O(1) since they are so core.

To solve for this, a few Mojo types "cache" their canonical form, allowing the
canonicalizer to stop recursing on them. This means that the canonicalizer will
typically look through a few nodes, but stops when it gets to specific common
large parts of the type graph. This is implemented by the
`KGEN::SugaredTypeInterface` MLIR interface, notably `LIT::StructType`.

For example, consider our SIMD type from above:

```mojo
comptime ideal_width = some_complex_calculation()*4
def get_data() -> SIMD[DType.int8, ideal_width]: ...
```

The return type of `get_data()` is represented as a StructType with two
parameter values. The second parameter is a SugarAttr so it prints as
`SIMD[DType.int8, ideal_width]` with the alias inline. In addition to this
StructType maintains a canonical form which has all sugar recursively removed,
which would print as `SIMD[DType.int8, some_complex_calculation()*4]`. This
cache allows the recursive canonicalization process to stop early.

If we see other compile time impacts, we can add this to other types, we should
likely add it to `KGEN::SignatureType` and `KGEN::StructType`. The former is a
large and complicated type, and the latter exists after lowering.

### Sugar Lowering

LIT->KGEN lowering is responsible for removing SugarAttr. The rewrite is
simple: all types and parameters are replaced with their canonical form. This
gives the invariant that `SugarAttr` never exists after LIT lowering is done.

However, please remember that most of the core KGEN dialect logic needs to work
before and after LIT lowering, so it mostly needs to be sugar tolerant.

## Working with Sugar in the Mojo Compiler

Ok, now that we know how sugar is represented, what does a journeyman compiler
engineer need to know when working on the Mojo compiler? Here are a few things
to keep in mind:

### Use the right kind of equality checks

Sugar inherently breaks pointer equality for semantic equality checks, so you
need to make sure not to use `==` unless you're looking for a "user visible
equality" - this is necessary (e.g. in the diagnostic subsystem) but isn't
what most semantic analysis needs to think about.

For these purposes, make sure to use the `isEqualCanon` style of methods. You
can also use `getCanonicalType` on both types and compare them, but please use
`isEqualCanon` where possible for consistency, and because we can make it
slightly more efficient.

### Use the right kinds of dynamic casts

The Mojo parser inherently has to handle different cases in some places, so it
needs to check things like "is this a struct type? Is this a trait type? is this
a function type?" etc. However, sugar can wrap any of these, and these semantic
checks need to look through sugar - `dyn_cast` and friends looks at the
structure of the type/parameter, not at the semantic meaning.

To solve this, KGENAttrs.h defines a bunch of these casts that work in a
sugar-tolerant way, including `sugarIsa`, `sugarIsaAndNonNull`, `sugarDynCast`,
`sugarDynCastIfPresent` etc. These check to see if the top level type/attr is
a `SugarAttr` and looks through it if so.

Note that `dyn_cast` (and friends) still works on attributes and types, and also
that it must be used on MLIR operations like `dyn_cast<StructDeclOp>(someOp)`:
the sugar-aware casts are almost always the right thing to do for type checking,
but there are some cases where you want to look at the structural form.

### Sugar is precious, don't ruthlessly throw it away

You might be tempted to just build some new feature with the philosophy of
"just give me the canonical form of this type, that ensures that I don't have
to worry about sugar". This will "work", but please don't do this - we want
our users to have a good time using Mojo and sugar preservation is an important
thing to do.

To help with this, there are a few tips:

1) `SugarAttr::strip` will strip off top level sugar without removing nested
   sugar. For example, it will remove an alias wrapping a function type if it
   is in the way, but won't remove all sugar from the argument types of the
   function.
2) `sugarDynCast` and related operations use `SugarAttr::strip` so they do
   preserve nested sugar.
3) We can reconstruct sugar in specific cases to propagate it (grep the codebase
   for `SugarAttr::get` to see examples of that. Please add a testcase for this
   when it comes up so we make sure not to regress.

Please continue to improve sugar preservation as Mojo evolves!

### Beware MLIR invariants

One last detail to know about - various bits of MLIR (correctly!)
expect pointer equality for things like types. One very simple example of this
is that the `lit.return` op is defined to check that its operand has the same
return type as the enclosing function.

As a consequence of this, there are a few places in IR generation that need to
introduce rebind operations/attributes. These are harmless (and removed by
LowerLIT).

You might wonder why we don't just fix these as they come up? This is a good
point and there are multiple different answers:

1) Some of the logic we work with are foreign dialects like the `index` dialect.
   We can't expect `index.add` to take an SugarAttr wrapping an index type. As
   a consequence of this, the processing of `__mlir_op` strips all sugar when
   forming arbitrary MLIR operations/attributes.
2) For specific cases like `lit.return`, we can and should go ahead and update
   it to use sugar equality, please do!
3) For other cases, we use structural-type-equality to simplify printing and
   parsing of the MLIR form. For example, in a ParamOperatorAttr::apply, we
   expect all the of the operands to be "the same type" as the contextual
   signature type. Allowing sugar in the way would make the printed KGEN
   representation a lot more verbose.

The solution to these problems are a few rebinds, which work well in practice.

### We choose to elide some sugar

While sugar is a good thing and the compiler should support it in arbitrary
places in the parameter/type graph, too much sugar isn't good for us either:
it can bloat the IR, it can be noisy for testcases, etc.

As such, we intentionally elide sugar from a few places:

1) `comptime` aliases that start with an underscore are assumed to be internal
   implementation details. Notably things like `_mlir_type` cause a lot of
   clutter for primitive types and are generally a nuisance.

2) We have some heuristics to collapse away things that simplify, e.g. if
   something simplifies to a parameter, it is omitted (e.g. the builtin function
   call for `X+0` gets turned into `X`).

3) Types can implement the `SugaredTypeInterface.canElideSugarFor` interface +
   hook to do custom logic. For example, it becomes onerous to sugar primitive
   origin expression and simple constants. The hook allows elision for
   builtin-function-only (so 4+5 always folds to 9) and also to omit aliases
   that are "simple enough" to not be worth propagating around.
