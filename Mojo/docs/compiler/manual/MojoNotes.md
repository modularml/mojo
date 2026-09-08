# Mojo🔥 Notes

Mojo is intended to evolve into a superset of Python, which adds
first-class support for static types, "structs" with zero-cost abstraction
features, and support for kgen-parameters and search.

That said, it is still in early development and is missing many features. This
document is intended to track notes about its ongoing development.

## Intentional differences from Python

Mojo is generally a superset of Python, but here are some intentional
incompatibilities as well as extensions. These are subject to discussion and
re-evaluation over time.

### Intentional Incompatibilities with Python already supports

1) We do not support all the deprecated features of the Python lexer.

2) We treat "soft" keywords like `case` as "hard" keywords in Mojo.
   Rationale is that we don't want things like "case = 42" to be supported.
   Python supports this for backwards compatibility reasons, but we can handle
   this in the future Python -> Mojo translator tool by (e.g.) adding an
   backticks around the keywords.

### New Features / Extensions that Python doesn't have

In addition to specific differences, we support the following extensions:

1) `struct` definitions, not just classes. Structs are stored inline in their
   containing type/frame instead of indirectly with a reference count. They
   don't support inheritance, and therefore are suitable for lots of
   optimization and things built with zero-cost abstractions.

2) We support type parameters being declared on function definitions, ala
   `def method[size: Int]`. This may be standardized into Python as [PEP
   695](https://peps.python.org/pep-0695/).

3) In addition to the builtin dynamic Python object types like "int" and "dict",
   we will have library-defined static versions named "Int" and "Dict" etc that
   are defined as Mojo structs in the library. These implementations will
   be very similar to the Python types in surface syntax, but will have type
   parameters and may have different behavior in some cases (TBD).

4) `var` definitions for local variables and instance properties in structs:
   We support implicit local variable definitions, but allow them to be
   optionally explicitly declared as well. We support the `alias` keyword for
   definition named parameters.

5) In addition to loosely typed `def` statements, we support a more strict `fn`
   statement, described below.

6) We support "backtick" identifiers
([like Swift](https://docs.swift.org/swift-book/ReferenceManual/LexicalStructure.html#:~:text=To%20use%20a%20reserved%20word,x%20have%20the%20same%20meaning.))
which allow using keywords and lexically invalid strings of text as a single
identifier. This is useful when migrating code from Python that uses new
keywords (e.g. fn), and is a generally useful escape hatch.

7) We support direct access to MLIR concepts including operations (with
   `__mlir_op`), types (`__mlir_type`) and attributes (`__mlir_attr`), described
   in more detail below.

8) We have more generous indentation rules, not requiring `\` at end of line
   in most cases, due to a more sophisticated lexer rule that allows expression
   continuation so long as the continuation is more indented than the start of
   the expression.

9) We support "dictionary subscripts" of the form
   `expr{key: value, key2: value2}`, used for type constructors. We aim to
   remove this over time now that initializers are more mature. We may instead
   support `List[1,2,3]` and `Dict{key: value}` sorts of expressions to create
   literals with a specific type, rather than having to default to one type.

## Various Design notes

### New `fn` introducer

Mojo supports a new `fn` introducer that can be used in place of `def`.
Both form of declaration have the same capabilities - a `fn` can include dynamic
operations and interact with Python objects directly.

`fn` is effectively a more strict `def` that has different defaults. Notably
a `fn` disables the following behavior that `def` maintains for Python
compatibility / familiarity, and to make it easier to migrate large swaths of
Python code to Mojo someday.

More specific differences:

1) `fn` definitions require type annotations on the function signature. In a
   `def`, they are optional and default to `object`.
2) Formal arguments in a `def` are mutable l-values, in a `fn` they are
   immutable.
3) `def` definitions allow local variables to be implicitly declared on their
   first use, a `fn` definition requires `let` and `var` introducers.

The choice of `fn` for this is arbitrary - we could have picked `func` (like
Swift/Go) or `function` like Javascript. `fn` aligns with Rust/Zig which are
cool kids on the block these days.

### Main

In Mojo the main function has signature `def main()`.
That function is exported automatically with the unique symbol "main" in the
final object file. This is subject to change as we close the gap with Python's
functionality.

### The full power of MLIR at your fingertips

A design goal of the Mojo is to provide "syntactic sugar" that makes it
much easier to write KGEN kernels than writing MLIR directly. Our approach to
solve this is to provide library developers full access to low-level MLIR
constructs, and allow them to build domain specific zero-cost abstractions on
top of them (wrapping all the low level MLIR stuff in lovely Python syntax). As
part of this, even "built in" types like `Bool` and `Int` are defined in the
standard library.

There are a few different components of this, which we explain here:

**Types:**

User defined types in Mojo are defined as structs (and eventually classes,
variants, etc). These all turn into an MLIR `lit.struct.decl` operation, and
references to them use the `!lit.struct<@Symbol>` type, e.g. this:

```python
struct EmptyStruct: pass
def test(a: EmptyStruct): pass
```

compiles into:

```mlir
lit.struct.decl @EmptyStruct {}
lit.fn @static(%x: !lit.struct<@EmptyStruct>) {..}
```

Beyond user defined types, the entire MLIR type system is exposed using the
`__mlir_type` magic identifier. With it, you can access named
types (in full MLIR syntax) as named properties. The backticks syntax for an
identifier are helpful for accessing things that don't fit into a standard
Python identifier syntax:

```python
def takeMLIRTypes(a: __mlir_type.f32,
                 b: __mlir_type.`!kgen.pointer<!kgen.scalar<ui32>>`): pass

struct Bool:
  var value : __mlir_type.i1
struct Int:
  var value : __mlir_type.index
struct F32:
  var value : __mlir_type.`!kgen.scalar<f32>`
```

The "takeMLIRTypes" function doesn't take values of user defined struct type, it
takes SSA values of the named MLIR types directly. This means that general
values in Mojo can have MLIR type, which is the basis for defining "nice"
user defined types like `Bool`, `Int` and `F32` respectively. Note that values
of MLIR type don't have any methods or properties on them, they are useful as
storage types and as input and outputs of MLIR operations.

**Attributes:**

MLIR's attribute system is a powerful, [extensible by
dialects](https://mlir.llvm.org/docs/AttributesAndTypes/), and a key part of the
domain abstraction in an MLIR abstraction. MLIR ODS provides a ton of syntactic
sugar for working with these, but they have a very simple core model. KGEN uses
MLIR attributes as the core representation for meta values that are determined
at compile time with elaboration and search.

Mojo exposes this whole system directly with the `__mlir_attr` magic
identifier, and you may use any `TypedAttr` as a general purpose meta or dynamic
value in Mojo. You can see this most easily when materializing a constant
value into a dynamic one, e.g. when storing into a variable:

```mlir
  # CHECK: %d = lit.varlet.decl "d" var : !lit.ref<mut i17, ...
  # CHECK: [[TMP:%.*]] = kgen.param.constant: i17 = <4>
  # CHECK: lit.ref.store [[TMP]], %d : !kgen.pointer<i17>
  var d = __mlir_attr.`4: i17`

  # CHECK: %dt = lit.varlet.decl "dt" var : !lit.ref<mut dtype, ...
  # CHECK: [[TMP:%.*]] = kgen.param.constant: dtype = <f32>
  # CHECK: lit.ref.store [[TMP]], %dt  : !kgen.pointer<dtype>
  var dt = __mlir_attr.`#kgen.dtype.constant<f32> : !kgen.dtype`
```

Note that this is making a 17-bit integer MLIR attribute, and a dialect specific
attribute of `!kgen.dtype` type.

When used in the attribute list of an operation (see below), you have access to
arbitrary attributes, you aren't limited to TypedAttr.

**Operations:**

Of course, a big part of MLIR is the definition of dialect operations, which
define the core compute plane. Mojo gives you direct access to this with
the `__mlir_op` magic identifier, which yields an MLIR operation identifier that
may optionally have attributes added to it (with subscript syntax) or be applied
to zero-or-more SSA values with call syntax, e.g. using the [`index.sub`
op](https://mlir.llvm.org/docs/Dialects/IndexOps/#indexsub-mlirindexsubop):

```mlir
def subtractIndexes(lhs: __mlir_type.index, rhs: __mlir_type.index) -> __mlir_type.index:
  return __mlir_op.`index.sub`(lhs, rhs)
```

Attributes can be specified by passing them as a list of key/value pairs using
subscript syntax (including specifying multiple attributes such as
`[a: 1, b: 42, c: 12]`). For example, to get a constant value using the
[`index.constant` op](https://mlir.llvm.org/docs/Dialects/IndexOps/#indexconstant-mlirindexconstantop),
you can use:

```mlir
  # CHECK: %idxConstant = lit.varlet.decl "idxConstant" var var = true : !lit.ref<mut index, ...
  # CHECK-NEXT: [[TMP:%.*]] = index.constant 42
  # CHECK-NEXT: lit.ref.store [[TMP]], %idxConstant
  var idxConstant = __mlir_op.`index.constant`[value: 42]()
```

Note that some dialects use a *lot* of syntactic sugar that can obscure the
low-level representation of an MLIR operation. If you're struggling with how
something is represented, it is helpful to ask MLIR to print IR in with the
`--mlir-print-op-generic` command line flag. For example, it was difficult to
understand how to do a comparison, which uses a custom attribute for the
comparison condition code. I used:

```sh
$ kgen-opt  --mlir-print-op-generic KGEN/test/pop-ir/pop-ops.mlir | grep pop.cmp
    %0 = "pop.cmp"(%arg0, %arg1) {pred = #pop<cmp_pred ge>} : (!kgen.scalar<f32>, !kgen.scalar<f32>) -> !kgen.scalar<bool>
...
```

which told me how to spell a comparison. This is accessible in Mojo like
this:

```Python
def cmp(lhs: __mlir_type.`!kgen.scalar<f32>`,
       rhs: __mlir_type.`!kgen.scalar<bool>`) -> __mlir_type.`!kgen.scalar<bool>`:
  return _mlir_op.`pop.cmp`[pred: __mlir_attr.`#pop<cmp_pred ge>`](lhs, rhs)
```

This is a bit grotty, but given low level access to this (with zero abstraction
cost!) you can define your own libraries on top of these things and only have
to understand this the first time you define the wrappers.

One final topic that we glossed over is that MLIR needs to know the result types
for an operation when you create it, and Mojo similarly needs to know the
type of an expression. Mojo handles this by using the
`InferTypeOpInterface` to figure out the result type whenever possible, which
handles many common cases very nicely. However, some operations don't have
an implementation of this interface because they can return multiple types.

To support this, Mojo allows you to define a `_type` attribute with the
result type to use, e.g. to cast an index value to `i1` you can use:

```mojo
  # CHECK: [[TMP:%.*]] = pop.load %idxConstant
  # CHECK: [[TMP2:%.*]] = index.castu [[TMP:%.*]] : index to i1
  var i1Cast = __mlir_op.`index.castu`[_type: __mlir_type.i1](idxConstant))
```

### Expression parsing happens in two phases

Python uses its expression grammar for value expressions and for types. This is
quite convenient for Mojo 🔥 given we want types to be parameter values!
That said, there are some annoyances to deal with in terms of how to handle
this, for example, Python allows:

1) Values may be lexically used before they are defined, e.g. in expressions
   like `[x*x for x in range(42)]`
2) Code generation does not follow order of emission, e.g. in expressions like
   `x() if cond() else y()` where the `cond()`
   expression is evaluated first, then x/y are evaluated conditionally based on
   that.
3) As mentioned above, the expression grammar may resolve into a type or value
   depending on context. Usually this doesn't matter, but we want to resolve
   expressions like `()` or `None` into different things in type and expression
   contexts.
4) As described below, we need to be able to parse the structure of a file
   before resolving types.

To handle all these problems we have a two phase resolution of expressions: we
first parse them into a bump pointer allocated tree data structure (defined
in `ExprNodes.h`) and we can then "codegen" them into SSA expressions or into
a type. This second phase is what performs name lookup etc, which means we can
parse the expression (and then ignore it) even before name binding.

### Structure of parsing + name binding + type checking

Python supports forward references to declarations in a file and/or module.
It handles this by making everything be dynamically executable (including `def`s
which are "executed" to install them in the dictionary for a class) and does not
actually type expressions statically. This works for Python, but won't work for
Mojo 🔥, and we can't give up support for forward references.

As such, we currently handle this by parsing the source file in three phases:

1) Declaration structure parsing.
2) Name binding + resolution of type expressions in declarations.
3) Parsing of values within those declarations (notably, function bodies and
   default initializers).

Let's take a look at an example to illustrate how this works:

```mojo
struct Int:   # Defined by stdlib.
  pass

def frolick(d: Doggie[42, Color(0, 255, 0)])
  print(d.furColor, d.numSpots*2)

struct Doggie[NumSpots: Int, FavoriteColor: Color]
  var furColor : Color
  cst numSpots : Int = NumSpots

struct Color:
  var r, g, b : Int

```

This example shows forward references of the `Color` type from the declaration
of the FavoriteColor parameter and `furColor` instance variable, as well as the
initializer expression for the parameter in the type list of `frolick` and
from the `d.color` usage in the `print`. The reference to `Doggie` first occurs
in the argument list for `frolick` etc. To support this, the first pass just
resolves the top level declaration names and structures, deferring parsing and
resolution of types and value expressions.

It parses and builds IR for these declarations:

```mojo
struct Int:
  SKIPPED

def frolick(d: SKIPPED)
  SKIPPED

struct Doggie[NumSpots: SKIPPED, FavoriteColor: SKIPPED]
  var furColor : SKIPPED
  cst numSpots : SKIPPED = SKIPPED

struct Color:
  var r, g, b : SKIPPED

```

In the second pass we reparse the type expressions (which is nicely efficient
given how our parser works), allowing us to "see" this much of the example:

```mojo
struct Int:
  SKIPPED

def frolick(d: Doggie[42, Color(0, 255, 0)])
  SKIPPED

struct Doggie[NumSpots: Int, FavoriteColor: Color]
  var furColor : Color
  cst numSpots : Int = SKIPPED

struct Color:
  var r, g, b : Int

```

Note that the deferred parsing and resolution of types cannot proceed in lexical
order: we need to resolve the type of `r/g/b` in the Color struct before we can
resolve the initializer expression in the parameter list of `d` in `frolick`.
This means 1) We need to do this in a worklist order, and 2) we can have cycles
which we need to identify and reject.

Once this is completed, we can parse and type check the remaining initializer
expressions / bodies which are all self contained in different scopes. This can
be done in parallel.
