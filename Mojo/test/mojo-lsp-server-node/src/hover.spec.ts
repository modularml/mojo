import * as assert from "assert";
import * as assertExtras from "./assertExtras";
import { Document, LanguageServer } from "./harness";
import { MarkupContent, Range } from "vscode-languageserver-protocol";

describe("hover", function () {
  let server: LanguageServer;

  beforeEach("start and connect to language server", async function () {
    server = new LanguageServer();
    await server.initialize();
  });

  afterEach("stop language server", async function () {
    await server.stop();
  });

  it("should get information for variables", async function () {
    let doc = new Document(
      server,
      "test:///test.mojo",
      `
def function():
  var something: Int = 100
  var other = 1 + \`something\`
  print(other)
`
    );

    await doc.open();

    assert.deepStrictEqual(
      await doc.hover(doc.findFirstPosition("something")),
      {
        contents: {
          kind: "markdown",
          value: "```mojo\n(variable) var something: Int\n```",
        },
        range: Range.create(2, 6, 2, 15),
      }
    );

    assert.deepStrictEqual(await doc.hover(doc.findFirstPosition("other")), {
      contents: {
        kind: "markdown",
        value: "```mojo\n(variable) var other: Int\n```",
      },
      range: Range.create(3, 6, 3, 11),
    });
  });

  it("should get information for function declarations", async function () {
    let doc = await Document.fromFile(
      server,
      "Mojo/test/mojo-lsp-server-node/data/standalone/functions.mojo"
    );
    await doc.open();

    assert.deepStrictEqual(await doc.hover(doc.findFirstPosition("__init__")), {
      contents: {
        kind: "markdown",
        value: `\`\`\`mojo
(function) def __init__(out self, borrowed_input: Int, init_arg: Int, var owned_input: Int, *init_kargs: Int)
\`\`\`
---

###
Init documentation.

#### Args:
&nbsp;&nbsp;borrowed_input: An imm argument.
\\
&nbsp;&nbsp;init_arg: An Int argument.
\\
&nbsp;&nbsp;owned_input: An owned argument.
\\
&nbsp;&nbsp;init_kargs: Multiple arguments.

`,
      },
      range: doc.findFirstRange("__init__"),
    });

    assert.deepStrictEqual(
      await doc.hover(doc.findFirstPosition("static_method")),
      {
        contents: {
          kind: "markdown",
          value: `\`\`\`mojo\n(function) def static_method() -> Int\n\`\`\``,
        },
        range: doc.findFirstRange("static_method"),
      }
    );

    assert.deepStrictEqual(
      await doc.hover(doc.findFirstPosition("async_function")),
      {
        contents: {
          kind: "markdown",
          value: `\`\`\`mojo\n(function) async def async_function(mut self)\n\`\`\``,
        },
        range: doc.findFirstRange("async_function"),
      }
    );

    assert.deepStrictEqual(
      await doc.hover(doc.findFirstPosition("non_capturing_nested_function")),
      {
        contents: {
          kind: "markdown",
          value: `\`\`\`mojo\n(function) def non_capturing_nested_function()\n\`\`\``,
        },
        range: doc.findFirstRange("non_capturing_nested_function"),
      }
    );

    assert.deepStrictEqual(
      await doc.hover(doc.findFirstPosition("parameter_nested_function")),
      {
        contents: {
          kind: "markdown",
          value: `\`\`\`mojo\n(function) def parameter_nested_function()\n\`\`\``,
        },
        range: doc.findFirstRange("parameter_nested_function"),
      }
    );

    assert.deepStrictEqual(
      await doc.hover(doc.findFirstPosition("another_nested_function")),
      {
        contents: {
          kind: "markdown",
          value: `\`\`\`mojo\n(function) def another_nested_function()\n\`\`\``,
        },
        range: doc.findFirstRange("another_nested_function"),
      }
    );

    assert.deepStrictEqual(
      await doc.hover(doc.findFirstPosition("function_that_raises")),
      {
        contents: {
          kind: "markdown",
          value: `\`\`\`mojo
(function) def function_that_raises(mut self, arg_in_function_that_raises: Int) raises -> String
\`\`\`
---

###
A function that raises.

#### Args:
&nbsp;&nbsp;arg_in_function_that_raises: An arg in a function with by-ref result.

`,
        },
        range: doc.findFirstRange("function_that_raises"),
      }
    );

    assert.deepStrictEqual(
      await doc.hover(doc.findFirstPosition("function_with_param")),
      {
        contents: {
          kind: "markdown",
          value: `\`\`\`mojo
(function) def function_with_param[Param1: Int, Param2: Int](mut self)
\`\`\`
---

###
A function with param.

#### Parameters:
&nbsp;&nbsp;Param1: An Int param.
\\
&nbsp;&nbsp;Param2: Another Int param.

`,
        },
        range: doc.findFirstRange("function_with_param"),
      }
    );

    assert.deepStrictEqual(
      await doc.hover(doc.findFirstPosition("exported_function")),
      {
        contents: {
          kind: "markdown",
          value: `\`\`\`mojo
(function) def exported_function()
\`\`\`
---

###
This is an exported function.

`,
        },
        range: doc.findFirstRange("exported_function"),
      }
    );

    assert.deepStrictEqual(
      await doc.hover(doc.findFirstPosition("def_function")),
      {
        contents: {
          kind: "markdown",
          value: `\`\`\`mojo\n(function) def def_function() raises -> Int\n\`\`\``,
        },
        range: doc.findFirstRange("def_function"),
      }
    );
  });

  it("should get information for structs", async function () {
    let doc = await Document.fromFile(
      server,
      "Mojo/test/mojo-lsp-server-node/data/standalone/functions.mojo"
    );
    await doc.open();

    assert.deepStrictEqual(
      await doc.hover(doc.findFirstPosition("SomeStruct")),
      {
        contents: {
          kind: "markdown",
          value: `\`\`\`mojo
struct SomeStruct[size: Int, other_param: Bool]
# Traits: AnyType, Deinitable, Movable
\`\`\`
---

###
Docstring for SomeStruct.

More docstring for SomeStruct.


#### Parameters:
&nbsp;&nbsp;size: The size of SomeStruct.
\\
&nbsp;&nbsp;other_param: Another param.

#### Constraints:
&nbsp;&nbsp;The constraints of SomeStruct.


`,
        },
        range: doc.findFirstRange("SomeStruct"),
      }
    );
  });

  it("should get information for aliases", async function () {
    let doc = await Document.fromFile(
      server,
      "Mojo/test/mojo-lsp-server-node/data/package/aliases.mojo"
    );
    await doc.open();

    assert.deepStrictEqual(await doc.hover(doc.findFirstPosition("IntAlias")), {
      contents: {
        kind: "markdown",
        value: `\`\`\`mojo
comptime IntAlias = 12
\`\`\`
---

###
Int alias summary

Int alias description.

`,
      },
      range: doc.findFirstRange("IntAlias"),
    });

    assert.deepStrictEqual(
      await doc.hover(doc.findFirstPosition("ExplicitIntAlias")),
      {
        contents: {
          kind: "markdown",
          value: "```mojo\ncomptime ExplicitIntAlias = Int(123)\n```",
        },
        range: doc.findFirstRange("ExplicitIntAlias"),
      }
    );

    assert.deepStrictEqual(
      await doc.hover(doc.findFirstPosition("AliasInsideFunction")),
      {
        contents: {
          kind: "markdown",
          value: '```mojo\ncomptime AliasInsideFunction = "sdfsdf"\n```',
        },
        range: doc.findFirstRange("AliasInsideFunction"),
      }
    );

    assert.deepStrictEqual(
      await doc.hover(doc.findFirstPosition("AliasToAlias")),
      {
        contents: {
          kind: "markdown",
          value: "```mojo\ncomptime AliasToAlias = IntAlias\n```",
        },
        range: doc.findFirstRange("AliasToAlias"),
      }
    );

    assert.deepStrictEqual(
      await doc.hover(doc.findFirstPosition("AliasInStruct")),
      {
        contents: {
          kind: "markdown",
          value: "```mojo\ncomptime AliasInStruct = Int\n```",
        },
        range: doc.findFirstRange("AliasInStruct"),
      }
    );
  });

  it("should get information for struct fields", async function () {
    let doc = new Document(
      server,
      "test:///test.mojo",
      `
struct SomeStruct:
    var a_field: Int
    """Summary of a_field."""

    def __init__(out self):
        pass


def main():
    var someStruct = SomeStruct()
    _ = someStruct.a_field
`
    );

    await doc.open();

    const hoverContents = {
      kind: "markdown",
      value: `\`\`\`mojo
(field) var a_field: Int
\`\`\`
---

###
Summary of a_field.

`,
    };

    // This hovers over the initial declaration
    assert.deepStrictEqual(await doc.hover(doc.findFirstPosition("a_field")), {
      contents: hoverContents,
      range: doc.findFirstRange("a_field"),
    });

    // This hovers over the usage of the struct field in main().
    // We differentiate from the declaration using .
    let usageRange = doc.findFirstRange(".a_field");
    // Skip the preceding .
    usageRange.start.character += 1;
    assert.deepStrictEqual(await doc.hover(usageRange.start), {
      contents: hoverContents,
      range: usageRange,
    });
  });

  it("should get information for function arguments", async function () {
    let doc = await Document.fromFile(
      server,
      "Mojo/test/mojo-lsp-server-node/data/standalone/functions.mojo"
    );
    await doc.open();

    assert.deepStrictEqual(await doc.hover(doc.findFirstPosition("self")), {
      contents: {
        kind: "markdown",
        value: "```mojo\n(argument) out self\n```",
      },
      range: doc.findFirstRange("self"),
    });

    assert.deepStrictEqual(
      await doc.hover(doc.findFirstPosition("borrowed_input")),
      {
        contents: {
          kind: "markdown",
          value: `\`\`\`mojo
(argument) borrowed_input: Int
\`\`\`
---

###
An imm argument.

`,
        },
        range: doc.findFirstRange("borrowed_input"),
      }
    );

    assert.deepStrictEqual(await doc.hover(doc.findFirstPosition("init_arg")), {
      contents: {
        kind: "markdown",
        value: `\`\`\`mojo
(argument) init_arg: Int
\`\`\`
---

###
An Int argument.

`,
      },
      range: doc.findFirstRange("init_arg"),
    });

    assert.deepStrictEqual(
      await doc.hover(doc.findFirstPosition("init_kargs")),
      {
        contents: {
          kind: "markdown",
          value: `\`\`\`mojo
(argument) *init_kargs: Int
\`\`\`
---

###
Multiple arguments.

`,
        },
        range: doc.findFirstRange("init_kargs"),
      }
    );

    assert.deepStrictEqual(
      await doc.hover(doc.findFirstPosition("owned_input")),
      {
        contents: {
          kind: "markdown",
          value: `\`\`\`mojo
(argument) var owned_input: Int
\`\`\`
---

###
An owned argument.

`,
        },
        range: doc.findFirstRange("owned_input"),
      }
    );

    assert.deepStrictEqual(
      await doc.hover(doc.findFirstPosition("arg_in_function_that_raises")),
      {
        contents: {
          kind: "markdown",
          value: `\`\`\`mojo
(argument) arg_in_function_that_raises: Int
\`\`\`
---

###
An arg in a function with by-ref result.

`,
        },
        range: doc.findFirstRange("arg_in_function_that_raises"),
      }
    );

    assert.deepStrictEqual(await doc.hover(doc.findFirstPosition("Param1")), {
      contents: {
        kind: "markdown",
        value: `\`\`\`mojo
(parameter) Param1: Int
\`\`\`
---

###
An Int param.

`,
      },
      range: doc.findFirstRange("Param1"),
    });

    assert.deepStrictEqual(await doc.hover(doc.findFirstPosition("Param2")), {
      contents: {
        kind: "markdown",
        value: `\`\`\`mojo
(parameter) Param2: Int
\`\`\`
---

###
Another Int param.

`,
      },
      range: doc.findFirstRange("Param2"),
    });
  });

  it("should get information for imported packages", async function () {
    let doc = await Document.fromFile(
      server,
      "Mojo/test/mojo-lsp-server-node/data/package/imports.mojo"
    );
    await doc.open();

    const simd = await doc.hover(doc.findFirstPosition("simd"));
    assert.ok(MarkupContent.is(simd!.contents));
    assert.ok(
      simd!.contents.value.indexOf(
        "Implements SIMD primitives and abstractions"
      ) !== -1
    );

    const simdAlias = await doc.hover(doc.findFirstPosition("_simd"));
    assert.deepStrictEqual(simd!.contents, simdAlias!.contents);

    assert.deepStrictEqual(await doc.hover(doc.findFirstPosition("aliases")), {
      contents: {
        kind: "markdown",
        value: "### module `aliases`\n",
      },
      range: doc.findFirstRange("aliases"),
    });

    assert.deepStrictEqual(await doc.hover(doc.findFirstPosition("function")), {
      contents: {
        kind: "markdown",
        value: "```mojo\n(function) def function() -> Int\n```",
      },
      range: doc.findFirstRange("function"),
    });

    assert.deepStrictEqual(
      await doc.hover(doc.findFirstPosition("StructWithAlias")),
      {
        contents: {
          kind: "markdown",
          value: `\`\`\`mojo
struct StructWithAlias
# Traits: AnyType, Deinitable, Movable
\`\`\``,
        },
        range: doc.findFirstRange("StructWithAlias"),
      }
    );
  });

  it("should get information for external symbols", async function () {
    let doc = await Document.fromFile(
      server,
      "Mojo/test/mojo-lsp-server-node/data/package/aliases.mojo"
    );
    await doc.open();

    assert.deepStrictEqual(await doc.hover(doc.findFirstPosition("LAZY")), {
      contents: {
        kind: "markdown",
        value: `\`\`\`mojo
comptime LAZY = 1
\`\`\`
---

###
Load library lazily (defer function resolution until needed).

`,
      },
      range: doc.findFirstRange("LAZY"),
    });

    assert.deepStrictEqual(
      await doc.hover(doc.findFirstPosition("ExternalAlias")),
      {
        contents: {
          kind: "markdown",
          value: "```mojo\ncomptime ExternalAlias = RTLD.LAZY\n```",
        },
        range: doc.findFirstRange("ExternalAlias"),
      }
    );
  });

  it("should get information for function calls", async function () {
    const doc = new Document(
      server,
      "test:///test.mojo",
      `
def print(x: StringLiteral):
    pass

def print(x: Bool):
    pass

def function[type: TrivialRegisterPassable](arg: type):
    print("string")
    print(arg)
`
    );
    await doc.open();

    assert.deepStrictEqual(
      (await doc.hover(doc.findFirstPosition("print(")))!.contents,
      {
        kind: "markdown",
        value: "```mojo\n(function) def print(x: StringLiteral)\n```",
      }
    );

    assert.deepStrictEqual(
      (await doc.hover(doc.findFirstPosition("print(arg")))!.contents,
      {
        kind: "markdown",
        value: `\`\`\`mojo
(function) def print(x: StringLiteral)
\`\`\`
---

\`\`\`mojo
(function) def print(x: Bool)
\`\`\``,
      }
    );
  });

  it("should handle complex function types", async function () {
    const doc = new Document(
      server,
      "test:///test.mojo",
      `
def function[
    func: def (Int) capturing -> Int
]() -> def (Int) capturing -> Int:
    pass
`
    );
    await doc.open();

    assert.deepStrictEqual(
      (await doc.hover(doc.findFirstPosition("function")))!.contents,
      {
        kind: "markdown",
        value:
          "```mojo\n(function) def function[func: def(Int) capturing thin -> Int]() -> def(Int) capturing thin -> Int\n```",
      }
    );
  });

  it("should handle named function types", async function () {
    const doc = new Document(
      server,
      "test:///test.mojo",
      `
def fn1[f: def [p1: DType](foo: Scalar[p1]) -> type_of(foo)]():
  ...


def fn2[f: def [dt: DType, dt2: Int](arg1: Scalar[dt], arg2: Int) -> None]():
  ...
`
    );
    await doc.open();

    assert.deepStrictEqual(
      (await doc.hover(doc.findFirstPosition("p1")))!.contents,
      {
        kind: "markdown",
        value: "```mojo\n(parameter) p1: DType\n```",
      }
    );

    assert.deepStrictEqual(
      (await doc.hover(doc.findFirstPosition("foo")))!.contents,
      {
        kind: "markdown",
        value: "```mojo\n(argument) foo: Scalar[p1]\n```",
      }
    );

    assert.deepStrictEqual(
      (await doc.hover(doc.findFirstPosition("arg2")))!.contents,
      {
        kind: "markdown",
        value: "```mojo\n(argument) arg2: Int\n```",
      }
    );
  });

  it("should display inferred parameters", async function () {
    const doc = new Document(
      server,
      "test:///test.mojo",
      `
@always_inline
def parametric[
    type: DType, simd_width: Int, //, other: Int
](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    return x * x


def foo():
    var v = SIMD[DType.float16, 4](33)
    _ = parametric[12](v)
`
    );
    await doc.open();

    assert.deepStrictEqual(
      (await doc.hover(doc.findLastPosition("parametric")))!.contents,
      {
        kind: "markdown",
        value:
          "```mojo\n(function) def parametric[type: DType, simd_width: Int, //, other: Int](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]\n```",
      }
    );
  });

  it("should pretty-print implicit __getitem__ invocations", async function () {
    const doc = new Document(
      server,
      "test:///test.mojo",
      `
from layout import IntTuple

struct Foo:
    var t: IntTuple

    def __init__(out self: Foo):
        self.t = IntTuple()

def baz():
    comptime f = Foo()
    comptime ft = f.t[0]
`
    );
    await doc.open();

    // Get hover info for `comptime ft = f.t[0]`
    const hover = await doc.hover(doc.findFirstPosition("ft"));
    assert.ok(MarkupContent.is(hover!.contents));
    // The hover info must not contain __getitem__; it should instead
    // pretty-print the indexing expression.
    assertExtras.contains(hover!.contents.value, "f.t[0]");
    assertExtras.doesNotContain(hover!.contents.value, "__getitem__");
  });

  it("should show conditional conformance conditions", async function () {
    const doc = new Document(
      server,
      "test:///test.mojo",
      `
trait Serializable:
    def serialize(self) -> Int:
        ...


@fieldwise_init
struct CondCopyable[T: Movable & Deinitable](
    Copyable where conforms_to(T, Copyable)
):
    var value: Self.T


@fieldwise_init
struct CondAnd[N: Int, T: Movable & Deinitable](
    Serializable where conforms_to(T, Serializable) and N > 4
):
    var value: Int

    def serialize(self) -> Int:
        return 0


@fieldwise_init
struct CondWhereClause[N: Int](Serializable where N > 4) where N < 10:
    var value: Int

    def serialize(self) -> Int:
        return 0
`
    );
    await doc.open();

    // The compiler-injected AnyType/Deinitable/Movable conformances stay
    // unconditional, so only the declared one carries a condition.
    assert.deepStrictEqual(
      (await doc.hover(doc.findFirstPosition("CondCopyable")))!.contents,
      {
        kind: "markdown",
        value: `\`\`\`mojo
struct CondCopyable[T: Deinitable & Movable]
# Traits: AnyType, Copyable (where conforms_to(T, Copyable)), Deinitable, Movable
\`\`\``,
      }
    );

    const condAnd = await doc.hover(doc.findFirstPosition("CondAnd"));
    assert.deepStrictEqual(condAnd!.contents, {
      kind: "markdown",
      value: `\`\`\`mojo
struct CondAnd[N: Int, T: Deinitable & Movable]
# Traits: AnyType, Deinitable, Movable, Serializable (where conforms_to(T, Serializable) and (N > Int(4)))
\`\`\``,
    });
    // The compiler lowers `a and b` to the ternary `b if a else a`. Recovering
    // the operator is what keeps that spelling out of the hover.
    assert.ok(MarkupContent.is(condAnd!.contents));
    assertExtras.doesNotContain(condAnd!.contents.value, " if ");

    // A conformance condition and the struct's own trailing `where` clause are
    // separate constraints, and each renders in its own place.
    assert.deepStrictEqual(
      (await doc.hover(doc.findFirstPosition("CondWhereClause")))!.contents,
      {
        kind: "markdown",
        value: `\`\`\`mojo
struct CondWhereClause[N: Int] where (N < Int(10))
# Traits: AnyType, Deinitable, Movable, Serializable (where (N > Int(4)))
\`\`\``,
      }
    );
  });
});
