import * as assert from "assert";
import { Document, LanguageServer } from "./harness";
import { SymbolKind } from "vscode-languageserver-protocol";

describe("document symbols", () => {
  let server: LanguageServer;

  beforeEach("start and connect to language server", async () => {
    server = new LanguageServer();
    await server.initialize();
  });

  it("should not crash when importing the current document", async function() {
    const doc = new Document(server, "test:///test.mojo", `
from . import test
`);
    await doc.open();
    // Nothing more to do here; we just need the server to not crash.
  });

  it("should get document symbols", async function() {
    const doc = new Document(
      server,
      "test:///test.mojo",
      `
comptime Value = 10

def foo(a: Pointer[mut=True, Float32, MutAnyOrigin]) -> Float32:
  var variable = 15
  def inner_fn():
    return
  def inner_closure(arg: Int, arg2: type_of(arg)) -> Float32:
    return a.load[width=1](arg)
  return inner_fn(variable)

struct struct_name:
  def struct_fn():
    return

  var field: Int

trait trait_name:
    def trait_fn(self):
        ...
`
    );
    await doc.open();

    assert.partialDeepStrictEqual(await doc.documentSymbols(), [
      { name: "Value", kind: SymbolKind.Property, detail: "10" },
      {
        name: "foo",
        kind: SymbolKind.Function,
        detail: "def foo(a: Pointer[Float32, MutAnyOrigin]) -> Float32",
        children: [
          { name: "inner_fn", kind: SymbolKind.Function, detail: "def inner_fn()" },
        ],
      },
      {
        name: "struct_name",
        kind: SymbolKind.Struct,
      },
      {
        name: "trait_name",
        kind: SymbolKind.Interface,
      },
    ]);
  });

  afterEach("stop language server", async () => {
    await server.stop();
  });
});
