import * as assert from "assert";
import * as path from "path";
import { Document, LanguageServer } from "./harness";
import {
  CompletionItemKind,
  CompletionList,
  CompletionRequest,
  DidChangeTextDocumentNotification,
  ErrorCodes,
  LSPErrorCodes,
  MarkupContent,
  Position,
  ResponseError,
} from "vscode-languageserver-protocol";

describe("completions", function () {
  let server: LanguageServer;

  beforeEach("start and connect to language server", async () => {
    server = new LanguageServer();
    await server.initialize();
  });

  afterEach("stop language server", async () => {
    await server.stop();
  });

  it("should provide completions for imports", async function () {
    let doc = new Document(
      server,
      "test:///test.mojo",
      `
import p

# this is a comment
`
    );

    await doc.open();

    let completions = await doc.complete(Position.create(1, 8));
    assert.ok(completions);
    assert.ok(
      completions.some(
        (i) =>
          i.label === "prelude" &&
          i.kind! === CompletionItemKind.Folder &&
          (i.documentation! as MarkupContent).value.includes(
            "Standard library prelude: fundamental types, traits, and operations auto-imported"
          )
      )
    );
  });

  it("should provide completions for nested imports", async function () {
    let doc = new Document(
      server,
      "test:///test.mojo",
      `
import std.builtin.
`
    );
    await doc.open();

    let completions = await doc.complete(Position.create(1, 19));
    assert.ok(completions);
    assert.ok(
      completions.some(
        (i) => i.label === "bool" && i.kind! === CompletionItemKind.Module
      )
    );
  });

  // Imports can resolve through regular directories, not just packages with
  // an __init__.mojo, so completion should offer plain source directories
  // too. `dir` here is a regular directory next to the opened document.
  //
  // The on-disk consumer.mojo is valid Mojo (so it stays lintable); we open
  // it with in-memory content containing the incomplete import, using the
  // real file:// URI so the document's directory is searched for imports.
  it("should complete source directories in imports", async function () {
    let file = path.resolve(
      "Mojo/test/mojo-lsp-server-node/data/import_dir/consumer.mojo"
    );
    let doc = new Document(
      server,
      `file://${file}`,
      `import d
`
    );
    await doc.open();

    let completions = await doc.complete(doc.findFirstRange("import d").end);
    assert.ok(completions);
    assert.ok(
      completions.some(
        (i) => i.label === "dir" && i.kind! === CompletionItemKind.Folder
      ),
      "expected directory 'dir' among: " +
        completions.map((i) => i.label).join(", ")
    );
  });

  it("should complete nested imports through source directories", async function () {
    let file = path.resolve(
      "Mojo/test/mojo-lsp-server-node/data/import_dir/consumer.mojo"
    );
    let doc = new Document(
      server,
      `file://${file}`,
      `import dir.
`
    );
    await doc.open();

    let completions = await doc.complete(
      doc.findFirstRange("import dir.").end
    );
    assert.ok(completions);
    assert.ok(
      completions.some(
        (i) => i.label === "module" && i.kind! === CompletionItemKind.Module
      ),
      "expected module 'module' among: " +
        completions.map((i) => i.label).join(", ")
    );
    assert.ok(
      completions.some(
        (i) =>
          i.label === "nested_dir" && i.kind! === CompletionItemKind.Folder
      ),
      "expected nested directory 'nested_dir' among: " +
        completions.map((i) => i.label).join(", ")
    );
  });

  it("should complete relative imports", async function () {
    let doc = await Document.fromFile(
      server,
      "Mojo/test/mojo-lsp-server-node/data/package/imports.mojo"
    );
    await doc.open();

    let completions = await doc.complete(
      doc.findFirstRange("from .aliases").end
    );
    assert.ok(completions);
    assert.ok(
      completions.some(
        (i) => i.label === "aliases" && i.kind! === CompletionItemKind.Module
      )
    );
  });

  it("should sort completion items", async function () {
    let doc = new Document(
      server,
      "test:///test.mojo",
      `
@fieldwise_init
struct Foo(Copyable, Movable):
  var __other: Int
  var ___another__: Int
  var __dunder__: Int
  var _sunder_: Int
  var _priv: Int
  var normal: Int
  def foo(self): pass
  def _foobar_(self): pass
  def _bar(self): pass
  def __baz__(self): pass

def function(arg: Foo):
  arg.
`
    );
    await doc.open();

    let completions = await doc.complete(Position.create(15, 6));
    assert.ok(completions);
    assert.deepStrictEqual(
      completions.map((i) => i.label),
      [
        "copy",
        "foo",
        "normal",
        "_bar",
        "_priv",
        "_foobar_",
        "_sunder_",
        "__baz__",
        "__deinit__",
        "__init__",
        "__dunder__",
        "__copy_ctor_is_trivial",
        "__del__is_trivial",
        "__move_ctor_is_trivial",
        "___another__",
        "__other",
      ]
    );
  });

  // MOCO-3124: a name bound by `from module import name` stays an unresolved
  // placeholder until first referenced, and used to complete with no kind.
  // Prelude re-exports (e.g. `Optional`) always reach completion as such
  // placeholders; the explicit imports cover the same path for user-written
  // imports.
  it("should resolve kinds for from-import completions", async function () {
    let doc = new Document(
      server,
      "test:///from_import_kinds.mojo",
      `
from std.collections import Deque
from std.collections import Deque as RenamedDeque
from std.hashlib import Hasher
from std.math import sqrt
from std import collections

def function() -> De
`
    );
    await doc.open();

    let completions = await doc.complete(doc.findFirstRange("-> De").end);
    assert.ok(completions);
    let kindOf = (label: string) =>
      completions!.find((i) => i.label === label)?.kind;
    assert.strictEqual(kindOf("Deque"), CompletionItemKind.Struct);
    assert.strictEqual(kindOf("RenamedDeque"), CompletionItemKind.Struct);
    assert.strictEqual(kindOf("Hasher"), CompletionItemKind.Interface);
    assert.strictEqual(kindOf("sqrt"), CompletionItemKind.Function);
    assert.strictEqual(kindOf("Optional"), CompletionItemKind.Struct);
    assert.strictEqual(kindOf("len"), CompletionItemKind.Function);
    assert.strictEqual(kindOf("Copyable"), CompletionItemKind.Interface);
    assert.strictEqual(kindOf("collections"), CompletionItemKind.Module);
  });

  it("should complete members", async function () {
    let doc = new Document(
      server,
      "test:///test.mojo",
      `
def function(arg: Int):
    arg.
`
    );
    await doc.open();

    let completions = await doc.complete(Position.create(2, 8));
    assert.ok(completions);
    assert.ok(
      completions.some(
        (i) => i.label === "__add__" && i.kind! === CompletionItemKind.Function
      )
    );
    assert.ok(
      completions.some(
        (i) => i.label === "_mlir_value" && i.kind! == CompletionItemKind.Field
      )
    );
  });

  it("should complete at the top-level", async function () {
    let doc = new Document(
      server,
      "test:///test.mojo",
      `
def function() -> Int:
    var value: Int = 10
    return value
`
    );
    await doc.open();

    let completions = await doc.complete(doc.findFirstPosition("nt"));
    assert.ok(completions);
    // TODO(MOTO-639): Check item.kind when we start resolving this correctly.
    assert.ok(completions.some((i) => i.label === "Int"));

    completions = await doc.complete(doc.findLastPosition("value"));
    assert.ok(completions);
    assert.ok(
      completions.some(
        (i) => i.label === "value" && i.kind! == CompletionItemKind.Variable
      )
    );
  });

  it("should not regress MOTO-767", async function () {
    let doc = new Document(
      server,
      "test:///moto-767.mojo",
      `
def main() raises -> :
  pass

comptime T = Tuple[StringLiteral, StringLiteral, StringLiteral]

def f[T: Equatable](s: T):
  pass
`
    );
    await doc.open();
    await doc.complete(doc.findFirstPosition("->"));
    // We need to simply not have crashed here.
  })

  describe("partial completions", function () {
    async function checkSnippet(doc: Document, startAt: string) {
      await doc.open();
      let completions = await doc.complete(doc.findFirstPosition(startAt));
      assert.ok(completions);
      assert.ok(
        completions.length > 0,
        "expected completion list to be non-empty"
      );
    }

    it("should complete for partial functions", async function () {
      let doc = new Document(
        server,
        "test:///def_no_colon.mojo",
        `
def function(arg: Int)`
      );
      await checkSnippet(doc, "nt");
    });

    it("should complete for partial if", async function () {
      let doc = new Document(
        server,
        "test:///if_no_colon.mojo",
        `
def function(arg: Int):
  if arg.value:`
      );
      await checkSnippet(doc, "value");
    });

    it("should complete for partial elif", async function () {
      let doc = new Document(
        server,
        "test:///elif_no_colon.mojo",
        `
def function(arg: Int):
  if False:
    return
  elif arg.value:`
      );
      await checkSnippet(doc, "value");
    });

    it("should complete for partial while", async function () {
      let doc = new Document(
        server,
        "test:///while_no_colon.mojo",
        `
def function(arg: Int):
  while arg.value`
      );
      await checkSnippet(doc, "value");
    });

    it("should complete for partial with", async function () {
      let doc = new Document(
        server,
        "test:///with_no_colon.mojo",
        `
def function(arg: Int):
  with arg.value`
      );
      await checkSnippet(doc, "value");
    });
  });

  describe("stale requests during rapid edits", function () {
    // When a client fires textDocument/completion while a burst of
    // textDocument/didChange notifications is still mid-debounce, the
    // server's currently-parsed document may not contain the position
    // being asked about. The correct response is ContentModified
    // (-32801), which tells spec-compliant clients to retry against the
    // updated document. Returning InvalidRequest (-32600) makes the
    // client discard the request outright, which manifested as
    // "completions silently don't work" in nvim-cmp.

    const source = `
struct Foo:
    var field_a: Int
    var field_b: Int
    def bar(self):
        self.
`;

    it("never returns InvalidRequest while a reparse is pending", async function () {
      let doc = new Document(server, "test:///burst.mojo", source);
      await doc.open();
      // Wait for the initial parse so the first didChange below is a
      // genuine update and not racing with the didOpen parse.
      await doc.diagnostics();

      // Cursor immediately after `self.` on the last non-empty line.
      let position = doc.findFirstPosition("self.");
      position.character += "self.".length;

      // Fire a burst of didChange notifications with no awaits between
      // them so the debouncer coalesces and a completion can land while
      // pendingDocContents is still populated.
      for (let i = 0; i < 8; ++i) {
        server.connection.sendNotification(
          DidChangeTextDocumentNotification.type,
          {
            textDocument: { uri: doc.uri, version: i + 1 },
            contentChanges: [{ text: source + `# burst ${i}\n` }],
          }
        );
      }

      // Completion request issued immediately — inside the debounce
      // window from the final didChange.
      try {
        const result = await server.connection.sendRequest(
          CompletionRequest.type,
          {
            textDocument: { uri: doc.uri },
            position,
            context: { triggerKind: 2, triggerCharacter: "." },
          }
        );
        // If the request won the race and ran against a parsed document,
        // the response must be a well-formed list with the expected
        // members — not some malformed payload.
        assert.ok(result !== null, "completion result was null");
        const items = Array.isArray(result) ? result : (result as CompletionList).items;
        assert.ok(
          items.some((i) => i.label === "field_a"),
          "expected field_a in completion list"
        );
      } catch (err) {
        const responseErr = err as ResponseError<unknown>;
        assert.notStrictEqual(
          responseErr.code,
          ErrorCodes.InvalidRequest,
          `completion during pending reparse returned InvalidRequest; ` +
            `expected ContentModified. Message: ${responseErr.message}`
        );
        assert.strictEqual(
          responseErr.code,
          LSPErrorCodes.ContentModified,
          `expected ContentModified (-32801), got ${responseErr.code}: ` +
            `${responseErr.message}`
        );
      }
    });

    it("returns a valid completion list once the reparse settles", async function () {
      let doc = new Document(server, "test:///burst_settle.mojo", source);
      await doc.open();
      await doc.diagnostics();

      let position = doc.findFirstPosition("self.");
      position.character += "self.".length;

      // Same burst, but this time wait for diagnostics from the final
      // change before asking for completion — the server's current
      // document should now reflect the last write, so completion must
      // succeed with real members.
      const diagPromise = server.awaitDiagnostics();
      for (let i = 0; i < 8; ++i) {
        server.connection.sendNotification(
          DidChangeTextDocumentNotification.type,
          {
            textDocument: { uri: doc.uri, version: i + 1 },
            contentChanges: [{ text: source + `# burst ${i}\n` }],
          }
        );
      }
      await diagPromise;

      const result = await server.connection.sendRequest(
        CompletionRequest.type,
        {
          textDocument: { uri: doc.uri },
          position,
          context: { triggerKind: 2, triggerCharacter: "." },
        }
      );
      assert.ok(result !== null, "completion result was null after settle");
      const items = Array.isArray(result) ? result : (result as CompletionList).items;
      const labels = items.map((i) => i.label);
      assert.ok(
        labels.includes("field_a") && labels.includes("field_b"),
        `expected field_a and field_b in completion list, got: ${labels.join(", ")}`
      );
    });
  });
});
