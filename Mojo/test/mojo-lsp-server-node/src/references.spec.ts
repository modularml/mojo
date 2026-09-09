import { Range } from "vscode-languageserver-protocol";
import { Document, LanguageServer } from "./harness";
import * as assert from "assert";

describe("references", function () {
  let server: LanguageServer;

  beforeEach("start and connect to language server", async function () {
    server = new LanguageServer();
    await server.initialize();
  });

  afterEach("stop language server", async function () {
    await server.stop();
  });

  it("should find variable references", async function () {
    const doc = new Document(
      server,
      "test:///test.mojo",
      `
def function(foo: Int):
    var bar: Int = foo + 420
    print(foo)
    print("foo")
`
    );
    await doc.open();

    assert.deepStrictEqual(
      await doc.references(doc.findFirstPosition("foo"), false),
      [
        { uri: doc.uri, range: Range.create(2, 19, 2, 22) },
        { uri: doc.uri, range: Range.create(3, 10, 3, 13) },
      ]
    );

    assert.deepStrictEqual(
      await doc.references(doc.findFirstPosition("foo"), true),
      [
        { uri: doc.uri, range: Range.create(1, 13, 1, 16) },
        { uri: doc.uri, range: Range.create(2, 19, 2, 22) },
        { uri: doc.uri, range: Range.create(3, 10, 3, 13) },
      ]
    );
  });
});
