import assert from "assert";
import { fileURLToPath } from "url";
import * as path from "path";
import { Document, LanguageServer } from "./harness";
import { Position } from "vscode-languageserver-protocol";

describe("definitions", () => {
  let server: LanguageServer;

  beforeEach("start and connect to language server", async () => {
    server = new LanguageServer();
    await server.initialize();
  });

  it("should have no definitions for an empty file", async () => {
    const doc = new Document(server, "test:///test.mojo", "");
    await doc.open();
    assert.deepStrictEqual(await doc.definition(Position.create(0, 0)), []);
  });

  it("should find definitions", async () => {
    const doc = new Document(
      server,
      "test:///test.mojo",
      `
def example():
  return 2

def main():
  example()

  var y = 123
  print(y)
`
    );
    await doc.open();

    assert.deepStrictEqual(
      await doc.definition(doc.findLastPosition("example")),
      [
        {
          uri: doc.uri,
          range: doc.findFirstRange("example"),
        },
      ]
    );

    assert.deepStrictEqual(
      await doc.definition(doc.findLastPosition("y")),
      [{
        uri: doc.uri,
        range: doc.findFirstRange("y"),
      }]
    );
  });

  it("should report all definitions of overloaded functions", async () => {
    const doc = new Document(
      server,
      "test:///test.mojo",
      `
def print(x: String):
    pass

def print(x: Bool):
    pass

def function[type: TrivialRegisterPassable](arg: type):
    print(arg)
`
    );
    await doc.open();

    assert.deepStrictEqual(
      await doc.definition(doc.findLastPosition("print")),
      [
        (await doc.definition(doc.findFirstPosition("print(x: String")))![0],
        (await doc.definition(doc.findFirstPosition("print(x: Bool")))![0],
      ]
    );
  });

  it("should find definitions across files imported via a relative -I directory", async () => {
    const importServer = new LanguageServer([
      "-I",
      "Mojo/test/mojo-lsp-server-node/data/import_dir",
    ]);
    try {
      await importServer.initialize();

      const doc = new Document(
        importServer,
        "test:///test.mojo",
        `
import dir.nested_dir.module

def main():
  dir.nested_dir.module.nested_fn()
`
      );
      await doc.open();

      const locations = await doc.definition(
        doc.findLastPosition("nested_fn")
      );
      assert.ok(locations);
      assert.strictEqual(locations!.length, 1);

      const filePath = fileURLToPath(locations![0].uri);
      assert.ok(path.isAbsolute(filePath));
      assert.ok(
        filePath.endsWith(
          path.join("import_dir", "dir", "nested_dir", "module.mojo")
        ),
        `expected definition in import_dir/dir/nested_dir/module.mojo, got: ${locations![0].uri}`
      );
    } finally {
      await importServer.stop();
    }
  });

  afterEach("stop language server", async () => {
    await server.stop();
  });
});
