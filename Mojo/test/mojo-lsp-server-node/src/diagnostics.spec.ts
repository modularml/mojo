import * as assert from "assert";
import { Document, LanguageServer } from "./harness";
import { Range } from "vscode-languageserver-protocol";

describe("diagnostics", () => {
  let server: LanguageServer;

  beforeEach("start and connect to language server", async () => {
    server = new LanguageServer();
    await server.initialize();
  });

  it("should report diagnostics", async () => {
    let doc = new Document(server, "test:///test.mojo", `
from a import b
`);
    await doc.open();

    assert.deepStrictEqual(await doc.diagnostics(), [
      {
        category: "Parse Error",
        message: "unable to locate module 'a'",
        range: Range.create(1, 5, 1, 6),
        severity: 1,
        source: "mojo",
      }
    ]);
  });

  afterEach("stop language server", async () => {
    await server.stop();
  });
});
