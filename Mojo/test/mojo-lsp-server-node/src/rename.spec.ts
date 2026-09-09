import { ErrorCodes, RenameParams, RenameRequest, ResponseError, WorkspaceEdit } from "vscode-languageserver-protocol";
import { Document, LanguageServer } from "./harness";
import * as assert from "assert";

describe("rename", function() {
  let server: LanguageServer;

  beforeEach("start and connect to language server", async function () {
    server = new LanguageServer();
    await server.initialize();
  });

  afterEach("stop language server", async function () {
    await server.stop();
  });

  it("should rename local variables", async function() {
    let document = new Document(server, "test:///test.mojo", `
def main():
  var someVar = 123
  print(someVar)
      `);

      await document.open();
      let result = await server.connection.sendRequest(RenameRequest.type, {
        textDocument: {
          uri: document.uri,
        },
        position: { line: 2, character: 8 },
        newName: "otherVar",
      });

      assert.deepStrictEqual(result, {
        changes: {
          ["/test.mojo"]: [
            {
              range: {
                start: { line: 2, character: 6 },
                end: { line: 2, character: 13 },
              },
              newText: "otherVar",
            },
            {
              range: {
                start: { line: 3, character: 8 },
                end: { line: 3, character: 15 },
              },
              newText: "otherVar",
            },
          ],
        },
      });
  });

  it("should not rename globals", async function() {
    let document = new Document(
      server,
      "test:///test.mojo",
      `
var someGlobal = 123
            `
    );

    await document.open();

    assert.rejects(() => {
      return server.connection.sendRequest(RenameRequest.type, {
        textDocument: {
          uri: document.uri,
        },
        position: { line: 1, character: 4 },
        newName: "otherGlobal",
      });
    }, (error) => {
      assert.ok(error instanceof ResponseError);
      assert.strictEqual(error.code, ErrorCodes.InvalidRequest);
      assert.strictEqual(error.message, "renaming is only available for local variables");
    });
  });

  it("should not rename function arguments", async function() {
    let document = new Document(
      server,
      "test:///test.mojo",
      `
def something(arg: Int):
  pass
            `
    );

    await document.open();

    assert.rejects(
      () => {
        return server.connection.sendRequest(RenameRequest.type, {
          textDocument: {
            uri: document.uri,
          },
          position: { line: 1, character: 15 },
          newName: "argument",
        });
      },
      (error) => {
        assert.ok(error instanceof ResponseError);
        assert.strictEqual(error.code, ErrorCodes.InvalidRequest);
        assert.strictEqual(
          error.message,
          "renaming is only available for local variables"
        );
      }
    );
  });

  it("should not rename imported symbols", async function() {
    let document = new Document(
      server,
      "test:///test.mojo",
      `
def main():
  print(1 + 2)
            `
    );

    await document.open();

    assert.rejects(
      () => {
        return server.connection.sendRequest(RenameRequest.type, {
          textDocument: {
            uri: document.uri,
          },
          position: { line: 2, character: 4 },
          newName: "pronto",
        });
      },
      (error) => {
        assert.ok(error instanceof ResponseError);
        assert.strictEqual(error.code, ErrorCodes.InvalidRequest);
        assert.strictEqual(
          error.message,
          "renaming is only available for local variables"
        );
      }
    );
  });

  it("should not rename keywords", async function() {
    let document = new Document(
      server,
      "test:///test.mojo",
      `
def main():
  print(1 + 2)
            `
    );

    await document.open();

    assert.rejects(
      () => {
        return server.connection.sendRequest(RenameRequest.type, {
          textDocument: {
            uri: document.uri,
          },
          position: { line: 1, character: 0 },
          newName: "func",
        });
      },
      (error) => {
        assert.ok(error instanceof ResponseError);
        assert.strictEqual(error.code, ErrorCodes.InvalidRequest);
        assert.strictEqual(
          error.message,
          "no identified symbol at this position"
        );
      }
    );
  })
});
