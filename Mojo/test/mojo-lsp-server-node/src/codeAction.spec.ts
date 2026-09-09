import * as assert from "assert";
import { Document, LanguageServer } from "./harness";
import { Range } from "vscode-languageserver-protocol";

describe("code actions", () => {
  let server: LanguageServer;

  beforeEach("start and connect to language server", async () => {
    server = new LanguageServer();
    await server.initialize();
  });

  it("should report all possible actions", async () => {
    let doc = new Document(
      server,
      "test:///test.mojo",
      `
def main() raises:
    var a = [1, 2, 3]
    var b = a

    print(repr(b))
`
    );
    await doc.open();

    const diagnostics = await doc.diagnostics();
    assert.ok(diagnostics.length > 0, "expected document to have at least one diagnostic");

    const actions = await doc.codeActions(diagnostics[0]);
    assert.partialDeepStrictEqual(actions, [
      {
        title: `consider transferring the value with '^'`,
        edit: {
          changes: {
            [doc.uri]: [{
              range: Range.create(3, 13, 3, 13),
              newText: '^',
            }]
          }
        }
      },
      {
        title: `you can copy it explicitly with '.copy()'`,
        edit: {
          changes: {
            [doc.uri]: [{
              range: Range.create(3, 13, 3, 13),
              newText: ".copy()",
            }]
          }
        }
      }
    ])
  });

  afterEach("stop language server", async () => {
    await server.stop();
  });
});
