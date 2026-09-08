import { InitializeParams } from "vscode-languageserver";
import { LanguageServer } from "./harness";

describe("test", () => {
  let server: LanguageServer;

  beforeEach("start and connect to language server", () => {
    server = new LanguageServer();
  });

  it("should initialize successfully", async () => {
    await server.connection.sendRequest("initialize", {
      processId: process.pid,
      capabilities: {
        window: {
          workDoneProgress: true,
        },
      },
    } as InitializeParams);
  });

  afterEach("stop language server", async () => {
    await server.stop();
  });
});
