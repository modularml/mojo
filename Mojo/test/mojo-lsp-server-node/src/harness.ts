import * as assert from "assert";
import * as path from "path";
import { ChildProcess, spawn } from "child_process";
import { once } from "events";
import { readFile } from "fs/promises";
import { setTimeout } from "timers/promises";
import {
  ClientCapabilities,
  CodeAction,
  CodeActionRequest,
  Command,
  CompletionItem,
  CompletionList,
  CompletionParams,
  CompletionRequest,
  Definition,
  DefinitionParams,
  DefinitionRequest,
  Diagnostic,
  DidOpenTextDocumentNotification,
  DocumentSymbol,
  DocumentSymbolRequest,
  Emitter,
  Event,
  Hover,
  HoverRequest,
  InitializeParams,
  Location,
  LocationLink,
  MessageConnection,
  Position,
  PublishDiagnosticsNotification,
  PublishDiagnosticsParams,
  Range,
  ReferencesRequest,
  SymbolInformation,
} from "vscode-languageserver-protocol";
import { createMessageConnection } from "vscode-languageserver-protocol/node";

export class LanguageServer {
  public connection: MessageConnection;
  private serverProcess: ChildProcess;
  private alive: boolean = true;
  private diagnosticEmitter: Emitter<PublishDiagnosticsParams>;
  public onDiagnosticsPublished: Event<PublishDiagnosticsParams>;

  constructor(extraArgs: string[] = []) {
    this.serverProcess = spawn(
      process.env["MODULAR_MOJO_MAX_LSP_SERVER_PATH"]!,
      ["-wait-on-shutdown", ...extraArgs],
      {
        stdio: ["pipe", "pipe", "inherit"],
      }
    );

    this.connection = createMessageConnection(
      this.serverProcess.stdout!,
      this.serverProcess.stdin!
    );
    this.connection.onError((err) => {
      console.error(err);
      this.alive = false;
    });

    this.connection.onClose(() => (this.alive = false));
    this.connection.onDispose(() => (this.alive = false));
    this.serverProcess.on("exit", () => (this.alive = false));
    this.serverProcess.on("error", () => (this.alive = false));

    this.diagnosticEmitter = new Emitter();
    this.onDiagnosticsPublished = this.diagnosticEmitter.event;
    this.connection.onNotification(PublishDiagnosticsNotification.type, (p) =>
      this.diagnosticEmitter.fire(p)
    );

    this.connection.listen();
  }

  async initialize(capabilities?: ClientCapabilities) {
    await this.connection.sendRequest("initialize", {
      processId: process.pid,
      capabilities,
    } as InitializeParams);
  }

  async awaitDiagnostics(): Promise<PublishDiagnosticsParams> {
    return new Promise((resolve) => {
      let conn = this.connection.onNotification(
        "textDocument/publishDiagnostics",
        (params: PublishDiagnosticsParams) => {
          resolve(params);
          conn.dispose();
        }
      );
    });
  }

  async awaitRequest<R>(method: string): Promise<R> {
    return new Promise((resolve) => {
      let conn = this.connection.onRequest(method, (params: R) => {
        resolve(params);
        conn.dispose();
      });
    });
  }

  async stop() {
    assert.ok(this.alive, "server terminated early");

    await this.connection.sendRequest("shutdown");
    assert.ok(this.serverProcess.kill());
    const result = await Promise.race([
      once(this.serverProcess, "exit"),
      setTimeout(5000, "timeout"),
    ]);

    if (result === "timeout") {
      console.error(
        "Timed out waiting for language server to exit, did server crash?"
      );
    }
  }
}

export class Document {
  public readonly uri: string;

  public get content(): string {
    return this._content;
  }

  private _content: string;
  private _lines: string[] = [];
  private _diagnostics?: Diagnostic[] = undefined;
  private _diagnosticsReceived: Emitter<Diagnostic[]> = new Emitter();
  private version = 0;

  constructor(private server: LanguageServer, uri: string, content: string) {
    this.uri = uri;
    this._content = content;
    this._lines = content.split("\n");
    this._diagnosticsReceived = new Emitter();

    server.onDiagnosticsPublished((p) => {
      if (p.uri !== this.uri) return;

      this._diagnostics = p.diagnostics;
      this._diagnosticsReceived.fire(p.diagnostics);
    });
  }

  public static async fromFile(
    server: LanguageServer,
    file: string
  ): Promise<Document> {
    file = path.resolve(file);
    let content = await readFile(file, {
      encoding: "utf-8",
    });

    return new Document(server, `file://${file}`, content);
  }

  async open() {
    return await this.server.connection.sendNotification(
      DidOpenTextDocumentNotification.type,
      {
        textDocument: {
          uri: this.uri,
          languageId: "mojo",
          version: this.version,
          text: this._content,
        },
      }
    );
  }

  /// Find the first position that a substring appears in the document. Throws
  /// if the substring is not within the document.
  public findFirstPosition(substr: string): Position {
    assert.doesNotMatch(substr, /\n/, "substr cannot contain a newline");

    for (let line = 0; line < this._lines.length; ++line) {
      let lineContent = this._lines[line];
      let offset = lineContent.indexOf(substr);
      if (offset !== -1) {
        return {
          line,
          character: offset,
        };
      }
    }

    throw new Error("substring not found in document content");
  }

  public findLastPosition(substr: string): Position {
    assert.doesNotMatch(substr, /\n/, "substr cannot contain a newline");

    for (let line = this._lines.length - 1; line > 0; --line) {
      let lineContent = this._lines[line];
      let offset = lineContent.indexOf(substr);
      if (offset !== -1) {
        return {
          line,
          character: offset,
        };
      }
    }

    throw new Error("substring not found in document content");
  }

  public findFirstRange(substr: string): Range {
    let pos = this.findFirstPosition(substr);

    return {
      start: pos,
      end: { line: pos.line, character: pos.character + substr.length },
    };
  }

  async hover(position: Position): Promise<Hover | null> {
    return this.server.connection.sendRequest(HoverRequest.type, {
      textDocument: {
        uri: this.uri,
      },
      position,
    });
  }

  public async references(
    position: Position,
    includeDeclaration: boolean
  ): Promise<Location[] | null> {
    return this.server.connection.sendRequest(ReferencesRequest.type, {
      textDocument: {
        uri: this.uri,
      },
      position,
      context: {
        includeDeclaration,
      },
    });
  }

  public async documentSymbols(): Promise<
    DocumentSymbol[] | SymbolInformation[] | null
  > {
    return this.server.connection.sendRequest(DocumentSymbolRequest.type, {
      textDocument: {
        uri: this.uri,
      },
    });
  }

  public async complete(position: Position): Promise<CompletionItem[] | null> {
    const response = await this.server.connection.sendRequest(
      CompletionRequest.type,
      {
        textDocument: {
          uri: this.uri,
        },
        position,
      }
    );

    if (Array.isArray(response) || response === null) {
      // tsc doesn't quite understand that we've narrowed the type of `response`
      // here so we have to help ita long.
      return response as CompletionItem[] | null;
    } else {
      // The protocol allows returning a CompletionList struct with a few fields
      // that we currently don't return. For convenience, we try to unwrap this
      // struct into its items array, but if/when the server starts returning
      // this information we want a signal to update the tests relying on it.
      assert.ok(
        !response.isIncomplete && response.itemDefaults === undefined,
        "CompletionList response contained unhandled fields"
      );

      return response.items;
    }
  }

  public async diagnostics(): Promise<Diagnostic[]> {
    if (this._diagnostics !== undefined) return this._diagnostics;

    return new Promise((resolve) => {
      const conn = this._diagnosticsReceived.event((d) => {
        resolve(d);
        conn.dispose();
      });
    });
  }

  public async definition(position: Position): Promise<Location[] | null> {
    const result = await this.server.connection.sendRequest(
      DefinitionRequest.type,
      {
        textDocument: {
          uri: this.uri,
        },
        position,
      }
    );

    if (result === null) return result;
    else if (!Array.isArray(result)) {
      // textDocument/definition can return a Location | Location[]. For
      // simplicity's sake, just wrap the Location in an array.
      return [result];
    } else {
      // LocationLink is an augmented version of Location returned when client
      // and server both support it. The Mojo language server doesn't support it
      // right now, so to simplify our current tests we want to return Location[].
      if (LocationLink.is(result[0])) {
        assert.fail("Tests cannot handle LocationLink results yet");
      }

      return result as Location[];
    }
  }

  public async codeActions(
    diagnostic: Diagnostic
  ): Promise<(Command | CodeAction)[] | null> {
    return await this.server.connection.sendRequest(CodeActionRequest.type, {
      textDocument: {
        uri: this.uri,
      },
      range: Range.create(0, 0, 99999, 99999),
      context: {
        diagnostics: [diagnostic],
      },
    });
  }
}
