import * as assert from "assert";
import * as path from "path";
import { Document, LanguageServer } from "./harness";

describe("package member completions", function () {
  let server: LanguageServer;

  beforeEach("start and connect to language server", async () => {
    server = new LanguageServer();
    await server.initialize();
  });

  afterEach("stop language server", async () => {
    await server.stop();
  });

  // Member access on a package (`pkg.<...>`) should complete that package's
  // public surface: the submodules and symbols its __init__ re-exports. Here
  // `sub`'s __init__ does `from . import leaf` and `from .leaf import leaf_fn`.
  //
  // The on-disk consumer.mojo is valid Mojo (so it stays lintable); we open it
  // with an in-memory edit that introduces the incomplete `sub.` member access,
  // since completion needs that incomplete form. The real file:// URI keeps the
  // document inside member_access_pkg so `from . import sub` resolves against
  // the on-disk `sub` sub-package.
  it("should complete a package's re-exported members", async function () {
    let file = path.resolve(
      "Mojo/test/mojo-lsp-server-node/data/member_access_pkg/consumer.mojo"
    );
    let doc = new Document(
      server,
      `file://${file}`,
      `from . import sub


def main():
    sub.
`
    );
    await doc.open();

    let completions = await doc.complete(doc.findFirstRange("sub.").end);
    assert.ok(completions);
    assert.ok(
      completions.some((i) => i.label === "leaf"),
      "expected re-exported submodule 'leaf' among: " +
        completions.map((i) => i.label).join(", ")
    );
    assert.ok(
      completions.some((i) => i.label === "leaf_fn"),
      "expected re-exported symbol 'leaf_fn' among: " +
        completions.map((i) => i.label).join(", ")
    );
  });
});
