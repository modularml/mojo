import { Document, LanguageServer } from "./harness";
import * as assert from "assert";

describe('MOTO-1127', function() {
  it('should not parse files twice', async function() {
    const server = new LanguageServer();
    await server.initialize({
      window: {
        workDoneProgress: false,
      },
    });

    let doc = new Document(server, "test:///test.mojo", `
def main():
  """
  This is a docstring.

  The nested code-block below will trigger a crash if this bug reproduces.
  \`\`\`mojo
  print("hi")
  \`\`\`
  """

  pass
`);

    await doc.open();
    await server.stop();
  });
})
