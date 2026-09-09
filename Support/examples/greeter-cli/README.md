# greeter-cli

`greeter-cli` is a command line interface (CLI) executable that says "hi" or
"bye." It is meant to demonstrate how CLIs can be built at Modular.

Notable features include:

- A top-level command, `greeter-cli`, and subcommands, such as `greeter-cli hi`.
- Very flexible command line argument parsing: `greeter-cli Joe` is equivalent
  to `greeter-cli Joe`.

You can implement your own CLI at Modular by copy-pasting this directory as a
starting point, and then reading through the code comments in
[`BUILD.bazel`].

For a more fundamental understanding of what's going on here, read the
[CLI Guide].

[CLI Guide]: ../../../docs/internal/CLIGuide.md
