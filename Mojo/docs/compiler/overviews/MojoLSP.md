# Mojo🔥 LSP

## Introduction

The Mojo Language Server is a productivity tool that enhances the authoring
experience of Mojo programs in editors that support the
[Language Server Protocol](https://en.wikipedia.org/wiki/Language_Server_Protocol).
It provides
[editing features](https://code.visualstudio.com/api/language-extensions/programmatic-language-features),
such as code completion, diagnostics, quick fixes, hover dialogs, jump to
definition, refactoring utilities, etc.

## Getting Started with VSCode or Cursor

Just run the `vscode-init` or `cursor-init` command on your terminal, which will
install and configure the **Mojo** extension on VSCode. The Language Server is
part of this extension and will be automatically launched whenever a `.mojo`
file is opened.

## Development

### Testing mojo-lsp-server

Our tests are currently split across three implementations. We will eventually
converge on the NodeJS-based test harness in `KGEN/test/mojo-lsp-server-node`.

#### NodeJS-based tests

These tests use Microsoft's reference implementation of the Language Server
Protocol, which is also used in VS Code. Unlike the Lit or C++-based tests,
these more closely mimic real-world conditions for the language server and are
preferred over the alternatives. Each test starts the language server and
communicates with it as an actual editor client would.

These tests live in `KGEN/test/mojo-lsp-server-node` and can be run using

```sh
bazel test //Mojo/test/mojo-lsp-server-node
```

If you are running these tests in a TTY, you can enable colorized output by
setting the `FORCE_COLOR` environment variable to 1. Doing so enables, among
other features, colorized diffs, which is helpful when a test produces incorrect
output.

```sh
FORCE_COLOR=1 bazel test //Mojo/test/mojo-lsp-server-node
```

#### Lit tests

These tests use llvm-lit to speak JSON-RPC to the language server and inspect
its output. They live in `KGEN/test/mojo-lsp-server` and can be run using

```sh
bazel test //Mojo/test/mojo-lsp-server
```

#### C++ unit tests

These tests are written in C++ using GTest and live in
`Mojo/unittests/mojo-lsp-server`. You can read more about GTests in this
[primer](https://github.com/google/googletest/blob/main/docs/primer.md).

Our tests live in `Mojo/unittests/mojo-lsp-server/` and are specified using the
C++ GTest framework.

You can read `Mojo/unittests/mojo-lsp-server/SampleTest.cpp` for a sample
test with some useful explanatory comments.

##### Inspecting the LSP traffic

You can invoke the tests with
`PRESERVE_LSP_IO_FILES=1 build check-mojo-lsp-server`, which will indicate the
test suite to print to stderr the IO files used to communicate with the Language
Server upon failures. In this case, these files are not cleaned up upon
termination, and you can inspect them to debug your issues or even invoke the
Language Server manually with
`cat /path/to/lsp_stdin | mojo-lsp-server -mojo-test`.

### mojo-lsp-simple-client

This little utility can be used to launch an LSP server and simulate some
actions that the user would do on the IDE. This tool can be extremely useful for
debugging issues.

### Debugging

`mojo-lsp-server` offers the `--attach-debugger-on-startup` argument
invocation to start a debug session on VSCode attaching to the Language Server.

There are two main ways to trigger a real debug session that uses this
capability:

- Via the VS Code command `Developer: restart the Mojo LSP Server and Attach the
  debugger to it`. This can be useful for debugging simple issues, but
  relaunching debug sessions requires several manual interactions.
- Via the `mojo-lsp-simple-client`, which offers the `-attach-debugger` option.
  This can be more convenient for automating a LSP session and rerun it
  repeatedly.
