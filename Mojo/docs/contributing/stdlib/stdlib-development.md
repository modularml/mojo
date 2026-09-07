# Mojo standard library development

This document covers the essentials of developing for the standard library.

If this is your first time contributing, first read everything in
the [Mojo contributor guide](../../../CONTRIBUTING.md).

## Set up your environment

To get started, you need to do the following:

1. Be sure that you meet the
   [system requirements](https://mojolang.org/docs/requirements/).

   If you're developing on macOS, you need Xcode 16.0 or later and macOS 15.0 or
   later. After upgrading macOS or Xcode you may need to run
   `xcodebuild -downloadComponent MetalToolchain`, which downloads the Metal
   utilities required for GPU programming.
2. If you're using VS Code,
   [Install the Mojo VS Code extension](https://marketplace.visualstudio.com/items?itemName=modular-mojotools.vscode-mojo).
3. [Fork the repo and create a branch](../../../../CONTRIBUTING.md#step-2-create-a-pull-request).

Now you're ready to start developing.

## Building the standard library

The Modular repository uses [Bazel](https://bazel.build/), a fast, scalable
build and test tool to ensure reproducible builds through dependency tracking
and caching. For more information on using Bazel in the Modular repo, see
[Using Bazel](../../../../bazel/docs/usage.md).

To build with Bazel, you'll need to choose one of two modes: either building
the Mojo compiler locally, or using a prebuilt Mojo compiler. To use the
prebuilt Mojo compiler, pass the `--config=prebuilt-mojo` option on every
`bazel` command. Or create a `local.bazelrc` file and add the following:

```text
build --config=prebuilt-mojo
```

The `bazel` commands shown in this document assume you have the configuration
already set in a `local.bazelrc` file.

To build and test your changes to the Mojo standard library, run the following
`./bazelw` commands from the top-level directory of the repository (where the
`bazel` folder is located).

To build the standard library, you can run:

```sh
./bazelw build //Mojo/stdlib/...
```

## Testing the standard library

To run the tests for the standard library, you can run:

```sh
./bazelw test //Mojo/stdlib/test/...
```

Tests build with assertions enabled, which compiles them with `-D ASSERT=all`
and activates every `debug_assert` in the standard library. A test can
therefore fail on an assertion that a release build skips. Individual test
files opt out through the `_DISABLED_ASSERTIONS` list in their `BUILD.bazel`.

## Testing only a subset of the standard library

You can run all the tests within a specific subdirectory by simply
specifying the subdirectory and using `/...`. For example:

```sh
./bazelw test //Mojo/stdlib/test/math/...
```

To find all the test targets, you can run:

```sh
./bazelw query 'tests(//Mojo/stdlib/...)'
```

If you have `pixi` installed, you can use the `pixi run tests` convenience
script to execute standard library tests within the mojo directory:

```sh
pixi run tests ./stdlib/test/bit
```

```sh
pixi run tests ./stdlib/test/bit/test_bit.mojo
```

This script automatically executes the equivalent bazelw command.

## Formatting changes

Please make sure your changes are formatted before submitting a pull request.
Otherwise, CI will fail in its lint and formatting checks. `bazel` setup
provides a `format` command. So, you can format your changes like so:

```sh
./bazelw run //:format
```

To avoid forgetting, we recommend setting up a `pre-commit` hook, which will
format your changes automatically at each commit. If you have the `pre-commit`
command installed, run:

```sh
pre-commit install
```

Or you can use a package manager like `pixi` or `uv` to run the
`pre-commit` command without installing it:

| Package manager | Command                     |
|-----------------|-----------------------------|
| Pixi            | `pixi x pre-commit install` |
| uv              | `uvx pre-commit install`    |

You can also consider setting up your editor to automatically format
Mojo, Python, and MDX files upon saving.

### Raising a PR

If you think you have a worthwhile change to propose, check the guidance in the
[Mojo contributor guide](../../../CONTRIBUTING.md).

If your changes are ready to go, follow the steps to
[create a pull request](../../../../CONTRIBUTING.md#step-2-create-a-pull-request).

Congratulations! You've now got an idea on how to contribute to the standard
library, test your changes, and raise a PR.

If you're still having issues, reach out on [Discord](https://modul.ar/discord)
or ask a question in the [Modular forum](https://forum.modular.com/).
