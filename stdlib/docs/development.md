# Mojo Standard Library Development

This document covers the essentials of getting started developing for the
standard library.

## Getting the Nightly Mojo Compiler

To get started, you need to install the latest nightly mojo compiler. The
Standard Library only guarantees compatibility with the latest nightly `mojo`
compiler.

```bash
modular auth
modular install nightly/mojo
```

If you already have an older `nightly/mojo` compiler, replace
`modular install nightly/mojo` with `modular update nightly/mojo`.

Then, follow the instructions from the `modular` tool in adding the `mojo`
compiler to your `PATH` such as:

```bash
echo 'export MODULAR_HOME="/Users/joe/.modular"' >> ~/.zshrc
echo 'export PATH="/Users/joe/.modular/pkg/packages.modular.com_nightly_mojo/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

## Cloning the Repository

```bash
git clone https://github.com/modularml/mojo/stdlib
```

## Building the Standard Library

To build the Standard Library, you can run the script
[`build-stdlib.sh`](../scripts/build-stdlib.sh) from the `scripts` directory in
the top-level of the repo. This will create a build artifacts directory,
`build`, in the top-level of the repo and produce the `stdlib.mojopkg` inside.

```bash
./scripts/build-stdlib.sh
```

## Testing the Standard Library

### Installing Unit Test Dependencies

To run the unit tests, you first need to install a few dependencies:

1. [`lit`](https://llvm.org/docs/CommandGuide/lit.html) - can be downloaded via
   `python3 -m pip install lit`
2. [`FileCheck`](https://llvm.org/docs/CommandGuide/FileCheck.html) - is part of
   the LLVM distribution you can obtain from your package manager

When you download `lit`, make sure you add the path to `lit` to your `PATH` if
needed. Some of the tests use `FileCheck` or `not` binaries that ship with LLVM.
For example, if you download LLVM via `homebrew`, these would be in
`/opt/homebrew/Cellar/llvm/<version>/bin` for example. You need to add this path
to your `PATH` when doing development for those tests. Over time, we’re looking
to move away from `FileCheck` in favor of writing the unit tests using our own
`testing` module and remove this dependency requirements for contributors. We
are happy to welcome contributions in this area!

### Running the Standard Library Tests

We provide a simple Bash script to build the Standard Library package and
`test_utils` package that is used by the test suite.  Just run
`./scripts/run-tests.sh` which will produce the necessary `mojopkg` files inside
your `build` directory and then run `lit -sv test/stdlib`.

```bash
./scripts/run-tests.sh
```

### Running a subset of the Standard Library Unit Tests

If you’d like to run just a subset of the tests, feel free to use all of the
normal options that the `lit` tool provides.  For example, to run just the
builtin and collections tests, you can

```bash
cd test/stdlib
lit -sv builtin collections
```

This can quickly speed up your iteration when doing development to avoid running
the entire test suite if you know your changes are only affecting a particular
area. We recommend running the entire test suite before submitting a Pull
Request.

Reminder that if you’re choosing to invoke `lit` directly and not use the
`run-tests.sh`, you need to ensure your `stdlib.mojopkg` and
`test_utils.mojopkg` are up-to-date. We’re not currently imposing any build
system right now to ensure these dependencies are up-to-date before running the
tests.

If you run into any issues when running the tests,
[please file an issue](https://github.com/modularml/mojo/issues) and we’ll take
a look.

## Formatting Changes

Please make sure your changes are formatted before submitting a Pull Request.
Otherwise, CI will fail in its lint and formatting checks.  The `mojo` compiler
provides a `format` command.  So, you can format your changes like so:

```bash
mojo format <file1> <file2> ...
```

You can also do this before submitting a pull request by running it on the
relevant files changed compared to the remote:

```bash
git diff origin/main --name-only -- '*.mojo' | xargs mojo format
```

You can also consider setting up your editor to auto format Mojo files upon
saving.
