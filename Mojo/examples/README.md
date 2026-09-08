# Mojo code examples

This directory contains a collection of sample programs written in the
[Mojo](https://mojolang.org/docs/manual/) programming language.

## Getting started

You can use `git` to clone the repository containing all of these sample
programs:

```bash
git clone https://github.com/modular/modular.git
```

For the most reliable experience building and running these examples, we
recommend using [Pixi](https://pixi.sh/latest/). It's both a package manager and
virtual environment manager—which alone makes development a lot easier—but it's
also fast, language agnostic, and provides lock files for package dependencies.

Each subdirectory of this directory is a self-contained project that
demonstrates features of the Mojo programming language and its standard library.
Each contains a `README.md` file and a `pixi.toml` file that specifies the
required dependencies. Simply follow the instructions in each `README.md` file
to use `pixi` to download and install the dependencies for the project and to
build and run the examples.

For more information on system requirements, installing Mojo and the Mojo
extension for VS Code, and getting started with Mojo programming, see the
[Install Mojo](https://mojolang.org/docs/manual/install/) section of the
[Mojo Manual](https://mojolang.org/docs/manual/).

## Example subdirectories

- `life/`: The
  [Get started with Mojo](https://mojolang.org/docs/manual/get-started)
  tutorial solution. A complete implementation of Conway's Game of Life cellular
  automaton, demonstrating Mojo basics including structs, modules, and Python
  interoperability.

- `python-interop/`: Calling Mojo functions from Python
  code, enabling progressive migration of Python hotspots to Mojo.

- `operators/`:
  [Implementing operators for a custom Mojo type](https://mojolang.org/docs/manual/operators#an-example-of-implementing-operators-for-a-custom-type).

- `testing/`: Writing and running unit tests using the [Mojo testing
  framework](https://mojolang.org/docs/tools/testing).

## License

The Mojo examples in this repository are licensed under the Apache License v2.0
with LLVM Exceptions (see the LLVM [License](https://llvm.org/LICENSE.txt)).

## Contributing

As a contributor, your efforts and expertise are invaluable in driving the
evolution of the Mojo programming language. The [Mojo contributor
guide](../../CONTRIBUTING.md) provides all the information necessary to make
meaningful contributions—from understanding the submission process to
adhering to best practices.
