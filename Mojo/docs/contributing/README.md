# Mojo contributor documentation

This documentation is for you if you're contributing to the Mojo compiler or the
Mojo standard library. Start with the
[Mojo contributor guide](../../CONTRIBUTING.md) for a short overview, then use
the pages below for the detail.

## Contributing to Mojo

- [contribution-areas.md](contribution-areas.md)—*Contribution areas*: Which
  parts of the codebase accept contributions, and which kinds of change each
  team accepts. Check this before you start work.

- [contribution-process.md](contribution-process.md)—*Contribution process*:
  The five stages of a contribution, from signaling intent through review,
  merge, and release.

- [proposal-process.md](proposal-process.md)—*Proposal process*: How to
  propose a significant change, and how the leads decide on it.

- [issue-pr-etiquette.md](issue-pr-etiquette.md)—*Issue and PR etiquette*:
  What we expect from you when engaging with the
  [modular/modular](https://github.com/modular/modular) repository, including
  our rules on AI-assisted contributions and pull request size.

## Standard library

- [stdlib/stdlib-development.md](stdlib/stdlib-development.md)—*Standard
  library development*: Set up your environment, fork and branch, build the
  library, and run tests. Start here if you're new to contributing.

- [stdlib/stdlib-code-style.md](stdlib/stdlib-code-style.md)—*Coding standards
  and style guide*: Conventions for standard library code: file layout,
  `mojo format`, naming, value lifecycle, API docstrings, and how to validate
  docstrings.

- [stdlib/docstring-style-guide.md](stdlib/docstring-style-guide.md)—*Mojo
  docstring style guide*: How to write API docs (docstrings) in Mojo: voice and
  tone, named sections (including `Safety:`), formatting rules, and
  per-declaration-level conventions for packages, modules, types, fields,
  aliases, and functions.

- [stdlib/adding-gpu-targets.md](stdlib/adding-gpu-targets.md)—*Adding a new
  GPU target*: How to extend `std/_gpu/host/info.mojo` with a new GPU
  architecture, covering the MLIR target configuration and the `data_layout`
  string format.

- [stdlib/faq.md](stdlib/faq.md)—*Frequently asked questions*: Contributor
  FAQ for the standard library (platform support, bug reporting, MLIR dialects,
  compiler runtime).

## Compiler

- [compiler/README.md](compiler/README.md)—*Compiler contributor docs*: Where
  to find the documentation you need to work on the compiler.

## Other docs

- [Mojo user documentation](https://www.mojolang.org/docs): Published from
  [`Mojo/docs/site`](../site).
- [`/max/docs`](/max/docs): Docs for developers working in the MAX framework.
- [`/max/docs/design-docs`](/max/docs/design-docs): Engineering docs that
  describe how core Modular technologies work.
- [max.modular.com](https://max.modular.com): All other developer docs.
