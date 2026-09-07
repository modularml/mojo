# Compiler contributor documentation

See [contribution areas](../contribution-areas.md) for whether the compiler is
accepting contributions and which kinds of change it accepts, and the
[contribution process](../contribution-process.md) for how a change gets
reviewed and merged.

There's no compiler-specific style guide yet. Match the conventions of the code
you're changing, and run `./bazelw run //:format` before you open a PR.

## Where to start

The compiler documentation lives in [`Mojo/docs/compiler`](../../compiler). The
pages most relevant to someone new to the codebase are:

- [Working with Mojo in the open source
  repo](../../compiler/WorkingInOSRepo.md): Build flags, Bazel aliases, and
  example commands. Read this first, because
  most other compiler docs assume you're working in Modular's internal
  monorepo.

- [Mojo compiler docs](../../compiler/README.md): The entry point to the
  compiler documentation, including the source layout and the tools the
  compiler builds.

- [Compiler testing](../../compiler/testing.md): Test tools, `FileCheck`
  conventions, and guidelines for writing compiler tests. Every bug fix needs a
  test.

- [Compiler walkthrough](../../compiler/MojoCompilerWalkthrough.md): A
  high-level tour of how a Mojo program becomes an executable.

- [Debugging the compiler with LLDB](../../compiler/MojoLLDB.md).
