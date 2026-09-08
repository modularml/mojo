
# Mojo Compiler Docs

This directory contains the Mojo compiler docs. This file is the
one-stop-shop entry point to all documentation relevant to the Mojo compiler.

> [!TIP]
>
> The name `KGEN` stands for "kernel generator". When you see KGEN, think Mojo.

## Working in the Open Source Repository

Most of the docs in this directory assume you're working in the Modular
monorepo. If you're working in the open source repository, you'll need to adjust
some of the commands. For details, see
[Working with Mojo in the open source repo](WorkingInOSRepo.md).

## File Overview

- `KGEN/` -- Mojo compiler sources, tests, and documentation
  - `docs/` -- You are here 👋. Main documentation for the Mojo compiler; links
    out to the other docs.
    - `docs/manual/` -- intro docs, written assuming no prior Mojo compiler
      knowledge; for newcomers to the compiler team, or folks making drive-by
      contributions.
    - `docs/overviews/` -- subsystem and cross-cutting behavior overviews, for
      those more familiar with the compiler.
      `docs/arcana/` -- more detailed docs, diving deep into nuanced behavior;
      useful for someone trying to debug the compiler, this has the vital hidden
      clues.
    - `docs/attic/` -- older compiler docs, that capture prior thinking and
      behavior. Occasionally useful to consult when doing code archeology.
  - `lib/` -- C++ sources for the Mojo compiler _libraries_, including parser,
    passes, MLIR dialects, and related tooling (e.g. debugger), etc.
  - `tools/` -- C++ sources for command-line interface _executables_ (CLI),
    including `mojo`, `kgen`, `kgen-opt`, `kgen-translate`, etc.
  - `test/` -- Compiler tests in the form of Mojo source programs that exercise
    features of the language, `mojo` CLI, and related tooling. See
    [testing.md](testing.md) for test tools and conventions.

## Artifacts

### Command-Line Tools

Public tools ship in the `mojo` package. Internal tools are available only in
the repository.

- `mojo` (_public_) — Main Mojo compiler executable.
  - Input: `.mojo`; Output: executables, `.a`, and `.dylib`
- `kgen` (_internal_) — Full compiler driver: parses Mojo (or reads MLIR),
  runs the KGEN pipeline, and emits an artifact. Can output various levels
  of IR or final build artifacts.
  - Input: `.mojo`, `.mlir`; Output: `.mlir`, `.mlirbc`, `.ll`, `.s`, `.o`,
    `.so`, C++ header, or program output
- `kgen-opt` (_internal_) — Apply specific optimization passes to KGEN IR.
  - Input: `.mlir`; Output: `.mlir`
- `kgen-translate` (_internal_) — Front end only, no compilation pipeline.
  `-import-mojo` parses and type-checks Mojo source and prints the resulting
  `lit` dialect MLIR; `-mlir-to-llvmir` translates already-lowered MLIR to
  textual LLVM IR. Use it to see what the parser produces, then pipe the output
  into `kgen-opt` to study individual passes.
  - Input: `.mojo`, `.mlir`; Output: `.mlir`, `.ll`

### Libraries

- `CompilerRT.a` — Runtime library for Mojo, linked in to every compiled
  Mojo programming, and providing facilities to the Mojo standard library.

**TODO:** The Mojo compiler produces dialect libraries(?) that are consumed
by the Graph Compiler so that it has knowledge of the dialects used by the
Mojo compiler.

## High-Level Walkthrough

- [Walkthrough Doc](MojoCompilerWalkthrough.md)

## Compiler Manual

- [Mojo Design](./manual/MojoNotes.md)
- [Rationale](./manual/Rationale.md)
