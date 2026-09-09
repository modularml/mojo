# Support Utility Library

## Introduction

This library is a loose collection of utilities that all other code in the
repository can use. In order to maintain proper build dependencies, it is
_required_ that code in the `Support/` directory cannot depend on code in
any other top level directory.

## Libraries

The libraries in `Support/` are not necessarily related to each other. They may
be independent, or they may depend on other libraries in `Support/`. New
libraries should only be added to `Support/` if they don't make sense to add
anywhere else in the repository.

Each library should have an associated documentation file to indicate its
purpose. The current libraries are:

- [ADT](ADT.md)
- [ASN1](ASN1.md)
- [Compiler](Compiler.md)
- [CrashReporting](CrashReporting.md)
- [Cryptography](Cryptography.md)
- [DebugInfoDialect](DebugInfoDialect.md)
- [Driver](Driver.md)
- [Filesystem](Filesystem.md)
- [Frameworks](Frameworks.md)
- [Globals](Globals.md)
- [HTTP](HTTP.md)
- [Logging](Logging.md)
- [MArchTarget](MArchTarget.md)
- [MDialect](MDialect.md)
- [ML](ML.md)
- [Profiling](Profiling.md)
- [Telemetry](Telemetry.md)
- [Threading](Threading.md)
- [Tracy](../../docs/internal/Tracy.md)

The `Support/` directory also includes a number of stand-alone libraries that
are not in subdirectories.
