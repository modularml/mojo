# Mojo project manifest and build tool

A *project manifest* is a file that describes the source code files that make up
a library or executable, and how those files are meant to be built, distributed,
and interacted with by language tooling (such as language servers and
debuggers). Some examples include Rust’s `Cargo.toml`, Python’s `setup.py`, or
language-agnostic formats such as Bazel and its `BUILD` files.

A *build tool*, on the other hand, is a program that can create the libraries or
executables described in a project manifest. For example, `cargo build` compiles
and links Rust executables and libraries described in a `Cargo.toml` file. In
Python, many different tools can process `setup.py` and `pyproject.toml` files,
in order to produce Python wheels or packages.

This proposal is meant to:

1. Announce our intent to define a project manifest format for Mojo.
2. Announce our intent to implement a build tool for Mojo projects.
3. Solicit community feedback and feature requests for the project manifest and
   build tool.

## Motivation

No project manifest format exists for Mojo as of Mojo SDK v0.7.0. In the present
situation, Mojo projects can be built by invoking `mojo package` (to produce a
`.mojopkg` library) or `mojo build` (to produce an executable) on the command
line.

This status quo has several drawbacks:

1. There is no standard for building a Mojo project from source. If we examine
   popular Mojo projects hosted on GitHub today, some are built via `docker`,
   where the Dockerfile invokes `mojo run` to execute the Mojo source file. Some
   are built via CMake, where project maintainers have used `add_custom_command`
   to invoke `mojo build` and `mojo package`. Some are built via a
   `mojo package` command that is documented only in the README. Some have build
   instructions that are only known to the maintainers. The lack of
   standardization makes it difficult to download and build a Mojo source
   repository, which inhibits collaboration among the Mojo community of
   developers.
2. Collaboration within the Mojo community aside, the lack of a project manifest
   inhibits Mojo language tooling, such as the language server and debugger,
   from functioning perfectly. For example, many Mojo projects make use of
   compile time definitions, such as `-D ENABLE_TILING`. Without knowing which
   definitions should be used when compiling Mojo source code, the language
   server cannot provide the same diagnostics that the user will see when
   actually building their project.

Therefore, we think that a project manifest and build tool specific to Mojo will
resolve these issues:

1. We aim to implement a single command that can be used to build any Mojo
   project from source, addressing the first issue listed above. This is
   analogous to how `cargo build` builds the default targets with the default
   settings for any Rust project, or how `zig build` does so for any Zig
   project.
2. Because a project’s manifest will specify which Mojo compiler options are to
   be used, language tools would make use of those and function as intended.
   This addresses the second issue listed above.

## Guiding principles

- As mentioned above, we aim for a single command to be capable of building any
  Mojo project from source.
- We believe the ability to *download* a project’s dependencies from the
  Internet — a “package manager” function — can be added at a later time. For
  example, `zig build` did not originally include this functionality, but added
  an implementation over six years later. The Mojo build tool will likely
  implement the downloading and building of project dependencies soon, but it
  will be the subject of a separate proposal.
- Although the project manifest and build tool we design is specific to Mojo, we
  will aim for the best possible integration with other build systems, such as
  Python setuptools, Bazel, Buck2, and CMake. We will make accommodations to
  better support these tools whenever possible.
- Our design will benefit from community input and contributions, so **we will
  develop this as an open-source tool, written primarily in Mojo**. We believe
  doing so will also serve to drive additions and improvements to the Mojo
  standard library.

## Request for feedback

As mentioned above, this proposal is primarily to announce our intent to develop
a project manifest format and build tool for Mojo, and to solicit feedback.
Below are some topics that we would love to hear community members’ opinions on:

- Whether you agree with the motivations and guiding principles in this
  proposal.
- Which project manifest formats and build tools you love, and why. We’re
  drawing inspiration from a broad set of language ecosystems, including Rust,
  Zig, Swift, and especially Python.
- Whether to adopt the [build server
  protocol](https://build-server-protocol.github.io). We think doing so may help
  with our guiding principle to integrate well into the existing ecosystem of
  tools.
- Whether to define the project manifest as an executable program. Analogous to
  how `build.zig` and `Package.swift` are programs that define a project, should
  we define a `project.mojo` or similar construct? There are many arguments in
  favor of doing so, but on the other hand, we see tradeoffs as well, and a
  purely declarative form could be used.
- Any other thoughts you wish to contribute — we are build systems and language
  tooling nerds! Share your thoughts in [our Discord server](https://modul.ar/discord),
  and let’s geek out.
