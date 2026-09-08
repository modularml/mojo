---
title: Mojo FAQ
sidebar_label: FAQ
description: Answers to questions we expect about Mojo.
---

We tried to anticipate your questions about Mojo on this page. If this page
doesn't answer all your questions, see [Mojo vision](/docs/vision/) for the
history and motivation behind the Mojo language, and the
[roadmap](/docs/roadmap) for a high-level view of what's next for Mojo.

## Motivation

### Why did you build Mojo?

We built Mojo to solve an internal challenge when building the [Modular
Platform](https://www.modular.com)—programming across the entire stack was too
complicated. We wanted a flexible and scalable programming model that could
target CPUs, GPUs, AI accelerators, and other heterogeneous systems that are
pervasive in the AI field. This meant a programming language with powerful
compile-time metaprogramming, integration of adaptive compilation techniques,
caching throughout the compilation flow, and other features that existing
languages don't support.

As a result, we're extremely committed to Mojo's long-term success and are
investing heavily in it. Our overall mission is to unify AI software and we
can't do that without a unified language that can scale across the whole AI
infrastructure stack. Our current focus is to unify CPU and GPU programming with
blazing-fast execution for the Modular Platform. That said, the north star is
for Mojo to support the whole gamut of general-purpose programming over time.

For more detail, and insight into why we built Mojo the way we did, see the
[Mojo vision](/docs/vision/).

### Why is it called Mojo?

Mojo means "a magical charm" or "magical powers." We thought this was a fitting
name for a language that brings magical powers to programmers, including
unlocking an innovative programming model for accelerators and other
heterogeneous systems pervasive in AI today.

## Functionality

### Where can I learn more about Mojo's features?

The best place to start is the [Mojo Manual](/docs/manual/). And if you want to
see what features are coming in the future, take a look at [the
roadmap](/docs/roadmap).

### Is Mojo only for AI, or can I use it for other things?

<!-- rumdl-disable MD033 -->
<!-- Use HTML anchor to link to community page so it doesn't
     get rewritten to /nightly/community/                    -->

Mojo's initial focus was to solve AI programmability challenges. However, our
goal is to grow Mojo into a general-purpose programming language. We use Mojo at
Modular to develop AI algorithms and GPU kernels, but you can use it for other
things like HPC, data transformations, writing pre/post processing operations,
libraries, and much more. See the <a href="/community/">community page</a> to
get inspired by the projects others are writing in Mojo!

<!-- rumdl-enable MD033 -->

### Is Mojo interpreted or compiled?

Mojo is a compiled language. [`mojo build`](/docs/cli/build/) and [`mojo
run`](/docs/cli/run/) both perform ahead-of-time (AOT) compilation.

### Does Mojo support distributed execution?

Not alone. Mojo is one component of the Modular Platform, which
makes it easier for you to author highly performant, portable CPU and GPU graph
operations, but you'll also need a runtime (or "OS") that supports graph-level
transformations and heterogeneous compute, which the
[MAX framework](https://max.modular.com) provides.

### How do I convert Python programs or libraries to Mojo?

See [Tips for Python devs](/docs/manual/python-to-mojo/) for a quick primer on
important differences between Python and Mojo. The
[Mojo AI skills](/docs/tools/skills/) can help your AI coding assistant
translate Python code into working Mojo code.

You can also migrate parts of a Python project to Mojo
by building Mojo bindings for Python. See the documentation about how to [call
Mojo from Python](/docs/manual/python/mojo-from-python).

### What about interoperability with other languages like C/C++?

Mojo code is interoperable with C code. For information, see the docs for the
[`ffi`](/docs/std/ffi/) module, the
[`@export`](/docs/reference/decorators/export/) decorator, and the
[`abi("C")` function effect](/docs/reference/function-declarations/#abi-c).

Mojo code is also interoperable with C++ code that uses `extern "C"`. We
believe we can deliver better C++ interoperability in the future.

### How does Mojo support hardware lowering?

Mojo leverages LLVM-level dialects for the hardware targets it supports, and it
uses other MLIR-based code-generation backends where applicable. This also
means that Mojo is easily extensible to any hardware backend.

### Who writes the software to add more hardware support for Mojo?

Mojo provides all the language functionality necessary for anyone to extend
hardware support. As such, we expect hardware vendors and community members to
contribute additional hardware support in the future.

## Performance

### Are there any AI-related performance benchmarks for Mojo?

Remember that we designed Mojo as a general-purpose programming language, and
any AI-related benchmarks rely heavily upon other framework components. For
example, we write all of the in-house CPU and GPU graph operations that power
the Modular Platform in Mojo. You can learn more about performance in our blog
posts on
[bringing the Modular Platform up on AMD MI355](https://www.modular.com/blog/achieving-state-of-the-art-performance-on-amd-mi355----in-just-14-days)
and
[optimizing matmul performance on the NVIDIA Blackwell GPU](https://www.modular.com/blog/matrix-multiplication-on-blackwell-part-4---breaking-sota).

## Mojo SDK

### How can I get the Mojo SDK?

You can get Mojo and all the developer tools by installing `mojo` with
any Python or Conda package manager. For details, see the
[Mojo installation guide](/install/).

### What's included in the Mojo SDK?

We actually offer two Mojo packages: `mojo` and `mojo-compiler`.

The `mojo` package gives you everything you need for Mojo development.
It includes:

- [`mojo` CLI](/docs/cli/) (includes the Mojo compiler)
- [Mojo standard library](/docs/std/)
- [`mojo` Python
  package](https://github.com/modular/modular/tree/main/mojo/python/mojo)
- Mojo language server (LSP) for IDE/editor integration
- [Mojo debugger](/docs/tools/debugging/) (includes LLDB)
- [Mojo code formatter](/docs/cli/format/)
- [Mojo REPL](/docs/cli/repl/)

The `mojo-compiler` package is smaller and is useful for environments where you
only need to call or build existing Mojo code. For example, this is good if
you're running Mojo in a production environment or if you're programming in
Python and [calling a Mojo
package](/docs/manual/python/mojo-from-python)—situations where you don't need
the LSP and debugger tools. It includes:

- [`mojo` CLI](/docs/cli/) (includes the Mojo compiler)
- [Mojo standard library](/docs/std/)
- [`mojo` Python
  package](https://github.com/modular/modular/tree/main/mojo/python/mojo)

If you're interested in GPU programming, install the `max` package, which
includes the MAX framework and Mojo. For details, see
[Get started with GPU programming](https://max.modular.com/gpu/intro-tutorial/)
in the MAX documentation.

### What are the license terms for the SDK?

The Mojo SDK is licensed under the Apache License v2.0 with LLVM Exceptions. For
details, see the
[LICENSE](https://github.com/modular/modular/blob/main/LICENSE).

### What operating systems does Mojo support?

Mojo supports Mac and Linux natively and supports Windows via WSL. For details,
see the [Mojo system requirements](/docs/requirements/).

### Is there IDE integration?

Yes, we've published an official Mojo language extension for
[Visual Studio Code](https://code.visualstudio.com/) and other editors that
support VS Code extensions (such as [Cursor](https://cursor.com/home)). The
extension supports various features including syntax highlighting, code
completion, formatting, hover, etc. It works seamlessly with remote-ssh and dev
containers to enable remote development in Mojo.

You can obtain the extension from either the
[Visual Studio Code Marketplace](https://marketplace.visualstudio.com/items?itemName=modular-mojotools.vscode-mojo)
or the
[Open VSX Registry](https://open-vsx.org/extension/modular-mojotools/vscode-mojo).

### Does the Mojo SDK collect telemetry?

Yes, the Mojo SDK collects some basic system information, tool invocation
events, crash reports, and some LSP performance events that enable us to
identify, analyze, and prioritize Mojo issues.

Specifically, we collect:

- **Invocation events**: Each Mojo tool (the `mojo` CLI, the Mojo language
  server, and the Mojo debugger) reports a single event when it starts. The
  event includes the tool name and subcommand (such as `build` or `run`), and
  whether crash reporting is enabled. It does not include your command-line
  arguments, file names, or source code.
- **Crash reports**: When a Mojo tool crashes, it uploads a crash report
  containing the stack trace of the crashed process, along with the tool name
  and the Mojo version, so we can attribute the crash to a specific release
  and fix it.
- **LSP performance metrics**: The Mojo LSP reports aggregate data on how long
  it takes to respond to user input (parsing latency). The report includes only
  the milliseconds between user keystrokes and when the Mojo LSP is able to
  show appropriate error or warning messages.

Every event also includes the Mojo/MAX version, basic system information (OS
type and version; CPU architecture, model name, core count, and supported CPU
features), and two anonymous identifiers: a machine identifier (a one-way
hash, which cannot be reversed to identify you or your hardware) and a
randomly generated per-session identifier. These identifiers let us count
active installations and connect a crash report to its invocation event - for
example, to compute a crash rate per release.

We never collect or transmit any user information, such as source code,
keystrokes, or any other user data.

This telemetry is crucial to help us quickly identify problems and improve our
products. Without this telemetry, we would have to rely on user-submitted bug
reports, and in our decades of experience building developer products, we know
that most people don't do that. The telemetry provides us the insights we need
to build better products for you.

Telemetry can be disabled by setting the environment variable
`MODULAR_TELEMETRY_ENABLED=false`.

## Versioning & compatibility

### What's the Mojo versioning strategy?

Starting with Mojo 1.0, Mojo follows semantic versioning for the core
language and stable portions of the standard library.

We consider language features stable unless we explicitly identify them as
experimental or unstable.

We consider standard library APIs **unstable** unless we explicitly identify
them as stable. The API documentation identifies the stable APIs.

For more information, see
[Mojo stability guarantees](/docs/api-docs/stability/).

See our
[roadmap](/docs/roadmap/) to understand where things are headed.

### How often do you release new versions of Mojo?

Mojo development is moving fast and we are regularly releasing updates. We aim
to produce stable releases every six weeks, and nightly builds almost every
night.

Join the [Mojo Discord channel](http://discord.gg/modular) for notifications and
[sign up for our newsletter](https://www.modular.com/blog#sign-up-for-our-newsletter)
(on the bottom of the Mojo blog page) for more coarse-grained updates.

## Open source

### Is Mojo open source?

Mojo is open source under the Apache License v2.0 with LLVM Exceptions. For
details, see the
[LICENSE](https://github.com/modular/modular/blob/main/LICENSE).

### Why didn't you develop Mojo in the open from the beginning?

Though we always intended to open source Mojo eventually, we started developing
it in private and took a gradual approach to open sourcing the entire language.

Mojo is a big project and has several architectural differences from previous
languages. We believe a tight-knit group of engineers with a common vision can
move faster than a community effort. Other projects that are now open source
(such as LLVM, Clang, Swift, MLIR, etc.) also followed this well-established
development approach.

## Community

### Where can I ask more questions or share feedback?

If you have questions about upcoming features or have suggestions for the
language, be sure you first read the [Mojo roadmap](/docs/roadmap/), which
provides important information about our current priorities.

<!-- rumdl-disable MD033 -->

To get in touch with the Mojo team and developer community, use the resources
on our <a href="/community/">community page</a>.

<!-- rumdl-enable MD033 -->
