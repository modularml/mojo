# Vision

## Principles

The following are “North Star” principles for the Mojo Standard Library.
These principles will inform multiple future decisions from what features we
work on to what bugs we prioritize during triage. In short, the standard library
vision is the ideal we may never reach, but we collectively show up every day
to work towards it.

- **Foster a vibrant community collaborating globally.** The community is
encour aged to engage with the standard library and language evolution. We
intend to ignite enthusiasm to contribute to the expanding Mojo package
ecosystem.

- The Standard Library prioritizes
  **Performance > Safety > Portability > Debuggability.**

- **Respectable performance by default.** Standard Library intends to provide
respectable performance by default — but not perfect performance. We support
low-level controls that enable performance tuning by systems engineers. While
providing consistently strong performance over time, we do so with minimal
regressions.

- **Portability by default.** The use of MLIR enables portability across a
variety of platforms without writing per-platform code in the standard library.

- **The standard library does not have special privileges.** Standard Library
types are not special and do not enjoy elevated privileges over user-contributed
code. This empowers Mojicians to create primitives equally as expressive as core
language primitives.

- **Fully utilize available hardware.** The Standard Library should not inhibit
using any available hardware on the system. On the contrary, standard library
primitives will enable users to maximize the utility of all available system
hardware.

- **Standard Library features prioritize AI workload optimizations.** Mojo
ultimately aims to be a multi-purpose programming language used to solve
problems in the systems programming domain. However, we will prioritize standard
library features and optimizations that improve the state of the art for AI.

## What's Not in the Vision

We reject the following vision statements, and the reasoning for each is written
inline.

- **Tensor operations are first-class citizens.** Mojo has a prime directive to
  optimize AI workloads. However, certain core AI primitives are tightly
  integrated with the MAX engine architecture, and will remain part of the MAX
  engine codebase.

## Objectives and Goals

- Make unsafe or risky things explicit. Software using unsafe constructs is
  inevitable, but it must be minimized and explicit to the reader. Safe things
  shouldn’t look like unsafe ones and unsafe constructs should leave artifacts
  to see in the code.

- **Unsafe operations support dynamic checking — where pragmatic.** It’s fine to
  have unchecked, unsafe operations for performance, but developers need the
  ability to check things before calling these unsafe APIs.

- **Advanced memory management features.** Provides a fleet of memory allocators
  out of the box for kernel developers to use. The language runtime provides a
  decently performing default global allocator implementation (eg. thread local
  caches, automatic slab-size scaling based on heuristics, virtual memory-based
  defragmentation, and more…).

- **First-class support for parallelism and concurrency.** To fully utilize
  available hardware, the standard library will provide a complete suite of
  primitives maximizing the parallelism potential of the system.

- **First-class debugging features.** Integration with Mojo debugger by
  incorporating LLDB visualizers for the Standard Library types and collections
  to make debugging easy.

- **Consistent API and behavior across all stdlib primitives.** A higher-level
  goal of the stdlib is to be an example of well-written mojo code. For example,
  all collections should behave consistently with:

  - Default-constructed collections do not allocate memory unless they are
    holding an element.
  - Collections provide no thread safety unless mentioned explicitly.
  - Commonly implemented collection operations use the same method names across
    all implementations.
  - Naming of public aliases/parameters common among collections (akin to
    `value_type` and friends in C++).

- **Interoperability with Python code** allows progressively migrating code to
  Mojo over time to not force an entire rewrite just to improve the performance
  of code where it matters most.

## Non-Goals

While some of these may be common goals of modern programming languages, the
value doesn’t outweigh the costs for us right now as we are moving fast to build
the language. While we don’t actively attempt to break the following we provide
no guarantees that they work — especially over multiple releases.

- Stable ABI between language/compiler and library.
- Backward or forward compatibility guarantees in APIs or semantic behaviors.
- Integrating domain specific functionality, such as GUI toolkits, game
  development frameworks, networking or data science libraries to name a few.
