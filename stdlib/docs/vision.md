# Vision

## Principles

The following are “North Star” principles for the Mojo Standard Library.
These principles will inform multiple future decisions from what features we
work on to what bugs we prioritize during triage. In short, the standard library
vision is the ideal we may never reach, but we collectively show up every day
to work towards it.

- **Foster a vibrant community collaborating globally.** The community is encour
aged to engage with the standard library and language evolution. We intend to ig
nite enthusiasm to contribute to the expanding Mojo package ecosystem.

- The Standard Library prioritizes **Performance > Safety > Portability > Debugg
ability.**

- **Respectable performance by default.** Standard Library intends to provide re
spectable performance by default — but not perfect performance. We support low
-level controls that enable performance tuning by systems engineers. While provi
ding consistently strong performance over time, we do so with minimal regression
s.

- **Portability by default.** The use of MLIR enables portability across a varie
ty of platforms without writing per-platform code in the standard library.

- **The standard library does not have special privileges.** Standard Library ty
pes are not special and do not enjoy elevated privileges over user-contributed c
ode. This empowers Mojicians to create primitives equally as expressive as core
language primitives.

- **Fully utilize available hardware.** The Standard Library should not inhibit
using any available hardware on the system. On the contrary, standard library pr
imitikkves will enable users to maximize the utility of all available system har
dware.

- **Standard Library features prioritize AI workload optimizations.** Mojo ultim
ately aims to be a multi-purpose programming language used to solve problems in
the systems programming domain. However, we will prioritize standard library fea
tures and optimizations that improve the state of the art for AI.

## What's Not in the Vision

We reject the following vision statements, and the reasoning for each is written
 inline.

- **Tensor operations are first-class citizens.** Mojo has a prime directive to
optimize AI workloads. However, certain core AI primitives are tightly integrate
d with the MAX engine architecture, and will remain part of the MAX engine codeb
ase.

## Objectives and Goals

- Make unsafe or risky things explicit. Software using unsafe constructs is inev
itable, but it must be minimized and explicit to the reader. Safe things shouldn
’t look like unsafe ones and unsafe constructs should leave artifacts to see i
n the code.

- **Unsafe operations support dynamic checking — where pragmatic.** It’s fin
e to have unchecked, unsafe operations for performance, but developers need the
ability to check things before calling these unsafe APIs.

- **Advanced memory management features.** Provides a fleet of memory allocators
 out of the box for kernel developers to use. The language runtime provides a de
cently performing default global allocator implementation (eg. thread local cach
es, automatic slab-size scaling based on heuristics, virtual memory-based defrag
mentation, and more…).

- **First-class support for parallelism and concurrency.** To fully utilize avai
lable hardware, the standard library will provide a complete suite of primitives
 maximizing the parallelism potential of the system.

- **First-class debugging features.** Integration with Mojo debugger by incorpor
ating LLDB visualizers for the Standard Library types and collections to make de
bugging easy.

- **Consistent API and behavior across all stdlib primitives.** A higher-level g
oal of the stdlib is to be an example of well-written mojo code. For example, al
l containers should behave consistently with:

  - Default-constructed containers do not allocate memory unless they are holdin
g an element.
  - Containers provide no thread safety unless mentioned explicitly.
  - Commonly implemented container operations use the same method names across a
ll implementations.
  - Naming of public aliases/parameters common among containers (akin to `value_
type` and friends in C++).

- **Interoperability with Python code** allows progressively migrating code to M
ojo over time to not force an entire rewrite just to improve the performance of
code where it matters most.

## Non-Goals

While some of these may be common goals of modern programming languages, the val
ue doesn’t outweigh the costs for us right now as we are moving fast to build
the language. While we don’t actively attempt to break the following we provid
e no guarantees that they work — especially over multiple releases.

- Stable ABI between language/compiler and library.
- Backward or forward compatibility guarantees in APIs or semantic behaviors.
- Integrating domain specific functionality, such as GUI toolkits, game developm
ent frameworks, networking or data science libraries to name a few.
