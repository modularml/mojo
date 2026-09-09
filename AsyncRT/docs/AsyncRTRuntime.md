# `M::AsyncRT::CPUDevice` Overview

This document introduces the `M::AsyncRT::CPUDevice`, some of the design points,
key configuration points and rationale for how it works. For more details on
datatypes and more specialized topics please see:

- [`AsyncValue` type documentation](AsyncValue.md)
- [Non-blocking work queues](WorkQueueNonblocking.md)

## Library-based Design

`M::AsyncRT::CPUDevice` is designed as a low-level concurrency library for
managing system resources on modern CPU systems. It has many peers that provide
similar functionality, such as Intel Thread Building Blocks, Apple Grand Central
Dispatch, and many others.

It has three major differentiating factors:

1) It follows proper library-based design approaches (for example there are no
   global parallel-for-each operations that act against an implicit thread
   pool).
2) It is designed to cooperate with other things within the application that
   are using CPU threads (including multiple instances of itself).
3) Its key policies (for example how is a thread pool implemented?) are
   abstracted from the code that is working with it.

This is very important: we want AsyncRT-based technology to compose into
existing applications with other things going on. A high performance console
video game is effectively its own operating system, talks to accelerators, and
has a lot of other things going on: we want AsyncRT-based tech to work in such
contexts.

Similarly, many numeric algorithms, compiler algorithms, and other interesting
concurrent workloads are independent of the underlying execution model. We want
these algorithms to be expressible in a way that isn't unduly exposed to the
operating system details - in fact, many of these can run on bare-metal systems.

On the other hand, we're not providing a virtual machine. If clients of
`M::AsyncRT::CPUDevice` want to be written in a system specific way, the full
machine is open and clients are not prevented from doing quirky and exotic
things as necessary.

## AsyncRT `WorkQueue` / Thread Pool Abstraction

The
[M::AsyncRT::WorkQueue](../include/AsyncRT/Runtime/WorkQueue.h)
class is an abstract interface for a work queue, which is usually implemented to
execute the submitted work in parallel with a thread pool. This class is
intentionally very simple, but has a few important design points:

1) It is an abstract interface that may be implemented in many different ways,
   for systems with different levels of thread abstractions or ones with
   familiar abstractions but different constraints.

2) The interface is minimal: you may add work, and may ask that a client thread
   is blocked until some `AsyncValue`s are computed, or until all work has
   completed.

3) The design assumes that work items never block.

Overall, an important mental model for `WorkQueue` is that the efficient way to
use a modern multicore system is to have one OS thread per CPU execution context
(core, hyperthread, etc) working away without interruption. You don't want to
spin up thousands of kernel threads and have context switches between them -
this is bad for cache efficiency and has other overheads.

The interface of `WorkQueue` is independent of this model, but is designed to
facilitate efficient implementation of it for different use-cases.

### An abstract interface with multiple implementations

`WorkQueue` providing an abstract interface is important for our goals of
AsyncRT as a root technology of various library-based designs. If the bottom of
the stack doesn't have a proper library-based design, nothing built on top of it
will either.

Furthermore, there are lots of ways to implement threading, including pthreads,
Windows threads, fibers, running in an unsynchronized single-thread context,
running on a multi-core bare-metal embedded system, in a Linux kernel, pinning
computation to just the "little cores" of a mobile device, pinned to one socket
of a multi-socket NUMA server, server processes with dynamically changing core
count (as more processes are loaded onto the machine), etc.

There is no "right" answer here - only the client can know the right way to
execute the work.

### Minimal interface

Keeping the interface minimal allows flexibility within the implementation, and
ensures that client algorithms (for example a "parallel for loop") are kept
orthogonal from the implementations of the `WorkQueue` implementations. The
algorithms that compose onto this interface are implemented separately, in
[Runtime/Algorithms.h](../include/AsyncRT/Runtime/Algorithms.h).

### Designed for non-blocking work

Implementations of `WorkQueue` are allowed to assume that no work items
submitted to the queue may block (for example on I/O). This allows a much
simpler implementation approach and allows more efficient use of the machine.
For an exploration of the issues involved here, please see [detailed document
on non-blocking work queues](WorkQueueNonblocking.md).

That said, it is very typical for top level clients to want to submit work and
not embrace the non-blocking approach themselves. As such, there is a top-level
`await` call which wait for a specified set of values to be computed. This
routines is designed to be used by the client only, and not by items in the work
queue or other things that are supposed to be non-blocking.

## AsyncRT `Allocator` Abstraction

The
[`M::AsyncRT::Allocator`](../include/AsyncRT/Runtime/Allocator.h)
class provides an interface for heap allocation. Similar to the `WorkQueue`
class, it is an abstract interface that allows algorithmic code to be kept
independent of client specific policies, allowing both to work together.

The core interface is similar to `malloc`/`free`, with a couple of
refinements: 1) the allocation interface takes an alignment specifier, allowing
the allocation of SIMD vectors and other types that require 16 or 32-byte
alignment in a composable way, and 2) the deallocation interface also takes a
size argument (making some allocator algorithms much more efficient).

This abstraction allows for some important opportunities. For example, AsyncRT
has a leak tracking and profiling allocator interface, which keep track of
allocation statistics while passing requests to another allocator. This allows
our unit test suite to default the leak tracking allocator, which is very handy
for finding bugs early.

Another use-case is for large scale server systems, which often have
multi-socket NUMA CPUs in them. In these cases, you really want to pin both the
compute (with `WorkQueue`) and the data (with `Allocator`) to the same socket to
avoid saturating the relatively low bandwidth inter-socket interconnect between
the CPUs. It also allows integration with "huge page" OS features which
[require fiddly logic to use](https://stackoverflow.com/questions/32652833/how-to-allocate-huge-pages-for-c-application-on-linux)
but can massively affect latency in some cases by reducing TLB misses.

When building data intensive applications (for example allocating tensor data
in a machine learning application), it is a good idea to allocate that data
with the `Runtime` you're executing within.

### Do not use `Allocator` for tiny allocations

While the `Allocator` interface is important for large-scale allocations, it
does provide a tiny bit of overhead (a vtable indirection) and isn't intended
to integrate into fine-grained C++ allocators. This means you shouldn't try to
funnel all `std::string` or `std::vector` allocations through it, nor should
every tiny linked list node go through it.

Focus on large scale allocations that can matter to the memory bandwidth of your
workload.

### Tracing allocations with `MODULAR_ALLOC_LOGGING`

The `TCMallocAllocator` and the Mojo heap allocator
(`KGEN_CompilerRT_AlignedAlloc/Free`) support optional `DEBUG`-level logging of
every allocation and free. Because these functions are in hot paths, the logging
is compiled out by default and must be opted in at build time:

```bash
./bazelw run //your:target --//AsyncRT:alloc_logging=true
```

Then enable the `DEBUG` log level at runtime:

```bash
MODULAR_LOG_LEVEL=DEBUG ./your_program
```

Example output:

```text
[13:41:13] [ DBG] tcmalloc alloc: ptr=0x7f1234560000 size=128 alignment=8
[13:41:13] [ DBG] tcmalloc free: ptr=0x7f1234560000 size=128
[13:41:13] [ DBG] mojo alloc: ptr=0x112f3fc00000 size=12582912 alignment=1
[13:41:13] [ DBG] mojo free: ptr=0x112f3fc00000
```

The compile-time guard ensures zero overhead in normal builds — no level check,
no branch.
