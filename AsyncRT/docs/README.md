# AsyncRT: Asynchronous Runtime

## Introduction

This library contains low level support for domain independent parallel CPU
computation. It is intended to support building high performance runtimes, as
well as hosting parallel compiler infrastructure like MLIR and LLD. AsyncRT
currently contains two major pieces of functionality:
AsyncRT/Support and [AsyncRT/Runtime](AsyncRTRuntime.md).

AsyncRT/Support provides low-level concurrent containers, reference counting
support, atomics support algorithm helpers etc - which are all dependency free
(just depending on the C++ standard library like `<atomic>`). These routines
are usable for a wide range of concurrent algorithms and applications.

AsyncRT/Runtime provides a more opinionated low-level concurrency library,
including a thread pool and machinery for partitioning memory allocation. It
is specifically designed as a proper library - it does not assume it has
complete control over the machine, allowing multiple instances of it can run on
the same machine, and allowing client control over many policies. For more
information on it, see its [dedicated documentation page](AsyncRTRuntime.md)

## Introduction to AsyncRT/Support

TO WRITE:

- Browse the API:
  [include/AsyncRT/Support](../include/AsyncRT/Support/)

- `RCRef<>`, `ReferenceCounted<>`. These compose but RCRef<> works
  with other types as well, including notably `AsyncValue`.

- RCRef is intentionally move-only, but has a `.copy()` method to avoid
  accidental implicit atomic reference counts.

- `Chain` is an empty marker type used in AsyncValue-based logic for expressing
  control dependencies into data dependencies. (link to a dedicated explainer
  doc on control dependencies).

- Location and Diagnostic.

## Introduction to AsyncRT/Runtime/Algorithms.h

TO WRITE:

- Generally use this instead of poking at WorkQueue directly.

- addTask, await. Parallel for loop, map reduce ...

- If this gets larger and more detailed, it may make sense to split it out to
  its own doc.
