# The nonblocking design of `M::AsyncRT::WorkQueue`

## Introduction

One of the key problems that a thread pool must solve is how they behave when
an item of work they execute blocks its thread (for example on I/O). When this
happens, the thread is implicitly taken out of the thread pool, and therefore
the machine ends up being over- or under-utilized.

For example, consider a 4-core machine: you can have 4 threads keeping the
machine busy, but if one of them blocks on disk or network I/O for 100ms, then
you've just given up 1/4 of your CPU cycles for 100ms that could be used to
execute other work in the work queue.

This is a challenging problem to deal with, particularly with large scale
software systems - most existing code in the world was built on top of existing
blocking APIs (for example even simple things like `printf` can block!). We
also face the unfortunate situation where some important OS's (for example like
older Linux kernels) [don't even support non-blocking async
I/O](http://davmac.org/davpage/linux/async-io.html).

There are two major approaches used to solve this problem: adaptive thread
pools... and what AsyncRT does. :)

## Adaptive thread pools

One classical way to try to solve for this is with adaptive thread pools, this
is how (for example) Apple's Grand Central Dispatch (GCD) API works.

Unfortunately, there are many problems with adaptive thread pools:

1. They end up firing up many more threads than the CPU has cores, relying on
   the kernel to switch between them, or with equivalent user-space
   functionality, or hybrids (M:N, fibers, etc). Regardless of the
   implementation approach, it is inefficient to lose processor caches and other
   state on each switch, and leads to poor latency stability.
2. You end up with weird edge cases where they run out of resources, for example
   crashing the system because you can't allocate enough thread stacks, or
   deadlock your app due to
   [other limitations](https://stackoverflow.com/questions/15150308/workaround-on-the-threads-limit-in-grand-central-dispatch).
3. The complexity of these systems escalate quickly because there is no
   structure to the problem. Tasks get markable with Quality of Service markers
   to help the scheduler, new kinds of queues get introduced for special cases,
   etc. The
   [source code for GCD](https://github.com/apple/swift-corelibs-libdispatch) is
   open source and relatively portable for anyone to inspect.
4. Uncooperative
   legacy code often talks to other concurrency approaches and has other
   non-compositional behavior beyond just blocking.
5. They don't provide an
   incentive for developers to move to non-blocking APIs.

Unfortunately, after many many years of trying to solve this problem, and a fast
ramp of complexity, it has become clear that there isn't a reasonable way to
solve this problem, even at Apple scale.

Partially as a consequence of these learnings, Apple has rolled out an entirely
new language-based concurrency approach in Swift built on async/await and
actors that eliminates blocking... but that isn't helpful to us in C++ land.

## AsyncRT's Approach for Blocking Tasks

Our approach for `WorkQueue` is to punt on the problem, and build on our
approach to library based design to allow clients to control things at a coarse
grain. This involves a few things:

1. The implementations of the `WorkQueue` interface can and should assume that
   the work items do not block. This allows them to be relatively simple and
   efficient in the important case where we have cooperative workloads.
2. Our runtime cooperates with other runtimes, threading models, and other stuff
   going on in the system. That non-cooperative stuff typically has other
   considerations, including their own synchronization primitives, etc. As such
   we just use those foreign systems to solve our problem: instead of running a
   potentially blocking operation on a `WorkQueue`, you should invoke it on the
   foreign thread pool, and have it add a completion handler work-item to the
   `WorkQueue` when it completes.
3. The top level client of `M::AsyncRT::CPUDevice` can then configure the system
   so the `WorkQueue` is balanced with the foreign runtimes at a granular level.
   For example, it is entirely reasonable for some use cases to configure
   `CPUDevice` to have N threads on an N CPU system, while also having other
   foreign runtimes with N (or more) threads for themselves. The OS kernel will
   handle context switching. It won't be ideal situations, but working with
   legacy code is not ideal.
4. This approach makes it slightly less efficient to use the legacy approach, so
   cases that matter will have pressure to move to the native design. This
   encourages the legacy code that matters to improve, instead of leaving it
   around with no incentive to invest in it.
5. Finally, there is no checking that workitems don't block, so an empirical
   approach to this is fine. For example, `printf` can theoretically block (for
   example when the output is redirected to a file) but we don't want to make
   `printf` debugging painful. Similarly, a `std::mutex` can block, but if you
   have a mutex that is protecting a tiny critical region, then it is safe
   enough to

This is a different approach than is used in many systems, but it strikes a
mix between being very useful for "completely pure" systems which scales down to
embedded systems where you want to control everything, while still working for
large scale systems with a lot of legacy.

Many clients of `M::AsyncRT::CPUDevice` will themselves be built with blocking
expectations (purely async-safe code is still relatively unusual) so there is a
top-level `await` call which waits until a specified set of values are complete
before returning. It doesn't block: it donates the client thread to running
work in the work queue until the values of interest are available.

That said, it is better to avoid using this - particularly if you are working
with accelerators - because you typically want the client thread to go off and
discover new work, not running existing background tasks.

### Handling `await` without blocking

As mentioned above, AsyncRT `await` doesn't block because the client thread
donates itself to run the tasks in the work queue.

Every threads accessing the work queue, including the "foreign" thread that
calls `await`, owns it's own semaphore. Mostly the semaphores are used to wake
the threads up when the new task is added to the queue so they can pick up and
run the task. For `await`ing threads, however, the semaphore is also posted
when the last value that the `await` is waiting for is fulfilled.

If we had a separate signaling mechanism to indicate the fulfillment of
awaiting values (e.g. have a separate semaphore for `await`), or had a single
shared semaphore that is shared across all the threads that are accessing the
work queue, we wouldn't be able to implement efficient non-blocking execution
with `await`. If we had a separate semaphore for `await` and let the foreign
thread wait on it when there's no more tasks in the work queue, the foreign
thread wouldn't wake up even when there is a new task added to the work queue.
If a single semaphore is shared across all threads, we wouldn't be able to
point the `await`ing thread to wake up when all the awaiting values are
fulfilled.

### Busy waiting with exponential backoff

While it's not ideal for threads to spin the wheel when there are no more tasks
in the work queue, threads going into sleep too early is also problematic
because waking them up again would waste time if new tasks are added to the
queue sooner than later. To get the right trade-off, we adopted busy-waiting
with exponential backoff. When the work queue is empty, threads are waiting for
the task without sleeping for the first few iterations of the busy-wait loop.
If an extra busy-wait time is specified, threads further waits for the given
amount of time. When the busy-wait time expires, threads go into sleep and wait
for the corresponding semaphore to be posted. Currently the default busy-wait
time is set to 1ms.

### The "Missing" I/O Subsystem in AsyncRT

One challenge of building high-performance infrastructure that needs I/O is that
there is no consistent and portable way to use
[asynchronous I/O](https://en.wikipedia.org/wiki/Asynchronous_I/O). You've got
things like AIO on Linux (which is
[surprisingly bad](http://davmac.org/davpage/linux/async-io.html)), Windows has
[reasonable async I/O](https://docs.microsoft.com/en-us/windows/win32/fileio/synchronous-and-asynchronous-i-o)
that just has
[a few edge cases](https://docs.microsoft.com/en-us/troubleshoot/windows/win32/asynchronous-disk-io-synchronous).
New Linux kernels have a new fancy new
[io_uring API](https://blogs.oracle.com/linux/post/an-introduction-to-the-io-uring-asynchronous-io-framework)
that is perfect for what we need. On the other hand, asynchronous I/O may not
even make sense for embedded systems.

At some point we will care enough about this to build a new async I/O subsystem
and build this into the AsyncRT. This should be an optional component that has
OS specific implementations, and allows clients of AsyncRT to access it
asynchronously. This will allow those algorithms to be written in a host OS
independent way, and allows us to centralize the complexity of this world into
one place.
