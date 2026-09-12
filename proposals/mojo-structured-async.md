# Mojo Structured Async

Owen Hilyard, Created December 17, 2024

## Background (Async/Await)

Async/Await is one of the programming paradigms developed in service of the
[C10K](https://en.wikipedia.org/wiki/C10k_problem) class of problems, where a
program needs to handle some large number of concurrent connections. Originally
this was around 10 thousand, now commonly 10 million or 100 million, but
generally it is large enough such that blocking IO is no longer sufficient. It
provides a mechanism through which the compiler can automate the creation of a
state machine, a formerly manual process, and as a result allows a user to write
sequential code that is broken up by "yield points", where the compiler may
suspend the task while IO is completed. However, as typically implemented this
results in a variety of issues.

## Motivation

Async/await has a few issues for high performance code as normally implemented,
but I think that Mojo can address those problems. Before I try to introduce
solutions, let's get on the same page as to the problems. The first one, which
prevented the adoption of this paradigm in many domains, is the lack of control
over memory. This is caused by heap allocations being made on behalf of the
developer, potentially inside of a hot loop, with no way to handle the failure
to allocate the coroutine frame. Many coroutines are also unable to be allocated
on the stack, due to the requirement to fully type erase coroutines to run them
in a generic executor, which is another reason for the heap allocations. Rust's
coroutine model helped to solve some of these issues, since it does not
inherently require type erasure or heap allocation. It allows for storing
futures in caller-allocated storage, giving the programmer control over the
placement of futures, and imposes no requirements on an async executor, allowing
for single-future executors such as `futures::block_on` from the `futures`
crate, or custom executors which execute futures from a closed set by storing
the in-progress futures as a sum type, or more general executors which perform
type erasure, such as `tokio` and `glommio`. However, in doing so Rust has
uncovered contention between the classic work stealing scheduler, thread safety,
and user ergonomics. Additionally, the general `async`/`await` model has also
been criticized for causing function coloring and viral annotations, causing
ecosystems to frequently drift towards forcing beginners to learn
`async`/`await` and multithreading before performing simple tasks such as
performing a database query due to the requirement for authors to duplicate
large parts of their libraries if they want to support both `sync` and `async`
code. After substantial discussions with other members of the community, I think
that the model presented here will help increase the expressiveness desired by
performance-sensitive and correctness-critical users, decrease the
implementation burden on library authors, make the Mojo ecosystem more friendly
to new users and be reasonable to implement inside of the Mojo compiler.

## Background: Rust Futures

```rust
pub trait Future {
    type Output;

    // Required method
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output>;
}
```

Rust treats [`Future`](https://doc.rust-lang.org/std/future/trait.Future.html)
as a trait. You may notice that Rust requires `self: Pin<&mut Self>`. This is
because Rust's futures are also almost always self-referential, and Rust does
not have move constructors, so
[`Pin`](https://doc.rust-lang.org/std/pin/struct.Pin.html) is needed to make
moving them around safe by giving them a chance to do pointer fixups before they
are invoked. We don't need that, so we can toss it out unless we want to discuss
whether it's a good idea to delay fixups until the last possible second to defer
work. Note that for any coroutine allocated on the heap (and not moved), `Pin`
is pure overhead. `Context` mostly serves to store a `Waker` which is used to
notify the async executor when a task is ready to be executed, primarily for
things like epoll where the executor isn't getting information from the OS about
when IO is done. The `Waker` type is implemented using a data pointer and a
vtable pointer, something which can likely be made generic so it can be
devirtualized where possible.

The executor calls `poll` on the pinned future, providing a context struct, and
gets back a `Poll` sum type, which is either a `Poll::Pending`, meaning the
future is still working but has given control back to the executor and doesn't
need to run until the `Waker` signals the executor, or a `Poll::Ready(T)`,
meaning that the coroutine is finished. For Mojo, we can expand this `Poll` enum
to allow `Poll::Pending` to have data attached to it, allowing the
implementation of async generators.

However, the most important part of this is that `Future` is a normal trait, and
can be attached to normal objects. This means that a user can write their own
futures timeouts (ex: a future that returns Ready if polled more than 1 second
after it was created), without much difficulty. This also makes it easier to add
extra capabilities on top of futures in user-space, for instance, a future may
be possible to run on both the CPU and the GPU if there is support from the
executor, allowing a program to take advantage of wherever there is free
compute. It also means there isn't any need for a full coroutine to manage
something like an epoll io request, you just implement `Future`.

This also allows storing the coroutine frame wherever you want. If you want to
make a vtable, this single-function trait turns into exactly what the current
Mojo `Coroutine` type looks like right now, a data pointer and a function
pointer, and the compiler-generated coroutines, which know the layout of the
vtable, can manipulate the vtable as needed to keep the same performance.
However, it leaves the ability for someone who can beat the compiler to
hand-write their own state machine in the good old "switch over a state tag"
style and still run in the normal async executor using this single entry point.

However, they have an issue as implemented in executors like `tokio`.

## Foundational Async

Since you have work stealing, any suspended task can be moved to another thread.
This means you get a bunch of rules about what can and cannot be carried across
suspend points, chief among them being locks and other synchronization
primitives, which happen to occur more frequently in async code than sync code.
In Rust, this leads to `Send + Sync + 'static` bounds on most async API
boundaries. `Send` means a type is safe to send between threads, and `Sync`
means that an immutable reference to the type is safe to send between threads.
`'static` means that the value lives forever, since the stack frame the
coroutine is temporarily using could go away before before new future runs. When
combined, this means that you suddenly end up with a lot of `Arc<Mutex<T>>`
floating around for your resources, even if you know that they live long enough.
There is a solution to this, but let's lay a foundation first.

This means we have an executor handle interface (or one hidden behind functions
which access static state), which looks something like this:

```mojo
trait Waker:
    # Wake up the associated coroutine
    fn wake(ref self):
        ...

# Holds ways for the coroutine to talk to the executor and schedulers.
# We can add things to this in the future via extension traits as needed
# to keep the coroutine trait signature intact.
trait CoroutineContext:
    alias WakerType: Waker
    
    fn get_waker(ref self) -> ref [self] Self.WakerType:
        ...

# Coroutine becomes a trait. Anyone who wants to hand-roll a state 
# machine can. The coroutine was already more or less a vtable anyway.
trait Coroutine:
    # This is likely to be a tagged union, but for "coroutines" which
    # hot run and are done, this can just be the return type.
    # We can create a version of the trait for "yields nothing until
    # it returns", and another for returning a tagged union. 
    alias RunReturnType: AnyType
    
    # State machine transition. Raises if the coroutine is finished executing.
    fn run[C: CoroutineContext](mut self, mut context: CoroutineContext) raises -> RunReturnType
        ...

# Swappable async schedulers. If I have strong opinions about how scheduling
# works, or so I can have sum types as instead of boxing my coroutines, I
# can make my own. 
trait AsyncScheduler:
    # Note the lack of Send/Sync bounds. This is intentional since all 
    # executors should be capable of handling local-only tasks. This 
    # frees us from many of the issues Rust has. 
    # This should be parametric.
    alias AsyncSchedulerTaskType: Coroutine

    fn get_next_task(mut self) -> Optional[Self.AsyncSchedulerTaskType]
        ...

trait AsyncTaskSpawner:
    # This should also be parametric
    alias AsyncTaskSpawnerTaskType: Coroutine

    fn spawn_task(mut self, owned co: Self.AsyncTaskSpawnerTaskType):
        ...

trait AsyncExecutor(AsyncScheduler, AsyncTaskSpawner):
    # More parametric
    alias TaskType: Coroutine + Movable
    alias AsyncSchedulerTaskType = Self.TaskType
    alias AsyncTaskSpawnerTaskType = Self.TaskType

    # default impl
    fn run_next_task(mut self, out ret: Self.TaskType.RunReturnType):
        var task = self.get_next_task()
        if task:
            out = Optional[Self.TaskType.RunReturnType](task.run())
            if not task.is_done():
                self.spawn_task(task)
            else:
                # end linear type
                destroy task^
        else:
            out = Optional[Self.TaskType.RunReturnType]()
```

This interface lacks error handling, since I think that another proposal I have
in the works will help with that, and if that proposal or some other form of
"generic over `raises`" is not accepted, then all of the functions need to
raise. The goal of this interface is to provide the **MINIMUM** traits that are
needed to represent an async executor. I don't think there's a reasonable
version of async/await where you don't have the ability to grab the next task to
run, and the ability to spawn a task. Remember that a coroutine which does not
return values can return a zero-sized value, such as an empty struct. Ideally,
most of these associated types will move into parameters on the trait so that
the trait can be implemented multiple times for a single executor (for instance
one which type erases). This is what I anticipate most coroutines will do, but
this allows for "run this single async function" async executors to avoid
indirection and still have return value placement optimizations. Of course, I
would go define a few other options very quickly:

```mojo
# Don't move the coroutine
trait RefAsyncScheduler(AsyncScheduler):
    alias RefAsyncSchedulerTaskType: Coroutine

    fn get_next_task_ref(mut self) -> \
        ref [__origin_of(self)] Self.RefAsyncSchedulerTaskType
        ...

trait TaskRemovableAsyncScheduler(AsyncScheduler):
    fn remove_task(mut self, owned task: Self.AsyncSchedulerTaskType) -> \
        Self.AsyncSchedulerTaskType:
        ...

trait WorkStealingAsyncTaskSpawner(AsyncTaskSpawner):
    alias WorkStealingAsyncTaskSpawnerTaskType: Coroutine + Send
    
    fn spawn_mt_task(
        mut self, 
        owned co: Self.WorkStealingAsyncTaskSpawnerTaskType
    ):
        ...

# A coroutine which can be cancelled and will clean up after itself 
trait AsyncCancellableCoroutine:
    async fn cancel(owned self):
        ...

# A sync version of above. If no async is needed to clean up a coroutine,
# then this lets you destroy the coroutine
trait CancellableCoroutine:
    fn sync_cancel(owned self):
        ...

# A coroutine which ran entirely in the "hot start" period. 
struct HotFinishCoroutine[T: AnyType](Coroutine):
    alias RunReturnType = T
    var value: Optional[T]

    fn run(mut self, mut context: CoroutineContext) raises -> RunReturnType
        if self.value:
            return self.value.take()
        else:
            raise "Empty!"
```

My hope is that, by adopting a highly modular approach to async capabilities,
library authors can specialize on the features offered by the async executor,
instead of trying to find a trade-off between "sufficiently featured as to not
be slow" and compatibility. This does mean "trait explosion", but I think that
can be mitigated by both adding trait unions to the language (ex: `Coroutine +
Send + Movable`) and choosing names that are readable for things we expect most
users to interact with. I think that decomposing this and keeping the standard
executor behind a wall of traits should help when people have specialized needs,
or if we try to bring async to GPUs in the future as a way to represent, say,
async copy operations, or reading from disk. Those executors will have radically
different capabilities than the CPU-based ones will, but we want libraries that
can work on both to be able to work on both. We can keep with the Box + type
erase approach the standard library currently uses, but we don't need to make it
mandatory for all coroutines. I also don't think it's a good idea to hide
coroutines being able to yield from the user. That is a very useful property,
and that's why I have exposed it here. It allows patterns like yielding
"Operation Requests" which tell the scheduler to do some kind of IO or other
operation and put the task back into the rotation for being scheduled when the
task is done.

## Coroutines from Functions

Given that `Coroutine` is now a trait, how do we handle coroutines defined by
async functions? Lets look at a few example functions:

```mojo
async fn async_read(mut fd: FileDescriptor, mut buffer: Span[UInt8, MutableOrigin]) raises -> Int:
    var pollfd = PollFd(fd, epoll.POLLIN | epoll.POLLERR)
    while True:
        # I am aware this is inefficient, this is done for illustrative 
        # purposes
        var result = poll(UnsafePointer.address_of(pollfd), 1, 0)
        if result == 0:
            __yield(None)
            continue
        
        if result < 0:
            # Make an error from errno and raise
            raise_errno()

        if pollfd.revents & epoll.POLLERR = epoll.POLLERR:
            # Will raise the error
            read(fd, buffer)
        
        return read(fd, buffer)

async fn async_open(path: Path, flags: Flags) raises -> FileDescriptor:
    # epoll doesn't really have an async file open, but pretend it does.
    # 
    # Also pretend that `open` has a similar poll/yield loop to what 
    # async_read has.
    return await open(path, flags)

async fn async_readall(path: Path) raises -> List[UInt8]:  
    var fd = await async_open(path, Flags::READ | Flags::NONBLOCK)
    var file_length = fstat(fd).size()
    
    var buffer = List[Uint8]()
    buffer.reserve(file_length + 1)
    for _ in range(file_length + 1):
        buffer.push_back(0)

    var cursor = 0
    while cursor < file_length:
        cursor += await async_read(fd, buffer[cursor:])

    fd.close()
    return buffer
```

The compiler can then generate something like this (boilerplate omitted for brevity):

```mojo
@value
struct __CO_async_readall_state:
    alias open_yield: __CO_async_readall_state(0)
    alias async_read_yield: __CO_async_readall_state(1)
    alias done: __CO_async_readall_state(2)

    var value: UInt8

struct __CO_async_readall_ctx_open_yield[origins: OriginSet]:
    var path: Path

struct __CO_async_readall_ctx_async_read_yield[origins: OriginSet]:
    var fd: FileDescriptor
    var file_length: UInt
    var buffer: List[UInt8]
    var cursor: UInt
    # borrow passed into async_read unless it can be elided
    var __async_read__borrow__fd: Pointer[
        FileDescriptor, 
        # No idea what the proper syntax for getting a member from an origin 
        # set is, so I'm using this.
        origins.fd,
    ]
    var __async_read__buffer: Span[UInt8, origins.buffer]

union __CO_async_readall_ctx[origins: OriginSet]:
    var open_yield: __CO_async_readall_ctx_open_yield[origins]
    var async_read_yield: __CO_async_readall_ctx_async_read_yield[origins]
    var done: NoneType

union __CO_async_readall__ReadReturnTypeUnion[Yield: AnyType, Return: AnyType]:
    var yield: Yield
    var ret: Return

struct __CO_async_readall__ReadReturnType[Yield: AnyType, Return: AnyType]:
    var state: UInt8
    var ret: __CO_async_readall__ReadReturnTypeUnion[Yield, Return]

struct __CO_async_readall[origins: OriginSet](Coroutine):
    alias RunReturnType = __CO_async_readall__ReadReturnType[None, List[UInt8]]

    var state: __CO_async_readall_state[origins]
    var ctx: __CO_async_readall_ctx[origins]

    fn run(mut self) raises -> Self.RunReturnType:
        # It's possible this state is never entered
        if self.state == 0:
            return ReadReturnType(self.__CO_async_readall__open_yield())
        elif self.state == 1:
            return ReadReturnType(self.__CO_async_readall__async_read())
        else:
            # programmer error in async executor
    
    fn __CO_async_readall__open_yield(mut self) raises:
        # Normal deconstruction of async function around a yield point
        ...

    # This function assumes a fairly smart decomposition of async fn to
    # coroutine. This works fine with less smart versions, but I want
    # to show that this allows keeping the zero-overhead principle.
    # It may be useful to have an MLIR-level representation of a 
    # "polling loop" to help the compiler with these transformations.
    fn __CO_async_readall__async_read(mut self) raises -> Self.RunReturnType:
        var result = poll(UnsafePointer.address_of(pollfd), 1, 0)
        if result == 0:
            return Self.RunReturnType(None)
        
        if result < 0:
            # Make an error from errno and raise
            raise_errno()

        if pollfd.revents & poll.POLLERR = poll.POLLERR:
            # Will raise the error
            read(fd, buffer)
        
        self.ctx.async_read_yield.cursor += read(fd, buffer)

        if self.ctx.async_read_yield.cursor < self.ctx.async_read_yield.file_length:
            # tail recursive
            return self.__CO_async_readall__async_read()
        
        self.ctx.async_read_yield.fd.close()

        var decomposed = self.ctx.async_read_yield
        var ret = Self.RunReturnType(decomposed.buffer^)
        self.state = __CO_async_readall_state.done

        # clean up the rest of the fields in decomposed and return

        return ret
```

## Benefits of user-defined coroutines

One major benefit of this style is that the caller controls where the coroutine
is allocated, and if they really need it to do something different (such as make
a particularly large context struct heap allocated to keep the average
allocation size small), they have an escape hatch via writing their own
coroutine. Also, note that this approach brings "free" coroutine fusion, at
least from a storage perspective. This also means we can handle the
non-existence of a heap and still have async, something C++-style coroutines
can't easily do. It also makes handling placement of coroutine context easy,
since the user can simply construct the coroutine where they want it to live.
While I'm not sure exactly how coroutines are treated inside of the compiler, I
this also allows things like async functions with captures to naturally fall
out.

This model also means that you aren't forced to do anything. Since we don't
force any behavior aside from what I think is a "minimum possible API", it
upholds the zero-overhead principle. As you start to ask for more capabilities
in your coroutines, like async cancellation at any state, you can gradually have
more of the complexity shown to you in terms of what you need to write. This
always leaves the door open to "beat the compiler", meaning less pressure on the
compiler team to make a "sufficiently smart async" that meets the needs of
everyone, including people who want to do things like store context in vector
registers (a thing I have personally done when I had 8 tasks I needed to very
rapidly switch between) or have the signal to make a coroutine runnable again be
a user-space interrupt. These are concerns I don't think we want in the
language, so my solution to this problem is to define a portable, extensible
interface, and then provide a default "good enough" implementation hidden behind
that interface which can either be swapped (at compile time or runtime) for a
newer, better executor (ex: doing IO with epoll -> io_uring -> DPDK + io_uring
or io_uring + SPDK -> DPDK + SPDK) or upgraded over time. This helps Mojo avoid
the trap that Go fell into, where to make use of new IO APIs or new advances,
they need to drag the entire ecosystem with them and make it fully backwards
compatible. For instance, an MPI-style program with a statically known set of IO
operations likely wants to do IO very differently than a latency-optimized
webserver or a throughput-optimized CDN server.

## Structured Async

Now that I've laid down a foundation, let's circle back around to the `Send +
Sync + 'static` problem.

One popular solution to this is to apply structured concurrency. Instead of
acting like each individual coroutine is an independent task, you allow the user
to draw the boundaries. Consider an async executor as managing trees of
coroutines. You have a single coroutine you spawn as the root, and then it stays
around until everything lower in the tree is finished, similar to how you don't
expect the stack frame above you to go away in synchronous programming. Each
tree is, at any given time, only on a single thread. If you want to do work on
multiple threads, you need to spawn new trees which point back to the original
tree. You can either make promise to keep the parent tree around for as long as
the new ones exist, or you need to do the `Send + Sync + 'static` thing. This
helps since many of the programs written with async are primarily composed of
tasks which are waiting on IO, so they don't consume very much CPU, and you can
spawn CPU-intensive parts of your program as subtrees manually if you feel a
need to balance that way. It is also my hope that having entire trees
represented this way opens up compiler optimizations, such as fusing all of the
state machines into one. This also means that when you want to cancel a task,
you need to throw out all of the work down the tree. This has its own hazards
not discussed here, and potentially has solutions in linear types. This means
that the scheduler can keep two queues. Work assigned to the current thread
which cannot move, and work which is eligible to run on other threads.
Alternatively, a single queue can be used if removals from be done from
locations other than the head atomically and safely, with a flag set for
thread-safe tasks, or if threads keep small local queues and any overflow is
placed in a global queue that is fetched from when no local work may be done.

I personally thing this is the best approach, mostly because of experience with
thread per core Rust async executors such as Glommio, which remove most of the
complaints I commonly see about Rust async requiring tons of `Arc<Mutex<T>>`. We
can work with authors of frameworks which will do a lot of IO, such as Lightbug,
to insert the calls to load balance different HTTP sessions across cores. This
would mean mostly thread per core semantics for each user, minimizing issues.

## Scoped Tasks

In Rust many async executors had a concept of "scoped tasks". This removes that
outer `Arc` layer by constraining all of the tasks to a lifetime, allowing you
to borrow inputs, leaving you with just a `Mutex` for shared resources. However,
this has a problem, which is that you end up with a case where the user simply
stops polling the parent future but the runtime keeps running the child. This is
part of what forces the `'static` bounds, satisfied by `Arc` in most cases. See
the following Rust code as an example:

```rust
// Example pulled from https://tmandry.gitlab.io/blog/posts/2023-03-01-scoped-tasks/

async fn evil_fanout(data: &Vec<Foo>) {
    // task::scope gives you an object you can use to spawn tasks into a 
    // scope, with the lifetime of `s` bounding how long tasks can exist for.
    // Since `s` is dropped before anything
    let scope_fut = task::scope(|s| async {
        for chunk in data.chunks(50) {
            s.spawn(|| async { process(chunk).await });
        }
    });

    // Rust needs to pin because we 
    let mut scope_fut = Box::pin(scope_fut);
    
    // spawn a single task
    futures::poll!(&mut scope_fut);
    // spawn a single task
    futures::poll!(&mut scope_fut);
    
    // std::mem::forget leaks a value. In rust, all values may be safely 
    // leaked 
    std::mem::forget(scope_fut);

    // borrow ends, list is freed, and the user gets a use-after-free
    return;
}
```

So, how does Mojo deal with this?

```mojo
async fn evil_fanout(data: List[Foo]):
    @parameter
    fn outer_lambda(mut s: ScopeManager):
        # Grab chunks of the list 50 at a time
        for chunk in data.chunks(50):
            @parameter
            async fn inner_lambda():
                process(chunk).await

            s.spawn(inner_lambda)

    # ChildTask is a linear type that implements Coroutine, and it has
    # an OriginSet for captures, gathered from `outer_lambda`
    var scope_fut: ChildTask[_] = task.scope(outer_lambda)

    # spawn a single task
    _ = scope_fut.run()
    # spawn a single task
    _ = scope_fut.run()

    # This does nothing, it never had a del
    __disable_del scope_fut
    return # Linear type error
```

As you can see, entirely by accident, Mojo has solved one of the largest
ergonomics issues with Rust async, that being the proliferation of `'static`
bounds. With the introduction of linear types, simply
making the `ChildTask` struct linear stops all of the issues that made this API
impossible to do soundly in Rust. In fact, members of the Rust community have
suggested adding linear types to Rust explicitly to solve this issue. All of the
ways that have been proposed to leak a linear type in Mojo involve making it
unsafe, so I think that we can safely say that this isn't an issue in "safe
Mojo".

## Why do we need `Waker`?

The reason that we need Waker is because it's a handle into the executor. It
allows decoupling async IO from the executor, which will be important since
otherwise designing an efficient async runtime for Mojo will require access to
the runtime. Consider the above example with poll, where we need to determine
when file descriptors won't block on read. The more efficient version of this
would accumulate a list of file descriptors it is waiting on, so it would make
sense to have an async mpsc channel which sends some struct which has a waker
and a work request to a single task in another thread which accumulates all of
the `poll`-based IO and submits all of it at once, then wakes back up when
something is ready, uses a provided "return" pointer to fill out the return
values for each of the ready requests before using the waker to tell the which
tasks are ready. Without a waker, this library using `poll` would need to have
enough details about the executor to dig into it and put the coroutine into a
list somewhere. Additionally, this enables busy-polled futures (for example, if
the user is using futures to round-robin between busy-polled tasks), so long as
the coroutine wakes itself.

## Summary

In summary, Mojo already has the tools needed to fix many of the ergonomics
issues that Rust-style async suffers from, and has the benefit of hindsight on
both avoiding the downsides of implicit work stealing and on the need to have a
robust, capability-based interface for async executors and schedulers that is
standardized to enable ecosystem interoperability. This approach gives the
programmer maximum control over async, letting programmers choose where to
allocate coroutines, allowing specialized executor/coroutine implementations
which map exactly to hand-optimized state machines with no loss in efficiency,
and it provides expandability over time. As the ecosystem determines new things
it would be useful for an executor to do, we can standardize these new
capabilities, allowing for ecosystem interoperability and preventing "tokio
lock-in" where substantial portions of the ecosystem are tied to a single
executor. For instance, `CancellableCoroutine` is reasonable to implement by
hand, and we can have that trait for hand-written coroutines until we have so
kind of `try_await` which lets an async task know if it has been cancelled so it
can handle it, at which point the compiler can start using it.

There are a few downsides to this. First, we would need trait objects or
hand-written vtables before this could be used for a general purpose async. I
think that feature is already in early stages, but if this proposal is accepted
it would need to be prioritized since it would be blocking a lot of IO APIs. The
second is that the "automatic fusion from 'stack' allocated coroutines" feature
may lead to some rather large, impossible to fragment allocations. I think that
this can be mitigated since users can add manual indirection if this impacts
them. Third, the coroutine fusion may also require a non-trivial amount of
analysis in the compiler when assembling the type. I personally think that a
fairly large amount of compile-time overhead is acceptable for this feature
since it should be in the "you only pay for it if you use it" category, and
because it should have runtime performance benefits. Another downside of this
fusion is that it prevents recursive coroutines without adding
indirection. For the most part, the only solutions I've seen to this issue
either involve forcing indirection (C++) or stackful coroutines (Go), neither of
which I am particularly fond of due to the overhead, pointer chasing for C++,
and memory for Go. There is also the "trait explosion" issue, but I don't think
that's avoidable with a capability-based API. Hopefully we can get trait aliases
and unions so that we can easily stack up common combinations of traits.

Finally, I have to acknowledge that this is a bit of a bump in user-facing
complexity over Go's "make everything async" approach. However, what I like to
call "the paint bucket approach to function coloring", has some downsides. You
end up hiding a lot of things from the developer, and I think that there are a
lot of places where no distinction is made between "accidental complexity" and
"inherit complexity". Accidental complexity is when something is made more
complicated than necessary (See
[INTERCAL](https://en.wikipedia.org/wiki/INTERCAL)). It should be avoided where
possible. Inherit complexity is for problems which are actually just hard, like
"any allocation can fail", concurrency, parallelism, register allocation, and
byzantine fault tolerance. Async in systems languages hits 3 of those hard
problems. The danger of inherent complexity, alongside the simple difficulty of
working with the problem, is that if you mistake inherent complexity for
accidental complexity and then remove some or all of said complexity when
solving the problem, you render your solution unworkable for some variants of
the problem. The best example of this is garbage collection, which solves a
subset of memory problems and causes endless headaches for others. Python ML
apps crash all the time when you run out of RAM or VRAM, because Python doesn't
really have a good way to handle OOM. Python also decided to use a global mutex
to make some problems easier, and they are now trying to dig themselves out of
that hole. When you try to reduce inherent complexity, you make important
decisions on the user. I'm fully in support of "I don't care about that" APIs,
but I think that Mojo, in it's aim to be a high performance language, needs to
start with the zero overhead principle. To me, "don't pay for what you don't
use" and "could't have written it better yourself" mean that we should offer a
very low level API with tons and tons of options, and then other APIs built on
that for common use-cases with the options pre-configured. To me, this is that
low level API for coroutines. All of the complexity is shown to you, you can
override the compiler and the standard library for any decision, you can make
your attempt at beating the compiler, and you can extend it as you want. Once an
API like this is in place, we can let the ecosystem try things out for a bit,
and slowly discover standard interfaces for the high-level APIs via widespread
experimentation. If someone has a better idea for how to do something 10 years
from now, we can offer a component which implements as much of the old interface
as possible, as well as the new idea, and nothing stops users from keeping the
old one around. The implementation could be in the stdlib or the ecosystem, but
as long as those minimal traits exist in the stdlib, the stdlib can be used as a
way to provide compatibility across the ecosystem.
