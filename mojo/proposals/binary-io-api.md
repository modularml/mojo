# Binary IO API

Author: Owen Hilyard

Date: May 25th, 2025

Status: Proposal

## Scope

This proposal addresses binary IO to both network sockets and files.

### Goals

#### Design for Performance First

Mojo already makes compute fast, so let's also make sure that we can keep that
compute fed. When possible, we should prefer offering more performance, since it
should always be possible to offer APIs which offer more ergonomics at the cost
of performance, but an ergonomics-first design risks making decisions that
render the API unusable to users who require performance. This includes
considering and making accommodations for state of the art (SOTA) APIs which may
not be suitable for all use-cases.

#### Let the user have opinions on how IO is done (`sync`/`async` agnostic where possible)

Some users may have security concerns about some APIs (`io_uring`) or not be
willing to make tradeoffs as aggressively as some APIs would like (DPDK/SPDK).
Others might just want to read in a config file and can't do much else until
that happens, so just do sync IO. The user might also want to bring their own
APIs, for instance for accessing NVMe drives or the network from a GPU or other
accelerator. We should strive to accommodate all of these use cases, and not
pick one as the default. In the Rust ecosystem, there have been problems with
`sync` and `async`, where users complain that they are forced into async IO
while still learning the basics of the language. I think that we should strive
to make it easy for users to choose the right API for their needs, rather than
trying to guess what they want. When possible, we should try to provide tools to
library authors to make their libraries as flexible as possible.

#### Make it extensible

Nobody can predict the future. I, the author, have done my best here, but we're
seeing IO modes which were inconceivable a decade or two ago. Users must be able
to bring their own IO APIs, and device vendors need to be able to expose
device-specific APIs. The APIs should make many small promises, ensuring that if
two pieces of functionality are bundled at the lowest level, it is because they
cannot exist without each-other, no matter how odd them not existing together
may seem. Mojo's trait system provides the ability to express these constraints
in a logical way, and to help us avoid breaking changes down the road. Not all
files may be possible to close, not all sockets support both send and receive,
so we break things up. If the user has a new type of IO, they can break it up
and put it in a library, or offer it to the standard library.

#### Expose alternatives to users

Linux does, in fact, have network protocols other than UDP and TCP. There are
also categories for reliably-delivered messages, reliably delivered messages
with a fixed max length (with more perf in exchange), and a way to add
congestion control to UDP. There's even
[TIPC](https://docs.kernel.org/networking/tipc.html), which provides a
substantial amount of the cluster communication services that Kubernetes does,
including service discovery, failure detection, pub/sub, globally broadcast
messages, and much more. Normally, users would reach for libraries for these
things which may be provided as OS-level services. Showing users that these
options exist undeniably increases complexity, but it also allows users to
consider what they really need, instead of reaching for the same thing they
always do.

### Non-Goals

#### Familiar APIs

As will be discussed further on, the classical POSIX APIs have some fairly
substantial issues with them, in part due to needing copies, and in part due to
being synchronous. IO is increasingly moving in the direction of the "IO Engine"
controlling the buffers and handing them back to you when IO is done. This is a
departure from common APIs, but has fairly large performance advantages.

#### Structured Data Formats

This proposal does not attempt to address the IO of structured data, instead
providing primitives which that API may be efficiently and performantly built on
top of.

## Background

The current binary IO API is designed as a quick "we probably need a way to do
IO" solution. Mojo has continued its development, and I believe that it's time
to consider what we want the end state of this API to be. In particular, I think
that we should not blindly follow POSIX APIs, given that higher performance, and
sometimes simultaneously more ergonomic, alternatives exist. Instead, this
proposal starts from a place of inspecting what the current state of the art is,
as well how long that state of the art has stood for. This is based on the
principle that an API which has been able to support state of the art
performance for a long period of time is more likely to continue to at least be
acceptable in the future, even if new techniques are developed which would
require an API break. Additionally, this proposal inspects "near future"
technologies, such as Compute Express Link (CXL), which may have a large impact
on what APIs are useful.

### IO APIs

#### CXL

Compute Express Link (CXL) is a new standard for devices with cache, memory and
compute, which allows them to, for lack of a better description, join in on CPU
cache coherence protocols. This works by switching the PCIe link into a new mode
after some negotiation, making it lower latency and capable of carrying cache
coherence traffic. Readers familiar with Nvidia Unified Shared Memory (USM) may
find this concept familiar, but USM is primarily a software solution.
"Fine-grained shared virtual memory", as OpenCL calls it, allows you to share
data between multiple GPUs, CPUs and accelerators without having to copy it
around. It does so by allowing the GPU driver to map pages directly into the
address space of other processes, including those running on different
processors. This was mostly implemented for on-die accelerators, such as iGPUs
which share a memory subsystem with the host CPU. However, CXL allows expanding
this concept to every device connected via CXL. As far as the host CPU is
concerned, new memory just shows up, and some devices can just poke at memory
the same way other CPU cores can. While this has large implications for GPUs and
other accelerators, it also has implications for storage hardware. There are
already several prototype "CXL SSDs", which allow a host to write to an SSD via
`store` instructions. This works well even for normal SSDs, since they can
perform write coalescing, but is a dramatic improvement for devices like Intel's
Optane drives and other "persistent memory" devices, which are nearly capable of
competing with DRAM but are persistent. While Intel Optane is discontinued,
storage vendors are working on versions without the downsides of Optane, namely
cost to manufacture, and I think it would be foolish of us to not consider these
devices when designing our IO API. Especially since in real-world use for some
tasks, Intel Optane still beats out many gen 5 NVMe SSDs in random read/write
benchmarks. This makes it useful for things like RAG databases which have a high
degree of random access.

CXL is the easiest to support of all of the storage APIs discussed in this
proposal, since it can be handled via `memcpy`-like API with a "base pointer"
handle and some form of "realloc" for writes, and early experiments make the
read half the API look similar to `mmap`.

#### The Data Plane Development Kit (DPDK)

DPDK is a library that aims to provide the highest possible performance for
network data plane applications. Example uses include using software to handle
hundreds of gigabits of packets in a software L2 switch or L3 router, L7
next-gen firewalls, Open vSwitch, which underpins many kubernetes deployments,
and high frequency trading. It has, to the author's knowledge, maintained its
position as the fastest vendor-portable production grade API for ethernet
networking since its creation in 2010. Even in comparisons to RDMA and RoCE,
which are implemented primarily in hardware, DPDK can still be competitive in
latency and throughput, albeit at a higher CPU consumption, and can outperform
RDMA in bandwidth-limited scenarios due to protocol-level flexibility (ex: ack
coalescing). The price for this performance is a that the API greatly diverges
from POSIX semantics, and kernel bypass which requires DPDK have exclusive
access over the PCIe device, although this is less of a concern in recent years
due to SR-IOV.

DPDK exposes the most complex interface, since it combines a wide variety of
hardware. It lets you string together a network card and a cryptographic
accelerator into a device which outputs (d)TLS records. It also exposes a very
low level set of capabilities, such as hardware packet steering and packet
parsing. Luckily, that is outside of the scope of this proposal, since that goes
into the io subsystem setup. What DPDK does bring as the first big addition to
the IO APIs is that it owns the network buffers. It tells you what buffer the
data landed in, and how much of that buffer is valid. When you want to send
data, you get some buffers from it, fill in the data (or use the indirect API),
and then give them back to it. This is a major divergence from the POSIX API,
which assumes the caller controls the buffers. This is necessary due to the
requirements to have buffers allocated in DMA-safe memory, and to have RX
buffers registered with the NIC.

#### Storage Plane Development Kit (SPDK)

SPDK is a something of a child of DPDK, using the primitives provided by DPDK to
control storage devices instead of network cards. Like DPDK, it also represents,
to my knowledge, the state of the art in its field, also using kernel bypass.
Now, most people have some level of attachment to their filesystems, and SPDK
is, in my opinion, best described as a tool for implementing your own
filesystem. Additionally, it is heavily tied to a coroutine-based API written in
C, which I think Mojo might be able to improve upon using first-class
coroutines. At present, I think that there is a minimal level of general purpose
usefulness in including SPDK in the Mojo stdlib directly, and that instead it
would be more useful to ensure that a raw NVMe drive be usable. This is feasible
because storage devices are better standardized than network cards.

#### `io_uring` (Linux)

`io_uring` is the newest IO API on Linux, and is quickly becoming an "async
syscall" interface on Linux. It operates via two queue pairs, a submission queue
and a completion queue. The submission queue contains requests, each of which
contain a file descriptor, offset, length, buffer pointers, etc. The kernel then
executes these requests asynchronously, and puts the results onto the completion
queue. This is, effectively, shared memory IPC with the Linux kernel instead of
using syscalls. The big benefit is that you completely bypass the overhead of
syscalls, a barrier which has been hardened more and more over time, adding more
and more overhead. It has support for all of the vectored, direct, async,
non-blocking, etc. features of POSIX IO, since it was a storage API first, but it
has also grown networking capabilities over time. This makes it an attractive
option for a general async IO API on Linux. `io_uring` is close to state of the
art solutions, such as DPDK and SPDK, but without the extreme methods those
libraries use to extract all of the possible performance out of the hardware,
taking inspiration from them while staying firmly inside of the kernel.

One large difference that `io_uring` has with many other APIs is the idea of
"multi-shot" operations. This allows you to do things like "accept every TCP
connection that comes in on this socket" or "read all of the data from from this
file/socket" by simply telling the kernel what you want to happen. These
operations often closely match the desires of users, and are something we may
want to consider.

`io_uring` is my recommendation for the general purpose file IO API on linux,
given that it isn't that much slower than SPDK and that rewriting the filesystem
is a much larger task than writing a new network protocol. It is also useful as
a fallback for general purpose networking in the absence of DPDK.

The main downside is that io_uring doesn't have full auditing and security
enforcement mechanisms in place at the moment, leading it to be disabled in some
environments. It is reasonable to assume that these issues will be addressed
over time, and many applications simply are not security sensitive enough for
this to matter, and nobody will bother to configure syscall auditing or some of
the other security measures impacted by this. There are also some issues with
older kernels not having full support, which may force us to implement fallback
paths for older kernels or fall the entire IO system back to `epoll`.

#### `epoll` (Linux)

Epoll is the current standard API for asynchronous IO on Linux. It acts as a
sensible fallback for `io_uring`. Unlike most of the other IO APIs this is a
readiness-based IO API. This means that the kernel will not do the IO for you,
and you still have to issue syscalls, it will only do its best to tell you when
the IO is non-blocking. Epoll also has poor support for async disk IO, meaning
that a fallback API using epoll would likely need a threadpool to hand disk IO
off to, which is what both Rust's Tokio, Golang, and libuv currently do.

#### `kqueue` (macOS/BSD)

`kqueue` is the `io_uring` equivalent on BSD systems. It has some disadvantages,
such as still requiring that a system call be made for every operation, but it
notifies the user when a file descriptor is available for reading/writing, and
provides a mechanism for waiting until events are available. Overall, kqueue
doesn't introduce many new concepts and most of those concepts are outside of
the scope of this proposal.

#### IOCP (Windows)

IOCP is the venerable async IO API on Windows. It has an interface which
somewhat resembles io_uring, that it wants you to pass in a pointer to a
particular type which is embedded in a struct. This means that in order to use
IOCP, we would need the ability to force the `OVERLAPPED` struct member to be
the first member of a struct. Although much of the documentation wants you to
use a background threadpool for managing the completions, the author things it
should be possible to map it to Mojo's async model instead, maintaining thread
per core.

#### IoRing (Windows)

Right now, IoRing is very early days. As far as I am aware, it only has basic
file IO implemented, and we will have to rely on IOCP for everything else. Part
of the stated goal of the project is to offer an `io_uring` equivalent, which
means that we should eventually expect it to become similar to `io_uring`,
especially once it supports networking. As a result, we can make some sacrifices
in IOCP performance in anticipation of IoRing being able to steadily move
operations over to IoRing.

#### POSIX IO APIs (macOS, BSD and Linux)

These are the basic APIs we all know. `read(2)`, `write(2)`, `open(2)`, etc for
disk IO, and `socket(2)`, `send(2)` and `recv(2)` for networking. They are all
synchronous APIs, and for a long time they have shaped the *nix community's
vision of what an async API should look like. As we have discovered, these APIs
are not ideal for modern hardware, and we should try to avoid them where
possible. All of them force a doubling of the memory bandwidth consumption
compared to more modern APIs by making "true zero copy" impossible. Given how
precious memory bandwidth is for API applications, and the designs of CPUs which
increasingly restrict the ability of a single core to pull the entire memory
bandwidth of the system. For example, 3rd Gen AMD EPYC (Turin) can only pull
about 30 GB/s of data from memory on a single core, meaning that in the common
case of the core requesting the `read` also doing the kernel work, that the core
is incapable of 200G networking. This would also mean that a single state of the
art NVMe drive (~12 GB/s of sequential read bandwidth) could nearly consume the
entire memory bandwidth of those cores. Although these APIs are "nice to have"
for quick scripts, I think we should discourage their use in serious
applications on these grounds. Even in less dramatic scenarios, doubling the
required memory bandwidth of anything isn't great.

However, I think that we still need to accommodate these APIs since some
projects won't want to have a full "Async IO Engine" (term introduced later) in
order to read in a config file, especially if they mostly read in a bit of data,
then do compute and exit. However, if a user is doing IO in this way, I think
that we can afford to let them pay other penalties shape the POSIX APIs into
something that works with the higher performance APIs.

#### `mmap`

While not often considered an IO API, `mmap` can be used for file IO, and is
often used in a similar way to how CXL may eventually be used, but with more
software glue. It allows the user to request that a file be mapped into their
virtual address space, and then the OS will use fault handling to make pages
"show up" in memory as needed, flushing them back to disk either as needed to
preserve memory or when the mapping is closed. We can design APIs for `mmap` and
they should apply to CXL reasonably well.

## IO Engines

So, how do we bridge the disparate forms of IO? We have four general groups:

- Memory operations and something else figures it out.
- Synchronous IO, where the program blocks until the operation completes.
- Readiness-based IO, where the OS tells the program when the operation can be
  executed.
- Completion-based IO, where the OS tells the program when the operation is
  complete.

With the exception of CXL, this is also a list from slowest to fastest.
`mmap`-based IO, while nice to use, is often slow and lacks control over when IO
actually happens, aside from the big flush at the end. Actually telling the
kernel what you want to do using the POSIX APIs is a bit faster, but you still
need to wait for the kernel. `epoll` and `kqueue` are a bit faster, since it can
do other things (like more IO) while it waits for the kernel to have the
kernel-side buffer ready for the transfer to userspace. Then we have `io_uring`,
IOCP, IoRing, SPDK and DPDK, which all use completion-based IO. However, DPDK
and SPDK deserve a call out since they are often communicating directly with
hardware, and thus have a lower abstraction level than the others.

This proposal suggest that Mojo fold readiness-based IO under the
completion-based API. We can still use coroutines to make things easier for
users, but in most cases the thing done directly after a resource is ready is
that the user does the IO they asked about. Additionally, we can put a similar
wrapper on top of the synchronous APIs as well as exposing them separately. In a
future version of Mojo with effect generics (or perhaps just async generics), we
can unify the API more. Memory-based APIs will likely still require some other
form of API to get the file mapped into the process, but we can still make the
same API work by sticking `memcpy` under the hood.

### Reader/Writer API

Parametric traits are assumed to exist as part of this proposal.

#### IOBuffer

```mojo
trait IoBuffer[mut: Bool, //, origin: Optional[Origin[mut]]]:
    """A trait representing a buffer for IO operations.
    
    It intentionally provides no guarantees other than marking the buffer as 
    something that should be used as an IoBuffer.

    Implementors of this type need not be a flat buffer, but might be a linked
    list, a slab list, a tree, or some exotic data structure.
    """
    pass
```

In order to handle a variety of IO APIs, the first thing we need to abandon is
the idea that buffers are always contiguous chunks of bytes with arbitrary
alignment. This base trait acts as a marker trait, and is intended to be used
primarily to help make error messages better. For example, "Given `$TYPE`, which
implements `IoBuffer`, but is not a `ContiguousIoBuffer` as required by this IO
Engine". This can be extended with various traits to specify different
properties, such as a guaranteed minimum alignment, being page aligned (whatever
that means on the platform), being a single allocation, or having particular
bits of metadata like a checksum.

#### Reader

```mojo
trait OwnedReader[
    mut: Bool, //, origin: Optional[Origin[mut]], BufferType: IoBuffer[origin]
]:
    """The Base Reader Trait.

    Implementers should implement as much of this trait family as possible.

    Users should depend only on the API surface they require.


    Parameters:
        mut: Whether the reader produces mutable or immutable buffers if the buffers are borrowed.
        origin: If the buffers are borrowed, and if they are the origin of the buffers.
        BufferType: The type of buffer returned by the reader.
    """

    fn read(owned self) raises -> self.BufferType:
        ...


trait RefReader[
    mut_buffers: Bool, //, origin: Optional[Origin[mut]], BufferType: IoBuffer[origin], self_mutability: Bool,
]:
    """The Base Reader Trait.

    Implementers should implement as much of this trait family as possible.

    Users should depend only on the API surface they require.

    Parameters:
        mut: Whether the reader produces mutable or immutable buffers if the buffers are borrowed.
        origin: If the buffers are borrowed, and if they are the origin of the buffers.
        BufferType: The type of buffer returned by the reader.
    """

    fn read[origin: Origin[self.self_mutability]](ref [origin] self) raises -> self.BufferType:
        ...
```

Right away, some of you are probably thinking about this API will require a
substantial amount of boilerplate code to implement. Although not in scope for
this proposal, I think that a way to be generic over the ownership and
mutability of the buffers would be beneficial, especially in a way that allows a
trait to specify that implementing any one of the variants is sufficient. For
the rest of this proposal, I will make a comment whenever an argument needs to
be generic over argument convention for the sake of brevity.

You will also notice that the buffer type is generic. This allows encoding
various API restrictions, such as `O_DIRECT`'s infamous alignment requirement,
for `io_uring` to make sure you're using registered buffers, or for vectored IO
to return a `List[IoBuffer]`, an `InlineArray[IoBuffer, N]` or something else
entirely. It is expected that most things which do IO will be generic over the
buffer type, and use trait constraints to require capabilities of the IO buffers,
such as slicing or the ability to copy data into a `List[UInt8]`.

```mojo
trait ModifyingOffsetReader[
    mut: Bool, //, origin: Optional[Origin[mut]], BufferType: IoBuffer[origin], *,
    off_t: DType = os.off_t.dtype
](Reader[origin, BufferType]):
    """A reader which supports reading relative to the start of the 
    stream/backing buffer, but may change the read position or "cursor" into 
    the stream/backing buffer."""

    # self is generic over owned/mut/read
    fn cursor_modifying_read_at_offset(
        mut self, owned offset: Scalar[off_t]
    ) raises -> self.BufferType:
        """Reads at an arbitrary offset within the backing buffer.

        **MAY** change the location of the "cursor" into the backing buffer to 
        an implementation-defined location."""
        ...

trait OffsetReader[
    mut: Bool, //, origin: Optional[Origin[mut]], BufferType: IoBuffer[origin], *, 
    off_t: DType = os.off_t.dtype
](ModifyingOffsetReader[origin, BufferType]):
    """A reader which supports reading relative to the start of the 
    stream/backing buffer, but **MUST NOT** change the read position or 
    "cursor" into the stream/backing buffer."""

    # self is generic over owned/mut/read
    fn read_at_offset(
        mut self, owned offset: Scalar[off_t]
    ) raises -> self.BufferType:
        """Reads at an arbitrary offset within the backing buffer.

        **MUST NOT** change the location of the "cursor" into the backing 
        buffer to an implementation-defined location."""
        ...
```

The first trait provides the ability to a "seek + read" combination, optionally
returning to the original location. This is useful on platforms without a
`pread` equivalent, whereas the second trait matches the API of `pread`. There
is also an accommodation made to deal with platforms which may use unsigned
offset.

```mojo
trait PreferredBlockSizeReader:
    """A reader which provides information about the preferred read size for
    optimal performance."""

    # self is generic over owned/mut/read
    fn preferred_block_size(self) -> UInt:
        """Returns the preferred block size for optimal performance."""
        ...

```

`PreferredBlockSizeReader` provides a trait for readers which report the
preferred block size for optimal performance. This is helpful both for
`O_DIRECT`, as well as for vectorized IO. Traits like this enable 2-way
communication between the user and the IO engine, allowing the user to optimize
their IO patterns based on the capabilities and preferences of the underlying IO
engine.

```mojo
trait NotifyReadAmountReader:
    """A reader which can make productive use of being told how much data you 
    intend to read, for example by batching allocations."""

    # self is generic over owned/mut/read
    fn notify_expected_read_amount(mut self, owned bytes: UInt) raises:
        """Notifies the reader of how much data will be read next (in bytes),
        in addition to all previous data.

        Note that this is not expected to be helpful except when buffering
        many individual read calls."""
        ...
```

`NotifyReadAmountReader` is a reader which has a prefetch API. This can enable a
background worker to perform IO on behalf of the user before the user wants to
actually put all of that data in buffers.

#### Writer

```mojo
trait Writer[
    mut: Bool, //, origin: Optional[Origin[mut]], BufferType: IoBuffer[origin]
]:
    """The Base Writer Trait.

    Implementers should implement as much of this trait family as possible.

    Users should depend only on the API surface they require.
    """

    # self is generic over owned/mut/read, as is data
    fn write_buffer(mut self, owned data: self.BufferType) raises -> UInt:
        """Maps to POSIX write(2) but does not allow writing partial buffers.

        Returns:
            The number of bytes written."""
        ...

    # self is generic over owned/mut/read, as is data
    fn write_all(mut self, owned data: self.BufferType) raises:
        """Wraps `write_buffer` to ensure all data is written, raising 
        otherwise."""
        ...
```

It should be noted that the the ability to write partial buffers is not
guaranteed, or is at best undefined behavior in some situations, such as
`O_DIRECT` or when interfacing closely with hardware which may require you to
adjust metadata attacked to the IO buffer instead. As a result, the base
`Writer` trait does not allow the traditional `write` API in any way.

```mojo
trait BufferSpanWriter[
    mut: Bool, //, origin: Optional[Origin[mut]], BufferType: IoBuffer[origin]
](Writer[origin, BufferType]):
    """A writer which supports writing partial buffers."""

    # self is generic over owned/mut/read, as is data
    fn write(mut self, owned data: self.BufferType, owned count: UInt) raises -> UInt:
        """Maps to POSIX write(2).

        Returns:
            The number of bytes written."""
        ...
```

And then there's a trait for adding the traditional posix

```mojo
trait FlushableWriter:
    """A writer with the capability to pass all buffered data down to it's
    inner write function. This is primarily intended to be used with buffered
    IO.

    Unbuffered writers may implement this trait trivially.
    """

    # self is generic over owned/mut/read
    fn flush(mut self) raises:
        """Flushes any buffered data via the normal write function."""
        ...


trait SyncableWriter:
    """A writer which can synchronize its state with a backing store."""

    # self is generic over owned/mut/read
    fn fsync(mut self) raises:
        """Synchronizes the current state of the writer with the underlying
        storage medium.

        Post-condition:
            It is expected that when this function is complete, that all data
            written is synchronized to a persistent data store. This may be a
            filesystem, database or other durable storage mechanism.

            This is a critical post-condition to uphold as users **MAY** rely
            on this post-condition for data integrity.
        """
        ...
```

These two traits encapsulate commonly desired functionality for writers. Note
the additional warnings around `fsync` since it is often used to uphold data
integrity and safety requirements in databases.

Writer also has a mirror of the `OffsetReader` traits above, called
`OffsetWriter`, as well as the block size trait, `PreferredBlockSizeWriter`.
These are similar to their reader equivalents and omitted for brevity. There is
also a `NotifyWriteAmountWriter` trait, which can be used to notify an IO engine
of an amount to be written soon, allowing it time to make space.

#### Additional API Notes

The API is designed to be expanded in later proposals with more functionality.

### Where are these buffers coming from?

Since this proposes that Mojo use special buffer types for IO, there also needs
to be a way to get those buffers. In this case, it's via a `BufferProvider`
trait family. This area may need some additional work once collections of types
with origins are better handled.

```mojo
trait BufferProvider[
    mut: Bool, //, origin: Optional[Origin[mut]], BufferType: IoBuffer[origin]
]:
    """Provides access to a mutable buffer into which data can be written."""

    # self is generic over owned/mut/read
    fn get_buffer(mut self) raises -> self.BufferType:
        """Gets a buffer for use in IO"""
        ...

trait MultiBufferProvider[
    mut: Bool, //, origin: Optional[Origin[mut]], BufferType: IoBuffer[origin]
]:
    """Provides access to multiple buffers into which data can be written."""

    # self is generic over owned/mut/read
    fn get_buffers[
        span_origin: MutableOrigin
    ](
        mut self,
        mut span: Span[UnsafeMaybeUninitialized[self.BufferType], span_origin],
    ) raises -> Int:
        """Gets a set of buffers for use in IO.

        Returns the number of buffers provided.
        """
        ...

    # self is generic over owned/mut/read
    fn get_all_buffers[
        span_origin: MutableOrigin
    ](
        mut self,
        mut span: Span[UnsafeMaybeUninitialized[self.BufferType], span_origin],
    ) raises:
        """Fills the span with buffers for use in IO."""
        ...
```

In the general case, it is expected that the buffer provider be the IO engine,
using registered buffers, but for more flexible IO engines it may instead be an
arena allocator of buffers, recycling buffers as needed assuming that reads and
writes are balanced. If they are not balanced then the following trait provides
a helpful API:

```mojo
trait BufferSink[
    mut: Bool, //, origin: Optional[Origin[mut]], BufferType: IoBuffer[origin]
]:
    """A sink for buffers which can be used to recycle buffers."""

    # self is generic over owned/mut/read
    fn recycle_buffer(mut self, owned buffer: BufferType) raises:
        """Recycles a buffer for reuse."""
        ...

trait MultiBufferSink[
    mut: Bool, //, origin: Optional[Origin[mut]], BufferType: IoBuffer[origin]
]:
    """A sink for buffers which can be used to recycle buffers."""

    # self is generic over owned/mut/read
    fn recycle_buffers[
        span_origin: MutableOrigin
    ](
        mut self,
        mut span: Span[UnsafeMaybeUninitialized[self.BufferType], span_origin],
    )
        """Recycles a buffer for reuse."""
        ...
```

Of course, these could just be wrappers over `malloc` and `free`, but they nudge
users into the "pit of success" by encouraging them to recycle buffers where
possible.

### IO Engine Traits

```mojo
trait IoEngine:
    """Acts as a base trait for IO engines. Intentionally, little is promised 
    in this high level trait aside from acting as a nice way to find 
    everything which you can use to do IO."""
    pass
```

The base trait acts as a marker, since, no matter what, there should be an easy
way to define "everything that can be used to do IO". This aids in user
discovery of IO Engines with wildly disparate capabilities and acts as a
documentation hub to explain why Mojo does IO this way.

```mojo
#alternative naming style with more clear name-spacing, please discuss. 
trait IoEngine_File_Open_ReadOnly[PathType: AnyType, FileType: AnyType]:
    """An IO engine which can open files in a read-only mode."""
    # self is generic over owned/mut/read
    fn open_readonly(self, read path: self.PathType) raises -> self.FileType:
        """Opens a file for read-only access."""
        ...

trait IoEngine_File_Open_WriteOnly[PathType: AnyType, FileType: AnyType]:
    """An IO engine which can open files in a write-only mode."""
    # self is generic over owned/mut/read
    fn open_writeonly(self, read path: self.PathType) raises -> self.FileType:
        """Opens a file for write-only access."""
        ...

trait IoEngine_File_Open_ReadWrite[PathType: AnyType, FileType: AnyType]:
    """An IO engine which can open files in a read-write mode."""
    # self is generic over owned/mut/read
    fn open_readwrite(self, read path: self.PathType) raises -> self.FileType:
        """Opens a file for read-write access."""
        ...

trait IoEngine_File_Open_Append[PathType: AnyType, FileType: AnyType]:
    """An IO engine which can open files in append mode."""
    # self is generic over owned/mut/read
    fn open_append(self, read path: self.PathType) raises -> self.FileType:
        """Opens a file for appending."""
        ...

trait IoEngine_File_CreateFile[PathType: AnyType, FileType: AnyType]:
    """An IO engine which can create new files."""
    # self is generic over owned/mut/read
    fn create_file(self, read path: self.PathType) raises -> self.FileType:
        """Creates a new file."""
        ...

trait IoEngine_File_Close[PathType: AnyType, FileType: AnyType]:
    """An IO engine which can close files."""
    # self is generic over owned/mut/read
    fn close(self, owned file: self.FileType) raises:
        """Closes a file."""
        ...

trait IoEngine_FileHandler[PathType: AnyType, FileType: AnyType](
    IoEngine_FileOpenReadonly[PathType, FileType],
    IoEngine_FileOpenWriteOnly[PathType, FileType],
    IoEngine_FileOpenReadWrite[PathType, FileType],
    IoEngine_FileOpenAppend[PathType, FileType],
    IoEngine_FileCreateNew[PathType, FileType],
    IoEngine_FileCloser[PathType, FileType]
):
    """An IO engine which can handle common file operations."""
    pass
```

This shows basic file operations, as well as one of the maintainability
downsides of this style of API. It does require a lot of initial boilerplate to
implement compared to normal "bundled" traits. However, LLMs can make stubbing
out traits like this much easier, especially given relevant context, and perform
fairly well given the repetitive nature of stubbing out the basics of the API.

```mojo
trait IoEngine_UDP_Socket[SocketType: AnyType]:
    """An IO engine which can send messages."""
    # self and socket are generic over owned/mut/read
    fn udp_socket_create(self, owned port: UInt16) raises -> self.SocketType:
        """Opens a UDP socket"""
        ...

trait IoEngine_UDP_Bind[SocketType: AnyType, UnderlyingAddressType: AnyType]:
    """An IO engine which can bind sockets."""
    # self and socket are generic over owned/mut/read
    fn udp_socket_bind(self, owned socket: self.InputSocketType, owned address: self.UnderlyingAddressType) raises:
        """Binds a socket to an address."""
        ...

trait IoEngine_UDP_Send[mut: Bool, //, origin: Optional[Origin[mut]], SocketType: AnyType, BufferType: IoBuffer[origin]]:
    """An IO engine which can send messages."""
    # self and socket are generic over owned/mut/read
    fn udp_socket_send(
        mut self,
        owned socket: self.SocketType,
        owned message: self.BufferType,
    ) raises -> UInt:
        """Sends a message to a remote host."""
        ...

trait IoEngine_UDP_Close[SocketType: AnyType]:
    """An IO engine which can close sockets."""
    # self is generic over owned/mut/read
    fn udp_socket_close(self, owned socket: self.SocketType) raises:
        """Closes a socket."""
        ...

trait IoEngine_UDP_Basic[SocketType: AnyType, UnderlyingAddressType: AnyType](
    IoEngine_UDP_Socket[SocketType],
    IoEngine_UDP_Bind[SocketType, UnderlyingAddressType],
    IoEngine_UDP_Close[SocketType]
):
    """An IO engine which can handle common UDP operations."""
    pass
```

### Example IO Engines

#### POSIX

```mojo
from pathlib import Path

@piecewise_init
struct UdpSocket:
    var fd: FileDescriptor

    fn bind(read self, address: IPv4Address, port: UInt16) raises:
        var sock_addr = SockAddrIn()
        sock_addr.sin_family = os.net.AF_INET
        sock_addr.sin_port = port
        sock_addr.sin_addr.s_addr = address.to_be_bytes()

        var result = external_call[
            "bind",
            c_int,
            c_int, UnsafePointer[SockAddrIn], c_socklen_t,
        ](self.fd, UnsafePointer(to=sock_addr), sizeof[socklen_t]())

        if result != 0:
            raise OSError.from_errno()

    fn bind(read self, address: IPv6Address, port: UInt16) raises:
        var sock_addr = SockAddrIn6()
        sock_addr.sin6_family = os.net.AF_INET6
        sock_addr.sin6_port = port
        sock_addr.sin6_flowinfo = 0
        sock_addr.sin6_scope_id = 0
        sock_addr.sin6_addr.s6_addr = address.to_be_bytes()

        var result = external_call[
            "bind",
            c_int,
            c_int, UnsafePointer[SockAddrIn], c_socklen_t,
        ](self.fd, UnsafePointer(to=sock_addr), sizeof[socklen_t]())
        
        if result != 0:
            raise OSError.from_errno()
    
    fn send(read self, read message: Span[UInt8, _]) raises -> UInt:
        var result = external_call[
            "send",
            c_ssize_t,
            c_int, UnsafePointer[UInt8], c_size_t,
        ](self.fd, UnsafePointer(to=message.unsafe_ptr()), len(message))

        if result < 0:
            raise OSError.from_errno()
        else:
            return result

    fn __del__(owned self):
        external_call["close", c_int, c_int](self.fd)

@piecewise_init
struct IPv4Address(IPAddress):
    var ip: UInt32
    ...

@piecewise_init
struct IPv6Address(IPAddress):
    var ip: UInt128
    ...

@piecewise_init
struct PosixIOEngine(
    IoEngine_FileHandler[Path, FileHandle],

    IoEngine_UDP_Basic[UdpSocket, IPv4Address],
    IoEngine_UDP_Basic[UdpSocket, IPv6Address],
    Movable,
    Copyable,
):
    fn open_readonly(read self, read path: Path) raises -> FileHandle:
        return os.open(path, os.O_RDONLY)

    fn open_writeonly(read self, read path: Path) raises -> FileHandle:
        return os.open(path, os.O_WRONLY)

    fn open_readwrite(read self, read path: Path) raises -> FileHandle:
        return os.open(path, os.O_RDWR)

    fn open_append(read self, read path: Path) raises -> FileHandle:
        return os.open(path, os.O_APPEND)
    
    fn create_file(read self, read path: Path) raises -> FileHandle:
        return os.open(path, os.O_CREAT | os.O_EXCL)

    fn close(self, owned file: FileHandle) raises:
        file.close()

    fn udp_socket_create(self, owned port: UInt16) raises -> UdpSocket:
        var fd = external_call[
            "socket",
            FileDescriptor,
            c_int, c_int, c_int,
        ](os.net.AF_INET, os.net.SOCK_DGRAM, 0)
        if fd < 0:
            raise OSError.from_errno()
        return UdpSocket(fd)

    fn udp_socket_bind[Address: IPAddress](self, read socket: UdpSocket, read addr: Address, read port: UInt16) raises:
        socket.bind(addr, port)

    fn udp_socket_close(self, owned socket: UdpSocket) raises:
        _ = socket^

# made up syntax, see the Rust equivalent, origins prevent implementing this trait without this capability right now.
impl[origin: MutableOrigin] IoEngine_UDP_Send[Some(origin), UdpSocket, Span[UInt8, origin]] for PosixIOEngine:
    fn udp_socket_send(read self, read socket: UdpSocket, read message: Span[UInt8, origin]) raises -> UInt:
        return socket.send(message)

impl BufferProvider[None, List[UInt8]] for PosixIOEngine:
    fn get_buffer(mut self) raises -> self.BufferType:
        """Gets a buffer for use in IO"""
        return List([UInt8(0) for i in range(4096)])
```

#### DPDK

I'm going to handwave a bunch of easy to create DPDK-related boilerplate here.

```mojo
alias DPDKUdpSocket = UInt16

@piecewise_init
struct DpdkIOEngine(
    IoEngine_UDP_Basic[DPDKUdpSocket, IPv4Address],
    IoEngine_UDP_Send[None, DPDKUdpSocket, rte_mbuf],
):
    var port_id: UInt16
    var rx_queue_id: UInt16
    var tx_queue_id: UInt16
    var ethdev_info: rte_eth_dev_info    
    var pool: UnsafePointer[rte_mempool]

    var open_ports: Set[DPDKUdpSocket]
    var addresses: Dict[DPDKUdpSocket, IPv4Address]
    var bind_map: Dict[DPDKUdpSocket, Tuple[IPv4Address, UInt16]]
    
    var arp_table: ArpTable

    fn udp_socket_create(mut self, owned port: UInt16) raises:
        self.open_ports.add(port)
    
    fn udp_socket_bind(mut self, owned socket: DPDKUdpSocket, read address: IPAddress, read port: UInt16) raises:
        self.bind_map[socket] = (address, port)

    fn udp_socket_send(mut self, read socket: DPDKUdpSocket, owned message: UnsafePointer[rte_mbuf], out result: UInt) raises:
        result = message[].pkt_len.cast()
        var cursor = message[].buf_addr
        var dest = self.bind_map[socket][0]
        var dest_addr = dest[0]
        var dest_port = dest[1]
        cursor.bitcast[EthernetHeader]().init_pointee_move(
            EthernetHeader(
                destination_mac=arp_table.lookup(dest_addr),
                source_mac=self.ethdev_info.mac_addrs[0],
                ethertype=(0x0800).to_big_endian(),
            )
        )
        cursor = (cursor.bitcast[EthernetHeader]() + 1).bitcast[IPv4Header]()
        # no moveinit for IPv4 due to needing to set bitfields for version and ihl
        cursor[].version = 4
        cursor[].ihl = 5
        cursor[].tos = 0
        cursor[].total_length = (message[].pkt_len.cast() - 34).to_big_endian()
        cursor[].id = 0
        cursor[].flags_fragment_offset = 0
        cursor[].ttl = 64.to_big_endian()
        cursor[].protocol = IPPROTO_UDP.to_big_endian()
        cursor[].checksum = 0
        cursor[].source_address = self.ethdev_info.mac_addrs[0]
        cursor[].destination_address = dest_addr

        var udp_cursor = (cursor + 1).bitcast[UDPHeader]()
        udp_cursor.init_pointee_move(
            UDPHeader(
                source_port=socket,
                destination_port=dest_port.to_big_endian(),
                length=message[].pkt_len.cast() - 42,
                checksum=0,
            )
        )

        var buffer = InlineArray[UnsafePointer[rte_mbuf], 1](message)
        
        # offload flags set at mempool init
        rte_eth_tx_prepare(self.port_id, self.tx_queue_id, buffer.unsafe_ptr(), 1)

        while rte_eth_tx_burst(self.port_id, self.tx_queue_id, buffer.unsafe_ptr(), 1) != 1:
           # if anything goes wrong enough for this to actually fail, a watchdog thread will kill the program. The most likely cause of this failing is either a NIC hardware failure or someone pulling the network cable. 
           pass


    fn udp_socket_close(mut self, owned socket: DPDKUdpSocket) raises:
        self.open_ports.remove(socket)
        self.bind_map.pop(socket) 

impl BufferProvider[None, UnsafePointer[rte_mbuf]] for DpdkIOEngine:
    fn get_buffer(mut self) raises -> self.BufferType:
        """Gets a buffer for use in IO"""
        return rte_pktmbuf_alloc(self.pool)
        
```

So, if we hand-wave the DPDK-isms and shortcuts in the implementation which could
easily be remedied, that's the lowest and highest performance networking APIs in
common use under a single API. There's one other benefit to this proposal, which
is that the blast radius of the author being wrong is limited to the function
which is wrong, so we can stamp out a v2 and slot that before the old, more
restrictive option, ideally throwing away whatever extra information was
necessary for the new version to make the new API work for old implementations.

### Example IO Engine Consumer

```mojo
struct TrafficGenerator[Engine: IoEngine & BufferProvider & IoEngine_UDP_Basic & IoEngine_UDP_Send]:
    var engine: Engine
    var socket: Engine.UdpSocket

    fn __init__(out self, owned engine: Engine):
        self.engine = engine
        self.socket = self.engine.udp_socket_create(9000)
        self.engine.udp_socket_bind(self.socket, Ipv4Address("192.168.1.1"), 443)

    fn generate_traffic(out self, n: UInt):
        for i in range(n) True:
            var buffer = self.engine.get_buffer()
            self.engine.udp_socket_send(self.socket, buffer^)
    
    fn __del__(owned self):
        self.engine.udp_socket_close(self.socket^)
```

As readers can see, this massive amount of complexity in the IO Engine
implementation results in much, much nicer code on the other side. The author
welcomes comparisons to other approaches which are capable of bridging the BSD
Sockets API and DPDK. The addition of async generics gets rid of the the `async`
variants of all of these functions, making it easy to also use normal OS async
IO APIs, and is the primary motivation for earlier proposals around effect
systems and async generics.

This API is, very clearly, incomplete, making this more of a philosophy
proposal, since this is already the longest Mojo proposal by a substantial
amount. However, it is intended to provide a solid foundation for future work.

### Why abstract this much?

Part of the reason why it's important for Mojo to have an IO API which lets you
swap out the backend is that Mojo is a language which aims to be able to run not
just on a collection of operating systems which all share vaguely similar IO
APIs, but also on AI accelerators. Nvidia has RDMA on GPUs, so we probably want
to have applications and libraries designed for RDMA be able to run there. The
DPDK API I showed as an example can, with very few source changes, be adapted to
run on GPUs as well (see [Nvidia L2 FWD](https://github.com/NVIDIA/l2fwd-nv)).
Mojo needs the ability to bridge Direct Storage with normal file APIs, or deal
with a compute in storage device which has far, far more access to the storage
than a CPU normally would. Abstracting to this level allows Mojo to deal with
whatever wacky way an accelerator talks to the outside world, and tries to still
solve the problem of portable code between those environments.
