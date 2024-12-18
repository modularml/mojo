# Add support for Time Stamp Counters (TSCs)

## Motivation

### What are Time Stamp Counters?

To understand why they are desirable, it must first be explained what this
somewhat unknown feature is. TSCs are low-level timers built into many CPU
architectures which enable a developer to cheaply get some sense of relative
time. This is most often used to answer  questions like "how long did this take
to run?" or for places where a timestamp that will only be compared to the same
clock needs to be acquired cheaply, such as in a hot loop. They are also
monotonic, meaning that they are unaffected by changes to system time such as
leap seconds. They are widely supported, with more recent architectures making
their usage easier (ex: x86 requires lookup tables to get the frequency, ARM
and RISC-V make you read a register).

### Why use Time Stamp Counters directly?

TSCs give you a much cheaper measurment of time than with most other methods.
In fact, most architectures implement their TSC as a single instruction, for
example, `rdtsc` on x86_64. This is valuable for benchmarking compute-bound
tasks, since it means you avoid introducing system calls into your benchmarking
loop. Although Linux has mechanisms like VDSO, the extra work of figuring out
what the real timestamp is a cost that doesn't need to be paid for relative
comparisons such as "how long did this take to run?". Additionally, careful
use of TSCs can create a single point of comparison which makes all further
timestamps in the program much cheaper to get.

### What are the costs of Time Stamp Counters?

Using the time stamp counter directly means that you give up the ability to
easily get a timestamp in return for a higher accuracy measurement. This can be
done by getting the current time and also dispatching a time stamp counter read
right after the system call returns, which gives decent accuracy, but then you
need to do some math to determine how long something took and it's not as
precise as asking the kernel for 2 timestamps in all scenarios.

Converting to real time can also be hard on some platforms. aarch64 (better
known as ARM) is a model citizen here, providing a register which gives you
the timer frequency in Hertz. x86_64 is less standardized, with Intel
providing enough information pubicly to determine the TSC frequency from CPUID,
but AMD relies on information which can't easily be retrieved in an OS-agnostic
manner (the P0 processor frequency, which is roughly the base clock of the CPU).
While this P0 frequency can be retrieved on Linux from userspace, that is not
true on all relevant operating systems.

### Library Fallbacks

In the event that the TSC is unreliable, not accessible from userspace, or
simply not present, the functions in this standard library will fall back to
the OS-provided monotonic clock (CLOCK_MONOTONIC on *nix OSes like Linux and MacOS).
OS, architecture and CPU combinations where this occurs will be listed below.

#### Library Fallbacks: Linux

Mojo defers to the Linux kernel, and if /sys/devices/system/clocksource/clocksource0/available_clocksource
does not contain a the string "tsc", then it falls back to CLOCK_MONOTONIC.
This allows Mojo to inherit the work done in the kernel to handle the LONG
list of historic TSC issues, such as consumer AMD being almost wholely
unreliable due to some BIOSes "helping" and subtracting time spent in system
management mode and older Intel CPUs counting at "whatever the processor
frequency is right now".

#### Library Fallbacks: MacOS

MacOS has historically had a hard requirement on a correctly functioning
TSC, and as such when targeting MacOS we assume it is reliable.

### Limitations, Hazards and Cautions

On most platforms, the TSC DOES NOT tick while the system is suspended. If you
are targeting platforms where the user may let their system go to sleep, it is
better to avoid these functions entirely, hook sleep events to recalibrate the
TSC, or advise the user to not allow the system to suspend while your program
is running.

On many platforms (ex: x86_64 and aarch64), each core has its own TSC which may
start at a different time. This can be an issue for multi-socket systems
where cores may come online at different times, and may also present issues for
chiplet-based CPUs which bring up their chiplets one by one. Chiplet-based CPUs
tend to have much smaller skew, and can generally be trusted, but programs
which may be run on multi-socket systems should utilize core pinning (sometimes
called "thread affinity") to ensure they stay on the same socket and you only
compare from similar clock sources. For very short measurements, consider
pinning to a single core.

Due to these hazards, `time.perf_counter_ns` should likely not be backed by the TSC
by default, but perhaps a compile-time flag could enable it for those who
are aware of the hazards and have mitigated them.

## Proposal

### A place for architecture specific features

The way to determine the TSC frequency on Intel CPUs is via the `cpuid`
instruction, which is decidedly not portable. There currently isn't a good
place for non-CPU-agnostic code to go, and as a result this proposal includes
the creation of a `sys._arch` module, containing `sys._arch.x86_64`,
`sys.arch.aarch64`, and future supported architectures. This balances the need to
use architecture specific inline assembly to expose some features that
developers may find valuable with the need to maintain a level of portability.

This _arch module would include eventually implementations of runtime CPU
feature discovery, raw time stamp counter reads, and the required
implementations for atomics, including atomics which may not exactly fit under
the C11 memory model. This module SHOULD NOT be used for anything which can be
implemented on other architectures, for instance, x86's `popcnt` can be
implemented as `vcnt.8` with ARM NEON or with a for loop, so it shouldn't be
there. In other words, every single function in this module and it's submodules
will have a constraint requiring a specific architecture.

As we want Mojo to be as portable as possible, the module shall be hidden by
default and the documentation of this module should discourage end-users using
its functions unless there are no other options.

### The creation of time._tsc

The time module is the most logical place for most of this functionality to
live, since they are time stamp counters. In this module, different
architectures will be unified into a single portable API consisting of the
following:

```mojo
@value
struct TscFrequency:
    """
    Contains the TSC frequency with architecture-defined units. 
    """
    var _tsc_frequency: UInt64

    fn get_frequency_hz(borrowed self) -> UInt64:
        """
        Get the frequency in Hertz. 
        """
        pass

@value
struct TscTimestamp:
    """
    Encapsulates the value of a TSC read. Units are architecture dependent.

    Implementation detail: Can only reliably hold up to 2^64-1 nanoseconds or ~580 years on most architectures. 
    """
    var _tsc_value: UInt64

    fn time_since_ns(borrowed self, borrowed other: Self, borrowed freq: TscFrequency) -> Int64:
        """
        Returns the time since the previous timestamp in nanoseconds using the provided frequency reference.
        """
        ...

    fn get_raw(borrowed self) -> UInt64:
        """
        Access the raw, architecture dependent value of the TSC. 
        """
        return self._tsc_value

struct Tsc:
    """
    Time Stamp Counter (TSC) container to hold the expensive initialization of 
    the TSC API, and also allows for converting tsc timestamps into values 
    usable by other APIs. Treats start_os_timestamp and start_tsc_timestamp as
    referring to the same instant in time.

    WARNING: There are hazards to the use of this API, please read the module
    documentation, specifically the "Limitations, Hazards and Cautions" section.

    freq: The frequency of the TSC, and where most of the expensive 
    initialization comes from.

    start_os_timestamp: Number of nanoseconds since an undefined point in time 
    (typically the platform epoch). Used to aid in converting TscTimestamp 
    instances to timestamps

    start_tsc_timestamp: A tsc timestamp taken right after the timestamp 
    syscall call returned. 

    tsc_is_reliable: Whether the TSC is usable as a time source. If false, a monotonic but fast to query OS clock is used.
    """

    var freq: TscFrequency
    var start_os_timestamp: UInt64
    var start_tsc_timestamp: TscTimestamp
    var tsc_is_reliable: Bool

    fn __init__(inout self):
        pass

    fn get_frequency_hz(borrowed self) -> UInt64:
        """
        Get a TscFrequency object which can be used to get the time delta between
        two TscTimestamp objects.
        """
        return self.freq.get_frequency_hz()

    fn get_tsc_timestamp(borrowed self) -> TscTimestamp:
        """
        Read from the TSC (or fallback clock). 
        """
        pass

    fn now(borrowed self) -> Time:
        """
        To be added once the standard library decides how to handle time.
        """
        pass
```

## Examples and Usecases

### Benchmarking a hot loop

This example shows a user gathering multiple datapoints per run, which is not
possible with the current benchmark library. Using a system call here would
cause `do_operation_2` to be started later than it otherwise would have,
causing `end` to deliver unreliable numbers.

```mojo
fn hot_fn():
    var tsc = Tsc()

    var middle_durations: List[Int64] = List[Int64]()
    var durations: List[Int64] = List[Int64]()

    for i in range(10):
        var start = tsc.get_tsc_timestamp()
        var phase1 = do_operation_1()
        var middle = tsc.get_tsc_timestamp()
        do_operation_2(phase1)
        var end = tsc.get_tsc_timestamp()

        middle_durations.append(tsc.get_time_delta(start, middle))
        durations.append(tsc.get_time_delta(start, end))
```

### Timestamps when logging

When outputting a lot of logs, timestamps are frequently a large fraction of
the cost to the degree that some logging libraries will actually treat whole
batches of messages as occuring at the same instant to save on system calls.
Using a TSC can help mitigate that by removing extra system calls entirely.

```mojo
struct Logger:
    var tsc: Tsc
    var prefix: String

    fn log(borrowed self, borrowed message: String):
        print(self.prefix, self.tsc.now(), " ", message)
```

### High precisions latency measurements

In networked applications it is often desirable to obtain as precise as
possible of a latency measurement. This is used for setting service level
agreements (SLAs) as well as for reasoning about the correctness of some
more exotic systems such as Google's Spanner database. The latency test
harnesses for these systems often have hundreds or thousands of messages
in flight at a time and want to make getting a timestamp as cheap as possible.

## Why the Standard Library?

Many other languages have a library for using TSCs portably, but they often are
architecture specific. Additionally, many people want this feature but are
unaware of its existence, making it hard for them to find the library. Finally,
there is a reasonable unification point of all relevant architectures in the
form of "give me the number of nanoseconds between these two points in time",
and most new architectures are implementing this feature in a similar way that
ARM does, with a frequency register and a count register on each core. As a
result, x86 will be the majority of the implementation difficulty with its past
difficulties around reliability since everyone else learned from x86. Most of
the x86 issues are for consumer processors, and servers are far more reliable
provided the aformentioned cautions are taken.
