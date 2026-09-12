# Library Provided Effect Handlers and an Implicit Parameter Store

Owen Hilyard, Created Jan 4, 2024

## Motivation

One of the greatest complaints about async is how viral it is. This is sometimes
known as the "function coloring problem", which I believe was coined by Bob
Nystrom (of "Crafting Interpreters" fame) in the blog post ["What Color is Your
Function?"](https://journal.stuffwithstuff.com/2015/02/01/what-color-is-your-function/).
That blog post makes very good, but necessary, background for this. The short
version of the blog post is that async is painful because it's usually viral.
This blog post is actually older than Javascript having promises or async/await,
so the problem was felt even more acutely then, with 'callback hell' being a
popular description of writing async Javascript at that time. While Mojo does
have a solution calling async functions inside of sync contexts, async still
blows a big hole in higher order functions and traits. In addition, time and
further reflection has brought us new colors! `raises` is a color which comes
with early function termination. `blocking` is an invisible color which causes
havoc in async code by suspending the thread. A hypothetical `pure` (D), `const`
(Rust) or `constexpr` (C++) which lets the compiler do anything which results in
the same input and output from the function. `gen` functions, which let you
create iterators using `yield`. Even `unsafe`, implicit in Mojo, explicit in
Rust, can be considered a color.

Lets say I want to make a higher order function of some sort, how many different
implementations do I need in Mojo? Well, you need 4 functions:

```mojo
fn hof(f: fn()):
    ...

fn hof(f: fn() raises) raises:
    ...

#assumes the existence of a way to declare an async function pointer
async fn hof(f: async fn()):
    ...

async fn hof(f: async fn() raises) raises:
    ...
```

This means if I want to implement, say, an iterator API, I need to implement
every higher order function 4 times. That is less than ideal. Now, lets say that
we add `constexpr` to Mojo as a way to tell the compiler it has total freedom to
make the output not even vaguely resemble the input source code so long as it
keeps the mapping of inputs to outputs (a more aggressive 'as-if'). Now we have
8 implementations of every single function in the iterator API. This becomes ~~a~~
problem very quickly, since each new color ends up doubling the size of the API
for some parts of the standard library. Traits suffer from this as well, since
we end up with `Iterator`, `RaisingIterator`, `AsyncIterator`,
`AsyncRaisingIterator` traits continuing the iterator example.~~~~

## The Classical Solution

This has a classic problem with a classic solution, an algebraic effect system!
We can define everything as effects, and each function gets a set of effects,
and we can just provide handlers! Except, for a systems language, a classical
effect system very difficult to do safely. You might write an effect handler
for, say, divide by zero, only to have it run inside of a segfault signal
handler or deep inside of a driver. If you can't pull in information from your
context, you have no idea what you can actually do about the problem, or you may
cause a worse problem by trying to handle the effect. Some people will want to
be notified on any FP error via an OpenTelemetry event, others won't want the
extra branches and icache bloat in their matmul kernel. Some types of effects
are very difficult to handle without local control flow, such as raising, which
goes up an unknown number of functions before it hits a catch block, but still
needs to clean up behind itself, running destructors. Then you get to head
scratchers, like how to "handle" a `constexpr`, or do we invert things and make
all non-`constexpr` functions have a `not_really_a_mathematical_function` effect
that gets handled by doing... Something?

## Library Provided Effect handlers

As an alternative, what if the library author could write the effect handlers,
and put them in with the other code? Lets look at an example:

```mojo
@value
struct ErrorHandling:
    var value: UInt8

    # Abort the program with a message on error
    alias ABORT = Self(0)

    # `raise` on error
    alias RAISE = Self(1)

    # Return a `Result[Ok, Err]` sum type
    alias RESULT = Self(2)

    # Assume it can never happen
    alias UNDEFINED = Self(3)


async=async fn read[async: Bool, block: Bool = True, raise: ErrorHandling = ErrorHandling.ABORT](...) raise=(raise is ErrorHandling.RAISE) -> c_size_t:
  @parameter
  if async:
    var ret = await epoll_read(...)
  else:
    @parameter
    if not block:
      compiler_warning("Blocking syscall made in context where user asked for nonblocking behavior", trace=True)
    
    var ret = libc.read(...)

  @parameter
  if not ErrorHandling.UNDEFINED:
    if ret < 0:
        @parameter
        if raise == ErrorHandling.ABORT:
            abort(ffi.strerr(-ret))
        elif raise == ErrorHandling.RAISE:
            raise LibCIoError(-ret)
        else:
            constrained["Unable to do `Result` error handling in this context"]()
    
  return ret
```

This is the start of a colorless read function. Instead of greenthreads (the
paint bucket approach) or having separate functions, instead we are a bit more
generic. Notice how `async` is a boolean flag controlled by a parameter, and
that `raise` is also a boolean flag, set by an expression? That capability to
set those flags with expressions is important for removing the need for a full
effect system. If it makes sense for the user to be able to provide a function,
they library author can add that, but this allows an author in a delicate
context to give the user options for how to handle various scenarios without
handing control flow over to the user. If the author thinks it's something the
user might want, they can take a function as a parameter. This massively reduces
the API surface that the standard library will need to expose, but there's a few
more things that will be needed to make it work well for most users.

First is the ability to set some of the "function flags" (`async`, `raises`,
etc) via an expression derived from parameters. While many "effects" have no
effect on how the compiler generates the function other than what a boolean
parameter could, those that can are necessarily part of the language. The
ability to abstract over this is almost the most important, since more users
will likely care about `async` and `raises` than whether blocking syscalls are
made or whether allocations happen. This first part alone is enough to make
large ergonomics improvements, but I think we can do better. Having to do `await
read[async=True, raise=Raise.RAISE](...)` gets very messy very quickly.
Additionally, this relies on every library remembering to propagate these
variables for every single function they call which may care, keeping up to date
with any new handlers which might be provided.

## [Nice to have] The Mojo Parameter Store

I think the best way to solve the issue of repetition is to provide a
"configuration" store, which acts as a map from parameter name to a stack of
Mojo values (and/or expressions if those are reasonable to accommodate). As the
compiler moves down the call tree, it can encounter expressions like this:

```mojo
# config / with config / withconfig / etc, we can bike shed if it's a good idea
config(async=True, blocking=False):
    ...
```

While these look like context managers, they are actually compiler directives of
the form `config PARAMETER=EXPRESSION:` which push the value of the expression
on top of a stack. When the block exits, pop from the stack and go back to the
old value. Using this, let's take another look at `read`:

```mojo
# async? is sugar for async=config[async], since it will be written a lot
async? fn read(...) raise=(config[raise] is ErrorHandling.RAISE) -> c_size_t:
  @parameter
  if config[async]:
    var ret = await epoll_read(...)
  else:
    @parameter
    if "block" in config and not config[block]:
      compiler_warning("Blocking syscall made in context where user asked for nonblocking behavior", trace=True)
    
    var ret = libc.read(...)

  @parameter
  if not ErrorHandling.UNDEFINED:
    if ret < 0:
        @parameter
        if config[raise] == ErrorHandling.ABORT:
            abort(ffi.strerr(-ret))
        elif config[raise] == ErrorHandling.RAISE:
            raise LibCIoError(-ret)
        else:
            constrained["Unable to do `Result` error handling in this context"]()
    
  return ret
```

Not that much of a difference, but now most users set their desired mechanism
for handling once per program. Users who don't want to make their code generic
over these "well known" effects don't need to be, but library authors can write
their library once using the well known effects, and, if the compiler overhead
isn't too high, provide their own effects. This also means that this is the only
"read" in the standard library. If this were to be attached to `File`, then no
matter how you wanted to IO done, you always do `File.read(..)`. This also means
that "script code" can simply abort on error, without needing to put `raises`
everywhere, but that a database storage engine can get all of the error handling
it needs to deal with full disks.~~~~

## Library config parameters

I think we can agree that being stringly typed is bad, so we want a way to have
strongly typed parameters in `config`. I propose the following syntax: `config
NAME: TYPE | config NAME: TYPE = DEFAULT`. config parameters which don't have a
default may be absent.

```mojo
config async: Bool = False
config block: Bool = True
config raise: ErrorHandling = ErrorHandling.from_string(env_get_string["DefaultErrorHandling"]) if is_defined["DefaultErrorHandling"]() else ErrorHandling.RAISE
```

This lets libraries define their own config parameters, with sensible default
values that are potentially pulled from param env.

## Compiler overhead

The main concern with a proposal like this is compiler overhead, since it
proposes the compiler haul around a map of string to stacks of mojo
values/expressions. I think that it should be possible to make this feature
reasonable in terms of compile cost. Information flow is one-way, similar to
normal parameters, as this feature could potentially be desugared to normal
parameters. I think that, under most circumstances, only very deep call chains
will set a config parameter more than a few times, so a "small stack"
optimization should be sufficient to keep memory usage down. By the time the
compiler is looking at each function, it should already have the full set of
input information, so the main areas of concern for me are the ability to be
generic over `async` and `raises`, since I'm not sure where the transformations
related to those flags occur.

## Struct layout concerns

One area the parameter store could cause issues is with struct layouts, since if
a struct is declared inside of a function it with one set of config parameters,
it may still exist afterwards and cause issues if run in a context with a
different config. As a result, I propose banning referencing `config` inside of
expressions used for types in structs until we find a compelling use-case.

## VTable Size Problems

Another potential issue is that this could lead to very large vtables, since
even with just boolean options you end up with 2^n entries per function, where n
is the number of boolean entries. There are a few options for dealing with this.
The first is to restrict the use of this functionality to effects defined by the
compiler, meaning we would have `async`, some built-in `raises` handling
options, and potentially some other usecases like `oom` and `div0`. We can keep
the size of vtables down by doing this, but it harshly limits the power of the
feature. Another option is to restrict the ability to change the function
signature to a few of the builtins (`async`, `raises`, etc) and move everything
else to runtime. This would keep the expressiveness intact, but severely harms
performance, especially for highly generic code. Another option is to have trait
objects capture the configuration as parameters. This would make trait objects
harder to use, but does preserve the power of the API for most code.
