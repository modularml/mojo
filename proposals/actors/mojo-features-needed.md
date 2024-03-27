## Mojo Features Needed

This section defines the things that a Mojo Actor System will need
from Mojo or its standard library.

Most of the things in this section have multiple ways they can
be implemented in a language. Treat the various topics as merely ways to
describe capabilities needed in the language, without being a specification
on the language. We need various mechanisms to implement actors effectively.
We don't care how the language supports them; we care that it *does* support
them.

While work can begin ahead of the availability of these language
features, there will likely be additional work without them. 

### Actor Representation
Mojo must support an efficient object-oriented programming model. This should
be intrinsic given its intention to be a superset of Python. Actors then can
be simply objects, classes or structures that can bind functionality (behavior)
to them.  Inheritance is not necessary but can be used to define an `Actor` base 
class in the Actor library and require actors to inherit from that class. This should 
be sufficient notification to the compiler without altering the language.

### Trait Implementations Not Needed At Usage Level
Actors should appear to be stateless to users of that actor. There is never a
a need for the *Actor* to implement any traits since you
cannot call methods on it anyway. In fact, sending messages to
an Actor only involves the message, the Actors address, and the
communication pathway to get it there. The actor itself is not
involved!

The definition of the actor is free to declare its state data, including
the use of traits and their implementations, but those are all hidden
and private details for the Actor, not its users.

### Concurrency
To support asynchronous, non-blocking message dispatch and execution, Mojo
will need to support a highly efficient and effective concurrency and
parallelism mechanisms. These mechanisms should serve both actor and non-actor
use cases. To do this well, actors need the capabilities described in the sections
below.

#### Low Cost Threading Model (Fibers or M:N Threading)
Because of the cost of context switching on an operating system
thread (e.g. pthreads), lightweight threading models were invented to run
in application space, without making round trips through the kernel. These
are variably known as Virtual Threads, Green Threads, or Fibers. We will call
them [Fibers](https://en.wikipedia.org/wiki/Fiber_(computer_science)). The key
difference between fibers and operating system threads is that fibers use
cooperative context switching, instead of preemptive time-slicing. But the cost
of context switching with fibers, in application space, at the boundaries of
runnable tasks, is dramatically lower than with preemptive kernel threading.

Fibers should be exceptionally lightweight, especially in terms of memory use. They
should merely wrap around the code (task) to be executed when it is allocated to
a CPU thread.  By lightweight, I mean it should be possible to create millions to
billions of them on a typical cloud server machine.

The software industry has done this many times and in many languages. Here are a
few references:
* IEEE: [Achieving Efficient Structured Concurrency through Lightweight Fibers in
  Java Virtual Machine](https://ieeexplore.ieee.org/document/9245253)
* [OpenJDK JEP 436](https://openjdk.org/jeps/436).
* [ZIO Fibers in Scala](https://zio.dev/reference/fiber/),

Fibers will be used by an actor system to dispatch and execute messages to actors.
Actors must not "hog" a kernel thread, especially when their mailbox is
exhausted. Instead, switching context must be lightweight so that messages can
be cooperatively executed without performance penalty.

#### Futures and Promises
[Futures & Promises](https://en.wikipedia.org/wiki/Futures_and_promises) refer to
computing abstractions used to synchronize program execution in various
[concurrent programming languages](https://en.wikipedia.org/wiki/Concurrent_computing#Concurrent_programming_languages).
A `Future` is a computing abstraction that encapsulates a function or lambda for
asynchronous execution purposes. Upon execution of the Future, a Promise is
returned as a proxy to the asynchronous result.  Execution of the Future's
encapsulated code may be eager (immediately) or lazy (when its computational
result is needed). A Promise is a proxy for the Future's computational result.
It can typically be polled for the result or waited upon for the result.

This is a very thoroughly and well researched topic. For example, you can:
* [Read more on Wikipedia](https://en.wikipedia.org/wiki/Futures_and_promises)
* [Review the Prior Work](https://gist.githubusercontent.com/lattner/31ed37682ef1576b16bca1432ea9f782/raw/dc3b70690a7ba3bed709d3043ad312eadb53d553/TaskConcurrencyManifesto.md)
* [The Swift asynch/await proposal](https://gist.github.com/lattner/429b9070918248274f25b714dcfc7619)

A similar (and likely more elegant) way to accomplish the same end goal is to
use [`async`/`await` as in many languages](https://en.wikipedia.org/wiki/Async/await)
to hide the `Futures` and `Promises` internal to the compiler to unburden the
developer from dealing with them. Here is
[the proposal for async/awat in Swift](https://gist.github.com/lattner/429b9070918248274f25b714dcfc7619)

Whichever mechanism is used, there is one golden rule that it must support:
> Thou shalt not block nor delay

During `await` or while running `Promise.await`, the fiber AND the thread
must not be idle waiting. The fiber that is waiting must return the thread
it is running on, so it can be used to execute other work. When the condition
is set, the fiber should resume again as soon as a thread is next available

#### Fork/Join Pool & Work Stealing
Concurrent programs are often recursively partitioned into smaller independent
tasks that run asynchronously. This leads to queues of tasks (fibers) that are
waiting to be executed on an available kernel thread (typically one OS thread
per CPU core) via a pool of worker kernel threads that execute one task at a time.
This is known as a Fork/Join pool.

When combined with Work Stealing, a fork/join pool can often make full use of
all processor cores. Work Stealing is simply when free threads "steal" work
(tasks) from the queue of other busy threads in the pool. By default, a worker
thread gets tasks from the head of its own deque. When it is empty, the thread
takes a task from the tail of the deque of another busy thread or from the global
entry queue since this is where the biggest pieces of work are likely to be
located.  This approach minimizes the possibility that threads will compete for
tasks. It also reduces the number of times the thread will have to go looking
for work, as it works on the biggest available chunks of work first.

More full treatments of this technique are here:
* [Baeldung Blog: Java-Fork-Join](https://www.baeldung.com/java-fork-join)
* [Java Fork Join Pool](https://docs.oracle.com/javase/8/docs/api/java/util/concurrent/ForkJoinPool.html)

### TLS or Better Support
Cluster and inter-cluster communications will need to be secured via
TLS (we don't care about insecure SSL) or a more advanced security
protocol between endpoints.  The quantum computing cryptographic
"deadline" of 2025 is fast approaching and alternative solutions to
TLS will be needed; or a more advanced cryptography scheme that
cannot be broken easily by quantum computers.

It is expected that whatever mechanism is chosen by Mojo to
implemented secure communications, it will not be part of the
actor specification.

### Sum-type of Product-types
Since actors are type safe, they must be able to define their API in terms of
the set of messages they accept; and we want to do that with a single type
name! This means, in type theory terms, that it wants a sum-type that contains
all the product-types of its API.  The important part is to introduce a
single type name that completely and exclusively represents a set of types
as the messages for the API of an actor. Doing this will permit significantly
better optimization of match statements when dispatching messages to functions.
Consider the following examples of this idea.

In Rust, a sum-type of product-types can be expressed easily enums, as shown
in [The Rust Programming Language](https://doc.rust-lang.org/book/ch06-01-defining-an-enum.html):
```rust
enum Message {
    Quit,
    Move { x: i32, y: i32 },
    Write(String),
    ChangeColor(i32, i32, i32),
}
```
Similarly, a sealed trait could accomplish this in Scala:
```scala
sealed trait Message
case object Quit extends Message
case class Move( x: Int, y: Int) extends Message
case class Write(s: String) extends Message
case class ChangeColor(r: Int, g: Int, b: Int) extends Message
```
And, in Haskell it is done with a data definition, like this:
```haskell
data Message = Quit | Move { x :: Int, y :: Int } | Write { s :: String } |
  ChangeColor { r :: Int, g :: Int, b :: Int }
```

I prefer Rust's syntax for this and suggest that Mojo use it directly but the
intent here is not the syntax but rather that Mojo's type system support the
declaration of a sum-type over a set of product-types, for reasons of
efficiency, expressiveness, and conciseness.

### Pattern Matching
Actor implementations will have one required method: `handle` which should
be a partial function that uses pattern matching to unfold messages and
dispatch them to message-specific behavior functions. If the language does
not support pattern matching, some kind of type-level introspection will
be needed to disambiguate the sum-type-of-product-types representing the
messages.

### Non-capture of actors state in closures
Since actors process messages one at a time; if they use a closure, that
closure should not capture any state of the actor because by the time the
closure runs (asynchronously), that state is stale and may have legitimately
changed through processing another message asynchronously. Furthermore, this
violates the [Actors Do Not Share State](about-actors.md#actors-do-not-share-state) 
rule.

In other actor systems (e.g. Akka) there is no language support for this in
Scala, and it is the common cause of exceptionally-hard-bugs-to-find. Mojo
needs to do better. When an actor uses any asynchronous code, the borrow
checker should disallow mutable access to the actor's state. This means that
certain constructs (e.g. Pointer, mutable reference) of the language should
be disallowed in the case that the target of those references is actor state.
This means the language needs to differentiate between struct an actor.

### Complete ownership of encapsulated data
The private data of an actor may never be modified by any code that is not part
of the behavior processing of the corresponding actor. Actors are like objects
in this sense; they bind functions and data together. It would be useful if
Mojo's ownership model was inherently aware of these tenets of actors,
specifically:
* Actor state should not even be referencable outside the actor.
* Concurrent changes to actor data can never happen.
* Actor data changes can only happen when an actor is processing a message.


```rust
trait MyActor 
```
### Efficient Abstract Read-Only Streams
Actor messages need to be sent in fire-and-forget fashion. The sender should
not block, and the message should not change even after delivery to the
target actor. As this is a generally useful construct, it should not be
a language feature but included in the standard library. Neither
message dispatch nor message delivery should be blocking operations, and they
must be re-entrant by multiple threads.

To make this efficient and safe:
* messages should be transmitted without copying data using an immutable
  reference
*

# After Here Is Copy of Prior Work To Aid Writing
(i.e. ignore, it will be deleted)
