## Actor System Features & Goals

The actors defined in [About Actors](about-actors.md) is constrained to 
the typical usage of actors on a single machine. However, this proposal
doesn't just stop there. In this section we want to consider various 
use cases in which Mojo Actors could be made to participate. 

### Design
We intend to outline a single coherent design for actors that can be:
* built in small increments over time
* scale from very small (processing units) to very large (distributed actor
  systems in multiple computing regions)
* to incrementally drive Mojo to support
  a powerful actor system at every level of scale.
* 
### Lower Cognitive Load
The actor model design should lower cognitive load for the programmer using it.
It should be easy to learn, easy to reason about

### Support Traceability
As a corollary to [Lower Cognitive Load](#lower-cognitive-load), the actor
system should efficiently support traceability of message flows and the effects they
have on the actors. This is to help with design comprehension and debugging.
Preferences:
* Works with distributed actor systems and across cluster and node boundaries
* Implemented with the traceability features of OpenTelemetry. 

### Safety
Mojo will support some form of ]lifecycle, ownership, and borrow checking
(ala Rust but easier). It is a design goal to use these features with 
Mojo Actor System to eliminate eliminate race conditions, deadlock, 
synchronization primitives, and other concurrency problems. 

These features should be used to prevent any kind of back-door access (e.g. 
an untyped pointer) to the actor's state. Such code generate a compile-time
error. This should also permit the compiler to generate efficient code since
only one code region may modify actor data. 

### Message Scalability
Actors require a communication pathway to transfer messages between actors.
On a given machine, there should be no additional transmission time cost 
based on the size of the message. Actor system message enqueuing and dispatch
times should be constant without regard to message size (on the same machine).
Measurements of those times must not factor in queue wait time since that is
dependent on how the actor processes its messages and how the dispatch
scheduler works.

### Communication Scalability
The communication medium(s) used by the actor system must scale its support 
for all 8 levels of interaction 
[as discussed here](about-actors.md#actors-are-asynchronously-message-driven).

### Resilience
The actor system should support multiple kinds of resilience:
- local supervision of parent actors to their child actors
- Mojo's Error type for error handling
- options to stop, restart or resume an actor upon failure
- persisting actor state with event sourcing or by only storing the most 
  recent state (CRUD) with the ability to rehydrate an actor automatically 
  should it fail
- ability to continue processing even through software update (ela Erlang)

### Performance
Actor models eliminate the need for concurrent or even atomic access to data
since only one thread will ever access its private data.  It is a goal to aid
the compiler in optimizing code construction by knowing that actor data cannot
cross task (thread, fiber) boundaries.

### Flexibility
We want to use low-level APIs to provide flexibility in the implementation of 
the actor model. For example, it should be possible to allow the programmer to
override or extend the default choice for many actor system components,
including: the choice of mailbox (queue), network communication system,
actor supervision mechanism, unique identity computation, & etc. Where possible,
it should be possible to override the default implementation of these 
algorithms by the application programmer without difficulty.

### Excellence
We should look to the existing actor models provided by other
languages (e.g. Erlang, Swift, Ruby, Dart) and frameworks such as 
[Akka](https://github.com/akka/akka), [CAF](https://github.com/actor-framework),
[Pykka](https://pykka.readthedocs.io/), [Akka.NET](https://getakka.net/),
[Orleans](https://learn.microsoft.com/en-us/dotnet/orleans/overview),
[Hactors](https://github.com/treep/hactors), and from these draw together the
best ideas so that Mojo offers a better actor model than all of them.

### Language Compatibility
To the greatest extent possible, the Mojo Actor System should be as compatible
with Mojo's existing concurrency constructs, design patterns, and APIs.

### Interoperability
To the greatest extent possible, the Mojo Actor System should be able to 
interoperate with non-actor systems, and actor systems using other languages
and frameworks.  We cannot build a conceptually beautiful new world without 
also building a pathway to get existing apps interoperating with it.

### Signals, Asynch I/O, GPU, DSP, ASICs, and Accelerators Are Actors
Mojo intends to provide a single programming language for heterogeneous
computing. This means that it will provide language or library features to
interact with various processors safely, efficiently, and abstractly. We would
like the actor system to provide a convenient way of interacting with these
processors.

Here are some examples of how this could work:
* A message could encapsulate a CUDA kernel. That message could then be sent to
  a CUDA-typed actor. It would schedule the execution of the kernel and collect
  the resulting computation. It would reply to the sender asynchronously with
  the result.
* The CPU may talk to an accelerator through asynchronous commands sent over
  DMA requests and interrupts or similar facilities in the available
  architecture
* Interrupt handlers in embedded systems could similarly be used to fire an
  asynchronous message to an actor
* Event-driven applications could use asynchronous signals to similarly
  fire asynchronous messages to an actor.
* Applications may have interest in being notified about OS signals, such as
  `SIGWINCH`, and it should be similarly easy to handle that signal quickly
  by sending a message

> *NOTE:* Using interrupt handling requires a high degree of efficiency in the
> handler code so that it is not involved in message delivery just capture of
> the interrupt data and placement in memory for a fiber/thread to process soon.
> This lets the cpu core get back to other business quickly.


We would like to use whatever Mojo language features support the above use cases
and (optionally) model them using actor facilities. That is the processors
would be modeled as actors, using asynchronous messages to send requests of the
device, and similarly using asynchronous messages to send responses back to
the requesting actor. Any actor definitions needed to support this should be
categorized to group common operations (messages) into a single actor type,
based on the interface of the kind of device; but allow the compiler to
choose which device to execute them upon.

### Actors Can And Should Form Clusters
Scalability and Resiliency are thwarted if actors cannot form clusters across
multiple nodes. Bad things happen to computing hardware, and the power, air
conditioning and networking they depend upon. Consequently, failures of actor
systems are inevitable; if you just wait long enough.  To keep a set of cooperating
actors alive continuously, with no interruptions or down-time,
means applying some strategies. The primary strategy is redundancy. Once
multiple nodes are involved, clusters of computers can become resilient, and
scalable, with only a few features:
* actor location transparency
* the [Raft consensus algorithm](https://raft.github.io/raft.pdf) for cluster membership consistency
* Phi-accrual failure detector
* split-brain (network partition) resolver
* conflict-free replicated data types (CRDTs)
* a remoting protocol/algorithm to send messages across the cluster

These features enable:
* CRDTs for client programs (useful at the application layer too)
* restart after failure (on a different node)
* workload partitioning/sharding (using N computers as one, N > 2)
* elastic scalability (growing or shrinking the set of nodes per workload
  demand)
* simplification of distributed parallel processing
* distributed publish & subscribe systems
* large scale (internet sized) applications
* many other things ... design your own

> TODO: provide mermaid diagram fo cluster sharding

### Actors Can Be Persistent
While memory-only actors are sufficient in many applications (because
replication over a reliable cluster is possible), some systems need to prevent
against total failure of all nodes, or the memory cost of the actor system
exceeds the available budget.

No worries, actors can store their state in a database like anything else.

### Actors Should Support CQRS
> TODO:
> CQRS - segregate read/write, support query-optimized database
> mermaid diagram of CQRS

### Actors Should Support Event Sourcing  
> TODO:
> event sourcing + snapshots

### Actors Should Support Auditing
With the above support of event sourcing, a complete record of change events
is persistently recorded. This should allow the actor system to support the use
of these change events in audit procedures. 

### Actor Systems Should Support Publish/Subscribe
Existing actor systems like Akka have historically been combined with
various Pub/Sub systems like [Apache Kafka](https://kafka.apache.org/) and
[Google Pub/Sub](https://cloud.google.com/pubsub/docs/overview) to support 
concurrent delivery of messages to multiple recipients (asynchronous broadcast). 
Recently, 
[Akka implemented its own brokerless Pub/Sub system](https://www.lightbend.com/blog/ditch-the-message-broker-go-faster) 
and we think this is a commendable feature. The brokerless nature of this
system means significantly less infrastructure and significantly increased
latency as [Lightbend documented here](https://www.lightbend.com/blog/benchmarking-kafka-vs-akka-brokerless-pub-sub).
