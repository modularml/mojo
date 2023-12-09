## Project Milestones

This section defines a list of actor system features, roughly in development
order based on the primary author's prior knowledge of feature dependencies.
It is _not_ a goal of this section to concern itself with:
* Actual dependencies between features
* What development tasks can be parallelized
* How to minimize project calendar time
* What the number of development teams should be
* Developing a Gantt chart
* Creating epic or story definitions.

### Create an Actor System Software Module (Class, Object)
An ActorSystem encapsulates all the functionality of a collection of actors that
cooperate to achieve some functionality. The Actor System is a heavy weight object
that needs to manage all of the following:
* Providing the API of the Actor System, mostly around startup and shutdown
* Providing the API of an Actor with private methods only the Actor System can call
* Providing APIs for message enqueuing and dispatch
* Implement the default version of all of the above APIs 
* having the ability to reach every actor, allocate new ones, deal with memory
  resource limitations (passivating unused actors to reclaim their memory)
* Providing implementations of the enqueuing and dispatching APIs to over varying
  CPU/memory usage tradeoffs or varying functionality
* Configuring and setting up light-weight threads (Co-routines, fibers) to perform
* Acting as the locus of configuration for all the actors
* Handling startup and termination processing
* Providing search, lookup, and navigation to actors
* Managing resource (memory typically) allocation and reclamation through the 
  lifecycle of actors. 
* Handling failures, dead-letters, etc. 

### Concurrent Message Enqueuing
When a message is sent to an actor, it is enqueued into the that actor's 
mailbox. The design should accommodate different kinds of mailbox queues 
and support only local enqueuing at this point. Distributed enqueuing will 
be added later. 

### Concurrent Message Dispatch
When an actor needs something to do (processing its current message has 
completed), the actor system must dispatch the next message, according to
mailbox semantics, for the actor to process. 

### Enforce Actor Message Type Safety
The set of messages passed to a kind of actor must be known at compile time 
and the compiler should generate an error message if an attempt is made to 
send an actor a message that it cannot process. 

### Composable Behavior
Actors should use the Mojo language features that make its behavior most naturally
composable. This needn't be done via inheritance. First class function composition
should suffice for an actor's behaviors. 

### Composable State
Like behavior, an actor's state should be composable too, using the facilities
of the language. State composition need not necessarily be attached to behavior
composition although that may turn out to be desirable to reduce cognitive load. 
It need not necessarily require the use of inheritance.

### Accelerated Message Delivery
Because the message delivery mechanism can turn out to be a large part of the
overhead of system of actors, it is necessary to accelerate or optimize this
portion of the Actor System.  Please see the 
[Message Transmission Acceleration](expected-optimizations.md#message-transmission-acceleration)
for further details

### System Scale Message Delivery
Putting all of the above together should provide a robust actor system for
single computers.  

### Actor Persistence via Event Sourcing
#### Pluggable Persistence API With Snapshotting
#### Reference implementation of Pluggable Persistence API
#### Rehydration from persistent stores
#### Implement Event Projections to R2DBC Databases

### Actor Persistence via Current State only (CRUD)

### Support for CQRS
#### Read/Write side separation
#### Projection from Write side to other actors

### Support for Pub/Sub Without A Broker 
#### Topic support
#### Client read commit and restart high-water mark

### Network Protocols With Actors
#### Aeron
#### TCP/IP,
#### UDP/IP,
#### gRPC

### Network Scale Message Delivery (Remoting)
#### Remote enqueuing
#### Distributed Actor Identity

### Cluster Management
#### Cluster Communications
#### RAFT Consensus Algorithm via Actors
#### Fault Tolerant Cluster Membership
#### Actor Partitioning/Sharding Across Cluster


### Use Actors To Implement Reactive Streams
### Use Actors To Implement RSockets
### Use Actors For Streaming

### Active-Active Multi-DataCenter Support

