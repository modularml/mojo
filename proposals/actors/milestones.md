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

### Create an Actor System Object
An ActorSystem encapsulates all the functionality of enqueuing and dispatching
messages to actors. This needs to define:
* The API of the Actor System, mostly around startup and shutdown
* The API of an Actor with private methods only the Actor System can call
* APIs for message enqueuing and dispatch
* Configuring and setting up threads to do enqueue and dispatch tasks in a
  work stealing manner.

### Concurrent Message Enqueuing
When a message is sent to an actor, it is enqueued into the that actor's 
mailbox. The design should accommodate different kinds of mailbox queues 
and support only local enqueuing at this point. Distributed enqueuing will 
be added later. 

### Concurrent Message Dispatch
When an actor needs something to do (processing its current message has 
completed), the actor system must dispatch the next message, according to
mailbox semantics, for the actor to process. 

### Make Mojo Aware of Actors
An easy way to do this is to use an `actor` keyword that defines a `struct`
with behavior functions. However, if class inheritance is supported in the
data model, it would be simpler to just define `Actor` base class in the
standard library and require actors to inherit it. This should be sufficient
notification to the compiler without altering the language.
### Enforce Actor Type Safety
#### Composable State
#### Composable Behavior
### Accelerator-Scale Message Delivery
### System Scale Message Delivery
### Use Actors For Network Protocols
#### Aeron
#### TCP/IP
#### gRPC
### Use Actors To Implement RSockets
### Use Actors To Implement Reactive Streams
### Use Actors For Streaming
### Add Actor Persistence via Event Sourcing
### Add Actor Persistence via Current State only (CRUD)
### Network Scale Message Delivery (Remoting)
### RAFT Consensus Algorithm via Actors
### Fault Tolerant Cluster Membership
### Distributed Actor Identity
### Persistence-Based Pub/Sub Channels
### Actor Partitioning/Sharding Across Cluster
### Event Projections to R2DBC Databases
### Persistent Projections
### Active-Active Multi-DC Support
