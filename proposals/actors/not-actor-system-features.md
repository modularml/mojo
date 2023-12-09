## Not Actor System Features 
This section details features of the Mojo Actor System and project goals
that we don't intend to implement. This is based on prior experience with Actor
Systems (Erlang, Akka) that have lead to confusion and weak performance.

### No Mojo Syntax or Semantics Requirements
We don't want to alter the Mojo syntax or semantics. We believe that
the actor model can be written directly in Mojo given its significant
planned capabilities.

### No Replacement Of Concurrence Features
We do not wish to replace existing or future concurrency features of Mojo.
But, we do intend to use them, however they are designed.

### Not Everything Is An Actor
In most of the research, academia assumes the use of a pure actor model
("everything is an actor"), and assumes a model of communication so
limited that it is not acceptable for this proposal.  This proposal does
not require such an all-or-nothing approach.

### No "actor" keyword in Mojo
Mojo, as a language, should be unaware of actors. The Mojo Actor System should
be able to be implemented in Mojo as a library without cluttering the semantics
or syntax of Mojo. Consequently, we don't feel the need to make Mojo have 
an `actor` keyword or any other facilitation of the concept. A suitable 
object-orientation based on its origins from Python should be sufficient.
Actors [combine state and behavior](about-actors.md#what-is-an-actor) much like 
the original intention for object-orientation<sup>[alan-kay-oop](#alan-kay-oop)</sup>,
mostly because the encapsulation of state is complete and behavior invocation
is only provided by message passing. 

However, the language will depend on numerous capabilities of the Mojo
language, which are [documented here](mojo-features-needed.md). 

### Non-specialized Language Features
Actors should be able to participate very efficiently in very
low-level interactions with things like coprocessors and accelerators (
[further details here](actor-system-features.md#signals-asynch-io-gpu-dsp-asics-and-accelerators-are-actors)).
Accomplishing this should not require specialized support from the language to
make it efficient. Since the language will undoubtedly need to provide low-level
access to various devices in order to support its Modular AI goals. The Mojo
Actor System should likewise be able to use the same facilities.

### No Support For Request/Response (Ask) Pattern
The [Prior Work](https://gist.github.com/lattner/31ed37682ef1576b16bca1432ea9f782)
suggested that:
> unidirectional async message sends are great, but inconvenient for some
> things.  We want a model that allows messages to return a value (even if
> we encourage them not to), which requires a way to wait for that value.
> This is the point of adding async/await.

Waiting for responses from an actor is currently regarded as an anti-pattern
in some actor systems, especially if the waiting occurs within an actor. Actor
implementations should *never* wait (hard rule). The reason is that it
delays processing of all pending messages in the actor's mailbox.
This leads to unresponsive applications.

Instead, an originating actor should *always* fire-and-forget a message to
a target actor. If a response is expected then that originating message should
include in its payload any contextual information (e.g. `ActorRef`) the
receiving actor may need when it later wants to send a response. This should
include the ability to publish the response to multiple recipients.

Should the target decide to respond (it might not!), the message must
originating contain the address of a *reply-to* actor so the target
actor knows to whom the response should be directed. This does not
need to be the same as the originating actor.

The *reply-to* actor must include the type of the response message in
its API. The response message should provide the originating actor's
contextual information in the response message for the *reply-to*
actor's convenience.

When the *reply-to* actor _is_ the originating actor, then the
response processing in the originating actor, should be treated as
wholly knew operation since intervening message processing may have
changed the actor's state.

This approach accommodates the need for a request/response pattern
but in a completely asynchronous and non-blocking fashion.

### Footnotes
#### Alan Kay OOP
Alan Kay, inventor of SmallTalk, had a conception of
object-oriented programming in 1967 when he coined the term. Read more about his take
on OOP[here](https://userpage.fu-berlin.de/~ram/pub/pub_jf47ht81Ht/doc_kay_oop_en)

