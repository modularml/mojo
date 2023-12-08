## Swift Actors

This section is a direct copy of a portion of the  [Swift Concurrency Manifesto](https://gist.github.com/lattner/31ed37682ef1576b16bca1432ea9f782)
for convenience of reference. It will be deleted when the proposal is completed. 


Actors are cheap to construct, and you communicate with an actor using efficient
unidirectional asynchronous message sends ("posting a message in a mailbox").
Because these messages are unidirectional, there is no waiting, and thus deadlocks are
impossible.  In the academic model, all data sent in these messages is deep-copied, which
means that there is no data sharing possible between actors.  Because actors cannot touch
each other's state (and have no access to global state), there is no need for any
synchronization constructs, eliminating all the problems with shared mutable state.

To make this work pragmatically in the context of Swift, we need to solve several problems:

- we need a strong computational foundation for all the computation within a task.  Good
  news: this is already done in Swift 1...4!

### Example actor design for Swift

There are several possible ways to manifest the idea of actors into Swift.  For the purposes of
this manifesto, I'll describe them as a new type in Swift because it is the least confusing way
to explain the ideas and this isn't a formal proposal.  I'll note right here up front that this is
only one possible design: the right approach may be for actors to be a special kind of class,
a model described below.

With this design approach, you'd define an actor with the `actor` keyword.  An actor can
have any number of data members declared as instance members, can have normal methods,
and extensions work with them as you'd expect.  Actors are reference types and have an
identity which can be passed around as a value.  Actors can conform to protocols and
otherwise dovetail with existing Swift features as you'd expect.

We need a simple running example, so let's imagine we're building the data model for an app
that has a tableview with a list of strings.  The app has UI to add and manipulate the list.  It
might look something like this:

```swift
  actor TableModel {
    let mainActor : TheMainActor
    var theList : [String] = [] {
      didSet {
        mainActor.updateTableView(theList)
      }
    }
    
    init(mainActor: TheMainActor) { self.mainActor = mainActor }

    // this checks to see if all the entries in the list are capitalized:
    // if so, it capitalize the string before returning it to encourage
    // capitalization consistency in the list.
    func prettify(_ x : String) -> String {
      // Details omitted: it inspects theList, adjusting the
      // string before returning it if necessary.
    }

    actor func add(entry: String) {
      theList.append(prettify(entry))
    }
  }
```

This illustrates the key points of an actor model:

- The actor defines the state local to it as instance data, in this case the reference to
  `mainActor` and `theList` is the data in the actor.
- Actors can send messages to any other actor they have a reference to, using traditional
  dot syntax.
- Normal (non-actor) methods can be defined on the actor for convenience, and
  they have full access to the state within their `self` actor.
- `actor` methods are the messages that actors accept.  Marking a method as `actor`
  imposes certain restrictions upon it, described below.
- It isn't shown in the example, but new instances of the actor are created by using the
  initializer just like any other type: `let dataModel = TableModel(mainActor)`.
- Also, not shown in the example, but `actor` methods are implicitly `async`, so they can
  freely call `async` methods and `await` their results.

It has been found in other actor systems that an actor abstraction like this encourage the
"right" abstractions in applications, and map well to the conceptual way that programmers
think about their data.  For example, given this data model it is easy to create multiple
instances of this actor, one for each document in an MDI application.

This is a straight-forward implementation of the actor model in Swift and is enough to achieve
the basic advantages of the model.  However, it is important to note that there are a number
of limitations being imposed here that are not obvious, including:

- An `actor` method cannot return a value, throw an error, or have an `inout` parameter.
- All the parameters must produce independent values when copied (see below).
- Local state and non-`actor` methods may only be accessed by methods defined lexically
  on the actor or in an extension to it (whether they are marked `actor` or otherwise).

### Extending the model through await

The first limitation (that `actor` methods cannot return values) is easy to address as we've
already discussed.  Say the app developer needs a quick way to get the number of entries in
the list, a way that is visible to other actors they have running around.  We should simply
allow them to define:

```swift
  extension TableModel {
    actor func getNumberOfEntries() -> Int {
      return theList.count
    }
  }
````

This allows them to await the result from other actors:

```swift
  print(await dataModel.getNumberOfEntries())
```

This dovetails perfectly with the rest of the async/await model.  It is unrelated to this
manifesto, but we'll observe that it would be more idiomatic way to
define that specific example is as an `actor var`.  Swift currently doesn't allow property
accessors to `throw` or be `async`.  When this limitation is relaxed, it would be
straight-forward to allow `actor var`s to provide the more natural API.

Note that this extension makes the model far more usable in cases like this, but erodes the
"deadlock free" guarantee of the actor model.  An await on an `actor` method suspends the
current task, and since you can get circular waits, you can end up with deadlock.  This is
because only one message is processed by the actor at a time.  The simples case occurs
if an actor waits on itself directly (possibly through a chain of references):

```swift
  extension TableModel {
    actor func f() {
       ...
       let x = await self.getNumberOfEntries()   // trivial deadlock.
       ...
    }
  }
```

The trivial case like this can also be trivially diagnosed by the compiler.  The complex case
would ideally be diagnosed at runtime with a trap, depending on the runtime implementation
model.

The solution for this is to encourage people to use `Void`-returning `actor` methods that "fire
and forget".  There are several reasons to believe that these will be the most common: the
async/await model described syntactically encourages people not to use it (by requiring
marking, etc.), many of the common applications of actors are event-driven applications
(which are inherently one way), the eventual design of UI and other system frameworks
can encourage the right patterns from app developers, and of course documentation can
describe best practices.

### About that main thread

The example above shows `mainActor` being passed in, following theoretically pure actor
hygiene.  However, the main thread in UIKit and AppKit are already global state, so we might
as well admit that and make code everywhere nicer.  As such, it makes sense for AppKit and
UIKit to define and vend a public global constant actor reference, e.g. something like this:

```swift
public actor MainActor {  // Bikeshed: could be named "actor UI {}"
   private init() {}      // You can't make another one of these.
   // Helpful public stuff could be put here to make app developers happy. :-)
}
public let mainActor = MainActor()
```

This would allow app developers to put their extensions on `MainActor`, making their code
more explicit and clear about what *needs* to be run on the main thread.  If we got really
crazy, someday Swift should allow data members to be defined in extensions on classes,
and App developers would then be able to put their state that must be manipulated on the
main thread directly on the MainActor.

### Data isolation

The way that actors eliminate shared mutable state and explicit synchronization is through
deep copying all the data that is passed to an actor in a message send, and preventing
direct access to actor state without going through these message sends.  This all composes
nicely, but can quickly introduce inefficiencies in practice because of all the data copying
that happens.

Swift is well positioned to deal with this for a number of reasons: its strong focus on value
semantics means that copying of these values is a core operation understood and known by
Swift programmers everywhere.  Second, the use of Copy-On-Write (🐮) as an
implementation approach fits perfectly with this model.  Note how, in the example above,
the DataModel actor sends a copy of the `theList` array back to the UI thread so it can
update itself.  In Swift, this is a super efficient O(1) operation that does some ARC stuff: it
doesn't actually copy or touch the elements of the array.

The third piece, which is still in development, will come as a result of the work on adding
[ownership semantics](https://github.com/apple/swift/blob/master/docs/OwnershipManifesto.md)
to Swift.  When this is available, advanced programmers will have the ability to *move*
complex values between actors, which is typically also a super-efficient O(1) operation.

This leaves us with three open issues: 1) how do we know whether something has proper
value semantics, 2) what do we do about reference types (classes and closures), and 3) what
do we do about global state.  All three of these options should be explored in detail, because
there are many different possible answers to these. I will explore a simple model below in
order to provide an existence proof for a design, but I do not claim that it is the best model
we can find.

#### Does a type provide proper value semantics?

This is something that many many Swift programmers have wanted to be able to know the
answer to, for example when defining generic algorithms that are only correct in the face of
proper value semantics.  There have been numerous proposals for how to determine this,
and I will not attempt to summarize them, instead I'll outline a simple proposal just to provide
an existence proof for an answer:

- Start by defining a simple marker protocol (the name of which is intentionally silly to reduce
  early bikeshedding) with a single requirement:
  `protocol ValueSemantical { func valueSemanticCopy() -> Self }`
- Conform all of the applicable standard library types to `ValueSemantical`.  For example,
  Array conforms when its elements conform - note that an array of reference types doesn't
  always provide the semantics we need.
- Teach the compiler to synthesize conformance for structs and enums whose members are
  all `ValueSemantical`, just like we do for `Codable`.
- The compiler just checks for conformance to the `ValueSemantical` protocol and
  rejects any arguments and return values that do not conform.

To reiterate, the name `ValueSemantical` really isn't the right name for this: things like
`UnsafePointer`, for example, shouldn't conform.  Enumerating the possible options and
evaluating the naming tradeoffs between them is a project for another day though.

It is important to realize that this design does *not guarantee memory safety*.  Someone
could implement the protocol in the wrong way (thus lying about satisfying the requirements)
and shared mutable state could occur.  In the author's opinion, this is the right tradeoff:
solving this would require introducing onerous type system mechanics (e.g. something like
the capabilities system in the [Pony](https://www.ponylang.org/) language).  Swift already
provides a model where memory safe APIs (e.g. `Array`) are implemented in terms of memory
unsafety (e.g. `UnsafePointer`), the approach described here is directly analogous.

*Alternate Design*: Another approach is to eliminate the requirement from the protocol:
just use the protocol as a marker, which is applied to types that already have the right
behavior.  When it is necessary to customize the copy operation (e.g. for a reference type),
the solution would be to box values of that type in a struct that provides the right value
semantics.  This would make it more awkward to conform, but this design eliminates having
"another kind of copy" operation, and encourages more types to provide value semantics.

#### Reference types: Classes

The solution to this is simple: classes need to conform to `ValueSemantical` (and
implement the requirement) properly, or else they cannot be passed as a parameter or result
of an `actor` method.  In the author's opinion, giving classes proper value semantics will not
be that big of a deal in practice for a number of reasons:

- The default (non-conformance) is the right default: the only classes that conform will be
  ones that a human thought about.
- Retroactive conformance allows app developers to handle cases not addressed by the
  framework engineers.
- Cocoa has a number of classes (e.g. the entire UI frameworks) that are only usable on the
  main thread.  By definition, these won't get passed around.
- A number of classes in Cocoa are already semantically immutable, making it trivial and
  cheap for them to conform.

Beyond that, when you start working with an actor system, it is an inherent part of the
application design that you don't allocate and pass around big object graphs: you allocate
them in the actor you intend to manipulate them with.  This is something that has been
found true in Scala/Akka for example.

#### Reference types: Closures and Functions

It is not safe to pass an arbitrary value with function type across an actor message,
because it could close over arbitrary actor-local data.  If that data is closed over
by-reference, then the recipient actor would have arbitrary access to data in the sending
actor's state.  That said, there is at least one important exception that we should carve
out: it is safe to pass a closure *literal* when it is known that it only closes over
data by copy: using the same `ValueSemantical` copy semantics described above.

This happens to be an extremely useful carveout, because it permits some interesting "callback"
abstractions to be naturally expressed without tight coupling between actors.  Here is a silly
example:

```swift
    otherActor.doSomething { self.incrementCount($0) }
```

In this case `OtherActor` doesn't have to know about `incrementCount` which is defined
on the self actor, reducing coupling between the actors.

#### Global mutable state

Since we're friends, I'll be straight with you: there are no great answers here.  Swift and C
already support global mutable state, so the best we can do is discourage the use of it.  We
cannot automatically detect a problem because actors need to be able to transitively use
random code that isn't defined on the actor.  For example:

```swift
func calculate(thing : Int) -> Int { ... }

actor Foo {
  actor func exampleOperation() {
     let x = calculate(thing: 42)
     ...
  }
}
```

There is no practical way to know whether 'calculate' is thread-safe or not.  The only solution
is to scatter tons of annotations everywhere, including in headers for C code.  I think that
would be a non-starter.

In practice, this isn't as bad as it sounds, because the most common operations
that people use (e.g. `print`) are already internally synchronizing, largely because people are
already writing multithreaded code.  While it would be nice to magically solve this long
standing problem with legacy systems, I think it is better to just completely ignore it and tell
developers not to define or use global variables (global `let`s are safe).

All hope is not lost though: Perhaps we could consider deprecating global `var`s from Swift
to further nudge people away from them. Also, any accesses to unsafe global global mutable
state from an actor context can and should be warned about.  Taking some steps like this
should eliminate the most obvious bugs.

### Scalable Runtime

Thus far, we've dodged the question about how the actor runtime should be implemented.
This is intentional because I'm not a runtime expert!  From my perspective, building on top
of GCD is great if it can work for us, because it is proven and using it reduces risk from the
concurrency design.  I also think that GCD is a
reasonable baseline to start from: it provides the right semantics, it has good low-level
performance, and it has advanced features like Quality of Service support which are just as
useful for actors as they are for anything else.  It would be easy to provide access to these
advanced features by giving every actor a `gimmeYourQueue()` method.

Here are some potential issues using GCD which we will need to be figure out:

**Kernel Thread Explosions**

Our goal is to allow actors to be used as a core unit of abstraction within a program, which
means that we want programmers to be able to create as many of them as they want, without
running into performance problems.  If scalability problems come up, you end up having to
aggregate logically distinct stuff together to reduce # actors, which leads to complexity
and loses some of the advantages of data isolation.  The model as proposed should scale
exceptionally well, but depends on the runtime to make this happen in practice.

GCD is already quite scalable, but one concern is that it can be subject to kernel thread
explosions, which occur when a
GCD task blocks in a way that the kernel and runtime cannot reason about.  In response,
the GCD runtime allocates new kernel threads, each of which get a stack... and these stacks
can fragment the heap.  This is problematic in the case
of a server workload that wants to instantiate hundreds of thousands of actors - at
least one for every incoming network connection.

Provably solving thread explosions is probably impossible/impractical in any runtime given
the need to interoperate with C code and legacy systems that aren't built in pure Swift.  That
said, perfection isn't necessary: we just need a path that moves towards it, and provides
programmers a way to "get their job done" when an uncooperative framework or API is hit
in practice.  I'd suggest a three step approach to resolving this:

- Make existing frameworks incrementally "async safe" over time.  Ensure that new APIs are
  done right, and make sure that no existing APIs ever go from “async safe” to “async unsafe”.
- Provide a mechanism that developers can use to address problematic APIs that they
  encounter in practice.  It should be something akin to “wrap your calls in a closure and
  pass it to a special GCD function”, or something else of similar complexity.
- Continue to improve perf and debugger tools to help identify problematic cases that occur
  in practice.

This approach of focusing on problematic APIs that developers hit in practice should work
particularly well for server workloads, which are the ones most likely to need a large number of
actors at a single time.  Legacy server libraries are also much more likely to be async friendly
than arbitrary other C code.

**Actor Shutdown**

There are also questions about how actors are shut down.  The conceptually ideal model is
that actors are implicitly released when their reference count drops to zero and when the last
enqueued message is completed.  This will probably require some amount of runtime
integration.

**Bounded Queue Depths**

Another potential concern is that GCD queues have unbounded depth: if you have a
producer/consumer situation, a fast producer can outpace the consumer and continuously
grow the queue of work.  It would be interesting to investigate options for
providing bounded queues that throttle or block the producer in this sort of situation.  Another
option is to make this purely an API problem, encouraging the use of reactive streams and
other abstractions that provide back pressure.

### Alternative Design: Actors as classes

The design above is simple and self consistent, but may not be the right model, because
actors have a ton of conceptual overlap with classes.  Observe:

- Actors have reference semantics, just like classes.
- Actors form a graph, this means that we need to be able to have `weak`/`unowned`
  references to them.
- Subclassing of actors makes just as much sense as subclassing of classes, and would
  work the same way.
- Some people incorrectly think that Swift hates classes: this is an opportunity to restore
  some of their former glory.

However, actors are not *simple classes*: here are some differences:

- Only actors can have `actor` methods on them.  These methods have additional
  requirements put on them in order to provide the safety in the programming model we seek.
- An "actor class" deriving from a "non-actor base class" would have to be illegal, because
  the base class could escape self or escape local state references in an unsafe way.

One important pivot-point in discussion is whether subclassing of actors is desirable.  If so,
modeling them as a special kind of class would be a very nice simplifying assumption,
because a lot of complexity comes in with that (including all the initialization rules etc).  If not,
then defining them as a new kind of type is defensible, because they'd be very simple and
being a separate type would more easily explain the additional rules imposed on them.

Syntactically, if we decided to make them classes, it makes sense for this to be a modifier
on the class definition itself, since actorhood fundamentally alters the contract of the class,
e.g.:

```swift
actor class DataModel : SomeBaseActor { ... }
```

alternatively, since you can't derive from non-actor classes anyway, we could just make the
base class be `Actor`:

```swift
class DataModel : Actor { ... }
```

### Further extensions

The design sketch above is the minimal but important step forward to build concurrency
abstractions into the language, but really filling out the model will almost certainly require a
few other common abstractions.  For example:

- [Reactive streams](https://en.wikipedia.org/wiki/Reactive_Streams) is a common way to
  handle communication between async actors, and helps provide solutions to backpressure.
  [Dart's stream design](https://www.dartlang.org/tutorials/language/streams) is one example.

- Relatedly, it makes sense to extend the `for/in` loop to asynchronous sequences - likely through the introduction of
  a new `AsyncSequence` protocol.  FWIW, this is likely to be added to
  [C# 8.0](https://channel9.msdn.com/Blogs/Seth-Juarez/A-Preview-of-C-8-with-Mads-Torgersen#time=16m30s).

- A first class `Future` type is commonly requested.  I expect the importance of it
  to be far less than in languages that don't have (or who started without) async/await,
  but it is still a very useful abstraction for handling cases where you want to kick off simple
  overlapping computations within a function.

### Intra-actor concurrency

Another advanced concept that could be considered is allowing someone to define a
"multithreaded actor", which provides a standard actor API, but where synchronization and
scheduling of tasks is handled by the actor itself, using traditional synchronization
abstractions instead of a GCD queue.  Adding this would mean that there is shared mutable
state *within* the actor, but that isolation *between* actors is still preserved.  This is
interesting to consider for a number of reasons:

- This allows the programming model to be consistent (where an "instance of an actor
  represents a thing") even when the thing can be implemented with internal concurrency.
  For example, consider an abstraction for a network card/stack: it may want to do its own
  internal scheduling and prioritizing of many different active pieces of work according to its
  own policies, but provide a simple-to-use actor API on top if that.  The fact that the actor
  can handle multiple concurrent requests is an implementation detail the clients shouldn’t
  have to be rewritten to understand.

- Making this non-default would provide proper progressive disclosure of complexity.

- You’d still get improved safety and isolation of the system as a whole, even if individual actors are “optimized” in this way.

- When incrementally migrating code to the actor model, this would make it much easier to
  provide actor wrappers for existing concurrent subsystems built on shared mutable state
  (e.g. a database whose APIs are threadsafe).

- Something like this would also probably be the right abstraction for imported RPC services
  that allow for multiple concurrent synchronous requests.

- This abstraction would be unsafe from the memory safety perspective, but this is widely
  precedented in Swift.  Many safe abstractions are built on top of memory unsafe
  primitives - consider how `Array` is built on `UnsafePointer` - and this is an important
  part of the pragmatism and "get stuff done" nature of the Swift programming model.

That said, this is definitely a power-user feature, and we should understand, build, and get
experience using the basic system before considering adding something like this.


## Part 3: Reliability through fault isolation

Swift has many aspects of its design that encourages programmer errors (aka software
bugs :-) to be caught at compile time: a static type system, optionals, encouraging covered
switch cases, etc.  However, some errors may only be caught at runtime, including things like
out-of-bound array accesses, integer overflows, and force-unwraps of nil.

As described in the [Swift Error Handling
Rationale](https://github.com/apple/swift/blob/master/docs/ErrorHandlingRationale.rst), there
is a tradeoff that must be struck: it doesn't make sense to force programmers to write logic
to handle every conceivable edge case: even discounting the boilerplate that would generate,
that logic is likely to itself be poorly tested and therefore full of bugs.  We must carefully
weigh and tradeoff complex issues in order to get a balanced design.  These tradeoffs are
what led to Swift's approach that does force programmers to think about and write code to
handle all potentially-nil pointer references, but not to have to think about integer overflow on
every arithmetic operation.  The new challenge is that integer overflow still must be
detected and handled somehow, and the programmer hasn't written any recovery code.

Swift handles these with a [fail fast](https://en.wikipedia.org/wiki/Fail-fast) philosophy: it is
preferable to detect and report a programmer error as quickly as possible, rather than
"blunder on" with the hope that the error won't matter.  Combined with rigorous testing (and
perhaps static analysis technology in the future), the goal is to make bugs shallow, and provide
good stack traces and other information when they occur.  This encourages them to be found
and fixed quickly early in the development cycle.  However, when the app ships, this
philosophy is only great if all the bugs were actually found, because an undetected problem
causes the app to suddenly terminate itself.

Sudden termination of a process is hugely problematic if it jeopardizes user data, or - in the
case of a server app - if there are hundreds of clients currently connected to the server at the
time.  While it is impossible in general to do perfect resolution of an arbitrary programmer
error, there is prior art for how handle common problems gracefully.  In the case of Cocoa,
for example, if an `NSException` propagates up to the top of the runloop, it is useful to try to
save any modified documents to a side location to avoid losing data.  This isn't guaranteed
to work in every case, but when it does, the
user is very happy that they haven't lost their progress.  Similarly, if a server crashes
handling one of its client's requests, a reasonable recovery scheme is to finish handling the
other established connections in the current process, but push off new connection requests
to a restarted instance of the server process.

The introduction of actors is a great opportunity to improve this situation, because actors
provide an interesting granularity level between the "whole process" and "an individual class"
where programmers think about the invariants they are maintaining.  Indeed, there is a bunch
of prior art in making reliable actor systems, and again, Erlang is one of the leaders (for a
great discussion, see [Joe Armstrong's PhD thesis](http://erlang.org/download/armstrong_thesis_2003.pdf)).  We'll
start by sketching the basic model, then talk about a potential design approach.

### Actor Reliability Model

The basic concept here is that an actor that fails has violated its own local invariants, but that
the invariants in other actors still hold: this because we've defined away shared
mutable state.  This gives us the option of killing the individual actor that broke its invariants
instead of taking down the entire process.  Given the definition of the basic actor model
with unidirectional async message sends, it is possible to have the runtime just drop any new
messages sent to the actor, and the rest of the system can continue without even knowing
that the actor crashed.

While this is a simple approach, there are two problems:

- Actor methods that return a value could be in the process of being `await`ed, but if the
  actor has crashed those awaits will never complete.
- Dropping messages may itself cause deadlock because of higher-level communication
  invariants that are broken.  For example, consider this actor, which waits for 10 messages
  before passing on the message:

```swift
  actor Merge10Notifications {
    var counter : Int = 0
    let otherActor = ...  // set up by the init.
    actor func notify() {
      counter += 1
      if counter >= 10 {
        otherActor.notify()
      }
    }
  }
```

If one of the 10 actors feeding notifications into this one crashes, then the program will wait
forever to get that 10th notification.  Because of this, someone designing a "reliable" actor
needs to think about more issues, and work slightly harder to achieve that reliability.

### Opting into reliability

Given that a reliable actor requires more thought than building a simple actor, it is reasonable
to look for opt-in models that provide [progressive disclosure of
complexity](https://en.wikipedia.org/wiki/Progressive_disclosure).  The first thing
you need is a way to opt in.  As with actor syntax in general, there are two
broad options: first-class actor syntax or a class declaration modifier, i.e., one of:

```swift
  reliable actor Notifier { ... }
  reliable actor class Notifier { ... }
```

When one opts an actor into caring about reliability, a new requirement is imposed on all
`actor` methods that return a value: they are now required to be declared `throws` as well.
This forces clients of the actor to be prepared for a failure when/if the actor crashes.

Implicitly dropping messages is still a problem.  I'm not familiar with the approaches taken in
other systems, but I imagine two potential solutions:

1) Provide a standard library API to register failure handlers for actors, allowing higher level
   reasoning about how to process and respond to those failures.  An actor's `init()` could
   then use this API to register its failure handler the system.
2) Force *all* `actor` methods to throw, with the semantics that they only throw if the actor
   has crashed.  This forces clients of the reliable actor to handle a potential crash, and do so
   on the granularity of all messages sent to that actor.

Between the two, the first approach is more appealing to me, because it allows factoring
out the common failure logic in one place, rather than having every caller have to write (hard
to test) logic to handler the failure in a fine grained way.  For example, a document actor could
register a failure handler that attempts to save its data in a side location if it ever crashes.

That said, both approaches are feasible and should be explored in more detail.

*Alternate design*: An alternate approach is make all actors be "reliable" actors, by making
the additional constraints a simple part of the actor model.  This reduces the number of
choices a Swift programmer gets-to/has-to make.  If the async/await model ends up making
async imply throwing, then this is probably the right direction, because the `await` on a value
returning method would be implicitly a `try` marker as well.

### Reliability runtime model

Besides the high level semantic model that the programmer faces, there are also questions
about what the runtime model is.  When an actor crashes:

- What state is its memory left in?
- How well can the process clean up from the failure?
- Do we attempt to release memory and other resources (like file descriptors) managed by that actor?

There are multiple possible designs, but I
advocate for a design where **no cleanup is performed**: if an actor crashes, the runtime
propagates that error to other actors and runs any recovery handlers (as described in the
previous section) but that it **should not** attempt further clean up the resources owned by
the actor.

There are a number of reasons for this, but the most important is that the failed actor just
violated its own consistency with whatever invalid operation it attempted to perform.  At this
point, it may have started a transaction but not finished it, or may be in any other sort of
inconsistent or undefined state.  Given the high likelihood for internal inconsistency, it is
probable that the high-level invariants of various classes aren't intact, which means it isn't
safe to run the `deinit`-ializers for the classes.

Beyond the semantic problems we face, there are also practical complexity and efficiency
issues at stake: it takes code and metadata to be able to unwind the actor's stack and release
active resources.  This code and metadata takes space in the application, and it also takes
time at compile time to generate it.  As such, the choice to provide a model that attempted
to recover from these
sorts of failures would mean burning significant code size and compile time for something
that isn't supposed to happen.

A final (and admittedly weak) reason for this approach is that a "too clean" cleanup runs the
risk that programmers will start treating fail-fast conditions as a soft error that
doesn't need to be handled with super-urgency.  We really do want these bugs to be found
and fixed in order to achieve the high reliability software systems that we seek.

## Part 4: Improving system architecture

As described in the motivation section, a single application process runs in the context of a
larger system: one that often involves multiple processes (e.g. an app and an XPC daemon)
communicating through [IPC](https://www.mikeash.com/pyblog/friday-qa-2009-01-16.html),
clients and servers communicating through networks, and
servers communicating with each other in "[the cloud](https://tr4.cbsistatic.com/hub/i/r/2016/11/29/9ea5f375-d0dd-4941-891b-f35e7580ae27/resize/770x/982bcf36f7a68242dce422f54f8d445c/49nocloud.jpg)" (using
JSON, protobufs, GRPC, etc...).  The points
of similarity across all of these are that they mostly consist of independent tasks that
communicate with each other by sending structured data using asynchronous message
sends, and that they cannot practically share mutable state.  This is starting to sound familiar.

That said, there are differences as well, and attempting to papering over them (as was done
in the older Objective-C "[Distributed
Objects](https://www.mikeash.com/pyblog/friday-qa-2009-02-20-the-good-and-bad-of-distributed-objects.html)" system)
leads to serious problems:

- Clients and servers are often written by different entities, which means that APIs must be
  able to evolve independently.  Swift is already great at this.
- Networks introduce new failure modes that the original API almost certainly did not
  anticipate.  This is covered by "reliable actors" described above.
- Data in messages must be known-to-be `Codable`.
- Latency is much higher to remote systems, which can impact API design because
  too-fine-grained APIs perform poorly.

In order to align with the goals of Swift, we cannot sweep these issues under the rug: we
want to make the development process fast, but "getting something up and running" isn't the
goal: it really needs to work - even in the failure cases.

### Design sketch for interprocess and distributed compute

The actor model is a well-known solution in this space, and has been deployed
successfully in less-mainstream languages like
[Erlang](https://en.wikipedia.org/wiki/Erlang_(programming_language)#Concurrency_and_distribution_orientation).
Bringing the ideas to Swift just requires that we make sure it fits cleanly into the existing
design, taking advantage of the characteristics of Swift and ensuring that it stays true to the
principles that guide it.

One of these principles is the concept of [progressive disclosure of
complexity](https://en.wikipedia.org/wiki/Progressive_disclosure): a Swift developer
shouldn't have to worry about IPC or distributed compute if they don't care about it.  This
means that actors should opt-in through a new declaration modifier, aligning with the ultimate
design of the actor model itself, i.e., one of:

```swift
  distributed actor MyDistributedCache { ... }
  distributed actor class MyDistributedCache { ... }
```

Because it has done this, the actor is now subject to two additional requirements.

- The actor must fulfill the requirements of a `reliable actor`, since a
  `distributed actor` is a further refinement of a reliable actor.  This means that all
  value returning `actor` methods must throw, for example.
- Arguments and results of `actor` methods must conform to `Codable`.

In addition, the author of the actor should consider whether the `actor` methods make
sense in a distributed setting, given the increased latency that may be faced.  Using coarse
grain APIs could be a significant performance win.

With this done, the developer can write their actor like normal: no change of language or
tools, no change of APIs, no massive new conceptual shifts.  This is true regardless of
whether you're talking to a cloud service endpoint over JSON or an optimized API using
protobufs and/or GRPC.  There are very few cracks that appear in the model, and the ones
that do have pretty obvious reasons: code that mutates global
state won't have that visible across the entire application architecture, files created in the file
system will work in an IPC context, but not a distributed one, etc.

The app developer can now put their actor in a package, share it between their app and their
service.  The major change in code is at the allocation site of `MyDistributedCache`, which
will now need to use an API to create the actor in another process instead of calling its
initializer directly.  If you want to start using a standard cloud API, you should be able to
import a package that vends that API as an actor interface, allowing you to completely
eliminate your code that slings around JSON blobs.

### New APIs required

The majority of the hard part of getting this to work is on the framework side, for example,
it would be interesting to start building things like:

- New APIs need to be built to start actors in interesting places: IPC contexts, cloud
  providers, etc.  These APIs should be consistent with each other.
- The underlying runtime needs to be built, which handles the serialization, handshaking,
  distributed reference counting of actors, etc.
- To optimize IPC communications with shared memory (mmaps), introduce a new protocol
  that refines `ValueSemantical`.  Heavy weight types can then opt into using it where it
  makes sense.
- A DSL that describes cloud APIs should be built (or an existing one adopted) to
  autogenerate the boilerplate necessary to vend an actor API for a cloud service.

In any case, there is a bunch of work to do here, and it will take multiple years to prototype,
build, iterate, and perfect it.  It will be a beautiful day when we get here though.

## Part 5: The crazy and brilliant future

Looking even farther down the road, there are even more opportunities to eliminate
accidental complexity by removing arbitrary differences in our language, tools, and APIs.
You can find these by looking for places with asynchronous communications patterns,
message sending and event-driven models, and places where shared mutable state doesn't
work well.

For example, GPU compute and DSP accelerators share all of these characteristics: the
CPU talks to the GPU through asynchronous commands (e.g. sent over DMA requests and
interrupts).  It could make sense to use a subset of Swift code (with new APIs for GPU
specific operations like texture fetches) for GPU compute tasks.

Another place to look is event-driven applications like interrupt handlers in embedded
systems, or asynchronous signals in Unix.  If a Swift script wants to sign up for notifications
about `SIGWINCH`, for example, it should be easy to do this by registering your actor and
implementing the right method.

Going further, a model like this begs for re-evaluation of some long-held debates in the software
community, such as the divide between microkernels and monolithic kernels.  Microkernels
are generally considered to be academically better (e.g. due to memory isolation of different
pieces, independent development of drivers from the kernel core, etc), but monolithic kernels
tend to be more pragmatic (e.g. more efficient).  The proposed model allows some really
interesting hybrid approaches, and allows subsystems to be moved "in process" of the main
kernel when efficiency is needed, or pushed "out of process" when they are untrusted or
when reliability is paramount, all without rewriting tons of code to achieve it.  Swift's focus on
stable APIs and API resilience also encourages and enables a split between the core kernel
and driver development.

In any case, there is a lot of opportunity to make the software world better, but it is also a
long path to carefully design and build each piece in a deliberate and intentional way.  Let's
take one step at a time, ensuring that each is as good as we can make it.

# Learning from other concurrency designs

When designing a concurrency system for Swift, we should look at the designs of other
languages to learn from them and ensure we have the best possible system.  There are
thousands of different programming languages, but most have very small communities, which
makes it hard to draw practical lessons out from those communities.  Here we look at a few
different systems, focusing on how their concurrency design works, ignoring syntactic and
other unrelated aspects of their design.

### Pony

Perhaps the most relevant active research language is the [Pony programming
language](https://www.ponylang.org).  It is actor-based and uses them along with other techniques
to provide a type-safe, memory-safe, deadlock-free, and datarace-free programming model.
The biggest
semantic difference between the Pony design and the Swift design is that Pony invests a
lot of design complexity into providing reference capabilities, which impose a high
learning curve.  In contrast, the model proposed here builds on Swift's mature system of
value semantics.  If transferring object graphs between actors (in a guaranteed memory safe
way) becomes important in the future, we can investigate expanding the [Swift Ownership
Model](https://github.com/apple/swift/blob/master/docs/OwnershipManifesto.md) to
cover more of these use-cases.


### Akka Actors in Scala

[Akka](http://akka.io) is a framework written in the [Scala programming
language](https://www.scala-lang.org), whose mission is to "Build powerful reactive,
concurrent, and distributed applications more easily".  The key to this is their well developed
[Akka actor system](http://doc.akka.io/docs/akka/current/scala/actors.html), which is the
principle abstraction that developers use to realize these goals (and it, in turn, was heavily
influenced by [Erlang](https://www.erlang.org).  One of the great things about
Akka is that it is mature and widely used by a lot of different organizations and people.  This
means we can learn from its design, from the design patterns the community has explored,
and from experience reports describing how well it works in practice.

The Akka design shares a lot of similarities to the design proposed here, because it is an
implementation of the same actor model.  It is built on futures, asynchronous message sends,
each actor is a unit of concurrency, there are well-known patterns for when and how actor
should communicate, and Akka supports easy distributed computation (which they call
"location transparency").

One difference between Akka and the model described here is that Akka is a library feature,
not a language feature.  This means that it can't provide additional type system and safety
features that the model we describe does.  For example, it is possible to accidentally [share
mutable state](https://manuel.bernhardt.io/2016/08/02/akka-anti-patterns-shared-mutable-state/)
which leads to bugs and erosion of the model.  Their message loops are also manually written
loops with pattern matching, instead of being automatically dispatched to `actor` methods -
this leads to somewhat more boilerplate.  Akka actor messages are untyped (marshalled
through an Any), which can lead to surprising bugs and difficulty reasoning about what the
API of an actor is (though the [Akka
Typed](http://doc.akka.io/docs/akka/2.5.3/scala/typed.html) research project is exploring
ways to fix this).  Beyond that though, the two models are very comparable - and, no, this
is not an accident.

Keeping these differences in mind, we can learn a lot about how well the model works in
practice, by reading the numerous blog posts and other documents available online,
including, for example:
- Lots of [Tutorials](http://danielwestheide.com/blog/2013/02/27/the-neophytes-guide-to-scala-part-14-the-actor-approach-to-concurrency.html)
- [Best practices and design patterns](https://www.safaribooksonline.com/library/view/applied-akka-patterns/9781491934876/ch04.html)
- Descriptions of the ease and benefits of [sharding servers written in Akka](http://michalplachta.com/2016/01/23/scalability-using-sharding-from-akka-cluster/)
- Success reports from lots of folks.

Further, it is likely that some members of the Swift community have encountered this
model, it would be great if they share their experiences, both positive and negative.


### Rust

Rust's approach to concurrency builds on the strengths of its ownership system to allow
library-based concurrency patterns to be built on top.  Rust supports message passing
(through channels), but also support locks and other typical abstractions for shared mutable
state.  Rust's approaches are well suited for systems programmers, which are the primary
target audience of Rust.

On the positive side, the Rust design provides a lot of flexibility, a wide range of different
concurrency primitives to choose from, and familiar abstractions for C++ programmers.

On the downside, their ownership model has a higher learning curve than the design
described here, their abstractions are typically very low level (great for systems programmers,
but not as helpful for higher levels), and they don't provide much guidance for programmers
about which abstractions to choose, how to structure an application, etc.  Rust also doesn't
provide an obvious model to scale into distributed applications.

That said, improving synchronization for Swift systems programmers will be a goal once the
basics of the [Swift Ownership
Model](https://github.com/apple/swift/blob/master/docs/OwnershipManifesto.md) come
together.  When that happens, it makes sense to take another look at the Rust abstractions
to see which would make sense to bring over to Swift.
