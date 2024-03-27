## Pain Points Addressed

Actors were invented to solve some problems in the 1970s world of uni-core
processors. As [William Clinger](https://en.wikipedia.org/wiki/William_Clinger_(computer_scientist)) said in 1981, the actor model's 
development was:
> "motivated by the prospect of highly parallel computing machines consisting 
> of dozens, hundreds, or even thousands of independent microprocessors, each
> with its own local memory and communications processor, communicating via
> a high-performance communications network."
> (William Clinger (June 1981). "Foundations of Actor bemantics". Mathematics 
> Doctoral Dissertation. MIT. hdl:1721.1/6935.)

We've been in that world since about 2005 (multi-core processors).

## What Actors Have Achieved
Here is a summary of how Actors have helped since 1973:
* avoid shared mutable state via complete data encapsulation (using the
  single writer principle)
* achieve parallelism by using multiple computers attached by networking
* achieve parallelism by using multicore processors
* prove that any system that required shared mutable state can be designed
  with actors to avoid the downsides of that approach.
* completely eliminate the possibility of deadlock (locks are not used)
* completely eliminate hard-to-debug concurrency problems that plague
  development because of semantic inconsistencies introduced when supporting
  shared mutable state goes wrong.
* completely eliminate race conditions that occur because of shared mutable state.
* completely eliminate the need for synchronization primitives (locks, mutexes,
  semaphores, etc.) and all their slow evils because there's no shared mutable
  state.
* completely eliminating the need to find the optimal `synchronized` block to
  reduce blocking latency.
* avoiding the complexity, fragility, unsafety and cognitive load of dealing with
  performance problems that synchronization primitives cause (mostly blocking
  CPUs while waiting for a lock). As the [prior work](https://gist.githubusercontent.com/lattner/31ed37682ef1576b16bca1432ea9f782/raw/dc3b70690a7ba3bed709d3043ad312eadb53d553/TaskConcurrencyManifesto.md#shared-mutable-state-is-bad-for-software-developers)
  suggests, this includes avoiding:
    - [readers-writer locks](https://en.wikipedia.org/wiki/Readers–writer_lock),
    - [double-checked locking](https://en.wikipedia.org/wiki/Double-checked_locking),
    - low-level [atomic operations](https://en.wikipedia.org/wiki/Linearizability#Primitive_atomic_instructions),
    - and advanced techniques like
      [read/copy/update](https://en.wikipedia.org/wiki/Read-copy-update).

You might have noticed how problematic shared mutable state is. So what do we
mean by that? "State" is simply data that is used by a program. "Mutable state"
is state that can be modified as opposed to being constant. "Shared mutable state"
is state that can be modified by multiple tasks, possibly in parallel. The
problems start when the "shared mutable state" is modified in parallel. Each task
may walk away from their mutation thinking they know the value of the state. As
long as there is only one writer, there can be many readers without issue. This
is why the readers-writer lock exists. Note that reader is plural, writer is not.
But why not just avoid the wide range of problem introduced by multiple-writers
and only have one? Known as the _single writer principle_, it has applicability
in systems programming, concurrent application programming, databases, CQRS,
CRDTs, cluster membership cohesion, and many other areas.

For more on this and related ideas, please consult:
* [Is Parallel Programming Hard, And, If So, What Can You Do About It?](https://mirrors.edge.kernel.org/pub/linux/kernel/people/paulmck/perfbook/perfbook.2022.09.25a.pdf)
* [Understanding Parallel Computing (Part 2): The Lawnmower Law](https://www.youtube.com/watch?v=ehyO7mxeU74&pp=ygUbTGludXggY29uY3VycmVuY3kgbGF3bm1vd2Vy)
* [Concurrency: the cause of, and solution to, lots of problems in computing.](https://www.youtube.com/watch?v=LDPUWGLjHIg)

## Reduced Developer Cognitive Load

Programming highly complex concurrent systems is hard for human brains. But, 
broken down into Actors, only the transition of state from one message to the
next need be considered. This aspect of Actors has helped reduce the cognitive
load on development teams, which has a side benefit of increased productivity. 
