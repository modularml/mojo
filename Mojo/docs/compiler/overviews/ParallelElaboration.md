# Parallel Elaboration

## Introduction

The elaborator is individually one of the most complex passes in the KGEN
pipeline. This is the pass that transforms parametric IR to concrete IR, by
performing generator instantiation. This is akin to "template instantiation" in
C++, but with more things going on. The core algorithm of the elaborator is
conceptually a graph traversal, one that can be parallelized. This document
explains how parallel elaboration is implemented, because there can be quite a
lot going on and it gets hairy in a few places. This document assumes basic
knowledge on what elaboration is and what parameters are.

Elaboration can be neatly divided into 3 mostly orthogonal components:

- Parameter resolution
- Callgraph instantiation and multi-versioning
- JIT and evaluation

### Parameter Resolution

Parameter resolution refers to parameter instantiation from the perspective of a
single generator and its body. The body of a generator contains a parameter
graph, which describes the relations between parameter uses and definitions.
The graph itself can have multiple nested "scopes", but that concept will be
ignored for the sake of simplicity.

The parameter graph in relation to the generator can be thought of as a set of
equations, where the value of each parameter can be determined by evaluating a
"parameter expression" -- `add(x, 2)` is a parameter expression, for example,
so is `apply(:(index) -> index @mul2, 4)` -- given a set of input parameters
(incoming edges) on the generator.

```mlir
kgen.generator @foo<a, b -> e>() {
  kgen.param.declare c = <add(a, b)>
  kgen.call @other<c -> d>() : () -> ()
  kgen.param.result_bind<sub(d, b)>
  kgen.return
}
```

In the above example, the parameter equations are:

- `c = a + b`
- `d = other(c)`
- `e = d - b`

The call to `@other` can be thought of as evaluating a parametric function.
Instantiating the generator with given input parameters is simply a matter of
plugging those input values into the equations and computing all other declared
parameters (and then substituting the computed values throughout the body of the
generator: inside ops, attributes, and types).

### Callgraph Instantiation and Multi-Versioning

The previous section describes how parameter resolution works at the level of a
generator, but a larger concern for the elaborator is how the callgraph is
processed and instantiated and how this interacts with multi-versioning, a
unique feature of KGEN generators.

The pre-elaboration callgraph forms edges between parametric generators. The
post-elaboration callgraph forms edges between concrete functions.

```mlir
kgen.generator @parametric<a>() -> index {
  %0 = kgen.param.constant = <a>
  kgen.return %0 : index
}

kgen.generator @main() {
  %0 = kgen.call @parametric<1>() : () -> index
  %1 = kgen.call @parametric<2>() : () -> index
  kgen.return
}
```

In the above example, the parametric callgraph has 2 nodes: `@main` and
`@parametric<a>`. The concrete callgraph will have 3: `@main`,
`@parametric<1>`, and `@parametric<2>`, with edges extending from `@main` to
the latter two. The "expansion" of the callgraph forms the backbone of the main
elaborator algorithm, and is what we're interesting in parallelizing.

The elaborator creates two versions of `@parametric<a>`, with two different
sets of input parameters. However, KGEN also allows a generator with a given
set of input parameters to have multiple (or zero) possible instantiations,
resulting from a fork:

```mlir
kgen.generator @parametric<a>() -> index {
  kgen.param.fork result = <[a, add(a, 1)]>
  %0 = kgen.param.constant = <result>
  kgen.return %0 : index
}

kgen.generator @main() {
  kgen.call @parametric<1> : () -> ()
  kgen.return
}
```

In this example, the generator `@parametric<a>` is forked on two possible values
for the parameter `result`. How is this resolved at `@main`? The fork is
propagated all the way up the "expansion" graph, resulting in two different
versions of main:

```mlir
kgen.generator @"parametric,a=1,result=1"() -> index {
  %0 = kgen.param.constant = <1>
  kgen.return %0 : index
}

kgen.generator @"parametric,a=1,result=2"() -> index {
  %0 = kgen.param.constant = <2>
  kgen.return %0 : index
}

kgen.generator @"main,parametric,a=1,result=1"() {
  kgen.call @"parametric,a=1,result=1"() : () -> ()
  kgen.return
}

kgen.generator @"main,parametric,a=1,result=2"() {
  kgen.call @"parametric,a=1,result=2"() : () -> ()
  kgen.return
}
```

The important thing is that the fork can happen during the parametric resolution
of the body of any generator, meaning that parameter resolution has to resume
on both resulting versions of the generator from the same place.

Processing the "expansion" graph in parallel is the core part of this document,
so I will skip over details for now.

### JIT and Evaluation

The astute will note that if the elaborator needs to propagate forks up the
callgraph, where does it end? It ends in evaluation! Evaluation is where one
concrete function is selected from multiple concrete functions of the same
signature. These "candidate" functions can be entirely different functions or
be different instantiations of the same function, like above. Evaluation
typically involves benchmarking (for speed) the different implementations, but
the evaluator itself is user-written code. The evaluator, as well as the
candidate functions, are JIT'd by KGEN `ExecutionEngine`, and then the functions
pointers of the evaluator are passed to the evaluator, which can then do
whatever it wants, but it must return the index of the selected candidate.

```mlir
kgen.generator @two_candidates() {
  kgen.param.fork N = <[1, 2]>
  kgen.return
}

kgen.generator @evaluator(%fns: !kgen.pointer<() -> ()>, %sz: index) -> index {
  // Always pick the second one.
  %idx1 = index.constant 1
  kgen.return %idx1 : index
}

kgen.generator @foo() {
  kgen.param.evaluate selected: () -> () = [@two_candidates]
    with [(!kgen.pointer<() -> ()>, index) -> index: @evaluator]
  kgen.call_param[() -> (): select]()
  kgen.return
}
```

In this example, the evaluator was stubbed out. In general, the only concern
with the evaluator is that the search must be performed in isolation as much as
possible. If the compiler is compiling code at the same time other code is
being benchmarked, the results will be inaccurate.

## Core Algorithm

Neither the pre-elaboration callgraph nor the expansion graph are fully known at
any point elaboration is not complete. What instantiations ultimately arise from
a given generator is a function of the input parameters. For example:

```mlir
kgen.generator @pickInstantiation<c: i1>() {
  kgen.param.if <c> {
    kgen.call @foo<1>() : () -> ()
    kgen.param.yield
  } else {
    kgen.call @bar() : () -> ()
    kgen.param.yield
  }
  kgen.return
}
```

We are given the root nodes of the callgraph and expansion graph, the so-called
"primary generators", which correspond to exported functions or, typically,
`main`. Visiting a node and performing parameter resolution reveals the outgoing
edges of the expansion graph. Importantly, generator instantiation is a process
naturally recursive with parameter resolution:

```mlir
kgen.generator @foo<() -> out>() {
  kgen.param.result_bind<1>
  kgen.return
}

kgen.generator @main() {
  kgen.call @foo<[] -> out>() : () -> ()
  %0 = kgen.param.constant = <out>
  kgen.return
}
```

Parameter resolution of `@main` will reveal `out = foo()` and will have to
recurse into the instantiation of `@foo`. Once `@foo` is instantiated, pop back
up to `@main` and continue parameter resolution. One can see that the basic
algorithm for elaboration is:

1. Visit a generator with given input parameters and start parameter resolution
   by creating a function with the same body.
2. Solve the parameter equation in the function given the concrete inputs.
3. Upon encountering a generator instantiation, typically a call operation,
   recurse into that generator if it has not already been visited.
4. When that generator is complete, pop back up and continue parameter
   resolution.

This will traverse and reveal the expansion graph in depth-first (DF) order. In
order to handle recursion, each generator instantiation can be indicated as
already in-progress upon visitation, and then visiting an in-progress
instantiation indicates recursion. Recursion can be broken appropriately: in the
simple case, by directly referencing the in-progress concrete function.

## Forking

In the previous section, we see how a parametric generator can spawn multiple
instantiations and that edges in the expansion graph are between instantiations:
a pair of a generator and its input parameters. However, with forking, each
generation instantiation itself can have multiple implementations. In the
elaborator, a generator instantiation is represented as a `ParamNode` and each
of its implementations is an `ImplNode`.

Processing of a generator instantiation is considered complete when processing
of all its children implementations is complete. When a generator instantiation
is visited the first time, a single `ImplNode` is created to represent the
initial state of elaboration. Suppose the elaborator encounters a fork:

```mlir
// The generator inside the `ParamNode`.
kgen.generator @foo<a>() {
  kgen.param.fork N = <[a, add(a, 1)]>
  %0 = kgen.param.constant = <N>
  kgen.return
}

// The state of `foo<1>` when the fork is encountered.
kgen.func @"foo,a=1"() {
  kgen.param.fork N = <[1, 2]>
  %0 = kgen.param.constant = <N>
  kgen.return
}
```

The elaborator clones the current `ImplNode`, including the current function
body and all its elaboration state:

```mlir
// The original implementation.
kgen.func @"foo,a=1,N=1"() {
  kgen.param.declare N = <1>
  %0 = kgen.param.constant = <N>
  kgen.return
}

// The forked implementation.
kgen.func @"foo,a=1,N=2"() {
  kgen.param.declare N = <2>
  %0 = kgen.param.constant = <N>
  kgen.return
}
```

The elaborator will continue processing the current `ImplNode`, but upon
completion, it will continue looping over other incomplete implementations until
no new forks occur and all implementations are done. At this point, the parent
`ParamNode` is considered complete.

When elaboration of a function encounters a generator instantiation with
multiple implementation, like `@foo<1>` above, that function is then forked in
the exact same way:

```mlir
// Multiple implementations of `@foo` are propagated by multi-versioning
// `@someFunc` as well.
kgen.func @"someFunc,foo,a=1,N=1"() {
  kgen.call @"foo,a=1,N=1"() : () -> ()
  kgen.return
}

kgen.func @"someFunc,foo,a=1,N=2"() {
  kgen.call @"foo,a=1,N=2"() : () -> ()
  kgen.return
}
```

As previously discussed, fork propagation can be broken at a "search root",
represented by a `kgen.param.evaluate` operation, that reduces, for example, a
generator instantiation with multiple implementations down to 1.

## Errors and Propagation

Elaboration can fail when processing an implementation and when processing
a generator instantiation. This can occur for a variety of reasons, but
typically occurs due to user-written static asserts, such as:

```mlir
kgen.generator @foobar<a>() {
  kgen.param.assert <gt(a, 1)>, "'a' must be bigger than 1!"
  kgen.return
}
```

The elaborator will create the first implementation, but when processing that
function, will encounter the static assert and it can fail. If it fails, the
implementation is considered failed. If all implementations of a generator
instantiation fail, it also fails. A function that references a generator
instantiation that fails also fails, etc. Note that if a generator has multiple
instantiations, only the succeeded implementations will be used for forks.

For instances, the following example will only produce 1 candidate:

```mlir
kgen.generator @baz() {
  kgen.param.fork a = <[1, 2]>
  kgen.call @foobar<a>() : () -> ()
  kgen.return
}
```

Processing `@baz` will create 2 implementations, but the one where `a = 1` will
fail because instantiation of `foobar<1>` will fail. Thus, `@baz` will only have
1 valid implementation.

### Constraints

Generators can directly carry "constraints", which are static asserts that are
functions only of the input parameters:

```mlir
kgen.generator @foobar<a>()
    constraints <[gt(a, 1), "'a' must be bigger than 1!"]> {
  kgen.return
}
```

This is essentially a compile-time optimization, because if a generator
instantiation can be checked for validity without cloning the body and starting
elaboration on an implementation, elaboration here will fail faster.

## State Saving and Restoration, Suspension, and Recursion

The previous sections outline the components of the full elaboration algorithm.
There are a few skipped details, like bindings, but the overall picture is
there. There are two problems:

1. Recursion will stackoverflow the compiler.
2. A recursive, DF algorithm is not easy to parallelize.

Recursing into generator instantiations is a natural way to write the algorithm,
but it can easily cause the elaborator to stackoverflow if, for instance, the
user writes lots of code; callgraphs can be very deep. It also is not a form
that is easy to parallelize. So the first step is to reformulate the algorithm
to be iterative.

As part of forking elaboration of an implementation node, we already save the
elaboration state of a node. With a bit of massaging, moving state from the
callstack onto `ImplNode`, we can rewire the recursive part of the algorithm to
"suspend" and bail out elaboration of an `ImplNode`, go process the
instantiation, and then re-queue the suspended `ImplNode` when it completes.
This gives us the basis of parallelization, because this step can be done
asynchronously.

To recap: when elaboration hits a novel generator instantiation, the current
implementation being processed is suspended and added as a waiter on the new
instantiation. That instantiation is put on the queue, and when it completes,
all the waiters are pushed back on the queue and resumed.

### `apply`

One wrinkle is that generator instantiations can be nested **anywhere** in a
parameter expression: inside types and attributes of any operation. This means
that elaboration of any operation could potentially suspend. In reality, this
does not happen, because `lift-and-fold-apply` will pull the `apply` operators
out to `kgen.param.apply`. This handling should get revisited.

## Recursion

In an earlier section, we discussed how to handle recursion when the graph is
traversed in DF order -- it's as simple as setting a "visited" flag. There are
several additional problems with recursion:

1. A recursive generator instantiation cannot have multiple implementations.
2. A recursive generator instantiation cannot have result parameters.
3. Bindings require fixed-point iteration to propagate in a cycle.

If a recursive generator instantiation has multiple implementations, forking
will blow up to infinity. This is regardless of how many are valid: the
elaborator doesn't know which implementations are valid until elaboration of the
recursive instantiation is complete, which requires assuming all are valid.

If a recursive generator has result parameters, then this introduces a cycle in
the parameter use-def graph at the expansion graph level. Technically, the
following is well-defined:

```mlir
kgen.generator @foo<() -> x>() {
  kgen.call @foo<[] -> y>()
  %0 = kgen.param.constant = <y>
  kgen.param.result_bind<2>
  kgen.return
}
```

However, separation of concerns means that the elaborator will not penetrate the
abstraction of the generator's parameter equations in order to support what is
an extremely rare/nonexistent use-case.

Finally, bindings in a cycle are something the elaborator "gives up" on because
correctly propagating bound functions in a cycle requires fixed-point iteration,
which is not possible when the full graph is not known!

The other problem is that detecting recursion when traversal is not performed
depth-first is quite tricky. And this is important because when parallelization
is introduced, the graph will not be traversed depth-first.

## Path to Parallelization

The core thing to decide when parallelizing the elaborator is what the core
"task" is. We have already established this: processing an implementation node
is the main task that will be parallelized. These tasks can be suspended,
spawn other tasks, and be resumed. If only C++17 had coroutines!

This means we can elaborator starting from the root nodes in parallel and
elaborate forks in parallel. The thinking here is straightforward: each
`ParamNode` keeps an atomic representing the number of in-progress `ImplNode`.
When each completes, the atomic is decremented and if it hits zero, the parent
`ParamNode` is completed and all tasks waiting on it (kept via the waiter list
of an `AsyncValueRef<Chain>` are resumed). Rinse and repeat. The status of a
`ParamNode` also has to be atomic, because only one task is allowed to kick off
specialization of a generator.

One of the big headaches comes with handling search and cycles. We can no longer
rely on depth-first traversal to find cycles, and we have to ensure that search
is performed in isolation at least in the process of the compiler. That is,
we cannot benchmark JIT'd code at the same time the elaborator is compiling
code! This requires the elaborator to ensure that at least with respect to
elaboration, no other tasks are running. This is called "exhausting the
workqueue"; the elaborator uses AsyncRT as a virtual workqueue, but has to track
how many active tasks there are.

Each time a task is scheduled, the number of active tasks is incremented. Each
time a task is completed, the number of active tasks is decremented. To prevent
the counter from transiently reaching zero, when a `ParamNode` completes, it has
to increment the number of tasks released before emplacing its chain. The
"number of waiters" and the status of a `ParamNode` have to be modified
transactionally: this is done by munging both together into an atomic.

### Exhausting the Workqueue and Search

The main thread of the elaborator runs until the worklist has been exhausted:
when the number of work items hits zero, a chain is emplaced. The main thread
awaits this chain. In order to ensure search is performed in isolation, when a
`kgen.param.evaluate` operation is being processed, it issues the compile
command in parallel to produce a functor that runs the search. This functor is
saved in the elaborator and the current task is suspended. This means that the
workqueue will eventually exhaust itself without completion elaboration of the
whole graph. The main thread can diagnose this by checking whether all primary
generators are done. If not, it will check if there are "deferred" search
functions, and the main thread processing them serially, releasing the suspended
tasks.

### Cycle Detecting and Recursion

The elaborator can no longer assume that visiting an in-progress generator
instantiation indicates recursion, because it likely means another thread is
processing the `ParamNode` at that time. This means a cycle will cause the
workqueue to exhaust without deferred search and without completing all primary
generators.

In this situation, the main thread can perform a trimmed DFS from the incomplete
primary nodes on to in-progress nodes to find where cycles occur and break them
(or throw an error if the aforementioned recursion requirements are not
satisfied). Breaking the edges releases more tasks, and the main thread runs in
a loop until this is complete.

### Unlocking Full Parallelism

As written, the parallel algorithm will process primary generators and forks in
parallel, meaning in the given example:

```mlir
kgen.generator @main() {
  kgen.call @foo<1>() : () -> ()
  kgen.call @foo<2>() : () -> ()
  kgen.call @bar() : () -> ()
  kgen.return
}
```

Elaboration of `@main` will encounter `@foo<1>`, suspend and wait for
elaboration of `@foo<1>` to complete, resume and encounter `@foo<2>`, suspend
and so on. This is leaving lots of parallelism on the table, because elaboration
of `@foo<1>`, `@foo<2>`, and `@bar` can be done in parallel. The final step is
to dispatch all generator instantiations encountered during elaboration of a
function asynchronously.

Notably, this can only be done for generators with no result parameters:
finer-grain parallelism can be achieved even within the parameter use-def graph,
but result parameters are such an uncommonly-used feature that it isn't
worthwhile.

Importantly, if any async generator instantiations result in multiple
implementations, the forking cannot be performed while the elaborator is still
processing the body. It must be deferred until that is completed. We introduce a
second kind of task: `ImplNode` completion. When elaborating a function,
asynchronously dispatched instantiations are tracked as incomplete
"dependencies" in an atomic counter starting at 1. Each time a dependency
completes, the counter is decremented. When the elaborator finishes processing
everything else in the function body, the counter is decremented once, allowing
the completion task to run. The completion task takes all the complete
dependencies and processing them in order, generating forks and scheduling
completion tasks for them as necessary. The completion tasks waiting on a
`ParamNode` also have to be added as one of the waiters.

## Conclusion

This document provides a high-level overview of the parallel algorithm powering
the elaborator. It hopefully provides enough detail for someone to navigate the
weeds of the elaborator. There are many details left out of the document, but
the core algorithm and how it works has been described.
