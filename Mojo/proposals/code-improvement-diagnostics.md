# Mojo's approach to code improvement diagnostics

Author: Chris Lattner, Laszlo Kindrat
Date: Aug 15, 2025
Status: Accepted.

This document aims to capture the philosophy behind Mojo's approach to code
improvement tools. The document tries to refrain from specific implementation
details wherever possible and instead focus on ideas, guidelines, and prior art.

## What are code improvement diagnostics?

There are several places when a compiler may want to tell the user that they
should change or upgrade their code. For example:

1. The context within a parser makes it easy to identify some suspicious code —
  e.g. something like `if (x = y) {` in a C-style language (probably intended to
  be a comparison instead of an assignment) or `some_var = foo()` in Mojo with
  no use of `some_var` (maybe it was intended to be `somevar = foo()`). The
  compiler should warn about these to help the user find likely-incorrect code
  and fix it fast.
2. There are some “lint” style warnings that could similarly be potential bugs,
   or could just be left-over stuff after refactorings, e.g. an unused variable
   or unused import. It is helpful if the compiler identifies these, and this
   can require significant symbolic analysis (e.g. integration with name lookup)
   to do correctly.

3. Mojo will eventually care about helping users upgrade their code as the
   language and library evolve. Instead of simply renaming a simple `foo` to
   `bar` unilaterally, we might want the parser to produce an error or warning
   on uses of `foo` saying that it is deprecated, plus offering a replacement to
   `bar`. Similarly, when renaming keywords (like `inout` to `mut`) the compiler
   should be able to help migrate code.

The common theme across all of these scenarios is that the compiler is able to
detect a likely issue and provide assistance to the user in the form of a
warning or error that contains a **direct code rewrite**. There is a very
general set of rewrites that can be possible — dozens or hundreds of these, and
there are also an orthogonal set of policies about what to do with these:
**we don’t want a compiler to unilaterally change code** on disk, but we do have
some cases where it would be very handy to auto-apply them.

This document explores these issues, proposing that we build on lessons learned
from the Swift and Clang compilers, utilizing existing infrastructure in Mojo to
structure this, and use this to guide the development of future source code
improvement tools.

Note that this document is scoped to compiler-integrated code rewrites. LLMs and
AI tools are also very important and powerful tools, but are out of scope of
this document. It is worth highlighting, though, that the above principles
dovetail nicely with AI coding assistants; these tools can pick warnings and
automate their resolution with more context taken into account than traditional
tooling (e.g. reason about when an unused import should be removed or the
warning suppressed locally, based on where the file is, or what the style guide
says).

## How Clang/Python are different than Swift/Mojo

When learning from the past, it is important to understand how different systems
we may learn from face different challenges. There are many existing systems out
there, but for the sake of this document, we’ll look at Clang/Python and argue
that Mojo faces challenges more similar to Swift (and Go etc).

### Clang/Python: support for mature and uncooperative ecosystems

The modern C++ and Python ecosystems face a common challenge: they are very
mature, with many millions of lines of code in existence. Furthermore, there are
many different standards for code in the wild, as well as multiple
implementations of C++ compilers and Python tools (e.g. CPython, formatters,
linters, type checkers) that do not agree on all the details.

As a consequence of this, the core Clang and CPython tools face a challenge:
both are rapidly evolving and want to continuously improve developer experience,
but both face the challenge of legacy code bases and fragmented communities.
These two languages address the problem in different ways: Clang uses language
dialect flags, and Python uses a federated set of tools (black, ruff, pylint,
flake8, etc).

#### Clang’s evolving sea of compiler flags

Clang addresses this problem by having an evolving sea of compiler flags that
control its policy. This includes big flags like what language standard is being
parsed, e.g. `-std=c++11`, myriad flags to
[enable and disable specific warnings](https://xs-labs.com/en/blog/2012/01/10/warning-flags-clang/),
as well as a wide range of other flags to support various
[team-specific policies](https://clang.llvm.org/docs/UsersManual.html#command-line-options)
and dialects of C++ — e.g. it is common for some code bases to use “C++, but
without exceptions or RTTI”.

This approach makes sense for Clang, because C++ is mature, there is no single
shared C++ implementation (Clang, GCC and MSVC all have significant market
share), and there is a huge body of code that “works” even if it is dubious. The
authors of Clang cannot just add a new warning unilaterally, because it may
trigger thousands of times on an existing codebase. As a consequence, Clang
grows a wide variety of flags to disable these. This
[enabled Clang to innovate](https://clang.llvm.org/diagnostics.html)
even back when the GCC compiler was the dominant implementation.

This is a pragmatic solution given the constraints that Clang faces, but using
this approach for Mojo has a downside — it encourages and enables house
dialects, makes the compiler more complicated, and reduces incentive to improve
the language and compiler.

#### Python’s evolving sea of external tools

The Python ecosystem faces a similar challenge of mature code bases with
team-specific idioms and house dialects. While there is a single primary CPython
implementation with significant market share (unlike C++), there are many
versions of CPython in the wild and code aspires to be compatible with older
versions for a long time. Python faces two additional challenges:

- It is an interpreted language without strong static types — it has grown a
  variety of typing systems that are useful in finding bugs in Python code, but
  are unsound and do not agree with each other.
- It encourages dialects (and even embedded DSLs!) via powerful reflection, and
  syntactic overload features (e.g. decorators, metaclasses, dunder methods).

The consequence of these is that the Python community has developed a wide range
of third party formatter and type checking tools. This allows teams to have
choice in their tooling, and decouples the core Python interpreter from having
to be opinionated about these things.

This approach is pragmatic and makes sense for Python, but it would have a
downside for Mojo: the ecosystem could become fragmented, and the community’s
effort dispersed, instead of being concentrated on a single superior solution.
Building and maintaining the canonical tooling will put more burden on us, but
open sourcing the compiler (along with the tooling suite) can help with this.

### Swift: A unified single-vendor community

Swift faced a different challenge than C++ or Python with pros and cons that led
to a different approach. The advantage that Swift has in this space is that it
is owned and driven by a single vendor, with the implementation of Swift setting
the standard instead of an external standards body. Furthermore, Apple
incentivizes developers to keep on the latest release of Swift by tying it into
their evolving Apple SDK.

While this is a great thing for Swift, Swift faced a challenge particularly in
its early days: the language did evolve extremely rapidly and in ways that broke
the software ecosystem: Swift 1.1 was incompatible with Swift 1.0 code, and
later versions of Swift made major changes to how the iOS SDK was imported into
Swift, requiring massive (but mechanical) changes to application code that had
to adopt new versions of the tools to get newer SDK features.

Swift tackled this problem with a number of approaches:

1. It maintained an *opinionated* stance on what “good” Swift code looked like
  (e.g. similar to how Go is opinionated about formatting) which kept the
  ecosystem more consistent.
2. It avoided adding many fine-grained flags to control compiler behavior,
   instead providing the ability to silence specific instances of warnings in
   source code. It did later add course grain settings like “build in Swift 5
   mode” (vs Swift 6).

3. The compiler uses a structured notion of a “FixIt” that the parser can
   optionally attach to any compiler error or warning message. FixIts indicate a
   mechanical source code rewrite that can resolve the issue. The compiler
   generates FixIts for many common issues as well as language changes.

4. Swift developed languages features enabling API authors to specify rewrites
   to use when evolving APIs, e.g. you can deprecate a symbol with information
   so the compiler knows how to change the code with a FixIt (in simple cases,
   like a rename).

5. Swift-aware tools like IDEs and build systems got features to integrate with
   these. For example, Xcode added a
   [“Fix” button to error and warning messages](https://www.dummies.com/article/technology/programming-web-design/app-development/how-to-use-fixit-to-correct-swift-code-144658/),
   and supported an “automatically apply FixIts” mode. Going further, when a
   developer moved to a new version of Xcode, it would offer to auto-apply all
   FixIts to automatically migrate your code to a new version of the Swift. This
   greatly reduced (but did not eliminate) the cost of language changes for
   Swift developers.

Swift and Xcode are not unique here.
[Clang also supports FixIts](https://stackoverflow.com/questions/49748996/apply-clangs-fix-it-hints-automatically-from-command-line)
and all common IDEs (as well as the LSP specifications) have been updated to
support these features as well

Mojo and Swift face structurally the same opportunity and challenge — we want to
keep the language and community consistent while the language is young, and we
can use great tools and a central implementation to enable that. Note that
Python’s success can be attributed to partly doing the opposite: the diversity
of stylistic and policy choices one can make when writing Python has enabled
developers of vastly different backgrounds and interests to use it (e.g. data
scientist vs. backend web developers). The key here is that we want the reduce
fragmentation while the language and the ecosystem is evolving the fastest.
Eventually, Mojo will blossom and may even have other implementations, and an
ecosystem with abundant choices for tooling.

## Proposal: Code improvement diagnostics for Mojo

This document advocates adopting the Swift approach, and describes how we
already have some of these features, but highlights some of the places that can
be expanded over time. Let’s dive into some of the components.

### Mojo FixIts: already supported

The Mojo compiler already has the infrastructure to process and propagate FixIt
hints, and does use them in a few places, here is one trivial example:

```python
$ cat a.mojo
def returns_int() -> Int: return 4
def test():
    returns_int()

$ mojo a.mojo
/tmp/a.mojo:3:16: warning: 'Int' value is unused
    returns_int()
    ~~~~~~~~~~~^~
  _ =
```

FixIts coming out of the compiler do so with a structured format, but the `_ =`
text on the final line is how the compiler is rendering this FixIt information
to the terminal. FixIts internal to the compiler handle source code insertions,
deletions and changes in a structured way using the `M::FixIt` C++ class.

FixIts are already plumbed
[through the Mojo LSP](https://github.com/modularml/modular/blob/2b6b6b8e1517b6f6c52e4902a9a0a5213121775c/KGEN/tools/mojo-lsp-server/MojoServer.cpp#L1027),
so VSCode and other IDEs can provide the same “fix” button and similar logic for
Mojo that it does for C++ and Swift.

### Extending `mojo build` to support an “auto-apply” mode

To support migration use-cases, we should add support for a bulk-apply mode
built into mojo build and similar tools. Clang has a
[collection of flags](https://stackoverflow.com/questions/49748996/apply-clangs-fix-it-hints-automatically-from-command-line)
that may be a source of inspiration.

## Error and warning policies

Given the core infrastructure and the tooling support to enable code migrations,
we can then build into these. We aim to provide a consistent experience for Mojo
developers and have to decide the severity of various issues and how to handle
them. This section explores some examples to help frame up a set of policies for
how we can handle different classes of issues.

To help motivate this section, let’s consider one simple warning that Mojo
already produces, diagnosing unused variables:

```python
$ mojo a.mojo
/tmp/a.mojo:5:11: warning: assignment to 'x' was never used; assign to '_' instead?
  var x = foo()
          ^
```

There are lots of errors and warnings, but this gives us one concrete example to
dig into some of the policies that we’ll have to hammer out.

### Diagnostics can always be improved

There is a never-ending question to improve diagnostics in the Mojo compiler,
and there will never be a “good enough” point. For example, this doesn’t include
a FixIt to rewrite it to `_ = foo()`. Mojo also doesn’t support diagnosis of
unused functions and structs (though can we? it isn’t clear that our packaging
system allows this, because they’re all exported), nor do we support diagnosis
of unused imports. Each of these can be fixed on-demand in a case-by-case basis
and independently prioritized based on impact.

### Warnings should be stable for a given compiler version

It is very valuable to users of a compiler for errors and warnings to be
*stable* for the compiler, across unrelated compiler options. Not all language
implementations have this property: GCC notably runs various optimization passes
(e.g. inlining) before doing data flow checks. This means that the warnings you
get out of GCC can vary *based on the optimization level* you build code.

There are reasons for the GCC behavior that apply the C code, but Clang made the
opposite decision and there was no downside. Furthermore, more modern languages
(Go, Swift, Rust etc) have kept stable errors and warnings as well, to good
effect. We should keep Mojo stable here, and do this by only producing warnings
and errors out of early stable diagnostic passes (e.g. the parser and
lifetime checker) that are run in predictable order without optimizations
ahead of them.

Note that there can be additional categories of diagnostics that may want deeper
analysis of the code, and may want to run deep in the optimizer. LLVM-based
compilers like Clang have
“[auto vectorization remarks](https://llvm.org/docs/Vectorizers.html#diagnostics)”
which necessarily must be produced late in the optimization pipeline, and thus
inherently depend on optimization level. Clang handles this by making them
“remarks” instead of “warnings” or “errors”.

### Should unused code be an error or warning?

Some languages, notably go-lang, treat unused variables like this as an error
instead of a warning (and expects developers to solve this with tooling that
automatically adds/removes the right imports). This is strictly a policy choice,
and while Mojo currently treats these as a warning, we could certainly choose to
follow go’s
([controversial](https://www.reddit.com/r/golang/comments/16mzz3n/unused_variable_rant/))
approach.

Here is the rationale for Mojo’s current behavior (which is also used by Swift):

- It is very common to iteratively refactor and “hack on” code in intermediate
  states, and it can be very annoying to get a build error, e.g. when you
  comment out some code temporarily to run an experiment.
- Much of the reason for rejecting this is because teams may not want dead code
  in production. That is a good reason to want to check this, but that rationale
  can be applied to anything that produces a warning.
- Unused code is primarily a linting problem (not correctness), and having
  unused import *errors* while refactoring packages would likely be extremely
  annoying.

For now, we advocate for not changing Mojo’s behavior, and keep these a warning
by default. At the same time, we should be intentional about when a warning
should become an error, e.g. using a `-Werror` style flag. This can keep
engineers productive while prototyping, while allowing these to be blocking
errors when in CI jobs.

### Silencing individual warnings

We want warnings to be deterministic, but sometimes it can be better for code
readability to use idioms that we’d normally want to warn about. For example:

```python
def loops():
    for i in range(100):
        var sum = 0
        # warning: assignment to 'j' was never used; assign to '_' instead?
        for j in range(100):
            for k in range(100):
                sum += process(i, k)
        use(sum)
```

In this case, Mojo will produce a warning that says that `j` is unused and
should be changed to `_`. This is a great warning, because maybe the code used
the wrong variable instead of `j` and there is a bug in the code! That said,
it may also be that the author intended this and the warning is just a nuisance.
We don’t want warnings in production, both because it will hide other issues and
clutter up continuous integration

There are two structural ways to solve this sort of problem:

1. We can provide a comment (or something similar) to disable warnings on a line
  of code, e.g. use `for j in range(100): # nowarn`.

   - This has the advantage that it can become a general feature for silencing
     an arbitrary warning, at the expense of having to parse comments.
   - Inline comments like this can hide multiple warnings on a single line,
     which can unintentionally lead to hiding issues that should be fixed some
     other way.

2. We can provide a per-warning idiom for disabling the code. In this case, for
  example, Mojo doesn’t warn about identifiers that start with underscore, so
  you can silence the warning with `for _j in range(100):`.

   - This does not require parsing comments, but it will require having a set
     of different idioms for different errors (see example below on how
     `(a = b)` warnings are silenced by clang). Any other tooling (e.g.
     formatter) will need to understand when these are intentional or not.

Both approaches can be good because they make it clear to the reader of the code
that this was done intentionally, not by an accident of omission. Either can be
attached as a FixIt to a note, so downstream tooling can use it.

### FixIt’s shouldn’t change code behavior

Given it is useful to have a “bulk apply all FixIts” mode in tooling, we need to
decide how to handle varying levels of certainty about a fix. One very important
invariant is that a FixIt should always maintain the behavior of the code as
written, even if it is likely to be a bug!

Let’s switch languages to look at a C++ example containing suspicious code that
probably meant to use comparison instead of assignment:

```cpp
$ cat a.cpp
void foo() {
  int x = 1, y = 2;

  if (x = y) { // Here.
  }
}
```

If you run Clang on this code, it helpfully points out that there is likely to
be a bug and gives you two different ways to solve the problem (with FixIts
for both):

```cpp
$ clang a.cpp
a.cpp:6:9: warning: using the result of an assignment as a condition without parentheses [-Wparentheses]
    6 |   if (x = y) {
      |       ~~^~~
a.cpp:6:9: note: place parentheses around the assignment to silence this warning
    6 |   if (x = y) {
      |         ^
      |       (    )
a.cpp:6:9: note: use '==' to turn this assignment into an equality comparison
    6 |   if (x = y) {
      |         ^
      |         ==
```

The behavior that Clang uses (and that Mojo should follow) is that a FixIt is
only attached to the primary diagnostic (a warning or error) when the
replacement is uncontroversial and behavior preserving — for example, replacing
an unused variable with `_` doesn’t change the behavior of the code as written.

In this case though, the use of an assignment in an `if` is almost certainly
wrong, but we can’t put the FixIt on the primary diagnostic. The code as written
is suspicious but could be correct and code be in production — we don’t want a
migration tool to change the behavior of the code.

For situations like this, Clang solves this by generating the warning and one
(or more) notes that show different ways to solve the problem. The migration
tool can then either ignore these sorts of cases, or prompt the user for which
way to resolve it. The key thing is to be consistent here when adding FixIts to
the compiler.

### Language Features for Code Migration

One of Mojo’s design goals is to push much of the user-exposed design complexity
into libraries: this enables powerful user-defined libraries, but also results
in migration and evolution complexity in the library space. It is common to want
to introduce new symbols into a library that are unstable, then stabilize them,
deprecate them, and rename or remove them.

In the case of Swift, Apple maintains long-term ABI stability of its core OS
APIs, and
[has a rich attribute allowing API authors to decorate their APIs](https://docs.swift.org/swift-book/documentation/the-swift-programming-language/attributes/#available),
this says that the struct is only available when using a specific swift version
and with specific platform version (and later):

```swift
@available(swift 3.0)
@available(macOS 10.12, *)
struct MyStruct {
    // struct definition
}
```

More relatedly to this document, Swift supports expressing that a symbol is
deprecated or removed, (e.g. by being replaced with another name). This allows
someone to start a library with:

```swift
// First release
struct MyStruct { ... }
```

and then rename it later to something like this:

```swift
// Subsequent release renames MyStruct
struct MyRenamedStruct { ... }

@available(*, unavailable, renamed: "MyRenamedStruct")
typealias MyStruct = MyRenamedStruct
```

The compiler sees the `renamed` field and produces a note or warning with a
FixIt hint that renames any uses of the old name to use the new name. Clang
supports the same concept with a much simpler
[extended deprecated attribute](https://clang.llvm.org/docs/AttributeReference.html#deprecated)
that looks like this:

```swift
void foo(void) __attribute__((deprecated(/*message*/"use bar instead!",
                                         /*replacement*/"bar")));
```

Clang generates a warning on any use of `foo` saying “use bar instead!” and
produces a FixIt that rewrites to “bar”. It would be great for Mojo to
eventually support something like this, probably following the idea of what
Clang supports.
