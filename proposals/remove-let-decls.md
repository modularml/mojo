# Simplifying Mojo🔥 - let's get rid of `let`

Chris Lattner, Dec 5, 2023, Status: **Accepted**, [discussion thread](https://github.com/modularml/mojo/discussions/1456#discussioncomment-8358722)

Mojo is still a new language, and is rapidly evolving.  We’re learning a lot
from other languages, but Mojo poses its own set of tradeoffs that indicate a
unique design point.

One of the early decisions made in Mojo's development is that it adopts the
`let` and `var` design point that Swift uses.  This whitepaper argues that we
should switch to a simpler model by jettisoning `let` and just retaining `var`
(and implicit Python-style variable declarations in `def`).  This has also been
[suggested by the community](https://github.com/modularml/mojo/issues/1205).

Note that immutability and value semantics remain an important part of the Mojo
design, this is just about removing "named immutable variables".  Immutable
references in particular are critical and will be part of a future "lifetimes"
feature.

## Current Mojo 0.6 design

Mojo initially followed the precedent of Swift, which allows a coder to choose
between the `let` and `var` keyword to determine whether something is locally
modifiable.  That said, the design in Mojo 0.6 and earlier has a number of
particularly surprising and unfortunate aspects:

1. The notion of immutable variables is entirely new to Python programmers, and
previous experience with Swift shows that this ends up being the [very first concept](https://docs.swift.org/swift-book/documentation/the-swift-programming-language/thebasics/#Constants-and-Variables)
a Swift programmer has to learn.  This is unfortunate, because named immutable
variables aren't a core programming concept, and not something required
to achieve Mojo's goals.

2. The naming of `let` caused a lot of early [heat and
debate](https://github.com/modularml/mojo/discussions/120). Other programming
languages have a wide range of design points (e.g. `const` in C/C++ and
Javascript) and there is a divergence of naming for all these things:
`let`, `val`, `const`, etc, etc.

3. Mojo also has a notion of compile time value (`alias`), which means there are
three concepts going around: `alias`, `let`, and `var`.  Most of the uses of
(e.g.) Javascript `const` is better served with `alias` than `let`.

4. Both Swift and Rust encourage immutable values - Swift (and currently Mojo)
warn about unneeded mutability, Rust makes mutability more verbose (`let mut`),
and some propose that Mojo [make mutability more
verbose](https://github.com/modularml/mojo/issues/451).  This cuts very hard
against a lot of the design center of Python, which doesn’t even have this
concept at all: it would be weird to make it the default, but if we don’t,
then why bother having it?

5. There is no performance benefit to the Swift/Rust design, and I personally
haven’t seen any data that shows that there is any safety or readability
benefit.  If anything, it is the argument when seeing `let x = foo()` that you
know `x` will never be reassigned, but any benefit here is small.

6. The immutability only applies to the local value, and in the case of
reference semantic types (e.g. types like `Pointer` in today's Mojo, but also
*all classes* in tomorrow's Mojo) this is super confusing.  We are often asked:
“Why do I get a warning that I should change a "`var` pointer" to `let` when I
clearly mutate what it is pointing to?”

7. Mojo does not currently allow `let`’s as struct fields, (only `var`’s) which
is inconsistent.  Swift has a very complex set of rules for how struct fields
get initialized that would be nice to not implement for Mojo.  There also isn’t
a great way to define defaulted field values, e.g.:

   ```mojo
   struct Thing:
       # This is not actually supported right now, but imagine it were.
       let field = 42
       fn __init__(inout self):
           self.field = 17  # shouldn't be able to overwrite field?
   ```

8. Mojo has a notion of ownership and will eventually have a notion of lifetimes
and safe references (including both mutable and immutable *references*) which
will be different from (but can compose with) the `let` vs `var` distinction.
It is unfortunate to have different forms of immutability floating around, and
we really do need immutable borrows and immutable references.

Speaking subjectively as one of the principal designers of Swift, I will say
that it has several pretty pedantic language features intended to increase
safety (e.g. requiring all potentially-throwing values to be marked with a `try`
keyword) and many of the decisions were made early without a lot of data to
support them.  I believe we should fight hard to keep Mojo easy to learn and
eliminate unnecessary concepts if we can.

## Proposal: eliminate ‘`let`’ declarations

The proposal here is straightforward: let’s just eliminate the concept of an
immutable named value entirely.  This won’t eliminate immutability as a concept
from Mojo, but will instead push it into the world of borrowed arguments and
immutable references.  This would have a number of advantages:

This directly simplifies the conceptual Mojo language model:

1. This eliminates one unfamiliar concept that a Python program would have to
   learn.
2. This eliminates confusion between `let` vs `alias` directly.
3. This eliminates a fertile source of keyword bikeshedding.
4. This eliminates confusion in workbooks where top level values are mutable
   even though they are declared `let`.

This would eliminate a bunch of complexity in the compiler as well:

1. Error messages currently have to make sure to say `let` and `var` correctly.
2. The IR representation needs to track this for later semantic checks.
3. This eliminates the need to implement missing features to support `let`’s.
4. This eliminates the complexity around detecting unmutated `var`s that warn
   about changing to `let`.
5. Due to ASAP destruction, CheckLifetimes has extra complexity to reject code
   like: “`let x: Int; x = 1; use(x); x = 2; use(x)`” even though the original
   lifetime of the first “`x=1`” naturally ended and “`x`” is uninitialized
   before being assigned to.  This has always been a design smell, and it
   [doesn’t work right](https://github.com/modularml/mojo/issues/1414).

This proposal will not affect runtime performance at all as far as we know.

### What about var?

If this proposal is accepted, I think we should leave `var` as-is.  Unlike
traditional Python behavior, `var` introduces an explicitly declared and
*lexically scoped* value: we need some introducer and do want scoped
declarations.

The name `var` is also less controversial because it clearly stands for
“variable” in a less ambiguous way than using `let` to stand for "named
constant".  If there is desire to rename `var` it would be an orthogonal
discussion to this one and should be kept separate.

## Rolling out this proposal smoothly

If we think this proposal is a good idea, then I think we should stage this to
make adoption more smooth and less disruptive. Rolling this out would look like
this:

1. Build consensus with the Mojo community to get feedback and additional
   perspective.
2. Do the engineering work to validate there is no performance hit etc, and
   eliminate the IR representation and behavior for `let`.  At this phase we
   will keep parsing them for compatibility: parse them into the same IR as a
   `var`, but emit a warning “let has been deprecated and will be removed in
   the next release” with a fixit hint that renames the `let` to `var`.
3. In a release ~1 month later, change the warning into an error.
4. In a release ~1 month later, remove the keyword entirely along with the error
   message.

## Alternatives considered

We can always keep this around and re-evaluate later.  That said, I don’t think
anything will change here - the Mojo user community (both external to Modular
and internal) has already run into this several times, and this will keep coming
up.
