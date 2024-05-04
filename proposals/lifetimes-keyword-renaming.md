# Keyword naming and other topics to discuss

This document is split off the [Provenance Tracking and Lifetimes in
Mojo](lifetimes-and-provenance.md) document to separate general syntactic
bikesheding issues from the core semantic issues in that proposal.

Assuming that proposal goes through, I think we should consider a few changes to
the current Mojo keyword paint:

## `borrowed` Keyword => `borrow` or `ref`

`borrowed` as a keyword doesn’t really make sense in our new world.  This is
currently used to indicate an argument that is a borrowed version of an existing
value.  Given the introduction of lifetimes, these things can now appear in
arbitrary places (e.g. you can have an array of references) so it makes sense to
use a noun.

Instead of reading an argument as “this function takes foo which is a borrowed
string”, we would read it as “foo is a borrow/ref of a string”.  This makes it
consistent with local borrows on the stack:

```mojo
fn do_stuff[a: Lifetime](x: ref[a] String): ...

fn usage():
    var str = String("hello")
    ref r = str   # Defines a local borrow of str.

    do_stuff(str)  # Bind a reference to 'str'
    do_stuff(r)    # Pass on existing reference 'str'
```

## `inout` Keyword => `ref` or `mutref` (??)

The primary argument for the ‘`inout`’ keyword being named this was that Chris
wanted to get off the former ampersand syntax we used, and that (in an argument
position) there is copy-in and copy-out action that happens with computed
LValues.  I think there is a principled argument to switch to something shorter
like `ref` which is used in other languages (e.g. C#), since they can exist in
other places that are not arguments, and those don’t get copy-in/copy-out
behavior.  One challenge with the name `ref` is that it doesn't obviously
convey mutability, so we might need something weird like `mutref`.

Note that copy-in/copy-out syntax is useful in more than function call
arguments, so the `inout` keyword may return in the future.  For example, we may
actually want to bind computed values to mutable references:

```mojo
for inout x in some_array_with_getitem_and_setitem:
    x += 1
```

This requires opening the reference with getitem, and writing it back with
setitem. We may also want to abstract over computed properties, e.g. form
something like `Array[inout Int]` where the elements of the array hold closers
over the get/set pairs.  If we had this, this could decay to a classic mutable
reference at call sites providing the existing behavior we have.

Given this possible direction and layering, I think we should go with something
like this:

1. `ref`: immutable reference, this is spelled “`borrowed`” today

2. `mutref`: mutable reference, this is spelled “`inout`” today. I’d love a
better keyword suggestion than `mutref`, perhaps just `mut`?

3. `inout`: abstracted computed mutable reference with getter/setter.

`inout` can decay to `mutref` and `ref` in an argument position with writeback,
and `mutref` is a subtype of `ref` generally.

## `owned` Keyword => `var`

People on the forums have pointed out that the “`owned`” keyword in argument
lists is very analogous to the `var` keyword.  It defines a new, whole, value
and it is mutable just like `var`.  Switching to `var` eliminates a concept and
reduces the number of keywords we are introducing.

## Allow `let` in argument lists ... or remove them entirely (!)

If we replace the `owned` keyword with `var`, then we need to decide what to do
with `let`.  There are two different paths with different tradeoffs that I see.

The easiest to explain and most contiguous would be to allow arguments to be
defined as `let` arguments, just like we define `var` arguments.  This would
keep these two declarations symmetrical, and appease people that like to control
mutation tightly.

The more extreme direction would be to remove `let` entirely.  Some arguments
in favor of this approach:

1. It has been observed on the forum that it adds very little - it doesn't
   provide additional performance benefits over `var`, it only prevents
   "accidental mutation" of a value.
2. Languages like C++ default to mutability everywhere (very few people bother
   marking local variables constant, e.g. with `const int x = foo()`.
3. The more important (and completely necessary) thing that Mojo needs to model
   are immutable borrows.  Removing `let` would reduce confusion about these two
   immutable things.
4. Mojo also has `alias`, which most programmers see as a “different type of
   constant” further increasing our chance of confusing people.
5. `let` declarations require additional compiler complexity to check them, Mojo
   doesn’t currently support struct fields market `let` for example because the
   initialization rules are annoying to check for.  Once you have them, it
   messes with default values in structs.

In my opinion, I think we are likely to want to remove `let`’s, but we should
only do so after the whole lifetime system is up and working.  This will give us
more information about how things feel in practice and whether they are worth
the complexity.

## More alternatives to consider

[@sa-
suggests](https://github.com/modularml/mojo/discussions/338#discussioncomment-6104926)
the keyword `fix` instead of `let`.

[@mojodojodev suggests](https://github.com/modularml/mojo/discussions/338#discussioncomment-6105688):

`ref[a]` - immutable reference
`mut[a]` - mutable reference
`let[a]` - immutable owned
`var[a]` - mutable owned

Having three letters for all of the keywords will allow the user to understand
"this is related to ownership and mutability". The problem with the proposed
removing let is that code ported from Python to Mojo won't behave the same,
keeping let and var is advantageous in that it says this is a Mojo variable so
you can add all the weird Python dynamic behavior when the keyword is elided.

[@mzaks
suggests](https://github.com/modularml/mojo/discussions/338#discussioncomment-6134220)
using numbers to identify lifetimes, e.g.:

```mojo
fn example['1_life](cond: Bool,
                    x: borrowed'1 String,
                    y: borrowed'1 String):
   # Late initialized local borrow with explicit lifetime
   borrowed'1 str_ref : String

   if cond:
      str_ref = x
   else:
      str_ref = y
   print(str_ref)
```
