## WARNING

Everything in this file is subject to revision on any bugfix or security
update. We (the stdlib team and contributors), reserve the right to remove,
change the API contracts of, rename, or cause to instantly crash the program,
any operation described in here. These are **PRIVATE** APIs and implementation
details for the Mojo stdlib and for MAX to use. **WE WILL CHANGE IT WHENEVER
WE FIND IT CONVENIENT TO DO SO WITHOUT WARNING OR NOTICE**.

## MLIR Documentation

### `always_inline("builtin")` vs `always_inline("nodebug")`

From:
[Chris on Discord](https://discord.com/channels/1087530497313357884/1339917438372020264)

`always_inline("builtin")` is the same as `always_inline("nodebug")` but affects
how tagged methods are handled when they are called in parameter expressions.
For example, consider: `def thing[a: Int, b: Int](x: T[a], y: T[b]) -> T[a+b]:
...` This is a dependent type, and the return type is represented internally to
the compiler in a completely symbolic way as `T[Int.add(a, b)]`. This is the
core of how Mojo supports arbitrary comptime evaluation of things. That said,
when you get to the caller site, you often know what a and b are; `value =
thing(someT3, someT17)`, and in this case, "it is obvious" that value should
have type `T[20]` (assuming the first arg had value=3 and second value=17). A
naive handling of this would actually end up with it having type `T[Int.add(3,
17)]` which is lexically non-equal to `T[20]` and so would require rebinds for
obvious cases, which massively undermines dependent type support.

To address this, early on, an engineer added a horrible hack to rewrite the
comptime interpreter a bunch to see if it could go fold and simplify
expressions. This solves this case, e.g. var value : `T[20] = thing(someT3,
someT17)` works... but it is really inexpensive in compile time and introduces a
significant layering problem: the comptime interpreter isn't supposed to run on
the intermediate IR the parser is producing - that IR doesn't have destructors
inserted, isn't fully checked for semantic validity, and has other problems.
This really only affects low level types like `Int` and `Origin` and stuff like
that.

The new solution for this is to introduce a new form of `always_inline`, which
is the same as `always_inline("nodebug")`, but is different when parsed into a
parameter expression. Instead of turning `T[a+b]` into `T[Int.add(a, b)]` (where
`Int.add` does extracts from the struct, then an `index.add` mlir operation,
then calls the `Int` initializer to reform an `Int`) we actually do a very
limited form of symbolic inlining and turn this into something like
`T[Int{index.add(a.value, b.value)}]`. When used in a caller context with
specific constants, this magically all "just works" through constant folding,
and doesn't involve the interpreter at all. This decorator is very limited in
terms of the IR forms it can handle, so it isn't a generally useful thing, but
is important for this narrow case and (more importantly) enables simplifying the
compiler and making it more reliable.
