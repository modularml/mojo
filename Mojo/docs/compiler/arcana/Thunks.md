This doc explains all the non-obvious thunk handling in the compiler.

(I’ll explain what a "thunk" is a few sections down.)

# Thunk handling

This doc explains all the non-obvious thunk handling in the compiler.

(I’ll explain what a "thunk" is a few sections down.)

## Baseline: Simple Non-Thunk Example

Here’s a basic program that requires no thunks:

```mojo
struct Ship:
    def __init__(out self: Ship):
        pass

def read_ship(s: Ship):
    pass

def test_1():
    alias my_alias: def(Ship) -> None = read_ship
    # Example usage:
    # my_alias(Ship())
```

That `alias` line works cleanly, because `read_ship`’s signature is

`def(Ship)->None`

and `my_alias` expects something of type

`def(Ship)->None`

and those are the exact same, so nothing interesting happens. Huzzah!

If you value your sanity, stop reading here.

If you want to see more gnarly cases… proceed.

## A Simple Thunk

Here’s a case which _seems_ simple, but is actually really complicated under the
hood, because it needs a thunk:

```mojo
struct Ship:
    def __init__(out self: Ship):
        pass

def read_ship(imm s: Ship):
    pass

def test_1():
    alias accepts_mut_ship: def(mut Ship) -> None = read_ship
    # Example usage:
    # z = Ship()
    # accepts_mut_ship(z)
```

Notice in the line

`alias accepts_mut_ship: def(mut Ship) -> None = read_ship`

we’re trying to hand in `read_ship` which is a

`def(imm Ship)->None`

into an alias which expects something of type

`def(mut Ship)->None`

This is actually fine, because if anyone calls `accepts_mut_ship` and hands in a
mutable `Ship`, then `read_ship` will accept that mutable `Ship`, which is fine.

They’re all just pointers in the end (and it doesn’t risk any memory unsafety),
so why not?

And in fact, this is desirable for various reasons.

HOWEVER, in practice this is pretty hard for compilers to allow, because this
will [presumably, haven’t tested it] run into errors in the elaborator, when it
notices that we’re handing in a mutable reference to something that expects an
immutable reference.

The solution? To make a wrapper function!

## A Manual Wrapper Function

The user could write something like this:

```mojo
struct Ship:
    def __init__(out self: Ship):
        pass

def read_ship(imm s: Ship):
    pass

def read_ship_wrapper(mut s: Ship):
    read_ship(s) # <-- implicit cast to `imm Ship` here

def test_1():
    alias accepts_mut_ship: def(mut Ship) -> None = read_ship_wrapper
    # Example usage:
    # z = Ship()
    # accepts_mut_ship(z)
```

Notice how there’s a new `read_ship_wrapper` function, and how we’re giving it
to the alias (the `= read_ship_wrapper` part).

A **"thunk"** is a wrapper function that adapts some arguments and hands them to
another function.

`read_ship_wrapper` is a manual thunk, but when people say "thunk", they usually
mean wrapper functions that the compiler automatically generates.

Before we talk about that, let’s see a slightly more generic thunk.

## A Manual Generic Thunk

Here’s a program with two manual thunks:

```mojo
struct Ship:
    def __init__(out self: Ship):
        pass

def read_ship_1(imm s: Ship):
    pass

def read_ship_1_wrapper(mut s: Ship):
    read_ship_1(s) # <-- implicit cast to `imm Ship` here

def read_ship_2(imm s: Ship):
    pass

def read_ship_2_wrapper(mut s: Ship):
    read_ship_2(s) # <-- implicit cast to `imm Ship` here

def test_1():
    alias accepts_mut_ship: def(mut Ship) -> None = read_ship_1_wrapper
    # Example usage:
    # z = Ship()
    # accepts_mut_ship(z)
```

We can write a _generic_ manual thunk instead:

```mojo
struct Ship:
    def __init__(out self: Ship):
        pass

def read_ship_1(imm s: Ship):
    pass

def read_ship_2(imm s: Ship):
    pass

def test_1():
    alias accepts_mut_ship: def(mut Ship) -> None =
        ship_func_wrapper[read_ship_1]
    # Example usage:
    # z = Ship()
    # accepts_mut_ship(z)

def generic_ship_func_wrapper[callee: def(imm Ship)->None](mut s: Ship):
    callee(s) # <-- implicit cast to `imm Ship` here
```

Now, whenever we want to cast a

`def(imm Ship)->None`

to a

`def(mut Ship)->None`

we can just use `ship_func_wrapper`.

Presumably we do this so the parser has to do less work.

## We Automatically Generate Thunks

Looking at a previous example:

```mojo
struct Ship:
    def __init__(out self: Ship):
        pass

def read_ship(imm s: Ship):
    pass

def test_1():
    alias accepts_mut_ship: def(mut Ship) -> None = read_ship
    # Example usage:
    # z = Ship()
    # accepts_mut_ship(z)
```

We manually made a thunk for that, but we actually don’t have to in today’s Mojo
because our compiler automatically generates it for us.

It generates the previous section’s code, repeated here:

```mojo
struct Ship:
    def __init__(out self: Ship):
        pass

def read_ship_1(imm s: Ship):
    pass

def read_ship_2(imm s: Ship):
    pass

def test_1():
    alias accepts_mut_ship: def(mut Ship) -> None =
        ship_func_wrapper[read_ship_1]
    # Example usage:
    # z = Ship()
    # accepts_mut_ship(z)

def generic_ship_func_wrapper[callee: def(imm Ship)->None](mut s: Ship):
    callee(s) # <-- implicit cast to `imm Ship` here
```

## Param Refs Don’t Cause Thunks

This is a snippet that _doesn’t_ cause a thunk. It’ll serve as good context for
the next section.

```mojo
struct Ship[ZA: int]:
    def __init__(out self: Ship[ZA]):
        pass

def read_ship[T: AnyType](s: T):
    pass

def foo[ZC: int](z: Ship[ZC]):
    alias my_func_alias: def(Ship[ZC]) -> None = read_ship[Ship[ZC]]
    # Example usage:
    # my_func_alias(z)
```

As you can see, we gave a `read_ship[Ship[ZC]]` into the `alias my_func_alias`.

In other words a param ref (`ZC`) doesn’t cause any thunks.

Why is that relevant? Read on!

## Param Refs Complicate Thunks (TAPRCT)

However, if a thunk is already happening, then param refs can complicate things.

This example now needs a thunk:

```mojo
struct Ship[ZA: int]:
    def __init__(out self: Ship[ZA]):
        pass

def read_ship[T: AnyType](imm s: T):
    pass

def foo[ZC: int](mut z: Ship[ZC]):
    alias my_func_alias: def(mut Ship[ZC]) -> None = read_ship[Ship[ZC]]
    # Example usage:
    # my_func_alias(z)
```

Because `read_ship[ZC]` now has type:

`def(imm Ship[ZC])->None`

and we’re passing it into an alias that now accepts a

`def(mut Ship[ZC])->None`

so we need a thunk.

However, this is the (problematic) thunk we would generate:

```mojo
struct Ship[ZA: int]:
    def __init__(out self: Ship[ZA]):
        pass

def read_ship_1[ZB: Int](imm s: Ship[ZB]):
    pass

def foo[ZC: int]():
    alias my_func_alias: def(mut Ship[ZC]) -> None =
        generic_ship_func_wrapper[read_ship_1[ZC]]
    # Example usage:
    # z = Ship[ZC]()
    # my_func_alias(z)

def generic_ship_func_wrapper[
    callee: def(imm Ship)->None
](mut s: Ship[ZC]): # <-- THERE IS A PROBLEM HERE
    callee(s) # implicit cast to imm
```

We followed the same steps as before when making our thunk… but there’s a
problem now.

Notice how that innocent-looking `Ship[ZC]`, previously our friend, is now our
downfall: `generic_ship_func_wrapper` has no `ZC` declared!

To fix this, we’re going to do something I personally call "bedazzling the
thunk", or since we’re professionals or something, we’ll call it: prepending
"clarifying" parameters to the thunk.

## Prepend Clarifying Parameters to the Thunk (TAPCPTTT)

We’ll change the above to this:

```mojo
struct Ship[ZA: Int]:
    def __init__(out self: Ship[ZA]):
        pass

def read_ship[ZB: Int](imm s: Ship[ZB]):
    pass

def foo[ZC: Int]():
    alias my_func_alias: def(mut Ship[ZC]) -> None =
        generic_ship_func_wrapper[ZC, read_ship[ZC]] # <-- Added ZC,
    # Example usage:
    # z = Ship[ZC]()
    # my_func_alias(z)

def generic_ship_func_wrapper[
    ZC: Int, # <-- Added this too
    callee: def(imm Ship[ZC])->None
](mut s: Ship[ZC]):
    callee(s) # implicit cast to imm
```

Notice how we’re adding a `ZC,` input-parameter to `generic_ship_func_wrapper`.

This makes it work!

We’ll call these "clarifying" parameters, because they clarify the thunk’s
argument types.

## Thunks for Generic Arg References (TATFGAR)

Can anyone guess why this snippet produces a thunk?

```mojo
def read_ship[T: AnyType](s: T):
    pass

def test_1[ZC: Int]():
    alias my_func_alias: def(Int) -> None = read_ship[Int]
    # Example usage:
    # z = Int()
    # my_func_alias(z)
```

I sure couldn’t, and it took many days of investigating to figure it out. Hark,
intrepid engineer, as I reveal the hidden reasons.

TL;DR: `read_ship` ’s `T` is an `AnyType`, so `read_ship` has to treat it as a
memory type and therefore has to take in a reference to it (a `!lit.ref`).
`read_ship[Int]` therefore takes in a reference to an `Int`, a `!lit.ref<Int>`.
That doesn’t match `my_func_alias` which knows it can take in a normal value
(not a reference), and therefore we need a thunk.

If the above doesn’t make sense, keep reading.

Firstly, `read_ship`’s `s` argument accepts a `T`, which is an `AnyType`.
`read_ship` doesn’t know whether that will be in a register or a reference yet,
so it conservatively requires the caller to pass in a reference (it can’t
require the caller pass in something via register, it might not be a
register-passable type).

In other words, functions can only take _references_ to "direct" generic
arguments (of the form `x: T` where `T` is an input-parameter). Note that other
generic arguments (like `x: Scalar[D]`) will still be register passable; those
work as expected because the generic function knows whether `Scalar` is
register-passable or not.

Here’s the MLIR from the above `read_ship` function, note how it takes in a
``%s: !lit.ref<:!AnyType T, imm *"s`"> read_mem``:

```mlir
lit.fn @"read_ship[function_types::AnyType]($0)"<
    T: !AnyType
>[imm *"s`"](
    %s: !lit.ref<:!AnyType T, imm *"s`"> read_mem
) -> !kgen.none attributes {sourceName = "read_ship", specialFnKind = 0 : i8} {
    %none = kgen.param.constant: none = <#kgen.none>
    lit.return %none : !kgen.none
    lit.end_fn
}
```

Second, when we say `read_ship[Int]`, the compiler does a simple substitution,
so the argument type

`!lit.ref<:!AnyType T, imm *"s`"> read_mem` becomes:

`!lit.ref<!Int, imm *[0,0]> read_mem`. In other words, a _reference_ to an Int.

This means `read_ship[Int]`'s signature `def(ref Int)->None` isn’t a subtype of
`my_func_alias`'s expected signature `def(Int) -> None` and therefore requires a
thunk to be able to assign into `

If for some reason we wanted to make `read_ship` not take in a reference, we
would make its input parameter a register-passable trait like this:

```mojo
trait AnyRegisterPassableType(TrivialRegisterPassable):
    pass

def read_rp_ship[T: AnyRegisterPassableType](s: T):
    pass

def test_1[ZC: Int]():
    alias my_func_alias: def(Int) -> None = read_rp_ship[Int]
    # Example usage:
    # z = Int()
    # my_func_alias(z)
```

And suddenly no thunk is generated.

## Partial Function Application Doesn’t Need Thunks

This is a snippet that _doesn’t_ cause a thunk. It’ll serve as good context for
the next section.

```mojo
struct Ship[X: int, Y: Bool]:
    pass

def read_ship[X: int, Y: Bool](s: Ship[X, Y]):
    pass

def test_1():
    alias my_func_alias: def[Y: Bool](Ship[42, Y]) -> None =
        read_ship[42]
```

As you can see, we "partially bound" `read_ship`; we said `read_ship[42]` and
not its remaining input-parameters like `read_ship[42, True]`.

`read_ship[42]` still has one input-parameter unbound; its type is

`def[Y: Bool](Ship[42, Y]) -> None`.

This is sometimes called "partial function application", since we’re kind of
_half_ calling ("apply"ing) a function.

Anyway, as it turns out, this does **not** require a thunk. The compiler is
smart enough to treat that the same way as a normal mention of
`def other_func[Y: Bool](s: Bar[Y])` that doesn’t have any partial application
going on.

However, those "remaining unbound input-parameters" (`[Y: Bool]`) do cause some
complication if a thunk is already being made for some other reason, see
TARIPNBITM for more.

## Remaining Input Parameters Not Bound In Thunk Mention (TARIPNBITM)

However, if a thunk is already happening, then any remaining unbound
input-parameters can cause a non-obvious thing to happen.

In this example, the `[Y: Bool]` on the `my_func_alias` is the "remaining
unbound input-parameters".

```mojo
struct Ship[X: int, Y: Bool]:
    pass

def read_ship[X: int, Y: Bool](imm s: Ship[X, Y]):
    pass

def test_1():
    alias my_func_alias: def[Y: Bool](mut Ship[42, Y]) -> None =
        read_ship[42]
```

When we generate the thunk for this (because of that `mut`/`imm` mismatch),
it’ll look something like this:

```mojo
struct Ship[X: int, Y: Bool]:
    pass

def read_ship[X: int, Y: Bool](imm s: Ship[X, Y]):
    pass

def test_1():
    alias my_func_alias: def[Y: Bool](mut Ship[42, Y]) -> None =
        generic_ship_func_wrapper[?, read_ship[42]] # <-- `?`, unbound

def generic_ship_func_wrapper[
    Y: Bool,
    callee: def[Y: Bool](imm Ship)->None
](mut s: Ship[ZC, Y]):
    callee[Y](s) # implicit cast to imm
```

Note that `?` there. That’s MLIR-speak for `UnboundAttr`.

That `?` corresponds to the `[Y: Bool]` on the alias’s type.

## A More Complicated Example (TAAMCE)

If you understand this example, you’ve won.

```mojo
struct Ship[X: int, Y: Bool]:
    pass

def read_ship[X: int, Y: Bool](imm s: Ship[X, Y]):
    pass

def foo():
    alias Z: int = 42
    alias my_func_alias: def[Y: Bool](mut Ship[Z, Y]) -> None =
        read_ship[Z]
```

It should generate a thunk that looks like this:

```mojo
struct Ship[X: int, Y: Bool]:
    pass

def read_ship[X: int, Y: Bool](imm s: Ship[X, Y]):
    pass

def foo():
    alias Z: int = 42
    alias my_func_alias: def[Y: Bool](mut Ship[Z, Y]) -> None =
        ship_func_thunk[Z, ?, read_ship[Z]] # <-- `?` means unbound

def ship_func_thunk[
    Z: int,
    Y: Bool,
    callee: def[Y: Bool](imm Ship[Z])->None
](mut s: Ship[Z, Y]):
    callee[Y](s) # implicit cast to imm
```

MLIR (some names and strings reduced for clarity):

```mlir
lit.fn @"read_ship"<X, Y: !Bool>[imm *"s`"](%s: !lit.ref<@function_types::@Ship<X, :!Bool Y>, imm *"s`"> read_mem) -> !kgen.none attributes {sourceName = "read_ship", specialFnKind = 0 : i8} {
    %none = kgen.param.constant: none = <#kgen.none>
    lit.return %none : !kgen.none
    lit.end_fn
}
lit.fn @"foo()"() -> !kgen.none attributes {sourceName = "foo", specialFnKind = 0 : i8} {
    lit.alias.decl *"Z`" = <42>
    lit.alias.decl *"my_func_alias`1": !lit.generator<<"Y": !Bool>[1](!lit.ref<@function_types::@Ship<42, :!Bool *(0,0)>, mut *[0,0]> mut, |) -> !kgen.none> = <rebind(:!lit.generator<<!Bool>[1](!lit.ref<@function_types::@Ship<42, :!Bool *(0,0)>, mut *[0,0]> mut) -> !kgen.none> @"ship_func_thunk"<:!Bool ?, :!lit.generator<<!Bool>[1](!lit.ref<@function_types::@Ship<42, :!Bool *(0,0)>, imm *[0,0]> read_mem, |) -> !kgen.none> rebind(:!lit.generator<<"Y": !Bool>[1]("s": !lit.ref<@function_types::@Ship<42, :!Bool *(0,0)>, imm *[0,0]> read_mem) -> !kgen.none> @function_types::@"read_ship"<42, :!Bool ?>)>)>
    %none = kgen.param.constant: none = <#kgen.none>
    lit.return %none : !kgen.none
    lit.end_fn
}
lit.fn @"ship_func_thunk"<[""]_0: !Bool, [""]callee: !lit.generator<<!Bool>[1](!lit.ref<@function_types::@Ship<42, :!Bool *(0,0)>, imm *[0,0]> read_mem, |) -> !kgen.none>>[mut *"0_unnamed`"](%0[*""]: !lit.ref<@function_types::@Ship<42, :!Bool _0>, mut *"0_unnamed`"> mut) -> !kgen.none always_inline_no_debug attributes {isSynthetic, sourceName = "...", specialFnKind = 0 : i8, thunkKey = [!kgen.generator<!lit.generator<<!Bool>[1](!lit.ref<@function_types::@Ship<42, :!Bool *(0,0)>, imm *[0,0]> read_mem, |) -> !kgen.none>>, !kgen.generator<!lit.generator<<!Bool>[1](!lit.ref<@function_types::@Ship<42, :!Bool *(0,0)>, mut *[0,0]> mut, |) -> !kgen.none>>]} {
    %1 = lit.ref.immut %0 : <@function_types::@Ship<42, :!Bool _0>, mut *"0_unnamed`">
    %2 = lit.call [!lit.generator<[1](!lit.ref<@function_types::@Ship<42, :!Bool _0>, imm *[0,0]> read_mem, |) -> !kgen.none>: bind_signature(:!lit.generator<<!Bool>[1](!lit.ref<@function_types::@Ship<42, :!Bool *(0,0)>, imm *[0,0]> read_mem, |) -> !kgen.none> callee, _0)][muttoimm *"0_unnamed`"](%1)
    lit.return %2 : !kgen.none
    lit.end_fn
}
```

## Thunks Support Mojo’s Function Subtyping (TTSMFS)

Thunks help us convert one kind of function to a similar function, where the
conversions of the arguments/return types are obvious (like the above where we
just cast from `mut` to `imm`).

In other words, these thunks let the user automatically convert one kind of
function to something that’s semantically equivalent.

In other other words, these thunks enable **"function subtyping"**: function
type A ("actual") (like `def(imm Ship)->None`) is a "subtype" of function type
E ("expected") (like `def(mut Ship)->None`) if an A can be used wherever an E is
expected.

Of course, this only works if the differences aren’t too much: we can easily
cast a `mut Ship` to a `imm Ship`, but we can’t:

- Do the opposite (`imm Ship` to `mut Ship`)
- Cast a `Ship` to a `Shovel`.
- Cast a `float` to an `int` (technically possible, but it loses too much
  information so we shouldn’t).

The rules for function subtyping are roughly: We can cast function type X to
function type Y if:

- Y’s argument types can be cast to X’s argument types (in other words, the
  arguments are **"contravariant"**)
- X’s return type can be cast to Y’s return type (in other words, the return
  types are **"covariant"**)

The functions can have some other differences too:

- Their positional arguments can have different names.
- The actual function can take values, even if the expected function takes
  references (this happens in TATFGAR)

(the above list is non-exhaustive, the code is probably still the best reference
for this)
