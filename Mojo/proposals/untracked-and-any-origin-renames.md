# Rename the `ExternalOrigin` and `AnyOrigin` origin aliases

**Status**: Accepted

**Author**: Nathan Ward

**Created**: June 2026

## Summary

This proposal renames the two origin aliases so their names describe how they
interact with the compiler's lifetime analysis:

- `ExternalOrigin` → `UntrackedOrigin` (and its `Mut`/`Immut` variants).
- `AnyOrigin` → `UnsafeAnyOrigin` (and its `Mut`/`Immut` variants).

`UntrackedOrigin` is a legitimate, supported tool, so it gets a plain,
descriptive name. `UnsafeAnyOrigin` is a temporary compiler escape hatch that
defeats lifetime and exclusivity analysis; it will never stabilize and is slated
for deprecation and removal, so its name warns developers off.

## Background: origins drive lifetime analysis

The Mojo compiler includes a lifetime checker that analyzes dataflow through
your program. It uses origins to answer two questions about every reference:

- What value does this reference point to, and what else might it alias?
- Is the reference mutable?

From those answers the lifetime checker does two jobs:

- **Lifetime extension**: it keeps a value alive until the last use of any
  reference derived from it, then destroys it as soon as possible (ASAP
  destruction).
- **Exclusivity checking**: it rejects programs that hold two mutable references
  to the same value at the same time across a function boundary (or a mutable
  and an immutable one).

Most origins are tied to a concrete value — `origin_of(x)`, the origin inferred
for `UnsafePointer(to=x)`, and so on — and the checker tracks them precisely.
`ExternalOrigin` and `AnyOrigin` are the two origins that step outside this
tracking, and they do so in opposite ways: `ExternalOrigin` is the empty origin,
so it aliases nothing, while `AnyOrigin` is the universal origin, so it might
alias anything.

## `ExternalOrigin` → `UntrackedOrigin`

When a type or reference carries an origin tied back to a value, the compiler
extends that value's lifetime until the reference's last use:

```mojo
@fieldwise_init
struct Loud(Writable):
    var s: String

    def __del__(deinit self):
        print(t"{self}.__del__")

def main():
    var loud = Loud("abc")

    print("Creating pointer to Loud")
    # __type_of(ptr) == UnsafePointer[Loud, origin_of(loud)]
    var ptr = UnsafePointer(to=loud)

    print("Printing...")
    print(ptr[])
    # `loud` is destroyed here, after `ptr`'s last use.
```

Because `ptr`'s type is `UnsafePointer[Loud, origin_of(loud)]` — that is,
`origin_of(loud)` is carried in `UnsafePointer`'s origin type parameter — the
compiler can tie `ptr` back to `loud` and won't destroy `loud` until after
`ptr`'s last use:

```output
Creating pointer to Loud
Printing...
Loud(s='abc')
Loud(s='abc').__del__
```

An untracked origin opts out of that: it promises the reference aliases no value
the compiler is managing, so there is nothing for the lifetime checker to track
or extend.

```mojo
def main():
    var loud = Loud("abc")

    print("Creating pointer to Loud")
    # `ptr`'s origin is cast away to `MutUntrackedOrigin`.
    var ptr = UnsafePointer(to=loud).unsafe_origin_cast[MutUntrackedOrigin]()
    # `loud` is destroyed here — the compiler sees no live reference to it.

    print("Printing...")
    print(ptr[])  # <-- OH NO 😱! Reading uninitialized memory!
```

Now `ptr` no longer carries `origin_of(loud)`, so the compiler does not know
`loud`'s lifetime depends on `ptr`. `loud` is destroyed earlier, while `ptr`
still points at it:

```output
Creating pointer to Loud
Loud(s='abc').__del__
Printing...
BOOM 😱! UB? Maybe prints the value?
```

The pointer type still has an origin, but the compiler does not track it in
lifetime analysis, since it is an `UntrackedOrigin` — hence the name. Casting a
compiler-managed value's origin away like this is unsafe, as it can leave `ptr`
dangling.

But that same untracked behavior is exactly what you want when interfacing with
memory from outside the Mojo program. For example, `alloc()` returns a pointer
with `UntrackedOrigin`: the allocated block aliases no Mojo-owned value, so the
checker should not extend any lifetimes on its behalf.

## `AnyOrigin` → `UnsafeAnyOrigin`

`AnyOrigin` is the opposite of `UntrackedOrigin`: it might alias anything, which
forces the lifetime checker into its most conservative behavior. It breaks
lifetime extension, hides diagnostics, and is incompatible with exclusivity
checking. We are renaming it to `UnsafeAnyOrigin` so every use site reads as
unsafe.

The examples below use the proposed names (`MutUnsafeAnyOrigin` is today's
`MutAnyOrigin`).

### It extends unrelated lifetimes

Because an `UnsafeAnyOrigin` reference might alias any live value, the compiler
keeps every other value in scope alive for as long as the `UnsafeAnyOrigin` is
live — even values the pointer never points to — effectively halting ASAP
destruction:

```mojo
def main():
    var loud = Loud("abc")

    print("Creating pointer to n")
    var n = 42
    var ptr = UnsafePointer(to=n).as_unsafe_any_origin()
    #                            ~~~~~~~~~~~~~~~~~~~~~~~ Casts the origin to `UnsafeAnyOrigin`

    print(ptr[])
    # `loud` should die immediately after it's created,
    # but is kept alive until here. 😞
```

```output
Creating pointer to n
42
Loud(s='abc').__del__
```

With the correct origin, ASAP destruction runs normally:

```mojo
def main():
    var _loud = Loud("abc")
    # Nothing references `loud`, so it's destroyed right away. 🙂

    print("Creating pointer to n")
    var n = 42
    var ptr = UnsafePointer(to=n)
    print(ptr[])
```

```output
Loud(s='abc').__del__
Creating pointer to n
42
```

### It hides unused-variable warnings

The same lifetime extension suppresses unused-variable diagnostics. Here
`unused` is never read, but because the `UnsafeAnyOrigin` pointer might alias
it, the compiler treats it as live and stays quiet:

```mojo
def main():
    # No warning 😞 — the compiler thinks `ptr` might alias `unused`.
    var unused = "I'm an unused string!"

    var n = 42
    var ptr = UnsafePointer(to=n).as_unsafe_any_origin()
    #                            ~~~~~~~~~~~~~~~~~~~~~~~ Casts the origin to `UnsafeAnyOrigin`
    print(ptr[])
```

Giving `ptr` a specific origin lets the warning correctly fire again:

```mojo
def main():
    var unused = "I'm an unused string!"

    var n = 42
    var ptr = UnsafePointer(to=n)  # origin inferred as `origin_of(n)`
    print(ptr[])
```

```text
warning: assignment to 'unused' was never used; assign to '_' instead?
var unused = "I'm an unused string!"
^
```

### It disables mutable exclusivity checking

When an argument's origin is `UnsafeAnyOrigin`, the compiler cannot prove which
value it aliases, so it does not diagnose exclusivity violations for that
argument. Below, two mutable pointers to `x` reach a function, but the violation
goes unreported:

```mojo
def two_mut_pointers(
    p1: UnsafePointer[mut=True, Int, _],
    p2: UnsafePointer[mut=True, Int, _],
):
    pass

def main():
    var x = 42

    # No error, even though two mutable pointers to `x` are passed. 😞
    two_mut_pointers(
        UnsafePointer(to=x).as_unsafe_any_origin(),
        #                  ~~~~~~~~~~~~~~~~~~~~~~~ Casts origin to `UnsafeAnyOrigin`
        UnsafePointer(to=x),
    )
```

With concrete origins on both arguments, the compiler catches the aliasing
violation:

```mojo
def main():
    var x = 42

    two_mut_pointers(
        UnsafePointer(to=x),
        UnsafePointer(to=x),
    )
```

```text
error: argument of 'two_mut_pointers' call allows writing a memory location previously writable through another aliased argument
two_mut_pointers(
^

note: 'x' memory accessed through reference embedded in value of type 'UnsafePointer[Int, origin_of(x)]'
two_mut_pointers(
^
```

### Why rename it?

`AnyOrigin` is not a capability we want people reaching for. It is a temporary
compiler construct introduced in Mojo's early days to paper over awkward
lifetime edge cases. Those behaviors are precisely how it breaks the guarantees
the origin system is supposed to provide. For that reason it will never be
stabilized, and it is slated for deprecation and removal in the future.

The path forward is to improve the origin system so that the patterns people
currently use `AnyOrigin` for can be expressed safely, with the compiler still
tracking lifetimes and exclusivity. Until then, the `Unsafe` prefix marks every
remaining use as a place to migrate away from, not a tool to adopt.
