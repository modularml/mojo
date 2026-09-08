---
markdown-notebook-data-directory: mdnb-data/manual-terminology/
---

# Terminology

In this page, we'll talk about the terminology we use internally for various
concepts in Mojo. Some of it will be unusual, marked with a ⚠️. Knowing these
will help you understand Modular conversations and code!

Using this snippet:

```mojo
struct Spaceship[N: Int]:
    var engine: Engine[N]

    def foo[T: Stringable](x: T):
        var i: Int = 42
```

Here are involved concepts in those lines (not in order).

- In `var x: Int = 42`:
  - `42` is the "runtime value". Try not to call it just a "value" because also
    have compile-time values.
  - `Int` is the "type"
- In `def foo[T: Stringable](x: T):`
  - `x` is an “argument”. `x` is **not** a “parameter”. ⚠️ Because...
  - `T` is a "parameter". ⚠️ Outsiders call them "generic arguments" or "generic
    parameters", but here they're just "parameters".
  - The first `T` is a “parameter decl”.
  - The second `T` is a “parameter reference” (or "param ref").
- In `struct Spaceship[N: Int]: ...`
  - `N` here is a "parameter decl".
- In `var engine: Engine[N]`
  - `Engine` is a "symbol ref" that refers to a "type".
  - `N` is a "parameter reference" (or "param ref").
  - `N`, `Engine[N]`, and a hypothetical `N + 1` are all "parameter expressions"
    because they happen at compile-time.
  - `N`'s value is a "parameter value".
  - Types are also parameter values (well not technically, but one just wraps a
    type in a `TypeParamAttr`, easy).
  - `N`'s value and the `Engine` type are both parameter values.
  - Anything that ever appears to the right of a `:` is a parameter expression
    made up of parameter values.
  - A “parameter value” is **not** always a “type”. ⚠️ For example, `N` is a
    value that’s not a type.
- Revisiting `def foo[T: Stringable](x: T):`:
  - In this case, the second mention of `T` is also a “type”, but more often
    you'll hear it referred to as a "trait" or a "metatype".
    - "Metatype" = "trait" is a good rule of thumb, though sometimes people say
      "metatype" as if it refers to a trait's set of methods. ⚠️
- Revisiting `var x: Int = 42`:
  - `Int` is the “type”, but it’s **not** a “value”. A type is only a “value”
    when it’s in a parameter; only parameters have values.
  - `x` is not a “value” (despite LLVM handling it with `LLVMValueRef`)

See also
[Modular Jargon, Slang and Lingo](https://www.notion.so/modularai/Modular-Jargon-Slang-and-Lingo-d71a8b9aad66401d914309cc2f3c3eca)

More terms:

- A "generator" is a function.
- Everything in MLIR is either an **operation** or an **attribute**.
  - An **operation** generally describes computation, like a a function or an
    expression.
    - And for some reason, structs and traits are also operations.
  - An **attribute** is metadata, values, and flags.
    - Pre-elaboration, an attribute is generally a parameter expression.
    - Post-elaboration, an attribute is generally a parameter value.
- "Bindings" are the mapping of a caller-supplied argument (or parameter) to the
  callee's argument (or parameter) declaration. When calling
  `foo[a: Int, b: Bool]()` like `foo[42, True]`, the bindings are `a=42` and
  `b=True`.
- Structs and functions can both have **signatures**. A signature is the name,
  parameter types, and (if for a function) argument types.

Some miscellaneous other terms:

- POC — Parameter Operator Code
- POG — Parameter or argument (also a joke by Jeff, as
  [“pog” is a gaming term](https://modular-ai.slack.com/archives/C03GM7S2VMZ/p1736555603471169?thread_ts=1736479123.803939&cid=C03GM7S2VMZ))

## "Metatype"

Sometimes, when we're talking about traits (or a set of "required methods"), we
call it a "meta type".

A useful (but probably inaccurate) mental model is that a metatype is a specific
trait (`Copyable`) or the type that describes all traits (`_AnyTypeMetaType`).

Note that a trait's supertrait != the trait's metatype. If you have a
`trait Spaceship(Launchable): ...`, `Launchable` isn't the metatype, it's the
supertrait.

For more on this interpretation, see
[Mojo Type Taxonomy](https://docs.google.com/document/d/1TqQjyiJogQ6gPjmUEtO6Q7gLFs0edkU3lWCLSOt8QyY/edit?tab=t.0#heading=h.djo6baws2lua).

However, if you want to go deeper than that mental model (I recommend not), then
know that **a metatype is not actually a trait**.

It's easy to confuse traits and metatypes because there are some similarities:

- Both metatypes and traits are a set of requirements.
- There are subtyping relationships between traits.
- They can both be used to the right of a `:` (see `VariadicPack`'s
  `_AnyTypeMetaType`)

But a metatype is actually a **set of requirements.**

For example, if we have this struct:

```mojo
struct Spaceship:
    var hp: Int
    def launch(mut self):
        ...
```

The metatype describes "what it takes for a value to be" a `Spaceship`. In this
case, it's that it must have been explicitly specifically declared to be a
`Spaceship`, like `var s: Spaceship`.

The metatype for this trait:

```mojo
trait Launchable:
    def launch(mut self):
        ...
```

...has a more complex metatype. To be a `Launchable`, a value can either be:

- Any struct with a similar `launch` function (until we remove implicit
  conformance, that is).
- Any struct that explicitly declares itself to conform to Launchable, like
  `struct Enterprise(Launchable): ...`
- Any struct that indirectly conforms to Launchable, like
  `struct Enterprise(Constitution): ...` and
  `trait Constitution(Launchable): ...`.

A trait (like `Launchable`) is a collection of metatypes with a set of shared
requirements. Here are some example metatypes for that trait:

- This metatype / these requirements:
  - `def launch(mut self)`
- This metatype / these requirements:
  - `def launch(mut self)`
  - `def land(mut self)`
- This metatype / these requirements:
  - `def launch(mut self)`
  - `def fire(mut self, num_missiles: Int)`

Any type conforming to any of these metatypes can satisfy that trait.

In a way, a trait is "all metatypes that have _at least_ these requirements".

### lit.meta, AnyStruct

`!lit.anystruct` (a.k.a. `LIT::AnyStructType`) is a `!lit.meta` of a struct.

`!lit.meta` is a trait (sort of) that only one specific type conforms to, and
has all the methods and fields of that type.

For example, `!lit.meta<@Int>` is a trait (sort of) that only `Int` conforms to,
and it has all the methods and fields that `Int` has.

These things only exist in LIT. They (and all traits) lower to kgen's
`kgen.type` (a.k.a. "type type", KGEN_TypeType in KGENTypes.td).

### Metatypes in KGEN

There is only one trait/metatype in KGEN, `!kgen.type`, also known as TypeType.
It's oftened shortened to just `type` in our MLIR.

Every trait in Mojo lowers to `!kgen.type`.

All traits are metatypes. All traits lower to `type`. Therefore, **`type` =
metatype**. Try not to think about it.

In KGEN, `<x: type>` is exactly equivalent to `template<typename T>` in C++.

Like C++, KGEN's templates are duck-typed.
