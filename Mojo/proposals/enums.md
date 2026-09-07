# Enums for Mojo

**Status**: Concept proposal.

Date: Aug 29, 2026

This is a set of notes to braindump thoughts on adding enums to Mojo. It’s not
meant to be canonical or strongly held opinions, it is meant to help guide
further design discussions. This builds on the
[Mojo Pattern Matching Notes](pattern-matching.md) doc - make sure you have
that in context before digging into this.

## Introduction

This proposal adds **enums** to Mojo: nominal types whose values may be one of a
fixed set of alternatives. Each alternative, called a **case**, may optionally
carry associated state.

Enums complement structs in the type system. A struct represents a product of
its stored values, while an enum represents a choice between alternatives.
A simple working example of an enum is:

```mojo
enum Color:
    case red
    case green
    case blue
    case rgb(r: Int, g: Int, b: Int)
```

A `Color` value contains exactly one of these cases at a time. The `red`,
`green`, and `blue` cases carry no associated state, while `rgb` carries three
integer values.

Enums should otherwise behave like normal Mojo nominal types. They may have
methods, conform to traits, be generic, contain nested declarations, and
participate in Mojo's ownership system. Indeed, enums are actually syntactic
sugar for structs in Mojo.

This proposal deliberately starts with a narrow but valuable model that allows
future extension as concrete use-cases arise.

## Enum Model

An enum is a nominal **sum type** with a statically known, closed set of cases.
Here is a simple example that we’d like to support:

```mojo
enum Result[T: Deinitable, E: Deinitable]:
    case success(value: T)
    case failure(error: E)
```

A value of type `Result[T, E]` is either a `success` containing a `T` or a
`failure` containing an `E` , there is always exactly one active case.

A case may contain zero or more associated values:

```mojo
enum Token:
    case eof
    case identifier(value: String)
    case integer(value: Int)
    case source_range(start: Int, end: Int)
```

The associated state is part of the case itself rather than stored state common
to the enum.

Enums may have methods, conform to traits, and declare comptime values within
them, but they may not have additional state, e.g. this is rejected by the
compiler:

```mojo
enum Foo:
    var timestamp: Int  # error: enums may not have additional state

    case first(value: Int)
    case second(value: String)
```

This can already be expressed explicitly with composition, and allowing it would
open a raft of semantic questions for initialization, pattern matching, etc. It
is better to start narrow an extend later: for reference, Swift and Rust never
have supported this.

## Declaring Enum Cases

Cases are introduced with the `case` keyword:

```mojo
enum Color:
    case red
    case green
    case blue
```

Payload-bearing cases use argument-like syntax:

```mojo
enum Color:
    case rgb(r: Int, g: Int, b: Int)
```

This syntax deliberately makes a case resemble the constructor it introduces,
specifying the keyword labels.

For now, each case must be declared independently, but we can allow multiple
cases in a single line, e.g. `case red, green, blue` in the future. Requiring
`case` keeps the grammar and declaration model narrow, and leaves room for cases
to independently acquire documentation, attributes, availability information, or
other modifiers in the future.

Associated values may use generic types:

```mojo
enum Optional[T: Fooable]:
    case none
    case some(value: T)
```

and cases may carry multiple values:

```mojo
enum ParseResult[T: Fooable]:
    case success(value: T, consumed: Int)
    case failure(offset: Int, message: String)
```

Case parameters describe the stored state associated with that alternative. They
are not independent fields of the enclosing enum.

The initial design should avoid unnecessarily extending the function parameter
model into cases. Features such as default arguments, variadic arguments, or
case overloading can be considered separately if compelling use cases arise.
Let’s stay minimal and focused.

## Constructing Enum Values

An `enum` declaration is syntactic sugar for a `struct`, so we can explain how
the pieces work through the code they synthesize. An enum case introduces a
constructor-like member of its enum type. Payload-less cases can be referenced
directly, through a generated `comptime`:

```mojo
var color = Color.red
```

Payload-bearing cases are constructed using call syntax (this is just invoking
a synthesized static method):

```mojo
var color = Color.rgb(r=255, g=128, b=0)
```

Similarly:

```mojo
var result = Result[Int, Error].success(value=42)
```

Case constructors already naturally work with contextual member inference,
allowing the enum qualification to be omitted or shortened where the surrounding
context determines the enum type. For example, this will “just work” like it
does today for structs:

```mojo
setBrushColor(.red)
```

and:

```mojo
setBrushColor(.rgb(r=255, g=0, b=0))
```

## Matching and Destructuring Cases

Enum cases introduce a corresponding kind of pattern that tests the active case
and decomposes its associated state. For example:

```mojo
match color:
case .red:
    print("red")

case .green:
    print("green")

case .blue:
    print("blue")

case var .rgb(r, g, b):
    use(r, g, b)
```

The case pattern first tests whether the value contains the specified
enumerator. If so, its associated-value patterns are recursively applied to the
case's payload.

Case patterns are therefore ordinary **refutable patterns**. They are not
specific to the `match` statement and should work in the proposed `if let`
syntax as well:

```mojo
if .rgb(var r, var g, var b) = color:
    use(r, g, b)
```

Case patterns should compose recursively with all other pattern forms:

```mojo
case .success(value=[var first, *rest]):
    ...
```

Likewise, enum cases themselves may contain enums or other destructurable types.

The pattern-matching proposal defines the general mechanics of pattern success,
failure, binding, and control flow. Enums merely add another refutable pattern
form to that system.

## Methods, Traits, and Generic Enums

Enums should be full nominal Mojo types rather than restricted tagged unions.

They should support methods in the same way as structs:

```mojo
enum Color:
    case red
    case green
    case blue
    case rgb(r: Int, g: Int, b: Int)

    def is_grayscale(self) -> Bool:
        match self:
        case var .rgb(r, g, b):
            return r == g == b  # Mojo loves Python, woo!
        case _:
            return False
```

Likewise, enums should support trait conformance:

```mojo
enum ResultT: Movable, E: Movable:
    case success(value: T)
    case failure(error: E)
```

and generic parameters and constraints should work according to the normal rules
for nominal types.

More generally, enum declarations should support the same capabilities as struct
declarations wherever doing so is semantically meaningful:

- instance and static methods;
- trait conformance and default implementation inheritance—e.g. enums should be
  able to be `Equatable` when all their state is;
- generic parameters and constraints;
- comptime values / type aliases;
- computed properties, nested declarations, etc (some day).

The important restriction is on **stored instance state**, not on the
capabilities of the nominal type itself.

This gives Mojo a simple guiding rule:

> An enum differs from a struct primarily in how its instance state is declared.
> Structs contain all of their stored fields simultaneously; enums contain the
> associated state of exactly one case.
>

Everything else should reuse the normal Mojo type system as much as possible.

## Ownership and Mutation

Enums should participate naturally in Mojo's ownership system.

The ownership properties of an enum depend on the state that its cases may
contain. For example:

```mojo
enum Optional[T: Deinitable]:
    case none
    case some(value: T)
```

must correctly represent the ownership semantics of `T` when the `some` case is
active.

Moving an enum moves the active payload. Destroying an enum destroys the payload
of the active case. Copyability should similarly depend on whether the possible
associated values support the required copying semantics.

An enum value may also be replaced with a value containing a different case:

```mojo
var result = Result.success(value)
result = Result.failure(error)
```

This destroys the old active payload and initializes the new one according to
Mojo's normal assignment semantics.

Pattern matching should provide the normal Mojo binding conventions for
associated state. For example:

```mojo
match value:
case .some(ref element):
    use(element)
```

should borrow the payload rather than copying or moving it.

Similarly, mutable access to an enum payload should follow the same ownership
and exclusivity rules as mutable access to ordinary stored state.

## The `SumType` Trait

The `enum` syntax is syntactic sugar for introducing a sum type, but the
underlying semantic abstraction is broader than syntactic enums. A type that has
a finite set of mutually exclusive alternatives, each with a statically known
payload shape, is a **sum type** regardless of how it was declared or
represented.

Mojo should capture this abstraction with a built-in `SumType` trait:

```mojo
trait SumType:
    def get_discriminator(self) -> Int:
        ...

    comptime payload_types: <type list>
    comptime case_names: <string list>
    # Other stuff too, not fully designed.
```

The compiler automatically synthesizes a `SumType` conformance for every `enum`,
but the trait should not be restricted to compiler-generated enum types.
User-defined types may conform as well when they provide equivalent semantics,
for example:

```mojo
struct MyCompactOptionalT: Fooable:
    ...
```

The runtime discriminator selects the active alternative:

```mojo
Color.red.get_discriminator() == 0
Color.green.get_discriminator() == 1
Color.blue.get_discriminator() == 2
Color.rgb(r=1, g=2, b=3).get_discriminator() == 3
```

The discriminator is a semantic case index, not necessarily a promise about the
physical representation or ABI of the type. An implementation may use an
explicit tag, a niche representation, or some other encoding while presenting
the same `SumType` interface.

### `SumType` as the Basis for Pattern Matching

An important goal of this design is that syntactic enums should not be
privileged by the pattern-matching implementation. Instead, the compiler
implements enum-style refutable patterns in terms of the `SumType` trait.

For when type checking an assignment into a pattern for a type conforming to
`SumType`, e.g.:

```mojo
if .rgb(var r, var g, var b) = color:
    ...
```

The compiler performs three steps:

1. Resolve `.rgb` to the corresponding case index using the static `SumType`
   metadata.
2. Compare that case index with `color.get_discriminator()`.
3. If it matches, project the payload and recursively apply the payload pattern.

This means the compiler's pattern-matching machinery operates on **sum types**,
rather than special-casing declarations spelled with `enum`.

### Reflection and Derived Conformances

The same interface provides a useful foundation for generic reflection. For
example, `Equatable` trait provides a default implementation for eq/ne. It can
notice sum types and first compare the active cases:

```mojo
if lhs.get_discriminator() != rhs.get_discriminator():
    return False
```

When the discriminators match, `payload_types` identifies the payload type
corresponding to that case, allowing the implementation to compare the
associated state when that state is itself `Equatable`.

Similarly, `case_names` enables reflective operations that need a human-readable
name for an alternative, such as formatting, debugging, serialization, or
generic inspection:

```mojo
comptime name = T.case_names[index]
```

Other derived behaviors such as hashing or serialization can use the same
metadata.

This gives the language a useful separation of concerns:

- `enum` is convenient syntax for declaring a nominal sum type.
- The compiler lowers that syntax onto Mojo's ordinary nominal-type machinery
  and synthesizes a `SumType` conformance.
- `SumType` defines the semantic and reflective interface for finite
  alternatives.
- Pattern matching is implemented against `SumType`, rather than giving
  syntactic enums special treatment.
- User-defined representations may participate in the same system by conforming
  to `SumType` themselves.

This is consistent with the broader implementation philosophy of the proposal:
enums should introduce useful high-level syntax and synthesized machinery
without creating a parallel type system underneath Mojo.

## Other Implementation Details

At the language implementation level, an `enum` should be treated as **syntactic
sugar for a struct with compiler-synthesized state and members**. This means
that it should become a `StructDeclOp` in the compiler—we don’t need a distinct
`EnumDeclOp`. Swift got this wrong, and it was the source of many challenges
over the years.

For example:

```mojo
enum Optional[T]:
    case none
    case some(value: T)

    def has_value(self) -> Bool:
        if .some(_) = self:
            return True
        return False
```

can conceptually be viewed as defining a nominal struct-like type containing
compiler-managed representation state together with synthesized operations
corresponding to its cases:

```mojo
struct Optional[T]:
    var _state: `!kgen.variant<...>`

    <synthesized case constructors>
    <synthesized destruction/move/copy behavior>
    <synthesized case inspection/decomposition support>

    def has_value(self) -> Bool: ...
```

The important point is the **semantic lowering model**: an enum does not require
an entirely separate kind of nominal type throughout the compiler.

This implementation strategy has several advantages:

- Existing machinery for nominal types, generics, trait conformance, methods,
  visibility, and member lookup can be reused unchanged.
- Ownership semantics can be expressed through synthesized operations rather
  than building a parallel ownership system for enums.
- Improvements to struct capabilities naturally apply to enums where
  appropriate.
- Enum-specific compiler machinery can remain focused on case representation,
  case transitions, and pattern decomposition.
- The surface language maintains a strong distinction between structs and enums
  even though their implementation shares substantial infrastructure.
- This is easy to implement!

In this sense, `enum` is analogous to other high-level language features that
synthesize otherwise tedious or unsafe implementation details. The syntax
provides a concise and semantically meaningful way to declare a sum type while
allowing the compiler to lower it onto Mojo's existing nominal-type
infrastructure.

## Initial Restrictions and Future Directions

The initial enum design should intentionally remain narrow.

In particular:

- Every alternative is explicitly introduced with `case`.
- Enum instance storage comes exclusively from case-associated values.
- Enums do not contain additional stored instance fields.
- Cases form a closed set known to the compiler.
- Cases may carry associated state.
- Enums otherwise support normal nominal-type capabilities such as methods,
  traits, and generics.
- The physical representation of an enum is unspecified by default. Let’s not
  worry about C interop or other advanced features.
- Enum case patterns integrate with Mojo's general pattern system rather than
  introducing separate destructuring semantics.

Starting from a smaller semantic core is intentional. A closed sum type with
associated state, normal nominal-type capabilities, and integration with Mojo's
ownership and pattern systems provides the important functionality while keeping
the language and implementation model straightforward. More specialized
capabilities can be added later without weakening that foundation.

### Future consideration: Anonymous Sum Types

Note: This is specifically NOT part of the initial proposal. This is included to
sketch out a possible next step that can only be considered after the basic
proposal lands and settles.

One possible extension is to allow **anonymous sum types** directly in type
position, rather than requiring every sum type to have a nominal `enum`
declaration.

For example, we could imagine writing:

```mojo
def activate_light(state: .on | .off):
    ...
```

rather than declaring a named type:

```mojo
enum LightState:
    case on
    case off

def activate_light(state: LightState):
    ...
```

This is appealing for small, locally useful sets of alternatives. It is
analogous to anonymous structural types in other parts of the type system and
avoids introducing a declaration whose only purpose is to name two or three
cases.

One possible implementation model would be to define a generic library
type—conceptually something like `Enum[...]`—and desugar the anonymous syntax
into an instantiation of it: `.on | .off`.  This could conceptually become
something like: `Enum["on", "off"]`.

However, this becomes substantially more complicated once the cases are expected
to behave like normal enum cases.

For example:

```mojo
def activate_light(state: .on | .off):
    activate_light(.on)
```

requires `.on` to participate in contextual member lookup even though there is
no nominal declaration containing an `on` member. This can be supported with a
suitable implementation of `Enum.__getattr__` , but would be complicated.

Payload-bearing cases make this more challenging:

```mojo
def handle(value: .none | .some(Int)):
    ...

handle(.some(42))
```

The compiler must somehow discover that `.some` is a constructor available on
this particular structural sum type, understand its parameter list, produce
useful diagnostics for misspelled cases, and make the same information available
to pattern matching and reflection.

Anonymous sum types also raise broader questions that nominal enums answer
naturally:

- What is the concrete syntax, how do we handle the ambiguities?
- What determines whether two anonymous sum types are the same type?
- Does case ordering matter for type identity?
- Are `.on | .off` and `.off | .on` equivalent?
- How are case names represented in generic signatures and mangling?
- How do documentation and diagnostics refer to the type?
- Where do methods or trait conformances live?
- How does contextual lookup find case constructors?
- How do anonymous cases participate in reflection and `SumType` metadata?
- How do source and ABI stability work when an alternative is added or
  reordered?

None of these questions make anonymous sum types inherently undesirable. In
fact, they could eventually be a useful lightweight facility, particularly for
APIs whose alternatives have no meaningful identity outside of a single
signature.

However, they are orthogonal to the fundamental enum feature and require
considerably more type-system and member-lookup machinery than nominal enums. If
there is enough well-motivated demand, we can investigate adding this in the
future.
