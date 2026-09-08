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
    case rgb(Int, Int, Int)
```

A `Color` value contains exactly one of these cases at a time. The `red`,
`green`, and `blue` cases carry no associated state, while `rgb` carries a
single payload of type `Tuple[Int, Int, Int]`.

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
    case success(T)
    case failure(E)
```

A value of type `Result[T, E]` is either a `success` containing a `T` or a
`failure` containing an `E`; there is always exactly one active case.

A case may carry zero or one payload. The payload is a single unlabeled type.
As a convenience, writing several types in the case parentheses declares a
payload of that tuple type:

```mojo
enum Token:
    case eof
    case identifier(String)
    case integer(Int)
    case source_range(Int, Int)  # payload type is Tuple[Int, Int]
```

There are no keyword labels on case payloads. Tuples do not have named
elements, so `case source_range(start: Int, end: Int)` is not part of the
model. If named fields are needed, you may define an explicit `struct` and use
that as the payload (then destructure with struct patterns).

The associated state is part of the case itself rather than stored state common
to the enum.

Enums may have methods, conform to traits, and declare comptime values within
them, but they may not have additional state, e.g. this is rejected by the
compiler:

```mojo
enum Foo:
    var timestamp: Int  # error: enums may not have additional state

    case first(Int)
    case second(String)
```

Rationale: This can already be expressed explicitly with composition, and
allowing it would open a raft of semantic questions for initialization, pattern
matching, etc. It is better to start narrow an extend later: for reference,
Swift and Rust never have supported this.

## Declaring Enum Cases

Cases are introduced with the `case` keyword:

```mojo
enum Color:
    case red
    case green
    case blue
```

Payload-bearing cases list a single unlabeled type (or an inline tuple):

```mojo
enum Color:
    case rgb(Int, Int, Int)  # same as rgb(Tuple[Int, Int, Int])
```

This syntax deliberately makes a case resemble the constructor it introduces,
without inventing a parallel labeled-parameter model for payloads.

For now, each case must be declared independently, but we can allow multiple
cases in a single line, e.g. `case red, green, blue` in the future. Requiring
`case` keeps the grammar and declaration model narrow, and leaves room for cases
to independently acquire documentation, attributes, availability information, or
other modifiers in the future.

Associated values may use generic types:

```mojo
enum Optional[T: Fooable]:
    case none
    case some(T)
```

and multi-element payloads are just tuples:

```mojo
enum ParseResult[T: Fooable]:
    case success(T, Int)           # Tuple[T, Int]
    case failure(Int, String)      # Tuple[Int, String]
```

The payload type describes the stored state associated with that alternative.
It is not a set of independent fields of the enclosing enum.

## Constructing Enum Values

An `enum` declaration is syntactic sugar for a `struct`, so we can explain how
the pieces work through the code they synthesize. An enum case introduces a
constructor-like member of its enum type. Payload-less cases can be referenced
directly, through a generated `comptime`:

```mojo
var color = Color.red
```

Payload-bearing cases are constructed using call syntax (this is just invoking
a synthesized static method). Arguments are positional, matching the unlabeled
payload type:

```mojo
var color = Color.rgb(255, 128, 0)
```

Similarly:

```mojo
var result = Result[Int, Error].success(42)
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
setBrushColor(.rgb(255, 0, 0))
```

## Matching and destructuring cases

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

case .rgb(var r, var g, var b):
    use(r, g, b)
```

The case pattern first tests whether the value contains the specified
enumerator. If so, its payload pattern is recursively applied to the case's
single associated value. Multi-element payloads are tuples, so ordinary tuple
patterns decompose them positionally:

```mojo
match some_token:
case .eof:
    ...
case .identifier(value):
    ...
case .integer(value):
    ...
case .source_range(start, end):
    ...  # tuple pattern against Tuple[Int, Int]
```

Keyword labels on the payload are not allowed—there are no labeled fields to
bind:

```mojo
case .source_range(start=x, end=y):  # error: not allowed
```

To destructure by name, store a named struct as the payload and use a struct
pattern:

```mojo
struct SourceRange:
    var start: Int
    var end: Int

enum Token:
    case eof
    case source_range(SourceRange)

match some_token:
case .source_range(SourceRange(start=x, end=y)):
    ...
```

The case pattern's payload subpatterns are ordinary **refutable patterns**. They
are not specific to the `match` statement and should work in the proposed
`if let` syntax as well:

```mojo
if var .rgb(r, g, b) = color:
    use(r, g, b)
```

Case patterns compose recursively with all other pattern forms:

```mojo
case .success([var first, *rest]):
    ...
```

Likewise, enum cases themselves may contain enums or other destructurable types.

The [pattern-matching proposal](pattern-matching.md) defines the general
mechanics of pattern success, failure, binding, and control flow. Enums add
another refutable pattern form to that system, implemented against the
`EnumLike` trait described below.

### How enum pattern matching works

From the programmer's point of view, there are two surface forms:

1. **Case-only patterns** name an alternative with no parentheses:
   `Optional.None`, `.red`, `Color.blue`. These succeed when the subject's
   active case matches the name. They introduce no payload bindings.
2. **Case-with-payload patterns** look like a call: `Optional.Some(ref elt)`,
   `.rgb(var r, var g, var b)`. These succeed when the case matches **and**
   the payload subpattern(s) match the associated value. A multi-element
   payload is one tuple value, so several subpatterns are a tuple pattern.

A few rules follow from that split:

- Cases with **no associated value** (payload type `NoneType`) must use the
  case-only form. Writing `Optional.None()` or `Optional.None(value)` is an
  error: there is nothing to destructure.
- Cases **with** associated value require a payload pattern when parentheses
  are used. `Optional.Some()` with empty parentheses is rejected; write
  `Optional.Some(_)` to ignore the payload, or `Optional.Some` (case-only) to
  test the discriminant without projecting it.
- Payload patterns are positional. There are no `name=` bindings against the
  case itself; use a struct payload when named fields are required.
- Leading-dot forms (`.Some(ref x)`) resolve the case name against the
  subject's type, the same way contextual member lookup works for
  construction.
- Qualified forms (`Optional.Some(ref x)`) require the type prefix to be the
  subject's nominal type (unbound `Optional` may match `Optional[Int]`).

Under the hood, the compiler never special-cases the `enum` keyword when
emitting these patterns. It asks whether the **subject type** conforms to
`EnumLike`, then:

1. Resolves the written case name (`.rgb` / `Color.rgb`) to a case index by
   searching `_enum_case_names`.
2. Emits a predicate comparing `_get_enum_discriminant()` to that index.
3. If the pattern has payload subpatterns, projects the active payload with
   `_unsafe_get_enum_payload[id]()` and recursively matches the payload
   pattern against that projected value (short-circuiting so the payload is
   only projected when the discriminant matches).

Payload bindings use the same `var` / `ref` / bare-binding rules as every
other pattern. For example:

```mojo
match value:
case .some(ref element):
    use(element)  # borrows the payload; mutability follows the subject
```

borrow the payload rather than copying or moving it, subject to the usual
ownership and exclusivity rules.

Unknown case names (`Optional.Nope`) are diagnosed against `_enum_case_names`
rather than ordinary member lookup: enum cases need not exist as real members
of the type for matching to work (hand-written `EnumLike` conformances can
expose logical cases that are not constructor APIs).

## Methods, traits, and generic enums

Enums should be full nominal Mojo types rather than restricted tagged unions.

They should support methods in the same way as structs:

```mojo
enum Color:
    case red
    case green
    case blue
    case rgb(Int, Int, Int)

    def is_grayscale(self) -> Bool:
        match self:
        case .rgb(var r, var g, var b):
            return r == g == b  # Mojo loves Python, woo!
        case _:
            return False
```

Likewise, enums should support trait conformance:

```mojo
enum Result[T: Movable, E: Movable](Movable):
    case success(T)
    case failure(E)
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
    case some(T)
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

## The `EnumLike` trait

The `enum` syntax is syntactic sugar for introducing a sum type, but the
underlying semantic abstraction is broader than syntactic enums. A type that has
a finite set of mutually exclusive alternatives, each with a statically known
payload shape, is **enum-like** regardless of how it was declared or
represented.

Mojo captures this abstraction with a built-in `EnumLike` trait:

```mojo
trait EnumLike:
    comptime _enum_case_length: Int
    comptime _enum_case_names: _MLIR.KGENParamListType[KGENString]
    comptime _enum_case_types: _MLIR.KGENParamListType[AnyType]

    comptime _enum_elt_type_for_case[id: Int]: AnyType = TypeList[
        Self._enum_case_types
    ]()[id]

    def _get_enum_discriminant(self) -> Int:
        ...

    # FIXME: Prefer an interior origin so payload refs preserve subject
    # mutability cleanly.
    def _unsafe_get_enum_payload[
        id: Int
    ](ref self) -> ref[self] TypeList[Self._enum_case_types]()[id]:
        ...
```

The compiler automatically synthesizes an `EnumLike` conformance for every
`enum`, but the trait is not restricted to compiler-generated enum types.
User-defined types may conform as well when they provide equivalent semantics.
`Optional` is the motivating example: it is a struct (not spelled `enum`) that
conforms to `EnumLike` so pattern matching can treat `None` / `Some` as logical
cases:

```mojo
struct Optional[T: Movable](Copyable, EnumLike):
    ...
    comptime _enum_case_length = 2
    comptime _enum_case_names = ParameterList.of[
        "None".value, "Some".value
    ].values
    comptime _enum_case_types = TypeList.of[NoneType, Self.T].values

    def _get_enum_discriminant(self) -> Int: ...
    def _unsafe_get_enum_payload[id: Int](ref self) -> ...: ...
```

The runtime discriminant selects the active alternative:

```mojo
Color.red._get_enum_discriminant() == 0
Color.green._get_enum_discriminant() == 1
Color.blue._get_enum_discriminant() == 2
Color.rgb(1, 2, 3)._get_enum_discriminant() == 3
```

The discriminant is a semantic case index, not necessarily a promise about the
physical representation or ABI of the type. An implementation may use an
explicit tag, a niche representation, or some other encoding while presenting
the same `EnumLike` interface.

Payload type `NoneType` means "this case has no associated value." That is how
`Optional.None` is modeled, and it is what causes the pattern matcher to reject
payload subpatterns for that case.

### `EnumLike` as the basis for pattern matching

Syntactic enums should not be privileged by the pattern-matching
implementation. The compiler implements enum-style refutable patterns in terms
of `EnumLike`.

When matching a subject whose type conforms to `EnumLike`, for example:

```mojo
if .rgb(var r, var g, var b) = color:
    ...
```

or:

```mojo
match opt:
case Optional.Some(ref elt):
    use(elt)
case Optional.None:
    handle_missing()
```

the compiler performs the steps in [How enum pattern matching
works](#how-enum-pattern-matching-works): resolve the case name through
`_enum_case_names`, compare `_get_enum_discriminant()`, and (when needed)
project with `_unsafe_get_enum_payload[id]()` before recursively matching
payload patterns.

This means the compiler's pattern-matching machinery operates on **enum-like
types**, rather than special-casing declarations spelled with `enum`.

### Reflection and derived conformances

The same interface provides a useful foundation for generic reflection. For
example, an `Equatable` default implementation can notice `EnumLike` types and
first compare the active cases:

```mojo
if lhs._get_enum_discriminant() != rhs._get_enum_discriminant():
    return False
```

When the discriminants match, `_enum_case_types` identifies the payload type
corresponding to that case, allowing the implementation to compare the
associated state when that state is itself `Equatable`.

Similarly, `_enum_case_names` enables reflective operations that need a
human-readable name for an alternative, such as formatting, debugging,
serialization, or generic inspection:

```mojo
comptime name = T._enum_case_names[index]
```

Other derived behaviors such as hashing or serialization can use the same
metadata.

This gives the language a useful separation of concerns:

- `enum` is convenient syntax for declaring a nominal sum type.
- The compiler lowers that syntax onto Mojo's ordinary nominal-type machinery
  and synthesizes an `EnumLike` conformance.
- `EnumLike` defines the semantic and reflective interface for finite
  alternatives.
- Pattern matching is implemented against `EnumLike`, rather than giving
  syntactic enums special treatment.
- User-defined representations may participate in the same system by conforming
  to `EnumLike` themselves (as `Optional` already does in the prototype).

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
    case some(T)

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
- Cases may carry a single unlabeled payload (including an inline tuple type).
  Named payload fields require an explicit struct type.
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

### Future consideration: Control over layout and discriminators

The initial proposal provides a pure algebraic data type. Languages like C and
C++ use a very different model, which is nonetheless very useful - enums can
have specified discriminator values, and supports efficient casts to integer
values, and support a specifier for layout information. For example:

```C++
enum MyColors : int32_t {
    red   = 0xFF0000,
    green = 0x00FF00,
    blue  = 0x0000FF
}
```

The base proposal doesn't include such affordances, but they can be layered on
top when and if demand appears and the complexity is justified. Mojo doesn't
currently provide fine-grain control of struct layout, which is something that
should be tackled exposing layout control for enums.

Note that C also supports "enums as random collections of values", such as:

```C++
enum {
  Value1 = 42,
  Value2 = 17,
  Value3 = 42
};
```

This is a fundamentally different construct than an algebraic data type. Mojo
supports such concepts with existing `comptime` values, enums do not need to
provide support for this use-case.

### Future consideration: Recursive/indirect enum cases

The initial proposal doesn't provide support for
[Swift-style indirect enum cases](https://www.hackingwithswift.com/example-code/language/what-are-indirect-enums).

We can evaluate adding such a thing in the future, but it would be preferred to
express this with existing language features if possible, e.g.:

```mojo
enum LinkedListItem[T: Copyable] {
    case endPoint(T)
    case linkNode(T, HeapBox[LinkedListItem])
}
```

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
- How do anonymous cases participate in reflection and `EnumLike` metadata?
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
