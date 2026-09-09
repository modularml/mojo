# Contextually inferred member references

Date: August 15, 2026
Status: Implemented

Mojo should allow a leading-dot member reference such as `.float64` or
`.red` when the base type is obvious from context, instead of requiring the
fully qualified form `DType.float64` or `Color.red`.

This is the same idea as Swift's "implicit member expression" (leading-dot
syntax for enum cases and static members). It is especially useful for enums,
flag-like types, and APIs that take many values of one nominal type.

## Motivation

Today, Mojo requires naming the type whenever you refer to a static member or
`comptime` alias on that type:

```mojo
struct Color(ImplicitlyCopyable):
  comptime red = Color()
  comptime green = Color()
  comptime blue = Color()

  @staticmethod
  def hsb_to_rgb(h: Int, s: Int, b: Int) -> Color:
    return Color()

def takes_color(c: Color):
  pass

takes_color(Color.red)
takes_color(Color.hsb_to_rgb(120, 100, 50))

var x: Color = Color.blue
var palette: List[Color] = [Color.red, Color.green]
```

When the expected type is already `Color`, repeating `Color.` adds noise
without adding information. The same pattern shows up constantly with
`DType`, status codes, and similar "closed set of named values" types.

In addition to verbosity, this also forces the caller to import the Color more
often than necessary. This doesn't impact `DType` (because it is implicitly
imported everywhere) but does affect user-defined enums.

With contextual inference, those call sites become:

```mojo
takes_color(.red)
takes_color(.hsb_to_rgb(120, 100, 50))

var x: Color = .blue
var palette: List[Color] = [.red, .green]
```

The leading `.` still signals "member of some type"; only the base name is
filled in from the contextual type.

## Goals

1. Allow `.member` wherever an expression has a known contextual type `T`, and
   resolve it as `T.member` (static / `comptime` members and static methods).
2. Support the common wrappers around that form: calls, parameter bindings,
   parentheses, and further attribute chains, so `.hsb_to_rgb(...)`,
   `.alpha_blended[42](...)`, `(.red)`, and `.red.opacity(0.5)` work the same
   way as the bare member.
3. Reuse Mojo's existing contextual-type machinery (typed `var`/`ref`
   initializers, call-argument expected types, return destinations, and
   collection-literal element types) rather than inventing a parallel
   inference system.
4. Produce a clear error when no contextual type is available, for example
   `_ = .red`.

## Non-goals

1. **Inferred members in type position**, such as writing `.Foo` where a type
   is required. This feature only works when there is a contextual type.
2. **Pattern-matching sugar** such as `case .red:`. That can build on this
   later if Mojo grows richer patterns; it is not part of this proposal.
3. **Changing overload resolution rules** beyond "materialize an inferred
   member against the already-chosen expected type." If the contextual type
   is ambiguous, inference fails; we do not search the overload set for a
   type that would make `.member` work.

## Proposal

### Syntax

A primary expression may begin with `.` followed by a member name:

```text
inferred-member-ref ::= '.' identifier
```

This parses like an attribute reference with the base omitted. Postfix
operators apply as usual, so these are all one expression:

```mojo
.red
.hsb_to_rgb(120, 100, 50)
.alpha_blended[42](1, 2)
(.hsb_to_rgb)(120, 100, 50)
.red.opacity(0.5)
```

No new keywords or sigils beyond the leading `.`.

### Resolution rule

An inferred member reference is an unresolved value until the expression is
emitted into a destination that supplies an expected type `T`. At that point
the compiler rewrites the expression by inserting `T` as the base and
continues type checking as if the programmer had written `T.member`:

```text
.member          =>  T.member
.member(args)    =>  T.member(args)
.member[params]  =>  T.member[params]
.member.other    =>  T.member.other
(.member)        =>  (T.member)
```

`T` comes from the same places other contextual types already come from:

- the declared type of a `var` / `ref` being initialized
- the parameter type of a call argument
- the expected return type of a `return`
- the element (or value) type when constructing a typed collection literal,
  such as `List[Color]` or `Dict[String, Color]`

If no expected type is available, the compiler diagnoses an error:

```text
cannot resolve inferred member without a contextual type
```

### What may appear after the leading dot

This intentionally only supports a limited grammar that allows describing
static attributes (just comptime values today) and static methods, plus
further attribute / call / subscript operations on the result. Static methods
can be parametric, so we need to support subscript. As such, we allow:

- `comptime` aliases and other type-scoped values (`Color.red`)
- static methods (`Color.hsb_to_rgb`)
- parametric static methods (`Color.alpha_blended[42]`)
- chains off an inferred member (`Color.red.opacity(0.5)`)

The rewrite inserts `T` only as the base of the leading inferred member. Later
segments are ordinary attribute / call / subscript operations on that value.
So `.red.opacity(0.5)` becomes `T.red.opacity(0.5)`: `red` is looked up on
`T`, then `opacity` is looked up on the value of `T.red` as usual.

### Interaction with collection literals

Collection literals already carry element types into constructor arguments
once the container type is known. Inferred members inside those literals
therefore resolve through the normal call-argument path:

```mojo
def takes_colors(colors: List[Color]):
  pass

takes_colors([.red, .green, .blue])
var palette: List[Color] = [.red, .hsb_to_rgb(120, 100, 50)]
```

No special case is required beyond treating `.member` as a value that can be
materialized against an expected argument type.

### Implementation sketch

The parser produces an `InferredAttributeRef` AST node for `.member`. Type
checking keeps that node (and wrappers such as call, attribute, subscript, and
paren) as an unresolved value until an expected type is available, then
rewrites the tree to a normal attribute reference whose base is a synthetic
type expression for `T`, and re-emits it.

Wrappers that must stay unresolved until the result type is known:

| Form      | Why it must defer                                  |
|-----------|----------------------------------------------------|
| `.m`      | Base type unknown until context exists             |
| `.m(...)` | Callee is `.m`; result type is the contextual type |
| `.m.n`    | Base is `.m`; used for chains like `.red.opacity`  |
| `.m[...]` | Base is `.m`; used for parametric static methods   |
| `(.m)`    | Parentheses must not force early resolution        |

## Examples

```mojo
struct Color(ImplicitlyCopyable):
  comptime red = Color()
  comptime green = Color()
  comptime blue = Color()

  @staticmethod
  def hsb_to_rgb(h: Int, s: Int, b: Int) -> Color:
    return Color()

  @staticmethod
  def alpha_blended[a: Int](x: Int, y: Int) -> Color:
    return Color()

  def opacity(self, amount: Float64) -> Color:
    return Color()

  def __init__(out self):
    pass

def takes_color(c: Color):
  pass

def takes_colors(colors: List[Color]):
  pass

def test():
  # Error: no contextual type.
  _ = .red

  takes_color(.green)
  var x: Color = .blue

  takes_color(.hsb_to_rgb(120, 100, 50))
  takes_color((.hsb_to_rgb)(120, 100, 50))
  takes_color(.alpha_blended[42](1, 2))
  takes_color(.red.opacity(0.5))

  takes_colors([.red, .green, .blue])
  var palette: List[Color] = [.red, .hsb_to_rgb(120, 100, 50)]
```

## Alternatives considered

### Require an explicit short alias

Programmers can already write `comptime C = Color` and then `C.red`. That
helps inside one function but does not remove repetition at API boundaries,
and it does not help call sites that never bind a local alias.

### Infer from the member name alone

Looking up `.red` in every visible type would be ambiguous and expensive.
Contextual typing from an already-known expected type is the smaller, clearer
rule.

### Only allow enum-like `comptime` aliases

Restricting the feature to data members and rejecting static methods would
make `.hsb_to_rgb(...)` illegal. Static factories are a major use case for
the same sugar, and they fall out of the same rewrite rule.
