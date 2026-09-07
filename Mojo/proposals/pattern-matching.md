# Mojo Pattern Matching

**Status**: Concept proposal.

Date: Aug 29, 2026

This is a set of notes to braindump thoughts on adding pattern matching to Mojo.
It’s not meant to be canonical or a set of strongly held opinions; it is meant
to help guide further design discussions.

A separate [Enums for Mojo Notes](enums.md) doc builds on the pattern system
described here, but enums are not required to implement or motivate the core
pattern-matching design.

## Introduction

**Pattern matching** is a general mechanism for testing whether a value has a
particular structure and, when it does, extracting pieces of that value into new
bindings. Python has rich support for pattern matching - Mojo should embrace
its existing core syntax, and extend it in ways that are specific to its needs
(e.g. due to ownership handling).

Let's look at an example. Here we can see a pattern that simultaneously tests
parts of a tuple and bind others:

```mojo
def inspect(point: Tuple[Int, Int]):
    match point:
    case (0, 0):
        print("origin")
    case (var x, 0):
        print("on the x axis:", x)
    case (0, var y):
        print("on the y axis:", y)
    case (var x, var y):
        print("point:", x, y)
```

The essential operation here is structural: compare parts of a value against
patterns while binding other parts to names.

Patterns are useful well beyond `match` statements. Python has rich pattern
matching, but it is not alone. Languages such as Rust and
Swift treat patterns as a general language concept that can be reused across
declarations and control-flow constructs. For example, Rust uses patterns in
ordinary destructuring:

```rust
let (x, y) = point;

if let [first, second] = values.as_slice() {
    use_values(first, second);
}

let Point { x, y } = point;
```

Swift similarly uses patterns in constructs such as `switch`, `if case`, `guard
case`, and `for case`:

```swift
if case let (x, 0) = point {
    print("x axis:", x)
}
```

These languages therefore suggest the same useful decomposition for Mojo:

1. **Patterns describe values.** They provide matching, decomposition, and
   binding semantics independently of any particular control-flow construct.
2. **Patterns compose recursively.** Binding, wildcard, tuple, sequence, value,
   struct, and future pattern forms should fit into a common model rather than
   being unrelated language features.
3. **Different language constructs can consume patterns differently.** A `match`
   statement may try several patterns, while an `if` may test one pattern and
   ordinary destructuring may require a particular structure.

The goal is therefore to establish **patterns as a first-class piece of the
language grammar and semantic model**, rather than designing `match` as an
isolated control-flow feature.

This also gives us a natural implementation progression. The language can first
define the grammar, binding behavior, refutability, control-flow semantics, and
lowering model for patterns over existing value forms. Other language features,
including the separate enum proposal, can then add new kinds of patterns without
requiring their own destructuring machinery.

### Destructuring in Mojo Today

Mojo already supports rich **destructuring** in declarations and assignments.
For example:

```mojo
# Simple assignment
var x = get_value()

# Tuple destructuring
var (x, y) = get_pair()

# Generalized assignment targets
(var x, ref y, _) = get_values()
(a[i], b[j]) = get_pair()

# These nest arbitrarily
(var x, (ref y, _)) = get_nested_values()
```

The important observation is that Mojo already has a composable language for
describing how the components of a value are distributed into bindings and
assignment targets. `var x`, `ref y`, `_`, generalized lvalues, and `(p1, p2,
...)` can all participate recursively in destructuring. As with Python, tuple
destructuring doesn't even require parentheses in contexts where the grammar is
unambiguous.

Mojo also supports destructuring in other assignment-like contexts, such as
`for` loop targets:

```mojo
for key, value in entries:
    ...
```

Today, destructuring is required to be statically known to succeed. For example,
given:

```mojo
(var x, var y) = value
```

the compiler must know that `value` has an appropriate two-element tuple
structure. A value with an incompatible structure is a compile-time error rather
than something that can fail dynamically.

> **💡 NOTE:** The current tuple unpacking support is extremely hard-coded to
> `Tuple` specifically; this should be generalized at some point.
>

### Destructuring vs. Match Patterns

Destructuring and pattern matching are closely related, but they are not quite
the same thing. This proposal follows Python's general design by keeping them as
distinct grammatical contexts.

A **destructuring target** describes where pieces of a value should go. A
**match pattern** instead asks whether a value satisfies some condition and may
introduce bindings when it does:

```mojo
match get_pair():
    # Bind x, check against 0.
    case (var x, 0):
        use(x)

    # Error: arbitrary expressions/lvalues aren't value patterns.
    case (var x, a[2]):
```

Match patterns intentionally use a restricted grammar rather than treating
arbitrary Mojo expressions as equality tests. They can include literals and
other explicitly supported pattern forms, but not arbitrary calls, operators,
subscripts, or generalized lvalues. This avoids surprising interactions with
expressions that produce lvalues or references and keeps patterns statically
understandable.

Ordinary assignment continues to use Mojo's existing assignment-target rules:

```mojo
(var a, 0) = get_pair()   # Error: `0` isn't an assignment target.
0 = get_value()           # Error.

a[2] = get_value()        # assignment into generalized lvalue is ok.
```

This distinction ensures that adding pattern matching does not change the
meaning of existing assignments. In particular, `a[2] = value` always means
"store into `a[2]`," never "match against the current value of `a[2]`."

Despite this difference, destructuring targets and match patterns share much of
the same recursive structure: tuple decomposition, `var`/`ref` bindings,
wildcards, sequence decomposition, and other forms can reuse the same underlying
compiler machinery.

## The `match` Statement

The most direct way to use match patterns is with a `match` statement. For
example, a dynamically-sized list can be classified by its shape. Here we use
the `[a, b, c]` pattern which matches against a list of a specific length:

```mojo
match values:
case []:
    print("empty")

case var [x]:
    print("one element:", x)

case var [x, y]:
    print("two elements:", x, y)

case _:
    print("many elements")
```

Conceptually, `match` evaluates its subject once and considers its cases in
source order.

For each `case`, the corresponding pattern is applied to the subject. If the
pattern succeeds, any bindings it introduces become available in the case body
and that body executes. If the pattern fails, execution proceeds to the next
case.

This introduces the important distinction between **irrefutable** and
**refutable** patterns.

An irrefutable pattern is statically guaranteed to match every value of its
input type. For example:

```mojo
case var x:
case _:
```

A refutable pattern may or may not match depending on the runtime value. A
fixed-length sequence pattern is one simple example:

```mojo
case var [x, y]:
```

If the subject is a dynamically-sized list, this succeeds only when it contains
exactly two elements.

Value patterns are another important refutable form:

```mojo
match point:
case (0, 0):
    print("origin")

case (var x, 0):
    print("on the x axis:", x)
```

Here `(var x, 0)` first decomposes the tuple, binds its first element to `x`,
and requires its second element to match `0`. Failure of either requirement
means that the next case is considered.

Patterns compose recursively, so refutability naturally composes as well:

```mojo
match values:
case [(var x, 0), var z]:
    handle_special_value(x, z)

case [var x]:
    handle_one_value(x)

case _:
    handle_other(values)
```

A pattern is therefore **irrefutable when the compiler can prove that it matches
every value of its input type; otherwise it is refutable**.

Patterns that can be proven to never match should instead be diagnosed at
compile time. For example, matching a statically known two-element tuple with a
three-element tuple pattern is not a runtime match failure—it is simply invalid
code.

> 💡 Patterns that can **NEVER** match are a compile-time error. We expect `case
> var (a, b):` to be a compile-time error if the matched value is a 3-element
> tuple.
>

While not described here, Mojo should also support a `comptime match`
statement when the match subject is a parameter expression.

### Exhaustiveness

One question is whether a `match` statement must cover every possible input
value, either because the compiler can prove its cases exhaustive or because a
final irrefutable pattern such as `_` is present:

```mojo
match values:
case []:
    ...
case var [x]:
    ...
case _:
    ...
```

Exhaustiveness is straightforward for some pattern forms but difficult to decide
in full generality. Mojo could initially use conservative analysis: when the
compiler cannot prove that the cases cover every possible value, require a final
irrefutable case. Alternatively, we could not require exhaustiveness and
instead warn about unreachable cases after an irrefutable pattern.

More sophisticated exhaustiveness and redundancy analysis can be layered on as
the pattern language becomes richer.

The important point is that `match` itself remains simple: evaluate the subject
once, try patterns in order, and execute the first case whose pattern succeeds.
The richness comes from the pattern language, whose forms recursively compose
and can be extended independently over time.

## Pattern Forms to Add

Mojo already has several building blocks that naturally carry over into match
patterns: `var` / `ref` bindings, `_` wildcards, tuple decomposition, and
recursive nesting. We should extend this foundation with the following pattern
forms.

1. **Value patterns** match against a literal:

    ```mojo
    case 0:
    case "hello":
    case (0, 1):
    ```

    Arbitrary expressions are not supported by Python and therefore, we
    shouldn’t accept them as part of the initial implementation. See “Future
    Directions” for why we don’t want to accept `case x:` where `x` is an
    existing value as part of this proposal.

2. **Sequence patterns** decompose sequence-like values:

    ```mojo
    case []:
    case var [x]:
    case var [x, y]:
    ```

    These are refutable when the runtime sequence may have a different length.
    The same underlying decomposition support can also be used by ordinary
    destructuring assignments. This should be tied into some trait that the
    collection conforms to.

3. **Variable-length sequence patterns** capture a prefix, suffix, or remainder,
   tied into the same trait:

    ```mojo
    case var [first, *rest]:
    case var [first, *middle, last]:
    ```

4. **OR patterns** match when any of several alternatives match:

    ```mojo
    case 0 | 1:
    case [var x, 0] | [var x, 1]:
    ```

    Alternatives that introduce bindings should be required to introduce
    compatible bindings.

5. **AS patterns** apply a pattern while also retaining the complete matched
   value:

    ```mojo
    case [var x, var y] as value:
        ...
    ```

    The exact ownership convention for the `value` binding needs to be
    determined.

6. **Struct patterns** decompose struct-like values:

    ```mojo
    case Point(x=var x, y=var y):
        ...
    ```

7. **Mapping patterns** match particular keys and decompose their values:

    ```mojo
    case {"name": var name, "age": var age}:
        ...
    ```

    These are useful, but likely lower priority than sequence and struct
    patterns.

Other language features may add additional pattern forms. In particular, the
separate enum/sum-type proposal adds patterns for selecting and decomposing
sum-type alternatives through the same underlying machinery.

Finally, **match guards** are useful but are not themselves patterns:

```mojo
case [var x, var y] if x < y:
    ...
```

The pattern matches first and introduces its bindings; the guard is then
evaluated to determine whether the case is selected.

## Conditional Pattern Bindings: “`if let`” for Mojo

Pattern matching also fits naturally into Mojo's existing `if` statement without
requiring a distinct `if let` construct.

A useful property of Mojo here is that `=` is already a **statement**, rather
than an expression.

We can extend the grammar of `if` conditions to accept either boolean
expressions or `pattern = value` conditional-match clauses:

```mojo
if condition:
    ...

if pattern = value:
    ...
```

In the second form, the left-hand side is parsed using the full **match-pattern
grammar**, rather than the more restricted destructuring grammar used by
ordinary assignments. For example:

```mojo
if var [a, 0] = get_list():
    use(a)
```

The right-hand side is evaluated once and matched against `[var a, 0]`. The
condition succeeds only if the list contains exactly two elements and its second
element matches `0`. On success, `a` is bound and the body executes; otherwise
the condition is false.

This is intentionally more expressive than ordinary destructuring:

```mojo
# Ordinary destructuring: valid, but throws if the length is wrong.
var [a, b] = get_list()

# Ordinary destructuring: invalid because `0` isn't an assignment target.
var [a, 0] = get_list()

# Conditional matching: valid; failure takes the false path.
if var [a, 0] = get_list():
    use(a)
else:
    handle_other_shape()
```

Bindings introduced by a successful conditional match are scoped to the
corresponding `if` body:

```mojo
if var [first, second] = get_list():
    use(first, second)

# `first` and `second` are not available here.
```

This provides the capability commonly spelled `if let` in languages such as Rust
and Swift, but without introducing a new `let`-specific construct. The `if`
itself establishes match-pattern context, so nested and refutable patterns
compose naturally:

```mojo
if (var x, [var y, 0]) = get_value():
    use(x, y)
```

Irrefutable patterns can technically appear in this position as well:

```mojo
if var x = get_value():
    use(x)
```

but such a condition is statically known to succeed and is likely a mistake. The
compiler should probably warn about this, just as it may diagnose other
conditions known to be constant.

This gives `if` conditions two natural forms:

- A boolean expression succeeds when it evaluates to `True`.
- A conditional pattern match succeeds when its pattern matches.

Pattern failure in an `if` is therefore ordinary control flow rather than an
exception.

The same treatment should apply to `while` loop conditions:

```mojo
while var [item, 0] = get_next():
    use(item)
```

This gives Mojo the ergonomics of `if let` and `while let` while reusing the
normal `if` / `while` syntax and the same match-pattern language used by `case`.

## Looking Ahead

Although this proposal describes a fairly broad pattern-matching system, it does
**not** need to be implemented as one large monolithic project. Most of the
pieces are orthogonal and can land independently as relatively small
subprojects.

The key implementation investment is a recursive representation for **match
patterns that may succeed or fail**, together with a lowering model that can
apply such a pattern to a value and either produce its bindings or report
failure.

A very small starting point is a literal value pattern composed with Mojo's
existing tuple decomposition:

```mojo
match get_tuple_pair():
case (var x, 0):
    use(x)
case _:
    ...
```

Mojo already understands tuple decomposition and `var x`; the new operation is
simply testing the second element against `0` with `==` semantics. This
exercises the core matching machinery:

- representing refutable patterns;
- evaluating the matched value once;
- recursively composing binding and value patterns;
- producing bindings only when the complete pattern succeeds;
- and transferring control to the next `case` when it fails.

From there, the rest of the language can grow incrementally.

A closely related project is **failable sequence destructuring**:

```mojo
var [a, b, c] = get_list()
```

This belongs to ordinary destructuring rather than the full match-pattern
grammar, but it can share the same underlying decomposition infrastructure.
Sequence support should not be hard-coded to `List`; we should define a trait
that sequence-like types conform to, exposing the operations necessary to
inspect runtime shape and project elements with the appropriate ownership
semantics.

The same protocol can then support sequence match patterns:

```mojo
match values:
case var [a, b, 0]:
    ...
case _:
    ...
```

and later variable-length forms:

```mojo
case var [first, *rest]:
case var [first, *middle, last]:
```

Other pieces likewise decompose naturally into relatively independent projects:

- **Conditional matching in `if` and `while`** interprets pattern failure as a
  false condition.
- **OR patterns** compose several existing patterns.
- **AS patterns** retain the complete matched value while recursively applying
  another pattern.
- **Struct patterns** add an extensible mechanism for decomposing struct-like
  values.
- **Mapping patterns** introduce another independently implementable structural
  protocol.
- **Match guards** add boolean filtering after a pattern has successfully
  matched.
- The separate enum/sum-type proposal adds another family of patterns using the
  same infrastructure.
- Exhaustiveness and redundancy analysis can become progressively more
  sophisticated as additional pattern forms land.

The important architectural point is that these features should compose through
common pattern and decomposition infrastructure without requiring one
coordinated implementation effort.

### Future Direction: Implicit Bindings

The initial design should require bindings in match patterns to be explicit:

```mojo
case var x, y: ...
```

This is verbose, and users will likely ask for the more natural spelling:

```mojo
case x, y:    ...
```

There is good precedent for this in Mojo itself. A `for` loop already implicitly
introduces its iteration variable, implicitly defaulting to an `imm` reference
binding:

```mojo
# x and y are an "imm" binding implicitly.
for x, y in values:  ...

# var and ref are also allowed, the later infers mutability
for var x in values: ...
for ref x in values: ...
```

We should therefore leave bare identifiers in match-pattern position available
for a future extension where they implicitly introduce bindings.

This is also why the initial pattern grammar should **not** interpret a bare
identifier as "match against the existing value with this name." Doing so would
consume the syntax we are likely to want for implicit bindings and produce the
same capture-vs-value ambiguity that Python has to work around.

Starting with explicit `var` bindings therefore keeps the initial feature
unambiguous while preserving an obvious ergonomic improvement for the future.

### Future Direction: Arbitrary Value Matching

If we eventually allow bare identifiers in patterns to implicitly introduce
bindings:

```mojo
case (x, y):  # Declare a new value "x" and "y"
```

then a bare name can no longer also mean “match against the existing value with
this name.” This is the same ambiguity Python faces: `case x` is a capture
pattern, while qualified names such as `case HttpStatus.OK` are interpreted as
value patterns. Python deliberately reserves dotted names for this purpose.

We should adopt the same useful convention for qualified values:

```mojo
case Color.red:
case .blue:      # Inferred version of the same syntax.
case SomeModule.special_value:
```

but this does not solve every case. In particular, types are themselves comptime
values in Mojo, and it would be useful to match on them:

```mojo
def storage_kindT: AnyType -> String:
    comptime match T:
    case Int:
        return "integer"
    case String:
        return "string"
    case _:
        return "other"
```

Semantically, this should require no special “type matching” feature: types are
values and equality is defined on them. The problem is purely syntactic. If bare
identifiers implicitly bind, `case Int` cannot simultaneously mean “compare
against the existing value `Int`.”

Python considered this general problem extensively. PEP 635 records proposals
for explicit constant markers such as `^CONSTANT`, `$CONSTANT`, and
`==CONSTANT`, but ultimately left such a marker as a possible future extension.
PEP 642 went further and proposed an explicit equality-pattern syntax:

```mojo
case == expected:
```

with the explicit meaning “match when the subject is equal to this value.” PEP
642 was rejected in favor of Python's simpler initial design, but this
particular idea is interesting for Mojo.

For example, Mojo could eventually support:

```mojo
comptime match T:
case == Int:
    return "integer"
case == String:
    return "string"
case _:
    return "other"
```

and more generally:

```mojo
case (x, == expected):
    ...
```

This gives a clean division of syntax:

```mojo
case x:              # Implicitly bind x.
case 42:             # Literal value pattern.
case Color.red:      # Qualified value pattern.
case == expected:    # Explicit value pattern.
```

An explicit marker also gives us a place to consider more general expressions in
the future:

```mojo
case == compute_expected():
    ...
```

We do not need to support arbitrary expressions initially. In Mojo, expressions
can have interesting reference and lvalue behavior, so keeping the first version
constrained to simple value references may be preferable. The important point is
that an explicit equality pattern provides a natural extension point if
generalized value matching proves useful.

This allows us to optimize the common syntax for bindings while still retaining
an unambiguous way to match arbitrary existing values. It also avoids
introducing special semantics for types: `== Int` is simply a value pattern
whose value happens to be a type.
