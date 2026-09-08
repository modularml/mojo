# [Mojo] Checking constrained methods at overload resolution time

**March 25, 2025**
Status: Proposed, agreement to explore but not committed, not implemented.

**Sept 17, 2025**
Status: Updated to include param decl constraints, and renamed to "where".
Implementation has been scoped out and prioritized.

**Oct 1, 2025**
Status: Updated to remove the error message field. Implementation in progress.

**Dec 4, 2025**
Status: Original feature implemented. Added `__comptime_assert` statement.

**Feb 3, 2026**
Status: `__comptime_assert` syntax finalized as `comptime assert`

**May 27, 2026**
Status: Inline `where` constraints in parameter lists deprecated in favor of
trailing `where` clauses. Trailing `where` extended to struct and `comptime`
alias declarations. Auto-predication allows trailing `where` clauses to
discharge constraints from constrained types appearing in a signature.

**July 22, 2026**
Status: Added an optional failure message on `where` clauses.
String-literal messages only for now; supported on trailing function, struct,
and `comptime` alias constraints and on struct conditional-conformance clauses.
See "Failure messages" below.

This document explores adding “where” clauses to Mojo, a major missing
feature that will allow more safety, expressivity, and APIs that work better
for our users.

## Introduction

Generic programming has always been a core part of the Mojo library design
ecosystem, and it consists of three phases: 1) parse time, and 2) elaboration
time (which evaluates “comptime parameter expressions”) and 3) runtime. We aim
to move error detection earlier in this list by improving the type system, but
we eschew complexity because we don’t want to turn into other systems that
require (e.g.) a full constraint solver or interpreter built into the parser.

All this said, Mojo has a weakness, which is the lack of `where` clauses
that allow constraining method availability (**at parser time**) based on
static-ly known **parser-time** information. For example, we have methods like
this on SIMD (and thus on core aliases like `Int8`):

```mojo
def _simd_construction_checks[type: DType, size: Int]():
    constrained[
        type is not DType.invalid, "simd type cannot be DType.invalid"
    ]()
    constrained[size.is_power_of_two(), "simd width must be power of 2"]()
    ...

struct SIMD[dtype: DType, size: Int]:
    @implicit
    def __init__(out self, value: FloatLiteral):
        ...
        _simd_construction_checks[dtype, size]()
        constrained[
            dtype.is_floating_point(), "the SIMD type must be floating point"
        ]()

        <actual code>
```

This method allows one to construct a SIMD type with a float literal, like
`1.0` but checks at *elaboration time* that type `dtype` is a floating point
type, that `size` is a power of two, etc. This is deeply unfortunate for a few
reasons:

1. This wastes comptime by having the interpreter have to eval things like
   `_simd_construction_checks`.
2. This makes error messages on misuse much worse, because you get a stack trace
   out of the elaborator instead of a comptime error message.
3. This prevents defining ambiguous overload sets that are pruned at parse time
   based on conditional information.

The reason the last one matters is that there can be many ways to solve a
problem if you’re a library developer working with a closed set of overloads,
particularly if generic. Why does this matter? This allows **algorithm
selection** at **overload-resolution time**. You can choose implementation
based on type capabilities, or remove a candidate based on lack of capabilities
(to resolve an ambiguous candidate set):

```mojo
def thing[size: Int](value: YourType[size])
  where size.is_power_of_2(): ...
def thing[size: Int](value: YourType[size])
  where not size.is_power_of_2(): ...
```

Furthermore, we want to enable more generic and powerful libraries, which
require boolean conditions to be successful. It would be wonderful to be able
to express things like this (evaluated at parse time) - but note, this syntax
is not holy, it is just made up:

**Parameter relationships**: Express constraints between multiple type
parameters:

```mojo
def convert[From:.., To:..](value: From) -> To where can_convert_from[From, To]:
```

**Property-based constraints**: Specify requirements beyond simple trait
conformance

```mojo
def safe_divide[T](a: T, b: T) where (T instanceof Numeric and T.min_value() < 0):
```

There are many possibilities, this is a pretty important feature for us to have
for both expressiveness and user experience. This feature should be able to
remove many uses of `constrained` , directly improving QoI for each one.

## Concrete syntax design

`where` constraints are expressed as trailing clauses on a declaration,
written after the parameter list and any other signature components (parent
types, return type) and before the colon or `=`. Multiple `where` clauses can
be chained on the same declaration. They are always checked at parameter
binding time.

### Trailing where clauses

Trailing `where` clauses are supported on function/method declarations, struct
declarations, and `comptime` alias declarations.

**Functions and methods**: a trailing `where` attaches to the function and is
checked at parameter binding time (i.e., when an overload candidate is being
evaluated). This is what enables algorithm selection by type properties:

```mojo
struct SIMD[dtype: DType, size: Int]:
  @implicit
  def __init__(out self, value: FloatLiteral)
    where dtype.is_floating_point():
      <actual code>
```

Multiple `where` clauses can be chained and each takes a boolean expression:

```mojo
def matmul[m: Int, n: Int, k: Int](
    a: Matrix[m, k], b: Matrix[k, n]
) -> Matrix[m, n]
    where m > 0 where n > 0 where k > 0:
    ...
```

**Structs**: a trailing `where` attaches constraints to the struct type and is
checked at parameter binding time:

```mojo
struct SIMD[
  dtype: DType,
  size: Int,
](Defaultable)
  where dtype != DType.invalid
  where size.is_power_of_two():
  ...
```

**`comptime` alias declarations**: the same trailing form applies:

```mojo
comptime PositiveOnly[N: Int]: AnyType where N > 0 = ...
```

All three forms accept boolean expressions and can chain multiple `where`
clauses with an implicit `and`.

Notice how this puts the constraints where they belong - put the constraints
for the SIMD type as a whole on the struct, and put the constraints for the
method on the method itself.

### Constraints in the type system

A parameterized entity (struct, function, or comptime expression) that has not
yet had all of its parameters bound is called a **generator type**. The `where`
constraints on a declaration become part of that generator type — they travel
with it and are checked at every binding site.

**Specialization** is the process of turning a generator type into a more
concrete one by supplying arguments for its parameters. Two things can happen
during specialization:

- **Parameter binding** — a parameter is given a concrete value (e.g.
  `Matrix[m=3, n=4]`). This has always been possible.
- **Constraint discharge** — a constraint from the generator type's `where`
  clause is *proved satisfied* given the current context, and therefore drops
  out of the specialized type entirely. The resulting type no longer carries
  that constraint.

For example, specializing `Matrix[n, n]` inside a scope that already has the
assumption `n > 0` discharges both of `Matrix`'s constraints (`m > 0` and
`n > 0`) — because both collapse to the same already-known fact. The
specialized type is concrete and carries no residual constraints.

### Auto-predication

When a constrained type appears as a type expression in a signature (e.g. as a
parameter declaration type, a function argument type, a return type, etc.), the
compiler must verify that the type's constraints are always satisfied by the
signature's own constraints.

Auto-predication resolves this: the compiler collects any unprovable parametric
constraints encountered in the signature and requires they be discharged by the
declaration's own trailing `where` clause. This means a single trailing `where`
simultaneously constrains the declaration *and* satisfies the requirements of
constrained types used within the same signature:

```mojo
struct Matrix[m: Int, n: Int] where m > 0 where n > 0:
  ...

# Matrix[n, n]'s constraints (n > 0) are deferred and discharged by the
# trailing 'where n > 0'.
def solve_linear_system[
  n: Int,
  a: Matrix[n, n],       # trailing where n > 0 makes this valid
  b: Vector[n],
]() -> Vector[n]
    where n > 0:
  ...
```

The same discharge mechanism applies to struct parameter lists:

```mojo
struct LinearSystem[n: Int, a: Matrix[n, n]] where n > 0: ...
```

If no trailing `where` discharges a deferred constraint, the compiler reports
an error and suggests the missing clause:

```text
error: invalid bindings in signature: lacking evidence to prove correctness
note: add a trailing 'where' clause that requires '(n > 0)'
```

This capability is an “obviously good” thing, but all the questions revolve
around the implementation - how invasive is this, what are the limitations, and
does it require building an interpreter back into the parser just after we
excised it?

### Failure messages

A `where` clause may carry an optional message that is surfaced in the
diagnostic when the constraint fails. The message is written as a parenthesized
pair, `where (condition, "message")`:

```mojo
def foo[sc: Int]() where (sc > 1, "scaling factor must be greater than 1"):
    ...
```

Calling `foo[0]()` then reports the message in the note:

```text
note: constraint declared here evaluated to False, expected '(sc > 1)':
scaling factor must be greater than 1
```

The message is supported everywhere a `where` clause is: trailing function,
struct, and `comptime` alias constraints, and struct conditional-conformance
clauses (`struct S(Trait where (cond, "message"))`).

**Why parentheses.** An earlier proposal used an unparenthesized trailing form,
`where condition, "message"`. The bare comma was ambiguous in a conformance
list — in `(Trait where cond, "msg", Trait2)` the comma also separates the next
entry. Putting the comma inside the parentheses removes the ambiguity: the
message is simply the second element of a two-element tuple expression, and the
entry-separator comma is unambiguous.

**String literals only, for now.** Unlike `comptime assert` (below) — whose
message is checked by the elaborator and so may be any comptime string
expression — a `where` message is captured by the parser, which has no
interpreter. A non-literal message such as `"need " + String(N) + " values"`
cannot be evaluated at the `where`-parsing phase, so only string literals are
accepted (with the usual adjacent-literal concatenation); anything else is a
targeted "the message in a 'where' clause must be a string literal" error.
Because the parenthesized syntax already makes a non-literal message
syntactically unambiguous, lifting this restriction later is backward
compatible.

**Where the message surfaces.** Body constraints — trailing `where` on
functions, structs, and aliases — are checked during overload resolution and
instantiation, and the message is appended to the "evaluated to False" and
"needs evidence" notes. Conditional-conformance messages are attached to the
"does not conform to trait" diagnostic instead; they are evaluated under the
caller’s assumptions, so a conformance the caller proved via its own `where`
clause is not reported — only the genuinely-unsatisfied ones.

## Implementation approach

The major thing required (given our quality goals) is that we are able to
implement a conformance check - is this member a valid candidate of an overload
set - **at parse time**. We need to decide that (and report a good error
message if the only candidate fails the boolean predicate) at parser time
**without interpreting the code** (because the Mojo parser has no interpreter).
For the sake of this discussion, we only consider simple boolean expressions.

This has three main parts: 1) collect function/method/struct/alias
requirements, 2) propagate assumptions across declarations (contextual
invariants), and 3) perform symbolic checking during overload resolution.

### Part #1: Function/Method/Struct/Alias Constraints

Method constraints are pretty simple - the constraints for a function are the
union (with an ‘and’) of the function constraints and the enclosing struct
constraints. The `where` constraints become part of the type itself, checked at
every binding site:

```mojo
struct SomeThing[size: Int]
  where size.is_odd()
  where size != 233:

  def thing(self) -> Int
     where size.is_prime():
```

In this example, calling `SomeThing.thing` requires `Self.size` to satisfy the
condition `size.is_odd() and size != 233 and size.is_prime()`. Note that this
builds on `@always_inline("builtin")` and the simplifications that
`ParamOperatorAttr` does, but does not utilize an interpreter. This means that
we’ll be able to inline and simplify very simple integer expressions (e.g. we
could simplify things like `size != 1 and size > 0` into just `size > 1` ) but
we can’t do symbolic manipulation of `size.is_prime() and size.is_even()` into
`size == 2`.

This behavior will be core to this feature - like our dependent type support,
we can have rough mastery over simple affine operations on basic types like
integers, float and bool, but just treat more complicated operations
symbolically: `ParamOperatorAttr` does know that `a.is_prime() and
a.is_prime()` canonicalizes to `a.is_prime()` because the trivially redundant
subexpressions.

How do we store this? Method and struct requirements should be stored as a new
list of `TypedAttr` on both function and struct declarations. This ensures
they’re serialized to modules etc. This is parser time only behavior, so these
do not need to be lowered to KGEN or later.

### Part #2: Contextual Invariants

Contextual invariants - something known true at the point in some code - is a
question asked by overload set resolution at some point in the program.
Consider an overly complicated example like:

```mojo
struct S[
  a: Int,
  b: Int,
  c: Int,
  d: Int,
] where pred1(a):

    def some_method(self)
       where pred2(b):

       comptime if pred3(c):

           def nested()
             where pred4(d):
                # Checking at this point.
                some_callee(self)
```

We need to determine what are the invariants we know at the point of
`some_callee(self)`. This is determined by walking the MLIR region tree and
unioning the set of contextual invariants together with an “and” operation. In
this case, we know that the contextual invariant is `pred1(a) and pred2(b) and
pred3(c) and pred4(d)` because of the invariants on the struct, functions, and
parameter if.

### Part #3: *Symbolic* constraint checking at overload resolution time

Finally, given we have these two bits of information, we can use it at overload
resolution time. Overload resolution has to do a bunch of stuff (parameter
inference, implicit conversion checking etc) to determine if a candidate is
valid. At the end of checking, it then looks to see what the “contextual
invariant” boolean expression is, and the “function requirement” boolean
expression is. The candidate is considered valid if, the following is true:

```mojo
(contextual_invariant and function_requirement) == contextual_invariant
```

In English, this is saying that `function_requirement` doesn’t impose any novel
requirements that `contextual_invariant` doesn’t already encode.

But what is “truth” here and how do we determine this? The expressions may
themselves be conjunctions of nested subexpressions, may have unresolved
operands, and we don’t have an interpreter in the parser. To address this, we
just allow `ParamOperatorAttr` to canonicalize and simplify the expressions, and
use pointer equality of the resultant `TypedAttr`’s. If they are identical, then
they are known to be safe, if not, it should be rejected. I implemented the
requisite symbolic manipulation at the KGEN level
([in June 2022](https://github.com/modularml/modular/commit/9fcf5c859adb9e282378fbd37344a0c49cf2c895))
and we can make other new specific cases fancier as needed.

In the case of a rejection, we can do a bit more digging for better error
message quality: we can figure out which clause is failing and report the
failed constraint. For example, if we have something like `SIMD[f32, 17]` and
the following definition:

```mojo
struct SIMD[
    dtype: DType,
    size: Int,
] where dtype != DType.invalid
  where size.is_power_of_two():
```

The logical way for the compiler to check this is to build up a big conjunction
`(dtype != DType.invalid and size.is_power_of_two())` (and the
`is_power_of_two()` function will be inlined into subexpressions so it will be
much lower level) and then fold it and fail the whole expression - we want
overload checking to be efficient, because it is normal for some overload set
candidates to fail without the expression type checker failing overall.
However, if the whole set fails, we want to report the first failing condition.
This can be done by adding a new failure kind to `OverloadFitness` which error
emission uses.

## Pain Point: Expressing “assumptions” is cumbersome

An “assumption” is the term we use for constraints we know to be true in a
given lexical scope. For example, a parameter-if provides its nested scope with
the assumption that the if-condition is true.

Often-times, users know that a given condition is satisfiable, but it’s not
directly provable from the code. E.g.

```python
def needs_prime[x: Int]() where x.is_prime():
  ...

def main():
  # Un-provable constraint: 2.is_prime().
  needs_prime[2]()
```

This example shows a fully concrete constraint expression that we cannot
evaluate today, but it extends to fully symbolic expressions too.

The current paradigm we’re forcing users to adopt is a parameter-if:

```python
comptime if 2.is_prime():
  needs_prime[2]()
else
  constrained[False, "This shouldn't happen"]()
```

There are two problems:

1. **Forced Verbosity**: The user is forced to write an `else` branch in order
to verify their assumption.

2. **Forced Scope**: The user is forced to introduce a scope. This can be
problematic for organizing subsequent code.

### Proposal: Explicit Assumption Injection

A dedicated statement that brings into scope an assumption, similar to
`constrained` but more powerful.

```python
comptime assert 2.is_prime()
needs_prime[2]()
```

This allows users to easily insert a *checked* assumption into the current
parameter scope. This is a meta-code statement by nature, so its order relative
to the base code is irrelevant (similar to `comptime`). It would serve two
purposes in one go:

- Check that the assumption holds.
- Inject the assumption into the current scope.

Over time, we plan to phase out the stdlib `constrained` function in favor of
this statement.

#### Syntax

This will be a simple statement that accepts one/two parameters:

- A Bool expr that represents the condition to assert for.
- [Optional] A StaticString expr that represents the error message to report
when the condition fails.
  - This does *not* need to be a string literal expr.

Examples:

```python
comptime assert 2.is_prime()

comptime assert 2.is_prime(), "2 should be a prime"

comptime assert x > 2, "x should be greater than 2, got " + x + " instead"
```

#### Semantics

**Checking**:

If the condition folds to False in the parser, the parser will report a local
error.

> error: failed comptime assert: condition is always False.
>

If the condition folds to True in the parser, the parser will report a warning
that this statement can be removed.

> warning: redundant comptime assert: condition is always True.
>

Otherwise, the condition is verified by the elaborator, and works exactly like
`constrained` does today (same error behavior).

**Assuming**:

The parser is able to assume that the condition is true in the current
parameter scope, and allow users to use this assumption when binding parameters
/ invoking functions in this parameter scope.

```python
def needs_prime[x: Int]() where is_prime(x):
  ...

def main():
 comptime assert is_prime(2)
 needs_prime[2]()   # This is OK.
```

## Logical extensions (not in scope for this proposal)

Once we have the core language feature in place, we can address various
extensions. Things that come to mind, but which aren’t fully scoped out:

1. We should implement a `T instanceof Trait` and `T == Type` boolean
   predicates, this would allow subsuming trait testing.

2. We should look at type rebinding so we can fully subsume the existing
   “conditional conformance” logic, without requiring a rebind in the function
   body:

```mojo
struct A[T: AnyType]:
    var elt : T

    # existing
    def constrained[T2: Copyable](self: A[T2]):
         var elt_copy = self.elt # invoke copyinit

    # desired:
    def constrained(self)
      where T instanceof Copyable:
         # need to know T is copyable even though declared AnyType
         var elt_copy = self.elt
```

1. Depending on how we implement #2, maybe we can make `comptime if T
  instanceof SomeTrait` refine the value of T within the body of the `if`.
  Theoretically if we did this, we could eliminate the need for `rebind` in a
  lot of kernel code. I’m not sure if this is possible though.

2. Implement other special case attributes where we want to. For example, I’d
   love metatypes to be comparable and `Variant.__getitem__` ’s type list to be
   something like `where T in Ts` , which would be straightforward to
   implement once metatypes are comparable. We could also support things like
   `where T in [Float32, Float64]` when we fix up list literals.

3. Remove the existing CTAD conditional conformance stuff. This would be
wonderful, simplifying some fairly weird code.

4. ✅ I’d like to add implicit conversions to types like `UnsafePointer` to
   another `UnsafePointer` which are only valid if the OriginSet+mutability are
   a superset of the source ones. These things would now be simple enough to
   implement. NOTE: We have since figured out how to do this with the existing
   hack.

## Primary Limitation

The primary limitation of this proposal is that it doesn’t have a parser time
interpreter, and I think it is important that we evaluate these things at
parser time. This should be fine for most symbolic cases, but will have one
annoying limitation. Consider the following:

```mojo
struct X[A: Int]:
   def example(self) where A.is_prime(): ...


def test(value: X[2]):
    # Error, cannot symbolically evaluate '2.is_prime()' to a constant.
    value.example()

    comptime if 2.is_prime(): # tell dumb mojo that 2 is prime.
       # This is ok.
       value.example()
```

Because we don’t have a parser time interpreter, there will be some “obvious”
cases that cannot be evaluated.

**When does this occur?**: This will occur with non-trivial functions like
`is_prime` and other general logic (e.g. you could have a comptime function
that computes a hash table, that won’t be a thing in this phase). On the other
hand, this won’t happen for a lot of the trivial cases - the
`@always_inline("builtin")` ones will always be foldable, so basic math and
simple predicates like `is_power_of_two()` are all fine. Additionally,
parametric aliases can/will be used to expression generic and guaranteed
foldable expressions - but you can’t do a loop or complicated logic needed for
“is_prime” in an alias.

**What can we do about it?**

This could be annoying for people pushing the limits of the dependent types and
conditional conformance feature, there will always be limitations to symbolic
evaluation. If this becomes important in the future, we can consider bringing a
(different than we had) interpreter back into the parser, potentially building
on the work to make it support parameterized functions correctly.

Specifically, there is a proposal to make the comptime interpreter support
direct-interpreting parametric IR instead of relying on the elaborator to
specialize it before interpretation. If we had that (and a couple other minor
things) we could bring it back and make it “actually work and be predictable”
into the parser, and solve this in full generality.

We could also look to expand the support of `@always_inline("builtin")` for
specific narrow cases that need to be supported, but I think it is important
that we keep this pretty narrow for predictability reasons.

## Alternatives considered

### Syntax

There are multiple ways to spell this keyword, for example:

- `require` is not plural.
- Swift and Rust use `where`
- C++ uses `requires`

### Check at elaboration time

The one significant constraint in this proposal comes from checking
requirements at parser overload resolution time. One might ask “why not check
later, e.g. at elaboration time?

The problem is that it really is the parser that needs to determine which
concrete method is called, because this affects type checking. For example:

```mojo
def your_function(a: SIMD[F32, _]) -> Int where a.size.is_prime(): ...
def your_function(a: SIMD[F32, _]) -> F32: ...
```

We really do need to resolve (at parser time) which candidate gets picked,
because otherwise we don’t know the result type of a function call. You could
“support elaboration time checking” if the result types are consistent, but
that reduces the quality of the error messages (to include a stack trace on
failures) and is just sugar for a parameter if in the body.
