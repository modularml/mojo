# Fixing Simple Literals in Mojo

Author: Chris Lattner
Date: March 1, 2025
Status: Implemented, complete

This doc proposes a redesign of the Int/Float/String literal types in Mojo to
make them lower correctly and define away a large category of bugs that people
run into. This feature depends on powerful dependent types support which Mojo
now supports. This is specific to Int/Float/String literals - `Bool` and
collection literals are not affected.

## Background

Mojo’s design includes support for “infinite precision” integer **literals** at
compile time, these values may be arbitrarily large, so long as they are
converted to a reasonable size before they are “materialized” at runtime:

```mojo
alias biggg_number = 2 << 255 # Very large value

def use_number() -> Int:
  # Ok, not as big at runtime.
  return biggg_number // (2 << 246)

def fail_to_use_big() -> Int:
  # error: integer value 115792089237316195423570985008687907853269984665640564039457584007913129639936 requires 258 bits to store, but the destination bit width is only 64 bits wide
  return biggg_number

```

Similarly, floating point literals may be arbitrarily large, and may have
arbitrary precision, which is convenient for manifest constants:

```mojo

# Lots of precision, just in case we are looking at the radius of an atom at galactic scale.
alias pi = 3.1415926535897932384626433832795028841971693993751058209749445923
alias tau = 2 * pi # maintains full precision
alias e = 2.7182818284590452353602874713526624977572470936999595749669676277

def fast_circle_area(r: Float16) -> Float16:
  return pi*r*r
def more_precise_circle_area(r: Float64) -> Float64:
  return pi*r*r
def fast_circle_diameter(r: Float16) -> Float16:
  return tau*r

```

Note that we wouldn’t want to represent these as some concrete type like
`Float64`, because we wouldn’t want to allow an **implicit** conversion from
`Float64` to `Float16` (that would throw away precision) but we do want to allow
conversions from literals.

String literals in Mojo are similar: we want to support arbitrary compile-time
calculate of string values, but then burn them into a final executable binary as
a literal constant. That allows us to use them as immortal “pointer + length”
values with a `StaticConstantOrigin`.

### The literal representation at compile time

Mojo implements this by representing these compile time values as MLIR
attributes. Integers are an arbitrary precision integer of type
`!pop.int_literal` and floating point values are `!pop.float_literal` and
strings use `!kgen.string` as you might expect.

The details of the representation are more complex than is obvious: besides
being infinite precision, they handle complex cases correctly. For example,
Mojo handles floating point constants internally by representing them as being
either a special form (NaN, infinity, negative zero, etc) or a precise value
represented as a rational of two infinite precision integers.

This representation is completely fine to use at compile time, but we’d never
want to do *runtime* computation on it!

## Current design, and the problems with it

Today, we have a “nearly” correct design, we model things like `IntLiteral` and
`StringLiteral` as “nonmaterializable” types that are intended for use at
compile time only. These types may be used at compile time, and then they
automatically promote to values of safe runtime type where necessary.

For example, the current behavior for integers is (floating point works the same
way):

```mojo
def integers():
   alias a1 = 42 # typeof(a1) is IntLiteral
   var v1 = 42   # typeof(v1) is Int

   alias a2 = a1-3 # typeof(a1.__sub__(3)) is still IntLiteral
   var v2 = v1-3   # typeof(v1.__sub__(3)) is still Int
```

This is good behavior “things that happen at compile time stay at compile time”,
where we can support fancy infinite precision math, but anything that gets into
the runtime domain “materializes” to a type that the literal implicitly converts
to, and if nothing else, it defaults to `Int` . This is handled by the
implementation of `IntLiteral` , a simplified version of which looks like this:

```mojo
@__nonmaterializable(Int)
struct IntLiteral:
    var value: __mlir_type.`!pop.int_literal`

    def __sub__(self, rhs: Self) -> Self:
        return Self(
            __mlir_op.`pop.int_literal.binop<sub>`(self.value, rhs.value)
        )
```

This implementation works because the comp-time interpreter knows how to
constant fold the `pop.int_literal.binop` MLIR operation when doing calculation
in the compile time domain.

This is simple enough, and works in many common cases, but unfortunately this
can’t work in full generality. Consider something like this:

```mojo
@export
def test(a: IntLiteral) -> IntLiteral: return a-1
```

In this case, the `IntLiteral` type is being used at runtime. This fails with a
rather inglorious error:

```text
example.mojo:3:4: error: failed to convert func signature
def test(a: IntLiteral) -> IntLiteral: return a-1
   ^
example.mojo:3:4: error: failed to legalize operation 'kgen.func' that was explicitly marked illegal
def test(a: IntLiteral) -> IntLiteral: return a-1
   ^
example.mojo:3:4: note: see current operation:
"kgen.func"() ({
^bb0(%arg0: !pop.int_literal):
  %0 = "kgen.param.constant"() {value = #pop.int_literal<1> : !pop.int_literal} : () -> !pop.int_literal
  %1 = "pop.int_literal.binop"(%arg0, %0) {oper = #pop<int_literal.binop_kind sub>} : (!pop.int_literal, !pop.int_literal) -> !pop.int_literal
  "kgen.return"(%1) : (!pop.int_literal) -> ()
}) {LLVMMetadata = {}, crossDeviceCaptures = #M<strings[]>, decorators = #kgen<decorators[]>, exportKind = #kgen.export<exported>, funcTypeGenerator = !kgen.generator<(!pop.int_literal) -> !pop.int_literal>, inlineLevel = 0 : i32, sym_name = "test"} : () -> ()
```

Ugh, what is that?

The problem here is that the `IntLiteral` type is infinite precision compile
time type - the compiler doesn’t know how to lower it to a runtime
representation. Furthermore, it doesn’t know how to codegen the
`pop.int_literal.binop` operation either.

We can definitely improve this error message, but there is still a core issue:
we are expressing computation with an MLIR operation that cannot be lowered to
LLVM.

String literals are designed similarly, but are in worse shape. `StringLiteral`
doesn’t materialize to `String` for historical reasons, so we get this behavior:

```text
$ cat test.mojo
@export
def example(a: Int):
    var x = "foo"   # typeof(x) is StringLiteral not String
    for i in range(a):
        x += "bar"
    print(x)

$ mojo test.mojo
std/builtin/string_literal.mojo:156:45: error: cannot use StringLiteral append methods at runtime, only in an alias
        return __mlir_op.`pop.string.concat`(self.value, rhs.value)
                                            ^
x.mojo:6:11: note: called from
        x += "bar"
          ^
Included from Mojo/stdlib/std/prelude/__init__.mojo:1:
Included from Mojo/stdlib/std/prelude/__init__.mojo:102:
Mojo/stdlib/std/builtin/string_literal.mojo:156:45: note: see https://github.com/modular/mojo/issues/3820 for more information
        return __mlir_op.`pop.string.concat`(self.value, rhs.value)
                                            ^
Included from Mojo/stdlib/std/prelude/__init__.mojo:1:
Included from Mojo/stdlib/std/prelude/__init__.mojo:102:
/Users/clattner/Projects/modular/Mojo/stdlib/std/builtin/string_literal.mojo:156:45: error: failed to legalize operation 'pop.string.concat' that was explicitly marked illegal
        return __mlir_op.`pop.string.concat`(self.value, rhs.value)
                                            ^
x.mojo:6:11: note: called from
        x += "bar"
          ^
Included from Mojo/stdlib/std/prelude/__init__.mojo:1:
Included from Mojo/stdlib/std/prelude/__init__.mojo:102:
Mojo/stdlib/std/builtin/string_literal.mojo:156:45: note: see current operation: %58 = "pop.string.concat"(%57, %28) : (!kgen.string, !kgen.string) -> !kgen.string
        return __mlir_op.`pop.string.concat`(self.value, rhs.value)
                                            ^
mojo: could not lower funcs to LLVM, run LowerToLLVMPipeline failed
```

This is… not exactly the quality experience we want to have for Mojo. I find
failure on basic string literals to be pretty embarrassing. While again we
could improve the error messages to be better for this failure, I’d rather fix
the model so things actually work correctly and can be implemented in a reliable
way!

## Proposal: Use parameters for compile-time-only values

The core problem with the existing design is that we are representing
compile-time only values (the underlying `!lit.int_literal` magic) as runtime
state. This *almost works*, because the comptime interpreter can execute this
stuff at comptime, but will **never work in generality**, and will always be a
source of problems.

The solution is clear - let’s just keep compile-time-only values as
**parameters**, instead of representing them as runtime state. Until recently
though, Mojo didn’t offer us this choice: a series of problems with dependent
types and parameter inference were in the way - but these have been solved in
order to support this work. Let’s explore what the solution looks like, using
`IntLiteral` as the exemplar. This document proposes moving the above design to
represent the `value` within `IntLiteral` as a parameter, not a var. We’ll dig
into this slowly and discuss the implications. First the top level of
`IntLiteral` shifts from:

```mojo
# Old design we are replacing:
# @__nonmaterializable(Int)
# struct IntLiteral:
#    var value: __mlir_type.`!pop.int_literal`
#    ...

# New design:
@__nonmaterializable(Int)
struct IntLiteral[value: __mlir_type.`!pop.int_literal`]:
    # no var or other state
    ...
```

This shift - of the state from being `var` storage on `IntLiteral` to being a
parameter, is the crux of this proposal. By encoding the comptime-only value as
a parameter, we know it will never need to be materialized at runtime.

### Compiler Literal Syntax Support

Given this design, we make a small change to the compiler’s code generation of
literal syntax, e.g. an expression like `42` or `4.0` . Formerly, when a
literal was used (e.g. `alias x = 42`) the compiler would generate a call to the
initializer like `alias x = IntLiteral(42)`. With the new design, the compiler
generates a call of `alias x = IntLiteral[42]()`.

We enable this easily in our literal type with an initializer:

```mojo
struct IntLiteral[value: __mlir_type.`!pop.int_literal`]:
    # no var or other state

    def __init__(out self): pass
```

Take a minute to grok what this says - because there is no state at all inside
of `IntLiteral`, the initializer doesn’t store anything. This initializer is
the only initializer that `IntLiteral` gets, because it has no state. We’ll
come back to this in a bit.

Understanding the “`IntLiteral` holds no runtime state” part of the design is
the most important thing to grok. It means that all computation on integer
literals must be done as part of the compile time domain, and therefore,
computation of dependent result types is mechanic that underlies math on
literals.

### Binary operations on literals

Let’s look at how `2-1` works on integer literals: through composition, we know
this is just sugar for `IntLiteral[2]().__sub__(IntLiteral[1]())` . Note that
while the LHS and RHS of the subtraction are both integer literals, the types
are different because the parameters are different. This means that the
`__sub__` operation needs to accept asymmetric argument types. Fortunately Mojo
makes this easy enough:

```mojo
struct IntLiteral[value: __mlir_type.`!pop.int_literal`]:
    ...
    def __sub__(
        self,
        rhs: IntLiteral[_]) -> IntLiteral[??]):
      ?????
```

Mojo allows us to write the signature in a nice and clear way - we work on any
`IntLiteral` (the parameter value is available as `Self.value` or `self.value`
which are both the same) and we can take a right side that is an `IntLiteral`
with any value: auto-parameterization makes this easy with `IntLiteral[_]` or
even just `IntLiteral` depending on your style.

So that explains how to write the input values, and we know we want to return a
new `IntLiteral`, but what do we use for the **result type parameter**?

Given we’re encoding the value transformation in the type, we know it has to be
inline computation, and it must be derived from `self.value` and `rhs.value`.
We can do this with low-level MLIR attributes (something that only the
implementation of `IntLiteral` needs to ever see. One possible implementation
of `__sub__` looks like:

```mojo
 struct IntLiteral[value: __mlir_type.`!pop.int_literal`]:
   def __sub__(
        self,
        rhs: IntLiteral[_]) -> IntLiteral[
            __mlir_attr[
                `#pop<int_literal_bin<sub `,
                self.value,
                `,`,
                rhs.value,
                `>> : !pop.int_literal`,
            ]
        ],
    ):
        return IntLiteral[
            __mlir_attr[
                `#pop<int_literal_bin<sub `,
                self.value,
                `,`,
                rhs.value,
                `>> : !pop.int_literal`,
            ]
        ]()
```

Ok, take this in, it is admittedly a bit yuck - we are using a KGEN attribute to
perform the computation in the parameter domain. This says that means the
parameter of the result type is a “kgen int literal add of self.value and
rhs.value”. Mojo (still, sigh) doesn’t support initializer lists, so we have to
utter the type once in the method signature, and then we repeat it again in the
return line.

While this is one way to write this, Mojo supports initializer lists, which
allow us to infer the type from context (the `return` in this case). Using it,
an improved implementation of `__sub__` looks like:

```mojo
 struct IntLiteral[value: __mlir_type.`!pop.int_literal`]:
     def __sub__(self, rhs: IntLiteral[_])
        -> IntLiteral[
            __mlir_attr[
                `#pop<int_literal_bin<sub `,
                self.value,
                `,`,
                rhs.value,
                `>> : !pop.int_literal`,
            ]
        ]:
        return {}
```

Before we move on, let’s look at that last line - we are **constructing the
result value, using the `()` constructor**! Recall that `IntLiteral` has
no runtime state - the value of the result is captured in the type - so this is
the right thing to do. In fact, given that it is stateless, `IntLiteral` only
gets one constructor.

### Mojo has full support for comptime determined parameters

While thinking about something like `2-1` is easy to unpack, how do we think
about something like `x-2` where we don’t know the value of `x`? It turns out
that this all composes, using dependent type support (and now with
[improved dependent type support](https://github.com/modular/modular/blob/main/Mojo/proposals/always_inline_builtin.md),
it even works correctly!). We have to write this function as a type function,
and we don’t want to utter that MLIR attribute, so we can write it like this:

```mojo
def sub_two(a: IntLiteral[_]) -> __typeof(a-2):
    return {}

...
    alias four = sub_two(6)  # four has IntLiteral[4] type
    # You can check it with:
    alias really_four : IntLiteral[4] = sub_two(6)
```

Furthermore, if you build chained functions on literals, even without constant
values, it all stacks up and composes correctly.

### Defining aggregate methods

It turns out that this is useful, because some of the operations on `IntLiteral`
are aggregates of more primitive things, for example, `-x` is just `0-x` and
`~x` is the same as `x^-1`, and we can write it like this:

```mojo
struct IntLiteral[value: __mlir_type.`!pop.int_literal`]:
    ...
    def __neg__(self) -> type_of(0 - self):
        return {}

    def __invert__(self) -> type_of(self ^ -1):
        return {}
```

Note how this stacks up: `0` and `1` are sugar for `IntLiteral[0]()`,
`0 - self` is just an invocation of `__sub__` which we see above. It all stacks!

### Literal Comparisons

Like all comparisons, literal comparisons return `Bool`. As before, this is
done with an MLIR attribute, but don’t require dependent types. An
implementation of less than looks like:

```mojo
struct IntLiteral[value: __mlir_type.`!pop.int_literal`]:
    ...
    def __lt__(self, rhs: IntLiteral[_]) -> Bool:
        return __mlir_attr[
            `#pop<int_literal_cmp<lt `,
            self.value,
            `,`,
            rhs.value,
            `>> : !pop.int_literal`,
        ]
```

Notice how this code is returning a parameter expression (an attribute) as the
result. This means that this comparison always instantiates into a `True` or
`False` value after elaboration. This is because of how parameters work - the
elaborator goes and instantiates all of these operations, so something like
`4<5` or `x<17` (where x is some parametric `IntLiteral`) are always known at
compile time. Neat and tidy.

### Floating Point Literals

That’s all you need to know to understand integer literals, so lets look at
float literals. They work exactly the same, and given we already know how the
pieces work, we can take more at once:

```mojo
@__nonmaterializable(Float64)
struct FloatLiteral[value: __mlir_type.`!pop.float_literal`]:
    # Create from a float literal parameter expression.
    def __init__(out self):
        pass

    def __sub__(self, rhs: FloatLiteral)
        -> FloatLiteral[
            __mlir_attr[
                `#pop<float_literal_bin<sub `,
                value,
                `,`,
                rhs.value,
                `>> : !pop.float_literal`,
            ]
        ]:
        return {}
```

Similarly, the comparison and other methods work the same as `IntLiteral` does.
`FloatLiteral` does have some new tricks though: we want to support conversions
from integer literals, and those conversions mean that we need reversed binary
operations: we want `4 - 1.0` to turn into `1.0.__rsub__(4)`. No problem, we
can implement these operations using the same pattern:

```mojo
struct FloatLiteral[value: __mlir_type.`!pop.float_literal`]:
...

    @implicit
    def __init__(
        value: IntLiteral[_],
        out result: FloatLiteral[
            __mlir_attr[
                `#pop<int_to_float_literal<`,
                value.value,
                `>> : !pop.float_literal`,
            ]
        ],
    ):
        return {}

    def __rsub__(self, rhs: FloatLiteral) -> type_of(rhs - self):
        return {}
```

This uses the “conditional conformance” trick to define the initializer (this
required some enhancements to parameter inference to get working) and uses
simple composed operations to define the reversed subtract.

Once these are put together, everything else just falls out.

## Implications and Analysis

The implications of this design are a bit subtle, so let’s go through them one
at a time.

### The compiler crashers are defined away

Let’s start with the good news: now that this design is unblocked (due to
dependent types being fixed etc) this all works and the patch that [implements
it is straight-forward](https://github.com/modularml/modular/pull/56959). Even
better, this defines away all the bugs that motivated this work in the first
place.

Let’s go back to one of our examples:

```mojo
@export
def test(a: IntLiteral) -> IntLiteral: return a-1
```

When you compile this with the new design, you get a reasonable compile time
error:

```text
$ mojo test.mojo
test.mojo:3:27: error: 'IntLiteral' missing required parameter 'value'
def test(a: IntLiteral) -> IntLiteral: return a-1
                          ^
```

This happens because `IntLiteral` takes a parameter. Ok, let’s try to break it
harder - we know how to make parametric functions with `IntLiteral` now. You’d
write this function like this:

```mojo
def test(a: IntLiteral) -> __typeof(a-1):
    return {}
```

This is perfectly valid Mojo code in the new design, and while you can’t
directly export this (because it is parameterized) you can force the compiler to
emit it if you use the `@noinline` decorator or similar. The nice thing about
this is that this is **completely valid code** (even if a bit weird) because
`IntLIteral[someval]` is defined and lowers correctly to LLVM: it is just an
empty struct after all, the same as returning an empty tuple or `None`.

### We can remove all the MLIR ops for literals

For proper dependent type support, the compiler needs to support the MLIR
attribute form of the computations on symbolic literals, but the old design is
also keeping around the MLIR operations. With this design we can drop all the
MLIR operations (and the code that links them) leading to a significant
simplification in the compiler internals.

So that is the good news, now let’s look at some of the more subtle aspects of
this.

### `IntLiteral` can’t conform to the standard arithmetic traits

One subtle thing is that we have a few arithmetic traits that assume symmetric
types on operations - one example of this is `CeilDivable` :

```mojo
trait CeilDivable:
    def __ceildiv__(self, denominator: Self) -> Self:
```

This operation cannot be implemented on `IntLiteral` or `FloatLiteral` because
the RHS and result type of the operation isn’t `__ceildiv__`. In practice, it
doesn’t seem like this matters: it did require defining one overload of the
global `ceildiv` function:

```mojo
def ceildiv[T: CeilDivable, //](numerator: T, denominator: T) -> T:
    # return -(numerator // -denominator)
    return numerator.__ceildiv__(denominator)

def ceildiv(numerator: IntLiteral, denominator: IntLiteral)
  -> type_of(numerator.__ceildiv__(denominator):
    return {}
```

However, while this one method was needed, it begs a question - why are we
defining something as a global function that is already an operator? Can we
just remove this and use the `//` operator instead?

I also believe that this is a good thing - the point of traits is to build
larger scaled composed generic algorithms based on these abstractions. The
previous design looked appealing but could never achieve this - they’d fail to
compile and generate nasty errors. If abstracting over arbitrary precision
integer/float values proves to be useful in the future, we can enable it at more
complexity: we’d need to provide a `BigInt` type that exists at runtime. Such a
type could also be used at compile time and materialize into `IntLiteral` using
the same pattern as `StringLiteral._from_string()` . I think we should get
`StringLiteral` settled before embarking on this.

### Type merging of literals promotes to nonmaterializable type

The other subtle thing I discovered is due to type merging, consider logic like
this:

```mojo
def do_thing[dt: DType]():
    alias value = 1e-06 if type == DType.float32 else 1e-1
    ...
```

The computation of `atol` needs to merge values of two different float literal
types: `FloatLiteral[1e-06]()` and `FloatLiteral[1e-1]()`. The Mojo compiler
promotes the result to `Float64` which is the shared nonmaterializable type for
these two types. This “just works”, but does require adding one cast in the
monorepo. This doesn’t seem anything more than a curiosity to me.

### Compile time impact: None

There is no expected measurable impact on compile time. This change reduces IR
bloat (by removing the MLIR operations) that the elaborator would need to make
working with them, and maintains the same amount of work to fold the attributes
to compute the values.

Furthermore, while literal expressions do support arbitrary generality, in
practice they are tiny, so I can’t imagine this change being measurable in any
case.

## Conclusion

This design doc explores a change to the integer/float/string literal types
which will allow us to define away a class of compiler crashes, clarify our
model, and simplify the internals of the compiler. It takes a moment to
understand how it works, but seems like a nice step forward, building on Mojo’s
maturing support for dependent types.
