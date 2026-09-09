# Customizable Type Merging in Mojo

Chris Lattner, Feb 28, 2025
Status: Implemented in 25.3

NOTE: this doc explains an expansion of the Mojo type checker. It is now
implemented, but the doc still explains it as "the old behavior" and then "the
new change". If you are reading this to understand Mojo's behavior, please
skip down to the "Proposed Solution" section.

One small but important decision the Mojo type checker needs to handle is “what
is the result type of merging two values, and how do we compute it”. For
example, consider the ternary operator `a if cond else b` : when `a` and `b`
have the same type, the result is obviously the same, but what happens when `a`
is an `Int` and `b` is an `Float64`? The answer is `Float64` in this example,
but deciding that is the problem of type merging. Mojo also currently lacks
support support for homogenous collection literals, but it needs to be able to
support things like `[1.0, 2, b, a]` - determining a single homogenous element
type (e.g. `Float64`) or rejecting it with an error.

## Type Merging in Mojo as of Feb 2025 (Note: Changed now)

As of this writing, Mojo’s type unification follows the following algorithm in
pseudo code:

```mojo
def get_common_type(typea, typeb) raises -> Type:
   # If the types are already identical, then we are done.
   if typea == typeb:
       return typea

   # Check for implicit conversions.
   a_impl_converts_to_b = is_implicitly_convertible(typea -> typeb)
   b_impl_converts_to_a = is_implicitly_convertible(typeb -> typea)
   if b_impl_converts_to_a and a_impl_converts_to_b:
       throw "ambiguous conversion"
   if a_impl_converts_to_b:
       return typeb  # Use implicit conversion
   if b_impl_converts_to_a:
       return typea  # Use implicit conversion

   # Elided: do similar test for @__nonmaterializable types.
   throw "no common type found"
```

This works great for simple cases, e.g. it allows merging an `Optional[Int]`
and an `Int` into an `Optional[Int]` and many other cases. However, this isn’t
enough for numeric conversions, e.g. consider:

```mojo
def int_example(a: Int8, b: Int16, cond: Bool):
  # Currently: error: value of type 'SIMD[int8, 1]' is not compatible with value of type 'SIMD[int16, 1]'
  c = a if cond else b
  # Desired type: Int16
```

This really should work, but we don’t want all `Scalar[dt1]` to be implicitly
convertible to `Scalar[dt2]` for any dt1 and dt2, and we don’t want Mojo to
have hard coded knowledge of library types like SIMD. Furthermore, the approach
above only works when one or the other types is the right answer.

## Type merging needs to be able to invent new novel types

Consider examples like this (Note: these all work now):

```mojo
def ptr_example(mut x: Int, mut y: Int, cond: Bool):
  alias s1 = "hello"    # Type = StringLiteral["hello"]
  alias s2 = "goodbye"  # Type = StringLiteral["goodbye"]

  # Currently error: can't merge two different types.
  alias someStr = s1 if cond else s2
  # Desired type: StaticString
```

Mojo would need to know that merging two different string literals should
promote to type `StaticString` (which type-erases the value of the literal).

Or pointers with origins that need to be unioned:

```mojo
def ptr_example(mut x: Int, mut y: Int, cond: Bool):
  xptr = Pointer.address_of(x) # Type: Pointer[Int, origin_of(x)]
  yptr = Pointer.address_of(y) # Type: Pointer[Int, origin_of(y)]

  # Currently error.
  xy_ptr = xptr if cond else yptr
  # Desired type: Pointer[Int, origin_of(x, y)]

  xy_ptr[] += 42
```

We want the code above to “just work”, but it presents a problem: the type of
`xy_ptr` needs to be inferable to something with the origins of x and y merged
(because the pointer could be to either argument)… but how does Mojo know that?
Today it doesn’t, and can’t. The type to use needs to be definable by the
library author (the author of `SIMD` or `Pointer` ).

## Proposed solution: a new `__merge_with__` dunder

As with other solutions in Mojo, we can solve these problems by giving library
authors another tool in their toolbox to tell the Mojo compiler how to handle
the situation. In this case, the compiler has a value on each side of the
conditional, and knows the type of the other side, and needs an operation that
a) determines the result type, and b) does the conversion.

We do this by defining a new dunder (we should support both the forward and the
"r" forms, as is typical for binary operators in Mojo):

```mojo
struct StringLiteral[value: ...]:  # slightly simplified from StringLiteral.mojo
   def __merge_with__[
         other_type: type_of(StringLiteral[_])
      ](self) -> StaticString:
        return self
```

Let’s break it down. The `__merge_with__` dunder takes two inputs and has one
result, and has an implementation:

1. It takes `self` which is the **value** to be converted.

2. It takes `other_type` which is a **type parameter** which is the type to
   convert to, so the parameter is either the metatype of a struct, or a Trait
   that all the destination types convert to. The parameter name must be
   `other_type`.

3. It returns a `result` which has a type that is computed from the input value
   and parameter. In this case, by unioning the origins of the two together.

4. The implementation can be anything, but would typically be invoking an
   explicit constructor or just inlining the low level details. In this case, a
   `StringLiteral[_]` value is implicitly convertible to `StaticString`.

The type unioning algorithm uses this to see if the left side can be converted
to the right side, and whether the right side can be converted to the left
side. If so, and if they both agree on a result type, then the compiler can
provide the right behavior.

For pointers it is the same thing, just a bit scarier looking because of
generics:

```mojo
struct Pointer[type, origin]:  # slightly simplified from pointer.mojo
   # TODO: '_' doesn't work right in parameter lists currently, so the unbound
   # params of Pointer need to be explicitly declared.
   def __merge_with__[other_type: type_of(Pointer[type, _])]
      (self) -> Pointer[type, origin_of(self.origin, other_type.origin)]:
        return {self._value}
```

Ok, that is a mouthful, but the same as before. The similar approach can be
used with `Span` and other origin-taking types.

This would also solve the numeric issue it would look like:

```mojo
struct SIMD[type: DType, size: Int](
    def __merge_with__[other_type: type_of(SIMD[_, size])]
      (self) -> SIMD[type.merged_with(other_type.type), size]:
        return {self} # Use explicit conversion ctor

struct DType:
   ...
   def merged_with(self, other: DType) -> DType:
      # ... decide how to merge two dtypes, or use pop operation to do it...

```

Of course, this only works if the width of the SIMD vector is the same, we
might also want to allow broadcasting. This could be supported in a few ways,
e.g. do the same thing as “merged_with” to decide the result type using
dependent types, or by using overloads of `__merge_with__`.

This works great for simple cases, e.g. it allows merging an `Optional[Int]`
and an `Int` into an `Optional[Int]` and many other cases. However, this isn’t
enough for numeric conversions, e.g. consider:

```mojo
def int_example(a: Int8, b: Int16, cond: Bool):
  # Currently: error: value of type 'SIMD[int8, 1]' is not compatible with value of type 'SIMD[int16, 1]'
  c = a if cond else b
  # Desired type: Int16
```

This really should work, but we don’t want all `Scalar[dt1]` to be implicitly
convertible to `Scalar[dt2]` for any dt1 and dt2, and we don’t want Mojo to
have hard coded knowledge of library types like SIMD. Furthermore, the approach
above only works when one or the other types is the right answer.

## Type Merging with this proposal

With this proposal, Mojo’s type unification follows the following algorithm in
pseudo code:

```mojo
def get_common_type(typea, typeb) raises -> Type:
   # If the types are already identical, then we are done.
   if typea == typeb:
       return typea

   # If either type has a __merge_with__ function that accepts the other type
   # then this completely overrides any other behavior.
   amerge = typea.lookup_merge_with, typea, typeb)
   bmerge = typea.lookup_merge_with, typea, typeb)
   if amerge and bmerge:
      if amerge.result_type != bmerge_result_type:
         throw "conflicting merge types"
      return amerge.result_type
   if amerge:
      if is_implicitly_convertible(typeb -> amerge.result_type)
        return amerge.result_type
      throw "cannot convert"
   if bmerge:
      if is_implicitly_convertible(typea -> bmerge.result_type)
        return bmerge.result_type
      throw "cannot convert"

   # Check for implicit conversions.
   a_impl_converts_to_b = is_implicitly_convertible(typea -> typeb)
   b_impl_converts_to_a = is_implicitly_convertible(typeb -> typea)
   if b_impl_converts_to_a and a_impl_converts_to_b:
       throw "ambiguous conversion"
   if a_impl_converts_to_b:
       return typeb  # Use implicit conversion
   if b_impl_converts_to_a:
       return typea  # Use implicit conversion

   # Elided: do similar test for @__nonmaterializable types.
   throw "no common type found"
```

To paraphrase, the overall behavior here has this order of resolution:

1) If either type implements a matching `__merge_with__` function, then that
   overrides all other behavior.

2) If not, the compiler checks for implicit conversions. This should cover
   almost all cases, because typically common types are one of the two types.

3) The compiler rejects things that are ambiguous.

Some other notes that may be helpful:

- Any given type is allowed to have multiple `__merge_with__` overloads for
  different cases. Each overload can produce different target types if they
  want to. This allows defining `A.mergewith(B)->C` but `A.mergewith(D)->E`.

- When dealing with a merge two incompatible types `A <-> B`, it is sufficient
  to implement `A.mergewith(B)` or `B.mergewith(A)` since a type is always
  considered implicitly convertible to itself.

- When merging two incompatible types to a third type: `merge(A, B) = C`,
  it is sufficient to define just `A.mergewith(B)->C` if B is already implicitly
  convertible to `C`.
