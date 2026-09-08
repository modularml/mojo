# Upgrading "Trivial"

**Status**: Partially Implemented.

Date: April 2025

The `@register_passable("trivial")` decorator on struct declarations were added
early in the evolution of Mojo as a way to bring up the language, when
memory-only types were added. Given other progress, it makes sense to rework
this to generalize them.

Here are some simple examples of both forms of `@register_passable`:

```mojo
@register_passable
struct RP:
  var v: Int
  var p: ArcPointer[String]

  # Can choose to provide __copyinit__ and __del__ if it wants.
  def __copyinit__(out self, existing: Self): ...

  # Can't define __moveinit__.

@register_passable("trivial")
struct RPTrivial:
  var x: Int
  var y: Float64

  # Can't define __copyinit__ or __moveinit__ or __del__
```

## What `@register_passable` does today

The `@register_passable` decorator has a few effects, because it changes the
fundamental nature of how values of that types are passed and returned from
functions: instead of being passed around by reference (i.e. a hidden pointer)
it is passed in an MLIR register. This is an important optimization for small
types like `Int` - this document doesn’t explore this behavior and doesn’t
recommend changing it.

As a consequence of being `@register_passable` :

- the value doesn’t have “identity” - you can’t take the address of `self` on
  `read` convention methods, this is because they are in a register, not in
  memory.

- You aren’t allowed to define an `__moveinit__`: Mojo needs to be able to move
  around values of your type by loading and storing them, so it doesn’t allow
  you to have custom logic in your. move constructor.

- Mojo makes your type implicitly conform to `Movable` and will synthesize a
  trivial move constructor for you.

- Mojo checks that any stored member (`var` ’s) are also `@register_passable` -
  it wouldn’t be possible to provide identity for a contained member if the
  container doesn’t have identity.

- The type can choose whether it wants to be `Copyable` or not, use the
  `@value` decorator etc.

I'm quite happy with all of this, I feel like it works well and provides
important performance optimizations that differentiate Mojo from languages like
C++ and Rust.

## What `@register_passable("trivial")` does today

This is an extension of the `@register_passable` decorator that indicates the
type is “trivial” which means "movable + copyable by memcpy" and "has no
destructor”. In addition to the behavior of the `@register_passable` decorator,
marking something trivial provides the following behavior:

- The type implicitly conforms to `Copyable` and is given a synthesized
  `__copyinit__` that does a memcpy. It is given a trivial `__del__` member as
  well (so can’t be a linear type).

- All declared members are required to also be trivial, since you can’t memcpy
  or trivially destroy a container if one of its stored members has a
  non-trivial copy constructor.

- You are not allowed to define a custom `__copyinit__` or `__del__`.

- Types that are both `@register_passable` and trivial have some internal IR
  generation benefits to reduce IR bloat and improve compile time, but these
  aren’t visible to users of Mojo.

This behavior has been very effective and valuable: both as a way to make it
more convenient to define a common class of types (not having to write a
moveinit explicitly) and as a performance optimization for very low level code.

## What should "trivial" mean?

There are currently three core operations relevant to this topic: destruction,
copying and moving. "Trivial" for destruction means that the destructor is a
no-op, trivial for copying means that the value can be copied with a `memcpy`
(no other side effects are required). Trivial for moving means that a value can
be moved by `memcpy`'ing its state and considering the original value.

These are all each orthogonal axes: it is common for types to be trivially
movable, but have a destructor (e.g. an Arc pointer).

## Desired improvements to `“trivial”`

Trivial has worked well, but it can be better. Some observations:

- The semantics of ”copyable with memcpy” and “no behavior in the destructor”
  really have nothing to do with register pass-ability, we would like for this
  to work with memory-only types too.

- We would like for algorithms (e.g. `List.resize` or `List.__del__`) to be
  able to notice when a generic type is trivial, and use bulk memory operations
  (`memcpy`) instead of a for loop or skip calling destructors for elements.

- We would like to be able to define that a type is trivial if its
  elements are trivial, even if one of the element types is generic.

- We want types like `InlineArray` to themselves be considered trivially
  copyable when their element is trivially copyable, like conditional
  conformance.

For all these reasons, we need to upgrade “trivial” and decouple it from the
`@register_passable` decorator.

## Background: synthesized methods

The Mojo compiler now [implicitly synthesizes
conformances](upgrading-value-decorator.md) to `AnyType` (providing a
destructor), to `Movable` (providing `__moveinit__`) and to `Copyable`
(providing `__copyinit__`). The compiler can know that each of these operations
are trivial if all contained fields of the type are trivial according to the
same operation: for example, a struct's destructor is trivial if its elements
destructors are all trivial.

## Proposed approach: introduce aliases to `AnyType`, `Movable`, and `Copyable`

The proposed approach is to extend each of these traits with a new alias
indicating whether the operation is trivial:

```mojo
trait AnyType:
  # Existing
  def __del__(owned self, /): ...
  # New
  alias __del__is_trivial_unsafe: Bool = False

trait Movable:
   # Existing
   def __moveinit__(out self, owned existing: Self, /): ...
   # New
   alias __moveinit__is_trivial_unsafe: Bool = False

trait Copyable:
   # Existing
   def __copyinit__(out self, existing: Self, /): ...
   # New
   alias __copyinit__is_trivial_unsafe: Bool = False
```

These aliases allow containers to use custom logic with straight-forward
`comptime if`'s to conditionalize their behavior:

```mojo
struct List[T: Copyable & Movable]: # Look, no hint_trivial_type!
    ...
    def __del__(owned self):
        comptime if not T.__del__is_trivial_unsafe:
            for i in range(len(self)):
                (self.data + i).destroy_pointee()
        self.data.free()

    def __copyinit__(out self, existing: Self):
        self = Self(capacity=existing.capacity)
        comptime if T.__copyinit__is_trivial_unsafe:
            # ... memcpy ...
        else:
            # ... append copies...
```

These aliases impose no runtime overhead, and allow collections to express this
sort of logic in a natural way. This also keeps each of these notions of
"trivial" cleanly orthogonal.

Let's explore some nuances of this.

### Mojo compiler can synthesize the alias correctly

When Mojo implicitly synthesizes a member, it can synthesize the alias as well.

```mojo
struct MyStruct(Copyable):
  var x: Int
  # Compiler already synthesizes:
  # def __copyinit__(out self, other: Self):
  #    self.x = other.x

  # Compiler newly synthesizes:
  # alias __copyinit__is_trivial_unsafe = True.
```

However, the compiler doesn't have to know anything about the types, it actually
just uses the alias to support more complex cases correctly:

```mojo
struct MyStruct2[EltTy: Copyable](Copyable):
  var x: Int
  var y: EltTy
  # Compiler already synthesizes:
  # def __copyinit__(out self, other: Self):
  #    self.x = other.x
  #    self.y = other.y

  # Compiler newly synthesizes:
  # alias __copyinit__is_trivial_unsafe =
  #     Int.__copyinit__is_trivial_unsafe & EltTy.__copyinit__is_trivial_unsafe
```

This builds on Mojo's powerful comptime metaprogramming features naturally.

### Types implementing explicit operations default correctly

Because these aliases have default values, types that implement their own
method would be handled correctly by default:

```mojo
struct MyStruct:
  # __del__is_trivial_unsafe defaults to false.
  def __del__(owned self):
    print("hi")
```

### Types can implement custom conditional behavior

Because these aliases are explicit, advanced library developers can define
smart custom behavior:

```mojo
struct InlineArray[ElementType: Copyable & Movable]:
    def __copyinit__(out self, other: Self):
        comptime if ElementType.__copyinit__is_trivial_unsafe:
            # ... memcpy ...
        else:
            # ... append copies...

    # InlineArray's copy is itself trivial if the element is.
    alias __copyinit__is_trivial_unsafe = T.__copyinit__is_trivial_unsafe
```

Note that a type author getting this wrong could introduce memory unsafety
problems, on both the definition and use-side of things. This is why the aliases
have the word "unsafe" in them.

### Removing `@register_passable("trivial")`

It appears that this would allow us to remove `@register_passable("trivial")`,
but keep `@register_passable`. The former would be replaced with
`@register_passable`+`Copyable`+`Movable` conformance. If that causes too much
boilerplate, then we can define a helper trait in the standard library that a
type can conform to:

```mojo
@register_passable
trait RPTrivial(Copyable, Movable):
  pass
```

### Potential Design / Implementation Challenges

Here are a couple design and implementation challenges that may come up:

1) We need to decide on a final name for the aliases, e.g. do we want to include
   the word "unsafe" in them.

2) We might not be able to use `Bool` as the type of these properties, because
   Bool itself conforms to these traits. We may run into cycling resolution
   problems. Mitigation: we can use `i1` or some purpose built type if
   required.

3) We don't actually have defaulted aliases yet. Mitigation: we can hack the
   compiler to know about this, since these traits already have synthesis magic.

## Alternatives Considered

The chief alternative that was previously discussed was to introduce one or more
subtraits like:

```mojo
trait TriviallyCopyable(Copyable): pass
```

There are a few reasons why the proposed approach is nicer than this one:

1) Being "trivial" is a property of a copy constructor, so it seems like it
   should be modeled as an aspect of its conformance (as proposed here), not
   as a separate trait.
2) That would require doubling the number of traits in the standard library.
3) In the short term, we don't have conditional conformance or comptime trait
   downcasting, so we couldn't implement the behavior we need to use these.
4) Even if we did have those, we could do custom conditional implementation
   like shown above for `InlinedArray`.

As such, the proposed approach is more pragmatic in the short term, but also
seems like the right approach in the long term.

## Conclusion

This proposal should expand the expressive capability of Mojo by building into
other existing features in a nice orthogonal way. It does so without
introducing new language syntax, and indeed allows removing
`@register_passable("trivial")`, which removes a concept from the language.
