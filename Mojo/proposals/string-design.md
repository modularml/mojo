# Mojo `String` type

**Status**: Implemented

## Introduction

This document provides a high-level design doc and details about the core Mojo
`String` type. It is meant to be a bit of a reference and answer some questions
about its design and non-obvious tradeoffs.

## Basic Features

The Mojo `String` type is designed to be an efficient owning datatype that can
be used for a range of purposes in application programming. It needs to be
efficient, it needs to have convenient APIs, and it needs to be able to
interoperate.

Some high-level capabilities it provides:

- Unicode support.

- Zero copy from constant data like string literals.

- "Short string optimization" to avoid allocating for small strings.

- Ability to interoperate with 'nul' terminated C string APIs.

- O(1) copyinit by using a reference counted representation with
  lazy-copy-on-mutation.

This document will examine some of the basics of how it works and why. It
can be extended over time.

## Representation implementation

The `String` type is designed to be three-machine words in size: 24 bytes on a
64-bit system, and 12 bytes on a 32-bit system. It has two primary forms: an
"inline" representation when the string data is short, and an "indirect"
representation when the string points to external data. Both formats use a
common set of "Flags" bits at the top of the object.

### Shared Flags

The shared layout looks like this on a 64-bit system (32-bit systems are the
same but have fewer bytes). This is the machine memory layout, where the first
`?` is the first byte in the start of the string:

```text
[ ?, ?, ?, ?, ?, ?, ?, ?,   # "_ptr_or_data": the first word in memory
  ?, ?, ?, ?, ?, ?, ?, ?,   # "_len_or_data": the second word
  ?, ?, ?, ?, ?, ?, ?, OF ] # "_capacity_or_data": the third word
```

The `OF` byte is the last byte in the memory object. It has three flags that
are valid or either representation:

- `FLAG_IS_INLINE` is the top-most bit (the sign bit of the word). This bit
  indicates whether the rest of the bytes are in the "inline" representation or
  the "indirect" representation. It is the most frequently accessed of all the
  flags, so it is important that it is fast. Being the top bit allows hardware
  to just check to see if the 3rd word is negative.

- `FLAG_IS_REF_COUNTED` is the next bit, which indicates if the string is
  pointing to a reference counted mutable buffer. This allows checks in methods
  such as `__del__` to quickly skip reference counting logic.

- `FLAG_HAS_NUL_TERMINATOR` is the next bit, which indicates if the string is
  known to have an accessible "NUL" byte just beyond its declared length. This
  allows inter-operating with C APIs that want nul-terminated strings. This bit
  is valid for both representations.

In the implementation, switching between representations is handled and mostly
encapsulated by the `_capacity_or_data` field, which is a `UInt` type. There are
several helper methods in the source code (under the `# Capacity Field Helpers`
section) that manage the implementation details for working with this field.

Given the common flags, we'll now look at the meaning of the other bytes in both
representations.

### Inline String Representation

For an inline string, the `?` bytes are all parts of the string data. For
example if you create a string with the value "abcd" (but not as a literal) it
can be stored as:

```text
[ a, b, c, d, ?, ?, ?, ?,    # "_ptr_or_data": holding data.
  ?, ?, ?, ?, ?, ?, ?, ?,    # "_len_or_data": holding data.
  ?, ?, ?, ?, ?, ?, ?, OF ]  # "_capacity_or_data": holding data + size + flags
```

This allows the `String` to store up to 23-bytes of text data inline on 64-bit
systems, and 11 bytes on 32-bit systems. This is enough to hold many common
strings inline, including most Grapheme clusters, most simple integers converted
to a string, etc.

When in this form, the length of the string is stored in the low 5 bits of `OF`,
and is accessed with the `INLINE_LENGTH_START` and `INLINE_LENGTH_MASK`
constants.

### Indirect String Representation

Small strings are important for performance, but we need indirect strings for
generality. When the "is_inline()" predicate returns false, the representation
of the string looks like this:

```text
[ <<pointer address>>,   # "_ptr_or_data": address of the target.
  <<string length>>,     # "_len_or_data": the size in bytes.
  <<capacity + flags>> ] # "_capacity_or_data": capacity + flags
```

The first word contains a pointer with the address of the start of the string
data. When in this representation, this is what `unsafe_ptr()` returns.
Similarly, the second word contains the number of bytes in the string, which is
returned by `len(str)` and `str.byte_length()` when in this representation.
These two fields allow us to point to arbitrary-sized strings.

The third field is a bit trickier - it holds the capacity of the string as well
as the three flags described above. To make sure we can hold an arbitrary
capacity string, the `String` type does a trick: it guarantees the actual
capacity will be a multiple of 8, which allows it to shift the capacity down by
three bits, making room for the flags.

If the string is pointing to static constant memory, the `_capacity_or_data`
field can be one of two bitpatterns: 0 with no flags set, or 0 with
`FLAG_HAS_NULL_TERMINATOR` set. When attempting to get a mutable pointer to
static constant memory the implementation will decide whether to inline the
string, or reallocate it to the heap based on the requested capacity. This
avoids having to allocate when initializing from a `StringLiteral` or
`StaticString`.

### Mutable Indirect String Representation

When a string is indirect and not mutable, the pointer field points to the
string data corresponding to the string, but there is an additional
`Atomic[Index]` header *before* the string data the pointer points to.
This header contains a reference count for the mutable string buffer.

This design allows the copy constructor for the String type to guarantee that it
is O(1): it just copies three words of data, and increments the reference count
if indirect and mutable. When checking to see if a string is mutable, the data
is copied to a new buffer when the pointer is to an immutable string or when it
points to a shared buffer. This has no performance impact for short strings
(an important common case) and has very low overhead for anything else.

## Unicode support

TOWRITE.

## Other design topics

This section contains implementation details about the `String` type that may be
non-obvious.

### References to constant strings

As mentioned in the "indirect" representation section, the pointer of an
indirect string may refer to static constant data. This optimization is
important because string literals are very common, and we don't want users to
have to worry about use of `String` vs `StaticString` for optimization purposes:
most APIs should just take `String` for consistency and generality.

The capacity is the length of the constant string, when requesting a mutable
pointer it will reallocate the string with a capacity based on that length.

Construction from short constant strings (like "foo") loads the string data at
`String` construction time, which keeps usage of the string direct where
possible.

### Mutable String Views

The `String` type may contain pointers to static-constant memory, but
nevertheless it is sometimes important to provides mutable slice and mutable
pointer access to the underlying string data. We do this by making lazy
(on-demand) copies of immutable data when the client needs a mutable view.

However, internal-mutation is somewhat rare - the most common string mutation
method is a simple append, and that often needs to reallocate the string. We
want to constrain the API so we don't make unnecessary copies, so we split these
API's into non-mutating with short names e.g. `str.unsafe_ptr()` and provide
explicit mutable access with `_mut` suffixed version of these:

```mojo
def test_copy():
    var s0 = "find"
    var s1 = s0       # Maintains a pointer to immutable memory
    s1.unsafe_ptr_mut()[3] = ord("e")
    assert_equal("find", s0)
    assert_equal("fine", s1)
```

This keeps the common case fast and simple and avoids allocating memory in many
cases.

All mutation of the string is routed through `str.unsafe_ptr_mut(capacity=128)`
where you can optionally request a larger capacity. This is where all the logic
to make the string mutable occurs, for example inlining or reallocating a static
constant string.

See the `test_string.mojo` unit test for more details.

### 'nul' terminated string support

Mojo is a pragmatic language and needs to interoperate with existing C APIs, and
many of them accept nul-terminated strings. To do so, String provides a
`str.unsafe_cstr_ptr()` method that will produce an `UnsafePointer` that is
guaranteed to be nul terminated.

`Nul` termination is a bit of a nuisance: it makes appending to strings slower,
it prevents referring to slices into the middle of static strings, it can be
incorrect if the string contains an embedded nul, and it is unnecessary for the
vast majority of strings. Nevertheless, this is still an important feature for
`String` to provide support for.

`String`'s approach is to take a lazy-approach: it only guarantees
nul-termination when the `str.unsafe_cstr_ptr()` method is called. This requires
the `str.unsafe_cstr_ptr()` method to be a mutating method (because it can
modify the underlying string data). `String` keeps track of whether a `nul` has
been added to make future queries more efficient with the
`FLAG_HAS_NUL_TERMINATOR`. The Mojo compiler guarantees that data pointed to
by the `StringLiteral` type is always `nul` terminated: these guarantees work
together so Mojo never needs to make a copy of a string literal when passed to
a C string - it remembers the nul is there, which avoids having to mutate it.
