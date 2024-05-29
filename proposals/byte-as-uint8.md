# Standardize the representation of byte sequence as a sequence of unsigned 8 bit integers

At this point in time, a sequence of bytes is often represented as a sequence of
signed 8 bit integers in Mojo standard library.  Most noticeable example is the
underlying data of string types `String`, `StringLiteral`, `StringRef` and
`InlinedString`, but also APIs like for example the hash function `fn
hash(bytes: DTypePointer[DType.int8], n: Int) -> Int:`.

## Motivation

Logically a byte is an integer value between `0` and `255`. Lots of algorithms
make use of arithmetic ground by this assumption.  A signed 8 bit integer on
the contrary represents values between `-128` and `127`. This introduces very
subtle bugs, when an algorithm written for unsigned 8 bit integer is used on a
signed 8 bit integer.

Another motivation for this change is that Mojo aims to be familiar to Python
users. Those Python users are familiar with the `bytes` class, which itself is
working with values between `0` and `255`, not values between `-128` and `127`.

## Examples

### Division

A value `-4` represented as `Int8` has the same bit pattern as value `252`
represented as `UInt8`.  `-4 // 4` equals to `-1` (`bx11111111`), where `252 //
4` equals to `63` (`bx00111111`) as we can see the bit patterns are different.

### Bit shift

Values `-1` and `255` have the same bit pattern as `Int8` and `UInt8`
`bx11111111` but `-1 >> 1` results in `-1` (same bit pattern), where `255 >> 1`
results in `127` (`bx01111111`)

## Proposal

A text based search for `DTypePointer[DType.int8]` and `Pointer[Int8]` on
current open-sourced standard library revealed 29 results for `Pointer[Int8]`
and 78 results for `DTypePointer[DType.int8]`.  Replacing
`DTypePointer[DType.int8]` with `DTypePointer[DType.uint8]` and `Pointer[Int8]`
with `Pointer[UInt8]` on case by case bases is a substantial refactoring effort,
but it will prevent a certain class of logical bugs (see
<https://github.com/modularml/mojo/pull/2098>). As it is a breaking change in
sense of API design, it is sensible to do the refactoring as soon as possible.
