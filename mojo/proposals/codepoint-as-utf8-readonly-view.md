# Codepoint as UTF-8 readonly view

date: 2026-03-18, status: proposed

## Motivation

Currently the `Codepoint` type stores Unicode codepoints as their UTF-32
representation. It is a niche type and not often used because the performance
hit of parsing a unicode codepoint from a UTF-8 iterator is significant.
Codepoint's API is also somewhat limited compared to `String`, `StringSlice`,
and `StringLiteral`.

## Concrete proposal

While I would prefer to parametrize it based on encoding, we can go in a simpler
direction first that still achieves the same purpose:

1. Let's switch the struct's internals to be the raw UTF-8 value
    - Removes the overhead from UTF-32 parsing unless explicitly requested by
        the developer (cost that can be easily pushed to compile time if known)
2. Add several stringlike APIs to it
    - It should interop nicely with string equality, etc.
    - It would help cleanup several parts of the stdlib where we use private
        utf-8 helpers.
    - If we end up parametrizing the struct based on encoding, this will also
        help streamline the transition.
3. Make the iterator for all stringlike types be just `Iterator[Codepoint]`.
    - This would make every iteration be immutable
    - This inlines the data pointed to by the string (that is dereferenced
        anyway during iteration)
    - This reduces the size of the iteration from `size_of[StringSlice]()` (8 -
        16 bytes depending on the platform) to `size_of[Codepoint]()` (4 bytes)
    - This will be a great building block to implement more iterator-friendly
        string APIs. Since we won't be dealing with origins, this enables
        concatenations of string operations that return arbitrary new codepoints
        from the original data.
