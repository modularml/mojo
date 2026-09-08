# `@unavailable` Decorator for Intentionally-Unavailable APIs

**Status**: Accepted (implemented).

**Author**: Nathan Ward

**Date**: June 2026

## Summary

`@unavailable` is a decorator for functions and methods that turns any
reference to the decorated declaration into a **compile-time error**. It is the
hard-stop sibling of `@deprecated` (which only warns) and shares its syntax. The
decorated declaration still participates in name and overload resolution, so the
compiler emits a *targeted, author-written* error at the use site instead of a
generic "unknown name" or "no matching overload."

```mojo
@unavailable("`parse` was removed in 25.7; use `parse_v2` instead")
def parse(s: String) -> Int:
    ...

def caller():
    _ = parse("123")   # error: `parse` was removed in 25.7; use `parse_v2` instead
```

## Motivation

When an API is removed, renamed, or should never have existed, an author has two
unsatisfying options today:

1. **Delete it.** Users then get a generic diagnostic
   (`use of unknown declaration 'parse'`, or `no matching function in call`)
   that says nothing about *why* it is gone or *what to use instead*.
2. **`@deprecated` it.** This keeps the API working and only emits a warning.
   That is the wrong tool when the API must *not* be used.

`@unavailable` fills the gap. The symbol stays visible to name and overload
resolution, but any use is a hard error carrying an author-written explanation,
and optionally a fix-it that renames the call to a replacement.

### Motivating example: `String` indexing

Mojo's `String` is UTF-8 encoded, so a positional index `s[i]` is ambiguous — it
could mean the i-th byte, the i-th Unicode code point, or the i-th user-visible
character (grapheme cluster). Silently picking one is an issue, and *removing*
`__getitem__` would produce an opaque "no matching overload" error for the very
natural-looking `s[i]`.

Instead, `String` declares the positional `__getitem__` overloads as
`@unavailable`:

```mojo
struct String:
    @doc_hidden
    @unavailable(
        "String does not support direct positional indexing like `s[i]`"
        " because Mojo strings are UTF-8 encoded, and the same position can"
        " mean three different things. Use one of: `s[byte=i]` for a raw"
        " UTF-8 byte, `s[codepoint=i]` for a Unicode code point, or"
        " `s[grapheme=i]` for a user-visible character (grapheme cluster)."
    )
    def __getitem__(
        self, _index: Some[Indexer], /
    ) -> StringSlice[origin_of(self)]:
        ...
```

Now `s[i]` resolves to this overload and the user gets a precise, instructive
error pointing them to `s[byte=i]` / `s[codepoint=i]` / `s[grapheme=i]`, while
the keyword overloads remain fully available.

## Usage

`@unavailable` mirrors `@deprecated`. It accepts either a reason message or a
replacement symbol.

```mojo
# Positional reason (must be a string literal).
@unavailable("don't use this; it never worked")
def old_api():
    ...

# `reason=` keyword form (identical meaning).
@unavailable(reason="don't use this; it never worked")
def old_api2():
    ...

# `use=` names a replacement symbol.
@unavailable(use=new_api)
def renamed_api():
    ...

def new_api():
    pass
```

The body of an `@unavailable` function must be `...`, even when the function has
a non-`None` return type (no `return` is required, because the body is never
reachable):

```mojo
struct StringLike:
    @unavailable(
        "no length for `StringLike`; use byte_length() or codepoint_length()"
    )
    def __len__(self) -> Int:
        ...     # `...` is required; `return 0` here is an error.
```

### Interaction with `@deprecated` and `@stable`

`@unavailable` is **mutually exclusive** with both `@deprecated` and `@stable`,
in either order.

## Alternatives considered

### Reuse `@deprecated` with an "error" severity flag

Rejected. The warn-vs-error distinction is fundamental enough to deserve its own
decorator name: intent is obvious at the declaration, and a distinct decorator
makes the mutual-exclusivity rule with `@deprecated`/`@stable` trivial to state
and enforce.

### Just delete the API

Rejected for the discoverability reasons in the motivation: deletion yields a
generic diagnostic with no guidance, and it cannot target a single overload
while leaving sibling overloads in place.

### Allow `@unavailable` on types and aliases

Deferred. The motivating cases are all functions and methods — especially
overload-level blocking, which is meaningless for a type. Restricting the
initial surface keeps the feature small; type- or alias-level unavailability can
follow if a concrete need appears.
