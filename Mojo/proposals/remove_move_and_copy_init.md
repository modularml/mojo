Erica Sadun, Date: 11 Dec 2025

Status: Implemented for Mojo 26.2

# Pitch: Init Unification

A clear and consistent API surface makes a language easier
to learn. When one pattern extends across the language,
skills learned in one place apply everywhere else.

Adding special methods that are spelled differently
from each other forces users to stop and ask:
“Is this a new semantic distinction?” or “Are these
just different names for the same concept?” When the
answer is the second, the language is adding cognitive load
without adding clarity.

This pitch proposes removing `__copyinit__` and
`__moveinit__` and replacing them with overloaded
variants of `__init__`. With required argument labels
for `move`, this provides a single, predictable
construction and initialization entry point and a simpler mental model.

- "Same idea; same spelling" beats "Same idea; different spellings".
- Uniform overload sets are easier to learn and remember.

The current approach expands the API surface without expanding
the job that needs doing.

## Current design and its challenges

Mojo defines three separate constructor/initializer methods, each
with different names and spelling:

| Initializer    | Copy value | Move value | Takes ownership of |
|----------------|------------|------------|--------------------|
| `__init__`     | ✓          | ✓          | value (copy/move)  |
| `__copyinit__` | ✓          |            | copied value       |
| `__moveinit__` |            | ✓          | moved value        |

These three methods essentially perform the same operation:
construct a new value from an existing one. The differences
lie in ownership.

The problem is this: these naming differences don't reflect
foundational differences. Copying and moving are properties
of *value flow* and not distinct categories of constructors.
Having different spellings requires users to remember
special cases from names rather than recognize intent
from the call-site.

This makes learning harder in three ways:

1. **Redundant names:** Users must ask whether each spelling
   encodes unique behavior, even when it does not.
2. **Fragmented learning:** Initialization requires remembering
   three separate entry points instead of one unified pattern.
3. **Documentation overhead:** Documentation must explain three
   constructor spellings instead of one mechanism.

In short, the current approach expands the API surface without
expanding the job that needs doing.

## Proposed design

This pitch replaces `__copyinit__` and `__moveinit__`
with overloaded variants of `__init__`. Instead of
three constructor methods, types that
adopt `Copyable` or `Movable` provide two `__init__`
overloads that differ only in argument convention:

```mojo
# Copy constructor/initializer: same behavior as today's __init__
# Constructs by copying from the source value (Copyable)
def __init__(out self, *, copy: Self)

# Move constructor/initializer: consume the source (Movable)
# Requires a keyword at the call site to indicate movement
def __init__(out self, *, deinit take: Self)
```

- `copy` is self-evident.
- `take` makes movement and ownership transfer clear.

This design unifies object construction and initialization under
a single, predictable entry point. It removes two magic method
names without removing any capability from the type system.
The keyword only approach ensures easier disambiguation within
the overload group.

## Impact

### Language and compiler

Types that currently define custom copy or move behavior
will migrate their logic into the corresponding
`__init__` overload.

- The compiler select behavior from the overload.

- The `^` operator keeps its existing meaning and behavior.
When used with `take:`, it selects the move initializer.

- Error messages that refer to `__copyinit__` / `__moveinit__`
update to refer to the corresponding `__init__` overload.

- Error messages about missing `__copyinit__` / `__moveinit__`
are removed.

### Standard library

Types in the standard library that define custom copy or move behavior migrate
from:

- `__copyinit__` to `__init__(out self, other: Self)`
- `__moveinit__` to `__init__(out self, *, deinit take: Self)`

The semantics don't change, just initializer spelling.

### User code

This is a breaking change for user-defined types that implement
`__copyinit__` and `__moveinit__`. Those implementations must
move to the new `__init__` overloads. In return, users get a smaller
and more predictable constructor/initializer surface: one name
(`__init__`) with clear value-flow conventions instead of
three separate magic methods.

This will affect sample code, documentation, as well as code
developed by our users, which will need migration/fixits.

## Spelling and terminology

| Term     | Meaning                                                                                                                                               |
|----------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| Copy     | Duplicate a value.                                                                                                                                    |
| Copyable | A value capability: the value may be safely duplicated.                                                                                               |
| Move     | Move a value. The source is consumed and no longer usable.                                                                                            |
| Movable  | A value capability: the value may be safely consumed and transferred.                                                                                 |
| Take     | “Move and own.” Construction moves the value and transfers<br />ownership to the new instance. The term already<br />appears in `UnsafePointer` APIs. |

This pitch adds a required argument label `take` for transfer and `copy` for
copy initialization. This terminology is inspired from `UnsafePointer`'s
`take_pointee`.
