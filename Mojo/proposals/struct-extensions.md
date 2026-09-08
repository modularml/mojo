# Struct Extensions Goals + Requirements

**Status**: Draft.

Author: Evan Ovadia

**TL;DR:** Struct extensions let library B add a `fly_to` method to library A's
`Spaceship`, and we want that. This doc talks about what they are, why we want
them, and what they should do.

Scope: This designs the language feature, not the implementation.

Goal: Agree on what struct extensions should do.

## Basic Use Case

In library A:

```mojo
struct Spaceship:
    var location: String
    def liftoff(self):
        ...
    def set_location(mut self, new_location: String):
        self.location = new_location
```

In library B’s `spaceship_extensions.mojo`:

```mojo
import A.Spaceship

extension Spaceship:
    def fly_to(mut self, new_location: String):
        self.liftoff()
        self.set_location(new_location)
```

And now, some code (also in library B) can use it like a method on any
`Spaceship`:

```mojo
# Pseudocode (actual syntax/semantics TBD in this doc):
import A's struct Spaceship
import B's extension Spaceship

def do_things(ship: Spaceship):
    ship.fly_to("Corneria")
```

(There are various options for how/what we import, we’ll explore that below.)

## More Advanced Cases + User Journeys

### 1: User wants to break up a large struct definition file

User starts with this:

```mojo
# Library L's spaceship.mojo

struct Spaceship:
    var location: String

    def liftoff(self):
        ...

    def set_location(mut self, new_location: String):
        self.location = new_location

    def fly_to(mut self, new_location: String):
        self.liftoff()
        self.set_location(new_location)
```

…but their file has grown too large. They want to break it up. For example:

```mojo
# Library L's spaceship.mojo

struct Spaceship:
    var location: String
    def liftoff(self):
        ...
    def set_location(mut self, new_location: String):
        self.location = new_location

# Library L's spaceship_extensions.mojo

extension Spaceship:
    def fly_to(mut self, new_location: String):
        self.liftoff()
        self.set_location(new_location)
```

Preferably, this wouldn’t be a breaking change, they’d like to do this with
minimal disruption to their users.

### 2: User wants to split off methods that have extra dependencies

User starts with this:

```mojo
# Library L's spaceship.mojo

import python.PythonObject

struct Spaceship:
    var location: String

    def liftoff(self):
        ...

    def set_location(mut self, new_location: String):
        self.location = new_location

    def __init__(init self, py_obj: PythonObject):
        self.location = py_obj.something()
```

…but they don’t necessarily want every `import Spaceship` to bring in all of
`python.PythonObject`'s dependencies.

So they’d break it up into this:

```mojo
# Library L's spaceship.mojo

struct Spaceship:
    var location: String
    def liftoff(self):
        ...
    def set_location(mut self, new_location: String):
        self.location = new_location

# Library L's spaceship_extensions.mojo

import python.PythonObject

extension Spaceship:
    def __init__(init self, py_obj: PythonObject):
        self.location = py_obj.something()
```

For example, these `def __init__(out self, obj: PythonObject) raises: ...` (to
conform to `ConvertibleFromPython`) methods exist:

- in the`Bool` struct in `Mojo/stdlib/std/builtin/bool.mojo`
- in the`Int` struct in `Mojo/stdlib/std/builtin/int.mojo`
- in the`String` struct in `Mojo/stdlib/std/collections/String.mojo`
- in the `SIMD` struct in `Mojo/stdlib/std/builtin/simd.mojo`

And also the `to_python_object` methods (from `PythonConvertible` trait):

- in the `StringSlice` struct in
  `Mojo/stdlib/std/builtin/string_literal.mojo`
- in the `StringLiteral` struct in
  `Mojo/stdlib/std/builtin/string_literal.mojo`
- in the`Bool`, `Int`, `String`, `SIMD` structs from above.

We want to move those out into a separate [file? package?] that can be imported
separately.

This could be helpful when dealing with embedded systems that can’t afford to
bring in a lot of dependencies. This approach might be brittle, as it’s easy to
accidentally indirectly import something. Something like Rust
[features](https://doc.rust-lang.org/cargo/reference/features.html) might be
better, for explicitly saying what you want to pull in. Specifically, it could
work like this:

- Configure an entire module to only be present for a certain target, or only be
present if a certain user-config flag is passed in. Examples:
  - (command line) `--enable_feature my_embedded_library.UseArduinoSupport`
  - (in our `Cargo.toml`-or-whatever equivalent)
    `[features]  my_embedded_library = ["UseArduinoSupport"]`
- The library method would `requires` it, like
  `def arduino_blink() requires UseArduinoSupport: ...`

Note this user journey would be somewhat foiled by decision 6 ("Are extensions
automatically imported?") option B ("Struct imports automatically import their
extensions").

### 3: User wants to break dependency cycles

Similar to B, but for the purposes of avoiding dependency cycles.

For example, today we have a circular dependency between `std.builtin` and
`std.python`, because `Int` depends on `ConvertibleToPython`:

```mojo
struct Int(
    Absable,
    CeilDivable,
    Ceilable,
    Comparable,
    ConvertibleFromPython,  # <-- notice this dependency on Python stuff
    ConvertibleToPython,  # <-- notice this dependency on Python stuff
    ...
```

…and `PythonObject` has this constructor that depends on `Int`:

```mojo
    @implicit
    def __init__(out self, value: Int):
        ...
```

This only works for us because the stdlib is one giant module. If we were to
scale up and split these into multiple modules, we’d have circular dependency
errors.

### 4: User wants to express multiple methods’ `requires` clauses

User starts with this:

```mojo
# Library L's spaceship.mojo

import python.PythonObject

struct Spaceship[engine_type: EngineTrait]:
    var engine: engine_type

    def warp(self) requires engine_type: WarpEngineTrait:
        ...

    def shear_spacetime(self) requires engine_type: WarpEngineTrait:
        ...

    def invert_polarity(self) requires engine_type: WarpEngineTrait:
        ...
```

…but that’s really tedious, so they want some way to specify the bounds only
once.

Perhaps they could use an extension:

```mojo
# All the same file

struct Spaceship[engine_type: EngineTrait]:
    var engine: engine_type

extension Spaceship requires engine_type: WarpEngineTrait:
    def warp(self) requires:
        ...

    def shear_spacetime(self):
        ...

    def invert_polarity(self):
        ...
```

### 5: User wants to be compatible with Python expectations

There are a handful of methods in Python that don’t conform to
[PEP8](https://peps.python.org/pep-0008/) naming standards. For example
`isdigit` rather than `is_digit` , or `assertEqual` in `testing` . To make more
python code “just work” when copy and pasted, one should be able to write:

```mojo
# in stdlib/python_compat/__init__.mojo
from logging import Logger, LogLevel

extension Logger:
   def setLevel(mut self, level: LogLevel):
       ...

   def isEnabledFor(self, level: LogLevel) -> Bool:
       ...

# In some arbitrary file
from python_compat import * # allows the extension to work for Logger()
from logging import Logger, LogLevel
import logging

def foo():
    var logger = Logger()
    logger.isEnabledFor(logging.INFO)
```

### 6: User wants to conditionally conform to a trait

The user wants `List[T]` to be `Copyable`, but only if `T` itself is `Copyable`.

They could use extensions for this, like:

```mojo
struct List[T: AnyType]:
    ...

extension List(Copyable) requires T: Copyable:
    def __init__(out self, *, copy: Self, /):
    self = List(capacity=len(other))
    # ...
```

The user could also extend multiple traits at once, like
`extension List(ImplicitlyCopyable, Copyable) requires T: Copyable: ...`.

Note that extensions aren’t strictly necessary for this. The user could express
this differently:

```mojo
struct Foo[T: AnyType](
    Movable,  # unconditional
    Copyable requires T: Copyable,
    Stringable,  # unconditional
): ...
```

…though the user would have to explicitly re-mention `requires` on the method
too:

```mojo
    def __init__(out self, *, copy: Self, /) requires T: Copyable:
    self = List(capacity=len(other))
    # ...
```

TBD which (or both) we’ll support.

## Decision 1: Where can conforming extensions be?

If the extension conforms to trait, like this:

```mojo
extension List(Copyable) requires T: Copyable:
    def __init__(out self, *, copy: Self, /):
        self = List(capacity=len(other))
        # ...
```

Where can it appear?

- Option A: Either in the struct’s module or the trait’s module.
- Option B: Either in the struct’s file or the trait’s file.
  - Pro: Decision 8 (”What if we import only an `extension` but not its target
    `struct`?”) becomes moot, because it’s in the same scope as either the
    struct or the trait.
  - Con: Against user journey 1, where user wants to break up a large struct
    into multiple files.
- Option C: Anyone can extend any struct for any trait.

Options A and B are are both the same as Rust’s orphan rule in spirit, just
with different scopes. Option C risks some conflicts, but maybe there's a way
we can mitigate those.

None of this would apply to non-conforming extensions if we allow them (pending

decisions 2 and 3).

## Decision 2: Should we allow non-conforming extensions?

Example of a non-conforming extension:

```mojo
# Library L's spaceship.mojo

struct Spaceship:
    var location: String
    def liftoff(self):
        ...
    def set_location(mut self, new_location: String):
        self.location = new_location

# Library L's spaceship_extensions.mojo

extension Spaceship:
    def fly_to(mut self, new_location: String):
        self.liftoff()
        self.set_location(new_location)
```

Options:

- Option A: Yes let’s allow them; user journeys 1, 2, 3, 4, 5 don’t seem to
  need any trait.
  - Pro: Doesn’t require an empty trait when the user wants to do this.
  - Con: Requires additional thinking for handling imports.
- Option B: No, let’s not allow them.
  - Pro: Doesn’t require require additional thinking for handling imports.
  - Con: Requires an empty trait when the user wants to do this.
- Option C: Conservatively start with option B, reevaluate later.

This affects decision 8 as well (”What if we import only an `extension` but not
its target `struct`?”), it means we always import either the struct or the trait
to get its extension.

## Decision 3: Where can non-conforming extensions be?

(Assuming we have them; see previous section)

Option A: We can add as many non-conforming extensions as we want, wherever we
want. They can be in the same file, or different files. If they’re in the same
file, they can be in any order; order shouldn’t matter.

Option B: Same restrictions as conforming extensions.

- Pro: It seems like the conservative option.
- Con: It makes case #5 a little harder, the user would have to introduce a
useless trait.
- Precedent: Swift had no restriction here.

Q: How do we handle conflicts between them, such as if multiple extensions
introduce the same method signature?

A: See next section!

## Decision 4: What can be in an extension?

These can be in an extension:

- Methods, including static methods and initializers.
- Aliases
- `requires` clauses

We won’t allow:

- Variables
- Additional parameter declarations
- Decorators on the extension itself, e.g. `@register_passable`. We can
  allowlist these case by case. (See also Decision 11 for what decorators we can
  add on the contained methods.)

## Aside: What’s an extension’s name?

This isn't part of the requirements, but it clarifies my mindset.

An extension is technically anonymous, though `import`s can refer to it by its
target struct’s name.

For example, in `library_b`:

```mojo
import library_a.Spaceship

extension Spaceship:
    ...
```

This extension doesn’t have a name, but if `main.mojo` imports “Spaceship” from
this library, they’ll import the extension. Like:

```mojo
import library_b.Spaceship  # <-- imports the extension

def main():
    ...
```

This mindset isn’t tangibly required yet, but it does help in a few subtle ways.

1: It makes this theoretical syntax make sense:

```mojo
# no import library_a.Spaceship

extension library_a.Spaceship:
    ...
```

This is out of scope, but seems like a reasonable ability to later enable.

2: In the implementation, multiple things can’t really have the same name,
they’ll get generated unique names.

3: Later on, we can introduce a “named extension” concept, like this
`library_b`:

```mojo
import library_a.Spaceship

extension Spaceship as MyCoolSpaceshipExtensions:
    ...
```

…and the user could import it explicitly like
`import library_b.MyCoolSpaceshipExtensions`.

This could theoretically help resolve conflicts, and might have other benefits
down the line.

There’s no super strong need for this now so it’s not in scope for the design of
this proposal, but let’s keep the door open for this option.

## Decision 5: Handling conflicts

Q: What if two extensions offer the same methods? Like:

```mojo
# Library L's spaceship_extensions.mojo

extension Spaceship:
    def fly_to(mut self, new_location: String):
        self.liftoff()
        self.set_location(new_location)

# Library X's spaceship_extensions.mojo

extension Spaceship:
    def fly_to(mut self, new_location: String):
        do_something_else()
```

A: This should be fine and allowed. It would show an error if the user tried to
call them though:

```mojo
from L import Spaceship
from X import Spaceship

def flamscrankle(mut ship: Spaceship):
    ship.fly_to("Corneria")  # Error: ambiguous call
```

## Decision 6: Are extensions automatically imported?

### Option A: Explicitly import extensions

(Not recommended, but it is the cleanest conceptually IMO)

Like structs, traits, etc., extensions must be imported to be used.

For example, this Library A:

```mojo
# Library A's spaceship.mojo

struct Spaceship:
    var location: String
    def liftoff(self):
        ...
    def set_location(mut self, new_location: String):
        self.location = new_location

# Library A's spaceship_extensions.mojo

extension Spaceship:
    def fly_to(mut self, new_location: String):
        self.liftoff()
        self.set_location(new_location)

# Library A's __init__.mojo

from .spaceship import Spaceship
from .spaceship_extensions import Spaceship
```

Notice how the library author explicitly re-exports both the struct and the
extension from their `__init__.mojo`.

The user can then `import` them, like in this `main.mojo`:

```mojo
from A import Spaceship

def main():
    var ship = Spaceship("Corneria")
    ship.fly_to("Korhal")
```

### Option B: Struct imports automatically import their extensions

(Reluctantly recommended)

Like the above, but the `from .spaceship_extensions import Spaceship` is
implicit.

```mojo
# Library A's spaceship.mojo

struct Spaceship:
    var location: String
    def liftoff(self):
        ...
    def set_location(mut self, new_location: String):
        self.location = new_location

# Library A's spaceship_extensions.mojo

extension Spaceship:
    def fly_to(mut self, new_location: String):
        self.liftoff()
        self.set_location(new_location)

# Library A's __init__.mojo

from .spaceship import Spaceship
# from .spaceship_extensions import Spaceship   <-- implicit
```

User’s `main.mojo` is the same either way:

```mojo
from A import Spaceship   # imports both

def main():
    var ship = Spaceship("Corneria")
    ship.fly_to("Korhal")
```

However, if the `extension` is in a different module, like this Library B:

```mojo
# Library B's spaceship_other_extensions.mojo

from A import Spaceship

extension Spaceship:
    def barrel_roll(mut self):
        self.roll(360)
        maniacal_laughter()
```

…then the user will have to explicitly import it, like:

```mojo
from A import Spaceship  # imports A's struct and A's extension
from B import Spaceship  # imports B's extension

def main():
    var ship = Spaceship("Corneria")
    ship.fly_to("Korhal")
```

In other words, when we import a struct, we **automatically import extensions
only from the struct’s own module.**

Drawbacks:

- There would be no such thing as a private extension, all extensions are by
  default exposed with the struct.
- This might be incompatible with user journey 2, the user wanted to make
  something explicitly importable to not bring in dependencies, but we’re now
  automatically importing it.

Still, I recommend this option.

## Decision 7: Need `import extension`?

Recall this Library B that had a lone extension, that extended a struct from
library A:

```mojo
# Library B's spaceship_other_extensions.mojo

from A import Spaceship

extension Spaceship:
    def barrel_roll(mut self):
        self.roll(360)
        maniacal_laughter()
```

and the user explicitly imported it:

```mojo
from A import Spaceship  # imports A's struct and A's extension
from B import Spaceship  # imports B's extension

def main():
    var ship = Spaceship("Corneria")
    ship.fly_to("Korhal")
```

Should the user have said:

`from B import Spaceship`

or should they say:

`from B import extension Spaceship`

I recommend the former, there doesn’t seem to be much need for the `extension`.

## Decision 8: What if we import only an `extension` but not its target `struct`?

(Decisions 1 and 2 affect this)

Same scenario, Library B that had a lone extension, that extended a struct from
library A:

```mojo
# Library B's spaceship_other_extensions.mojo

from A import Spaceship

extension Spaceship:
    def barrel_roll(mut self):
        self.roll(360)
        maniacal_laughter()
```

What happens if the user *only* imports the B extension? Like:

```mojo
# from A import Spaceship  # user didn't import either of these
from B import Spaceship  # imports B's extension

def main():
    var ship = Spaceship("Corneria")
    ship.barrel_roll()
```

Specifically, should there be an error on `var ship = Spaceship("Corneria")`
because they didn’t import the actual struct?

Option A: Error. They must import the original struct.

Option B (recommended): Automatically import the target struct. (Note: depending
on Decision 6, it might automatically import any extensions in the target
struct's module.)

It’s unnecessary to mention the struct as well, but it’s allowed:

```mojo
# User's main.mojo

import A.Spaceship  # unnecessary but allowed
import B.Spaceship

def do_things(ship: Spaceship):
    ship.fly_to("Corneria")
    ship.barrel_roll()
```

## Decision 9: Support importing multiple extensions?

We should be able to pull in as many extensions as we want. If we imagine
modules `C` and `D` with more `Spaceship` extensions, this should work:

```mojo
# User's main.mojo

# This automatically pulls in A's spaceship.mojo's struct, and A's extension.
import B.Spaceship
import C.Spaceship
import D.Spaceship

def do_things(ship: Spaceship):
    ship.fly_to("Corneria")
    ship.barrel_roll()
    ship.some_c_method()
    ship.some_d_method()
```

Recommended: yes let’s allow it. Don’t see much reason not to.

## Decision 10: Proposed Syntax

```mojo
extension List(Copyable) requires T: Copyable:
    def __init__(out self, *, copy: Self):
        ...
```

Highlights:

- The extension uses the same parameter declarations as the struct.
- `(Copyable)` means the extension conforms to `Copyable`.

Alternatives:

- `extend` vs `extension`
- Parameterizing the extension? `extend[T: AnyType] List[T]`
- Specializing the struct? `extend List[Int]`, or `extend List where T == Int`?
- Conforming to a trait? `extend List : Copyable`?

## Decision 11: What decorators can be on extensions' methods?

- Option A: All of them
- Option B: Allowlist some of them, starting with `@inline`, `@no_inline`,
  `@implicit`, `@staticmethod`.

Recommended: Option A. IMO, struct extensions should be decoupled from methods'
APIs and details.

## Trait Extensions

This is not in scope for the Struct Extensions project, but lets mention them
anyway.

We’ll one day want something like this:

```mojo
# In list_iterator_extensions.mojo

extension Iterator requires T: Copyable
  def to_list(self) -> List[T]:
      var result = List[T]()
      for x in self:
          result.append(x)
      return result
```

Trait extensions’ methods must have bodies; we cannot use an extension to add an
additional requirement to an existing trait.

## Notes

(These are just some of my thoughts, feel free to disregard)

Central mindset: An extension is a bundle of methods, but it really is just a
bunch of normal vanilla methods.

“These are just functions.” should be our starting point for all of this.

Anything we can do with extensions, we should also consider how we’d do it
without extensions.

For example, this:

```mojo
extension Spaceship:
    def fly_to(mut self, new_location: String):
        self.liftoff()
        self.set_location(new_location)

    def barrel_roll(mut self):
        self.spin(3)
```

Is basically just two functions that can add themselves to an existing scope,
like (pseudocode):

```mojo
def Spaceship.fly_to(mut self, new_location: String):
    self.liftoff()
    self.set_location(new_location)

def Spaceship.barrel_roll(mut self):
    self.spin(3)
```

So from that perspective, these are just functions, and should generally follow
normal function rules.

There are two other aspects to `extension` even in this mindset; `extensions`s
can:

- Declare a conformance between a struct and a trait
- Serve as a subject for importing (`import Spaceship` can import a Spaceship)

We should consider separate mechanisms for those. Separate composable mechanisms
is often better than a monolithic feature to solve a bunch of unrelated problems
at once (I’m looking at you, Java’s `extends` keyword).

This mindset has precedent in Vale, especially with its UFCS support. Even if we
don’t mimic what Vale did, it serves as a great conceptual framework IMO.
