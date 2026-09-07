# Upgrading the `@value` decorator

April 2025. Status: Implemented in Mojo 25.4

The `@value` decorator on struct declarations were added really early in the
evolution of Mojo as a way to bring up the language and get the lifecycle
methods ergonomic enough to use. That said, it has a number of problems, and we
have traits now, which provide a better way to solve this.

This whitepaper explores the problem with them and recommends a solution that
involves traits (which didn’t exist when these were added).

Here’s a simple example of this decorator:

```mojo
@value
struct Simple:
  var x: Int
  var y: String
```

## What it does today

When the `@value` decorator is applied to a struct, it has the following
effects:

- If all the declared `var` members in the struct are copyable, it adds a
  `__copyinit__` (unless it was already explicitly declared by the user) and
  conformance to `Copyable`. It also adds a `copy()` member and conformance to
  `ExplicitlyCopyable` as well

- Similarly, if the members are all movable, it adds a `__moveinit__` when
  missing.

- Finally, unless the user declared it already, it adds a “fieldwise”
  `__init__` method that allows one to construct the struct by specifying all of
  the fields.

On the example above, it is as if you wrote out this:

```mojo
struct Simple(Copyable, Movable, ExplicitlyCopyable):
  var x: Int
  var y: String

  def __init__(x: Int, owned y: String):
     self.x = x
     self.y = y^

  def __copyinit__(out self, existing: Self):
     self.x = existing.x
     self.y = existing.y

  def copy(out self, existing: Self):
     self.x = existing.x
     self.y = existing.y

  def __moveinit__(out self, owned existing: Self):
     self.x = existing.x
     self.y = existing.y^
```

This saves a lot of boilerplate!

## Problems with the `@value` decorator

This decorator is very handy and has served us well, but has some problems,
mostly that it is “all or nothing”:

- Many types don’t want a fieldwise initializer, because this leaks
  implementation details, but they do want synthesized move/copy members. You
  face a choice of a) not using `@value` and write them explicitly or b) use
  `@value` and get stuff they don’t want.

- It is difficult to understand whether a type is copyable or movable by looking
  at its declaration, because the traits aren’t explicitly declared.

- It isn’t a great thing to extend, because adding a feature to it will affect
  everything.

You might say that it isn’t very “Modular”.

## Proposed solution

The solution I’d propose is to move the synthesis of these members to trait
specifications and introduce a new `@fieldwise_init` decorator (it isn’t
useful or expressible as a trait). The example above would be expressed like
this (replacing the `@value` decorator) to get all the code describe above:

```mojo
@fieldwise_init
struct Simple(Copyable, Movable, ExplicitlyCopyable):
  var x: Int
  var y: String
```

This requires a couple steps:

1. Introduce a new `@fieldwise_init` decorator, that only introduces the
   fieldwise constructor.

2. Change trait conformance checking for `Copyable` and `Movable` and
   `ExplicitlyCopyable` to synthesize the member when missing.

3. Move the standard library over to use the new features, and deprecate the
   value decorator.

4. Eventually remove the value decorator.

## Non-goal - for this proposal: `#derive`

Note that #2 could be the start of a much more general language feature (e.g.
like the `#derive` feature in Rust), and is potentially related to the “default
member implementations in traits” feature.

I recommend we just move the existing logic hard coded into the compiler for
`@value` to be triggered in new ways, making the scope of this feature quite
small and incremental. We can generalize it over time.

## FAQ: How does `@value` related to `@dataclass`?

One potential FAQ: why don’t we just use `@dataclass` for this? In Python,
`@dataclass` is widely used for structs that are “bags of fields”, one example
is:

```python
@dataclass
class InventoryItem:
    name: str
    unit_price: float
    quantity_on_hand: int = 0

    def total_cost(self) -> float:
        return self.unit_price * self.quantity_on_hand
```

For reference, the same thing in Mojo would look like this in our proposal:

```mojo
@fieldwise_init
struct InventoryItem(Copyable, Movable, ExplicitlyCopyable):
    var name: String
    var unit_price: Float64
    # unrelated to this proposal, we want to support default field values someday
    var quantity_on_hand: Int = 0

    def total_cost(self) -> Float:
       return self.unit_price * self.quantity_on_hand
```

The [behavior of the `@dataclass`
decorator](https://docs.python.org/3/library/dataclasses.html) is to synthesize
a fieldwise constructor, methods like `__eq__` and `__repr__` and a bunch of
other things. Python classes have reference semantics, so they don’t have
`__copyinit__` or `__moveinit__` like Mojo.

This is definitely in the same neighborhood as the `@value` decorator (which
this proposal removes), but is quite different from this proposal:

- First, the name `@dataclass` is incorrect for a `struct`! Mojo will
  eventually support classes, and it may make sense to support `@dataclass` on
  them, but classes have reference semantics and address-based identity, which
  structs don’t have. Using the same name for something subtly different would
  be confusing: we’d get the benefit of familiarity, but the trap of not being
  the same behavior.

- Second, the point of this proposal is to get rid of monolithic things like
  `@value` and `@dataclass` and replace them with more modular things that
  better reflect the behavior: trait conformance. Python doesn’t have traits,
  so this solution doesn’t work for it.

There are a fairly common set of traits that “record like” types need to
support, and it can be burdensome to have to write: `(Copyable, Movable,
ExplicitlyCopyable)` on each type, particularly if/when we extend Mojo to
support things like synthesizing `Representable` or `Comparable` etc. Fear not,
this can be solved in the library by adding something like:

```mojo
# Theoretical thing that could be done some day: not part of this proposal.
alias RecordLike = Copyable & Movable & ExplicitlyCopyable & Representable
...

@fieldwise_init
struct InventoryItem(RecordLike):
```

Which uses existing language features (trait composition) to solve this problem.
