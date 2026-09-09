# Closures

> [!NOTE]
> This arcana describes _unified_ closures, not the old closure implementation (
> sometimes referred to as capturing, escaping, and parameter closures).

## Anatomy

Consider a closure defined like this:

```mojo
def main():
  var y = 1
  def closure(x: Int) unified {var y} -> Int:
    return x + y

  print(closure(1)) # 2
```

The implementation of this closure consists of:

- a trait (the **closure trait**), `def(x: Int) -> Int`, that captures the
  interface of all closures with this signature. In particular, this contains
  the signature of the `__call__` method used by this closure. This is shared
  across all closures with the same signature.
- a struct decl (the **struct wrapper**) that defines the shape of a specific
  closure definition's captures. This can conform to one or more closure trait
  (see the section on rebinds below), as well as other traits depending on the
  captures.
- a struct generator that acts as the compile-time vtable for both the wrapper
  and the closure op (note that struct generators are not unique to closures).
- a `lit.closure.init` op that represents the instantiation of the struct
  wrapper, which holds the values captured by that particular closure instance.
- a variable declaration that allows referencing the created closure instance.

## Conformances

Closures (specifically, their struct wrappers) conform to several traits
automatically:

- `AnyType`
- `Deinitable`
- `Movable`

Additionally, they conform to several other types conditionally:

- `Copyable` and `ImplicitlyCopyable` (together, if the closure captures are
  compatible with the operation)
- `DevicePassable` (if all the closure's captures are trivially
  register-passable, and the closure function is marked with the
  `register_passable` effect)

### Rebinds

Closure wrappers are rebound to other closure traits in some scenarios. This
involves appending a witness table with a conformance entry to the other trait's
`__call__` method.
