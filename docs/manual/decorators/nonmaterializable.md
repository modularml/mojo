---
title: '@nonmaterializable'
description: Declares that a type should exist only in the parameter domain.
codeTitle: true

---

You can add the `@nonmaterializable` decorator on a struct to declare that the
type can exist only in the parameter domain (it can be used for metaprogramming
only, and not as a runtime type). And, if an instance of this type does
transition into the runtime domain, this decorator declares what type it
becomes there.

To use it, declare your type with `@nonmaterializable(TargetType)`, where
`TargetType` is the type that the object should convert to if it becomes a
runtime value (you must declare the `TargetType`). For example, if a struct is
marked as `@nonmaterializable(Foo)`, then anywhere that it goes from a
parameter value to a runtime value, it automatically converts into the `Foo`
type.

For example, the following `NmStruct` type can be used in the parameter domain,
but the `converted_to_has_bool` instance of it is converted to `HasBool` when it's
materialized as a runtime value:

```mojo
@value
@register_passable("trivial")
struct HasBool:
    var x: Bool

    fn __init__(out self, x: Bool):
        self.x = x

    @always_inline("nodebug")
    fn __init__(out self, nms: NmStruct):
        self.x = True if (nms.x == 77) else False

@value
@nonmaterializable(HasBool)
@register_passable("trivial")
struct NmStruct:
    var x: Int

    @always_inline("nodebug")
    fn __add__(self, rhs: Self) -> Self:
        return NmStruct(self.x + rhs.x)

alias still_nm_struct = NmStruct(1) + NmStruct(2)
# When materializing to a run-time variable, it is automatically converted,
# even without a type annotation.
var converted_to_has_bool = still_nm_struct
```

:::note

A non-materializable struct must have all of its methods annotated
as `@always_inline`, and it must be computable in the parameter domain.

:::
