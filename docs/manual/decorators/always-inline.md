---
title: '@always_inline'
description: Copies the body of a function directly into the body of the calling function.
codeTitle: true

---

You can add the `@always_inline` decorator on any function to make the Mojo
compiler "inline" the body of the function (copy it) directly into the body of
the calling function.

This eliminates potential performance costs associated with function calls
jumping to a new point in code. Normally, the compiler will do this
automatically where it can improve performance, but this decorator forces it to
do so. The downside is that it can increase the binary size by duplicating the
function at every call site.

For example:

```mojo
@always_inline
fn add(a: Int, b: Int) -> Int:
    return a + b

print(add(1, 2))
```

Because `add()` is decorated with `@always_inline`, Mojo compiles this program
without adding the `add()` function to the call stack, and it instead performs
the addition directly at the `print()` call site, as if it were written like
this:

```mojo
print(1 + 2)
```

## `@always_inline("nodebug")`

You can also use the decorator with the `"nodebug"` argument, which has the
same effect to inline the function, but without debug information. This means
that you can't step into the function when debugging.

This decorator is intended to be used on the lowest-level functions in a
library,   which may wrap primitive functions, MLIR operations, or inline
assembly. Marking these functions as "nodebug" prevents users from accidentally
stepping into low-level non-Mojo code when debugging.
