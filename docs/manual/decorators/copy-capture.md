---
title: '@__copy_capture'
description: Captures register-passable typed values by copy.
codeTitle: true

---

You can add the `__copy_capture` decorator on a parametric closure to capture
register-passable values by copy. This decorator causes a nested function to
copy the value of the indicated variable into the closure object at the point
of formation instead of capturing that variable by reference. This allows the
closure to be passed as an escaping function, without lifetime concerns.

```mojo
  fn foo(x: Int):
      var z = x

      @__copy_capture(z)
      @parameter
      fn formatter() -> Int:
          return z
      z = 2
      print(formatter())

  fn main():
      foo(5)
```
