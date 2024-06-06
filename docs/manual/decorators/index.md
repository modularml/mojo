---
title: Mojo decorators
sidebar_label: Decorators
sidebar_position: 1
description: A reference of Mojo's built-in decorators
hide_table_of_contents: true
listing:
  - id: docs
    contents:
      - always-inline.ipynb
      - copy-capture.ipynb
      - nonmaterializable.ipynb
      - parameter.ipynb
      - register-passable.ipynb
      - staticmethod.ipynb
      - value.ipynb
    type: grid
    page-size: 99
---

A Mojo decorator is a [higher-order
function](https://en.wikipedia.org/wiki/Higher-order_function) that modifies or
extends the behavior of a struct, a function, or some other code. Instead of
actually calling the higher-order function, you simply add the decorator (such
as the `@value` decorator) above your code (such as a struct). The Mojo
compiler then uses the decorator function to modify your code at compile time.

:::note No custom decorators

The creation of custom decorators is not yet supported. The available ones are
built directly into the compiler.

:::

The following pages describe each built-in decorator with examples.

:::{#docs}
:::
