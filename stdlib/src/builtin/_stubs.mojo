# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


@register_passable("trivial")
struct __MLIRType[T: AnyRegType](Movable, Copyable):
    var value: T
