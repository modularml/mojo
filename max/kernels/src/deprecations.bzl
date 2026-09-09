"""Deprecated stdlib/kernel Pointer APIs that `max/kernels` and `Kernels`
intentionally still call.

Splat `IGNORED_POINTER_DEPRECATIONS` into a `mojo_library`'s
`ignore_deprecated` param, or `IGNORED_POINTER_DEPRECATIONS_COPTS` into a
`mojo_test`/`mojo_binary`/`mojo_filecheck_test`'s `copts`, to suppress just
these declarations' deprecation warnings for that target, while leaving all
other deprecation warnings visible.
"""

IGNORED_POINTER_DEPRECATIONS = [
    "UnsafePointer",
    "alloc",
    "Pointer.__add__",
    "Pointer.__sub__",
    "Pointer.__iadd__",
    "Pointer.__isub__",
    "Pointer.__getitem__",
    "Pointer.load",
    "Pointer.store",
    "Pointer.free",
    "Pointer.bitcast",
    "Pointer.address_space_cast",
]

IGNORED_POINTER_DEPRECATIONS_COPTS = [
    "--ignore-deprecated=" + decl_name
    for decl_name in IGNORED_POINTER_DEPRECATIONS
]
