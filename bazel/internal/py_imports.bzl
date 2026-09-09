"""Auto import path for max/python/max packages.

Separate from ``modular_py_library`` so ``modular_py_test`` can use it without a
load cycle (``modular_py_library`` loads ``modular_py_test``).
"""

_MAX_PYTHON_ROOT = "max/python/max/"
_IGNORED_PACKAGES = [
    "max/python/max/_core/internal/mlir_nanobind/tblgen",
]

def compute_py_imports(package_name, imports):
    """Return the single import path for a py_library/py_test target.

    For packages under ``max/python/max`` the path is computed automatically
    and passing ``imports`` explicitly is an error. Elsewhere the caller's
    ``imports`` (at most one entry) is returned unchanged.

    Args:
        package_name: The target's Bazel package, from native.package_name().
        imports: Caller-provided import paths; must be empty for packages
            under max/python/max, where the path is computed automatically.

    Returns:
        A single-element list with the computed import path, or the
        caller-provided imports unchanged.
    """
    if (package_name + "/").startswith(_MAX_PYTHON_ROOT) and package_name not in _IGNORED_PACKAGES:
        if len(imports) > 0:
            fail(
                "Do not pass 'imports' for packages under {}. ".format(_MAX_PYTHON_ROOT) +
                "The imports path is automatically computed.",
            )
        relative_path = package_name.removeprefix("max/python/")
        depth = len(relative_path.split("/"))
        return ["/".join([".."] * depth)]
    if len(imports) > 1:
        fail("Only a single import path is supported.")
    return imports
