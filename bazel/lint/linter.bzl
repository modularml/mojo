"""Wrapper for defining a lint target"""

load("@rules_multirun//:defs.bzl", "command")

def linter(base_name, binary, has_fix = True, has_fast = True):
    """Defines a lint target.

    Args:
        base_name: Base target name
        binary: The binary to run
        has_fix: Whether the linter provides a "fix" mode
        has_fast: Whether the linter provides a "fast" mode
        """
    if not has_fast and not has_fix:
        fail("Don't define a linter() macro for a linter with only one configuration")

    check_opts = [True, False] if has_fix else [True]
    fast_opts = [True, False] if has_fast else [False]
    for is_check in check_opts:
        for is_fast in fast_opts:
            command(
                name = base_name + (".check" if is_check else ".fix") + ("" if is_fast else "-all"),
                command = binary,
                environment = {
                    "CHECK": str(is_check),
                    "FAST": str(is_fast),
                },
                tags = ["maybe-unused", "manual"],
            )
