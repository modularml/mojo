# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements functions and variables for interacting with execution and system
environment.

You can import these APIs from the `sys` package. For example:

```mojo
from sys import argv
```
"""

from sys import external_call

from memory.unsafe import Pointer


# TODO: When we have global variables, this should be a global list.
fn argv() -> VariadicList[StringRef]:
    """The list of command line arguments.

    Returns:
        The list of command line arguments provided when mojo was invoked.
    """
    var result = VariadicList[StringRef]("")
    external_call["KGEN_CompilerRT_GetArgV", NoneType](
        Pointer[VariadicList[StringRef]].address_of(result)
    )
    return result
