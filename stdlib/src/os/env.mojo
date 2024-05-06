# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Implements basic routines for working with the OS.

You can import these APIs from the `os` package. For example:

```mojo
from os import setenv
```
"""

from sys import external_call, os_is_linux, os_is_macos

from memory import DTypePointer

from utils import StringRef


fn setenv(name: String, value: String, overwrite: Bool = True) -> Bool:
    """Changes or adds an environment variable.

    Constraints:
      The function only works on macOS or Linux and returns False otherwise.

    Args:
      name: The name of the environment variable.
      value: The value of the environment variable.
      overwrite: If an environment variable with the given name already exists,
        its value is not changed unless `overwrite` is True.

    Returns:
      False if the name is empty or contains an `=` character. In any other
      case, True is returned.
    """
    alias os_is_supported = os_is_linux() or os_is_macos()
    if not os_is_supported:
        return False

    var status = external_call["setenv", Int32](
        name.unsafe_ptr(), value.unsafe_ptr(), Int32(1 if overwrite else 0)
    )
    return status == 0


fn getenv(name: String, default: String = "") -> String:
    """Returns the value of the given environment variable.

    Constraints:
      The function only works on macOS or Linux and returns an empty string
      otherwise.

    Args:
      name: The name of the environment variable.
      default: The default value to return if the environment variable
        doesn't exist.

    Returns:
      The value of the environment variable.
    """
    alias os_is_supported = os_is_linux() or os_is_macos()

    if not os_is_supported:
        return default

    var ptr = external_call["getenv", DTypePointer[DType.uint8]](
        name.unsafe_ptr()
    )
    if not ptr:
        return default
    return String(StringRef(ptr))
