# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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
"""Provides functions for working with environment variables.

You can import these APIs from the `os` package. For example:

```mojo
from std.os import setenv
```
"""

from std.ffi import c_int, external_call
from std.sys import CompilationTarget


def setenv(var name: String, var value: String, overwrite: Bool = True) -> Bool:
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
    var status = external_call["setenv", Int32](
        name.as_c_string_slice(),
        value.as_c_string_slice(),
        Int32(1 if overwrite else 0),
    )
    return status == 0


def unsetenv(var name: String) -> Bool:
    """Unsets an environment variable.

    Args:
        name: The name of the environment variable.

    Returns:
        True if unsetting the variable succeeded. Otherwise, False is returned.
    """
    return external_call["unsetenv", c_int](name.as_c_string_slice()) == 0


def getenv(var name: String, default: String = "") -> String:
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
    var ptr = external_call[
        "getenv", OptionalPointer[UInt8, ImmUntrackedOrigin]
    ](name.as_c_string_slice())
    if not ptr:
        return default
    return String(unsafe_from_utf8_ptr=ptr.value())
