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
"""Implements functions for retrieving compile-time defines.

You can use these functions to set parameter values or runtime constants based on
name-value pairs defined on the command line. For example:

```mojo
  from std.sys import is_defined

  comptime float_type = DType.float32 if is_defined["FLOAT32"]() else DType.float64

  # Use `float_type` as a constant.
```

And on the command line:

```
  mojo -D FLOAT32 main.mojo
```

For more information, see the [Mojo build docs](https://mojolang.org/docs/cli/build/#d-keyvalue).
The `mojo run` command also supports the `-D` option.


You can import these APIs from the `sys` package. For example:

```mojo
from std.sys import is_defined
```
"""

from std.collections.string.string_span import _get_kgen_string


def is_defined[name: StaticString]() -> Bool:
    """Return true if the named value is defined.

    Parameters:
        name: The name to test.

    Returns:
        True if the name is defined.
    """
    return __mlir_attr[
        `#kgen.param.expr<get_env, `,
        _get_kgen_string[name](),
        `> : i1`,
    ]


def _is_bool_like[val: StaticString]() -> Bool:
    return (
        val == "0"
        or val == "1"
        or val == "true"
        or val == "True"
        or val == "TRUE"
        or val == "false"
        or val == "False"
        or val == "FALSE"
        or val == "on"
        or val == "On"
        or val == "ON"
        or val == "off"
        or val == "Off"
        or val == "OFF"
    )


def get_defined_bool[name: StaticString, default: Bool = False]() -> Bool:
    """Get a boolean-valued define. If the name is not defined, return a default
    value instead. If the name is defined this will return `True` if the value
    is ("1", "true", "True", "TRUE", "on", "On", "ON") otherwise it will return
    `False`.

    Parameters:
        name: The name of the define.
        default: The default value to use.

    Returns:
        A boolean parameter value.
    """
    comptime if is_defined[name]():
        comptime val = get_defined_string[name]()
        return (
            val == "1"
            or val == "true"
            or val == "True"
            or val == "TRUE"
            or val == "on"
            or val == "On"
            or val == "ON"
        )
    else:
        return default


def get_defined_int[name: StaticString]() -> Int:
    """Try to get an integer-valued define. Compilation fails if the
    name is not defined.

    Parameters:
        name: The name of the define.

    Returns:
        An integer parameter value.
    """
    return Int(
        mlir_value=__mlir_attr[
            `#kgen.param.expr<get_env, `,
            _get_kgen_string[name](),
            `> : index`,
        ]
    )


def get_defined_int[name: StaticString, default: Int]() -> Int:
    """Try to get an integer-valued define. If the name is not defined,
    return a default value instead.

    Parameters:
        name: The name of the define.
        default: The default value to use.

    Returns:
        An integer parameter value.

    Example:
    ```mojo
    from std.sys.defines import get_defined_int

    def main() raises:
        comptime number = get_defined_int[
            "favorite_number",
            1 # Default value
        ]()
        parametrized[number]()

    def parametrized[num: Int]():
        print(num)
    ```

    If the program is `app.mojo`:
    - `mojo run -D favorite_number=2 app.mojo`
    - `mojo run -D app.mojo`

    Note: useful for parameterizing SIMD vector sizes.
    """

    comptime if is_defined[name]():
        return get_defined_int[name]()
    else:
        return default


def get_defined_string[name: StaticString]() -> StaticString:
    """Try to get a string-valued define. Compilation fails if the
    name is not defined.

    Parameters:
        name: The name of the define.

    Returns:
        A string parameter value.
    """
    var res = __mlir_attr[
        `#kgen.param.expr<get_env, `,
        _get_kgen_string[name](),
        `> : !kgen.string`,
    ]
    return StaticString(res)


def get_defined_string[
    name: StaticString, default: StaticString
]() -> StaticString:
    """Try to get a string-valued define. If the name is not defined,
    return a default value instead.

    Parameters:
        name: The name of the define.
        default: The default value to use.

    Returns:
        A string parameter value.
    """

    comptime if is_defined[name]():
        return get_defined_string[name]()
    else:
        return default


def get_defined_dtype[name: StaticString, default: DType]() -> DType:
    """Try to get a DType-valued define. If the name is not defined,
    return a default value instead.

    Parameters:
        name: The name of the define.
        default: The default value to use.

    Returns:
        A DType parameter value.
    """

    comptime if is_defined[name]():
        comptime parsed = DType._from_str(get_defined_string[name]())
        comptime assert parsed, "invalid DType define for '" + name + "'"
        return parsed.value()
    else:
        return default


struct MojoVersion(ImplicitlyCopyable, TrivialRegisterPassable):
    """Represents the Mojo version as major, minor, and patch numbers."""

    var major: Int
    """The major version number."""
    var minor: Int
    """The minor version number."""
    var patch: Int
    """The patch version number."""

    @always_inline("nodebug")
    def __init__(out self):
        """Initializes the version by reading it from the compiler at compile time.
        """
        self.major = Int(mlir_value=__mlir_op.`lit.mojo.version.major`())
        self.minor = Int(mlir_value=__mlir_op.`lit.mojo.version.minor`())
        self.patch = Int(mlir_value=__mlir_op.`lit.mojo.version.patch`())


comptime MOJO_VERSION = MojoVersion()
"""The version of the Mojo language used for the current compilation, available at compile time."""
