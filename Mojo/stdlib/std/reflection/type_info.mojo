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
"""Provides function name introspection utilities.

For type-name introspection, use the static methods on `Reflected[T]`
(obtained via `reflect[T]`):

- `reflect[T].name()` - returns the name of a type.
- `reflect[T].base_name()` - returns the unqualified name of a type's base
  type.

This module exposes the function-side counterparts:

- `get_function_name[func]()` - returns the source name of a function.
- `get_linkage_name[func]()` - returns the symbol/linkage name of a function.

Example:

```mojo
from std.reflection import reflect, get_function_name

struct Point:
    var x: Int
    var y: Float64

def my_function():
    pass

def main():
    print(reflect[Point].name())             # "Point"
    print(get_function_name[my_function]())  # "my_function"
```
"""

from std.sys.info import _current_target, _TargetType
from std.collections.string.string_span import get_static_string

from .reflect import reflect


def get_linkage_name[
    func_type: AnyType,
    //,
    func: func_type,
    *,
    target: _TargetType = _current_target(),
]() -> StaticString:
    """Returns `func`'s symbol name.

    Parameters:
        func_type: Type of func.
        func: A mojo function.
        target: The compilation target, defaults to the current target.

    Returns:
        Symbol name.
    """
    var res = __mlir_attr[
        `#kgen.get_linkage_name<`,
        target,
        `,`,
        func,
        `> : !kgen.string`,
    ]
    return StaticString(res)


def get_function_name[
    func_type: AnyType, //, func: func_type
]() -> StaticString:
    """Returns `func`'s name as declared in the source code.

    The returned name does not include any information about the function's
    parameters, arguments, or return type, just the name as declared in the
    source code.

    Parameters:
        func_type: Type of func.
        func: A mojo function.

    Returns:
        The function's name as declared in the source code.
    """
    var res = __mlir_attr[`#kgen.get_source_name<`, func, `> : !kgen.string`]
    return StaticString(res)


# TODO: This currently does not strip the module name from the inner type name.
# For example, Generic[Foo] should return "Generic[Foo]" but currently returns
# "Generic[module_name.Foo]".
def _unqualified_type_name[type: AnyType]() -> StaticString:
    comptime name = reflect[type].name()
    comptime parameter_list_start = name.find("[")
    comptime if parameter_list_start == -1:
        # HACK: Split is evaluated twice because `List[StringSlice]` cannot
        # be materialized to runtime from a comptime context.
        comptime n = len(name.split("."))
        return name.split(".")[n - 1]
    else:
        comptime split = name[byte=:parameter_list_start].split(".")
        comptime base_name = split[len(split) - 1]
        comptime parameters = name[byte=parameter_list_start:]
        return get_static_string[base_name, parameters]()
