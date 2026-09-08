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
"""Compile-time reflection over function values.

`reflect_fn[func]` is a `comptime` alias for the `ReflectedFn[func]` handle
type, exposing function introspection through static methods. This is the
function-side counterpart to `reflect[T]` / `Reflected[T]` for types.

```mojo
from std.reflection import reflect_fn

def my_func(x: Int) -> Int:
    return x + 1

def main():
    print(reflect_fn[my_func].display_name())   # "my_func"
    print(reflect_fn[my_func].linkage_name())   # mangled symbol name
```
"""

from std.sys.info import _TargetType, _current_target


# FIXME(MOCO-3929): Merge with `reflect` once function types are better
# supported in the parameter system.
comptime reflect_fn[func_type: AnyType, //, func: func_type] = ReflectedFn[func]
"""A compile-time alias for the reflection handle type of function `func`.

Resolves to `ReflectedFn[func]`, a type whose static methods expose function
introspection. Use it as `reflect_fn[my_func].display_name()` rather than
constructing an instance.

Parameters:
    func_type: The function's type (inferred).
    func: The function value to introspect.

Example:
    ```mojo
    from std.reflection import reflect_fn

    def add(a: Int, b: Int) -> Int:
        return a + b

    def main():
        print(reflect_fn[add].display_name())  # "add"
    ```
"""


struct ReflectedFn[func_type: AnyType, //, func: func_type]:
    """A compile-time reflection handle type for a function value.

    All methods are `@staticmethod` — the handle has no runtime state, only
    compile-time parameters. Spell access as `reflect_fn[func].display_name()`
    (preferred) or `ReflectedFn[func].display_name()`.

    Parameters:
        func_type: The function's type. Inferred from `func`.
        func: The function value being introspected.

    Example:
        ```mojo
        from std.reflection import reflect_fn

        def my_func(x: Int) -> Int:
            return x

        def main():
            print(reflect_fn[my_func].display_name())  # "my_func"
        ```
    """

    @staticmethod
    def display_name() -> StaticString:
        """Returns the function's name as declared in source.

        The returned name does not include any information about the function's
        parameters, arguments, or return type.

        Returns:
            The function's source-level name.

        Example:
            ```mojo
            from std.reflection import reflect_fn

            def my_func(): pass
            print(reflect_fn[my_func].display_name())  # "my_func"
            ```
        """
        var res = __mlir_attr[
            `#kgen.get_source_name<`, Self.func, `> : !kgen.string`
        ]
        return StaticString(res)

    @staticmethod
    def linkage_name[
        *, target: _TargetType = _current_target()
    ]() -> StaticString:
        """Returns the function's mangled linkage / symbol name.

        Parameters:
            target: The compilation target (defaults to the current target).

        Returns:
            The symbol name as it appears in the linker.
        """
        var res = __mlir_attr[
            `#kgen.get_linkage_name<`,
            target,
            `,`,
            Self.func,
            `> : !kgen.string`,
        ]
        return StaticString(res)

    # TODO(reflection): Extend with `parameter_count`, `parameter_names`,
    # `parameter_types`, `return_type`, `is_raising`, `is_async`,
    # `is_exported`, `parent_qualified_name` once KGEN gains the
    # corresponding `#kgen.get_function_*` attribute primitives.
