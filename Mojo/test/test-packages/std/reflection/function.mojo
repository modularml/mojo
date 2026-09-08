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
"""Provides compile-time reflection over function values."""


comptime _TargetType = __mlir_type.`!kgen.target`


@always_inline("nodebug")
def _current_target() -> _TargetType:
    return __mlir_attr.`#kgen.param.expr<current_target> : !kgen.target`


comptime reflect_fn[func_type: AnyType, //, func: func_type] = ReflectedFn[func]
"""A compile-time alias for the reflection handle type of function `func`."""


struct ReflectedFn[func_type: AnyType, //, func: func_type]:
    """A compile-time reflection handle type for a function value."""

    @staticmethod
    def display_name() -> StaticString:
        var res = __mlir_attr[
            `#kgen.get_source_name<`, Self.func, `> : !kgen.string`
        ]
        return StaticString(res)

    @staticmethod
    def linkage_name[
        *, target: _TargetType = _current_target()
    ]() -> StaticString:
        var res = __mlir_attr[
            `#kgen.get_linkage_name<`,
            target,
            `,`,
            Self.func,
            `> : !kgen.string`,
        ]
        return StaticString(res)


def get_linkage_name[
    func_type: AnyType,
    //,
    func: func_type,
    *,
    target: _TargetType = _current_target(),
]() -> StaticString:
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
    var res = __mlir_attr[`#kgen.get_source_name<`, func, `> : !kgen.string`]
    return StaticString(res)
