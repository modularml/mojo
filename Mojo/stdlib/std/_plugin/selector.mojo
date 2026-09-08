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
"""Shared compile-time plugin registration and selection utilities."""

from std.sys.info import _TargetType, _current_target


trait Plugin:
    """Defines the identity required by a compile-time plugin registry."""

    comptime name: __mlir_type.`!kgen.string`
    """The stable identifier matched against a target's plugin field."""


struct PluginSelector[
    Trait: type_of(Plugin),
    //,
    *PluginTypes: Trait,
]:
    comptime index_for_target[
        target: _TargetType
    ]: __mlir_type.index = Self._find[target, __mlir_attr.`0 : index`]()

    comptime current: Self.Trait = Self.PluginTypes._get_type_at_index[
        Self.index_for_target[_current_target()]
    ]

    comptime for_target[target: _TargetType]: Self.Trait = (
        Self.PluginTypes._get_type_at_index[Self.index_for_target[target]]
    )

    # Avoid `TypeList.length` to keep selection in raw parameter space so it is
    # safe during stdlib bootstrap, before `Int` and `SIMDLength` are available.
    comptime _length = __mlir_attr[
        `#kgen.param_list.size<:`,
        Self.PluginTypes._mlir_type,
        ` `,
        +Self.PluginTypes.values,
        `> : index`,
    ]

    @staticmethod
    def _matches[target: _TargetType, idx: __mlir_type.index]() -> Bool:
        """Returns whether plugin `idx` matches `target`."""
        return __mlir_attr[
            `#kgen.param.identical<`,
            __mlir_attr[
                `#kgen.param.expr<target_get_field,`,
                target,
                `, "stdlib_plugin" : !kgen.string`,
                `> : !kgen.string`,
            ],
            `,`,
            Self.PluginTypes._get_type_at_index[idx].name,
            `> : !kgen.scalar<bool>`,
        ]

    @staticmethod
    def _find[
        target: _TargetType, idx: __mlir_type.index
    ]() -> __mlir_type.index:
        """Finds the plugin matching `target` through parameter recursion."""
        comptime if not _index_lt[idx, Self._length]():
            __mlir_op.`llvm.intr.trap`()
            return idx
        elif Self._matches[target, idx]():
            return idx
        else:
            return Self._find[
                target,
                __mlir_attr[
                    `#kgen.param.expr<add,`,
                    idx,
                    `, 1 : index> : index`,
                ],
            ]()


def _index_lt[lhs: __mlir_type.index, rhs: __mlir_type.index]() -> Bool:
    """Returns whether raw parameter index `lhs` is less than `rhs`."""
    return __mlir_op.`index.cmp`[pred=__mlir_attr.`#index.cmp_predicate<ult>`](
        lhs, rhs
    )
