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
"""Provides the unified `reflect[T]` / `Reflected[T]` reflection API."""


comptime _field_types_of[T: AnyType] = TypeList[
    __mlir_attr[
        `#kgen.struct_field_types<`, T, `> : !kgen.param_list<`, AnyType, `>`
    ]
]


comptime reflect[T: AnyType] = Reflected[T]
"""A compile-time alias for the reflection handle type of `T`."""


struct Reflected[T: AnyType]:
    """A compile-time reflection handle type for a Mojo type."""

    @staticmethod
    @always_inline("builtin")
    def is_struct() -> Bool:
        return __mlir_attr[`#kgen.is_struct_type<`, Self.T, `> : i1`]

    @staticmethod
    def name[*, qualified_builtins: Bool = False]() -> StaticString:
        return StaticString(
            __mlir_attr[
                `#kgen.get_type_name<`,
                Self.T,
                `, `,
                qualified_builtins._mlir_value,
                `> : !kgen.string`,
            ]
        )

    @staticmethod
    def base_name() -> StaticString:
        return StaticString(
            __mlir_attr[
                `#kgen.get_base_type_name<`,
                Self.T,
                `> : !kgen.string`,
            ]
        )

    @staticmethod
    @always_inline("builtin")
    def field_count() -> Int:
        return _field_types_of[Self.T]().length

    @staticmethod
    def field_types() -> _field_types_of[Self.T]:
        return {}

    @staticmethod
    def field_index[name: StringLiteral]() -> Int:
        comptime str_value = name.value
        return Int(
            mlir_value=__mlir_attr[
                `#kgen.struct_field_index_by_name<`,
                Self.T,
                `, `,
                str_value,
                `> : index`,
            ]
        )

    comptime field[name: StringLiteral] = Reflected[
        __mlir_attr[
            `#kgen.struct_field_type_by_name<`,
            Self.T,
            `, `,
            name.value,
            `> : `,
            AnyType,
        ]
    ]

    comptime field_at[idx: Int] = Reflected[_field_types_of[Self.T]()[idx]]

    @staticmethod
    @always_inline("nodebug")
    def field_ref[
        idx: Int
    ](ref s: Self.T) -> ref[s] _field_types_of[Self.T]()[idx]:
        return __get_litref_as_mvalue(
            __mlir_op.`lit.ref.struct.ger`[
                index=idx.__mlir_index__(),
                _type=__mlir_type[
                    `!lit.ref<`,
                    _field_types_of[Self.T]()[idx],
                    `, `,
                    origin_of(s)._mlir_origin,
                    `>`,
                ],
            ](__get_mvalue_as_litref(s))
        )
