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
# RUN: %parse-mojo-isolated %s | FileCheck %s

# ===----------------------------------------------------------------------=== #
# Helpers
# ===----------------------------------------------------------------------=== #

comptime struct_field_index_by_name[
    T: AnyType,
    name: StringLiteral,
]: Int = Int(
    mlir_value=__mlir_attr[
        `#kgen.struct_field_index_by_name<`,
        T,
        `, `,
        name.value,
        `> : index`,
    ]
)

comptime struct_field_type_by_name[
    StructT: AnyType,
    name: StringLiteral,
] = __mlir_attr[
    `#kgen.struct_field_type_by_name<`,
    StructT,
    `, `,
    name.value,
    `> : `,
    AnyType,
]

comptime struct_field_types[
    T: AnyType,
] = TypeList[
    __mlir_attr[
        `#kgen.struct_field_types<`, T, `> : !kgen.param_list<`, AnyType, `>`
    ]
]

comptime struct_field_names_raw[
    T: AnyType,
] = ParameterList[
    __mlir_attr[
        `#kgen.struct_field_names<`, T, `> : !kgen.param_list<!kgen.string>`
    ]
]

# ===----------------------------------------------------------------------=== #
# Parse-Time Folding
# ===----------------------------------------------------------------------=== #


trait HasInt:
    def __init__(out self):
        ...

    def get_int(self) -> Int:
        ...


@fieldwise_init
struct MyStruct(HasInt, ImplicitlyCopyable):
    def get_int(self) -> Int:
        return 42


@fieldwise_init
struct MyParam[x: Int](HasInt, ImplicitlyCopyable):
    def get_int(self) -> Int:
        return Self.x


# CHECK-LABEL: lit.struct.decl @MyWrapper
@fieldwise_init
struct MyWrapper[x: Int](Movable where False):
    # CHECK: lit.struct.field nested : !MyStruct
    var nested: MyStruct
    # CHECK: lit.struct.field nested_param : !lit.struct<#MyParam <:!Int x>>
    var nested_param: MyParam[Self.x]


# CHECK-LABEL: lit.fn @"main()"
def main():
    # Test struct_field_types folding
    # CHECK: lit.alias.decl *"fieldType0`{{[0-9]*}}": !AnyType = <!MyStruct>
    comptime fieldType0 = struct_field_types[MyWrapper[37]]()[0]
    # CHECK: lit.alias.decl *"fieldType1`{{[0-9]*}}": !AnyType = <@struct_type_reflection_attrs::@MyParam<:!Int {:scalar<index> 37}>>
    comptime fieldType1 = struct_field_types[MyWrapper[37]]()[1]
    # CHECK: lit.alias.decl *"nestedParamFields{{.*}}:param_list<!AnyType> []>
    comptime nestedParamFields = struct_field_types[fieldType1]()

    # Test struct_field_names folding
    # CHECK: lit.alias.decl *"fieldName0`{{[0-9]*}}": string = <"nested">
    comptime fieldName0 = struct_field_names_raw[MyWrapper[37]]()[0]
    # CHECK: lit.alias.decl *"fieldName1`{{[0-9]*}}": string = <"nested_param">
    comptime fieldName1 = struct_field_names_raw[MyWrapper[37]]()[1]

    # Test struct_field_index_by_name folding
    # CHECK: lit.alias.decl *"fieldIdx0`{{[0-9]*}}": !alias_Int1 = <sugar_builtin({{.*}}, 0), rebind(:!Int {:scalar<index> 0}))>
    comptime fieldIdx0 = struct_field_index_by_name[MyWrapper[37], "nested"]
    # CHECK: lit.alias.decl *"fieldIdx1`{{[0-9]*}}": !alias_Int1 = <sugar_builtin({{.*}}, 1), rebind(:!Int {:scalar<index> 1}))>
    comptime fieldIdx1 = struct_field_index_by_name[
        MyWrapper[37], "nested_param"
    ]

    # Test struct_field_type_by_name folding
    # CHECK: lit.alias.decl *"fieldTypeByName0`{{[0-9]*}}": !AnyType = <!MyStruct>
    comptime fieldTypeByName0 = struct_field_type_by_name[
        MyWrapper[37], "nested"
    ]
    # CHECK: lit.alias.decl *"fieldTypeByName1`{{[0-9]*}}": !AnyType = <@struct_type_reflection_attrs::@MyParam<:!Int {:scalar<index> 37}>>
    comptime fieldTypeByName1 = struct_field_type_by_name[
        MyWrapper[37], "nested_param"
    ]

    # Test is_struct_type parsing (positive cases - Mojo struct types)
    # At parse time, the attribute is not yet evaluated, so we check the unevaluated form.
    # CHECK: lit.alias.decl *"isStructMyStruct`{{[0-9]*}}": i1 = <#kgen.is_struct_type<!MyStruct>>
    comptime isStructMyStruct = __mlir_attr[
        `#kgen.is_struct_type<`, MyStruct, `> : i1`
    ]
    # CHECK: lit.alias.decl *"isStructMyWrapper`{{[0-9]*}}": i1 = <#kgen.is_struct_type<!lit.struct<#MyWrapper <:!Int {:scalar<index> 42}>>>>
    comptime isStructMyWrapper = __mlir_attr[
        `#kgen.is_struct_type<`, MyWrapper[42], `> : i1`
    ]
    # CHECK: lit.alias.decl *"isStructInt`{{[0-9]*}}": i1 = <#kgen.is_struct_type<!Int>>
    comptime isStructInt = __mlir_attr[`#kgen.is_struct_type<`, Int, `> : i1`]

    # Test is_struct_type parsing (negative cases - MLIR primitive types)
    comptime intFieldTypes = struct_field_types[Int]()
    # CHECK: lit.alias.decl *"isStructMLIR`{{[0-9]*}}": i1 = <#kgen.is_struct_type<:!AnyType scalar<index>>>
    comptime isStructMLIR = __mlir_attr[
        `#kgen.is_struct_type<`, intFieldTypes[0], `> : i1`
    ]
