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
"""Implements compile-time constraint helpers used by trait conformance checks.
"""
from std.collections.string.string_span import _get_kgen_string
from std.reflection import reflect
from std.reflection.type_info import _unqualified_type_name


@always_inline("nodebug")
def _constrained_conforms_to[
    cond: Bool,
    *,
    Parent: AnyType,
    Element: AnyType,
    ParentConformsTo: StaticString,
    ElementConformsTo: StaticString = ParentConformsTo,
]():
    comptime parent_type_name = reflect[Parent].name()
    comptime elem_type_name = reflect[Element].name()
    # TODO(MOCO-2901): Support traits in reflect[T].name()
    #   comptime trait_name = reflect[ParentConformsTo].name()
    comptime parent_conforms_to_trait_name = ParentConformsTo
    comptime elem_conforms_to_trait_name = ElementConformsTo

    # Construct a message like:
    #     List(Equatable) conformance requires Foo(Equatable) conformance, which
    #     is not satisfied.
    comptime assert cond, StaticString(
        _get_kgen_string[
            parent_type_name,
            "(",
            parent_conforms_to_trait_name,
            ") conformance requires ",
            elem_type_name,
            "(",
            elem_conforms_to_trait_name,
            ") conformance, which is not satisfied.",
        ]()
    )


@always_inline("nodebug")
def _field_conforms_to_error[
    *,
    Parent: AnyType,
    FieldIndex: Int,
    ParentConformsTo: StaticString,
    FieldConformsTo: StaticString = ParentConformsTo,
]() -> StaticString:
    """Asserts that a struct field conforms to a trait at compile time.

    This helper is used in default trait implementations that use reflection
    to operate on all fields of a struct. It produces a clear error message
    when a field doesn't conform to the required trait.

    Parameters:
        Parent: The struct type being checked.
        FieldIndex: The index of the field in the struct.
        ParentConformsTo: The trait the parent is trying to conform to.
        FieldConformsTo: The trait the field must conform to
            (defaults to ParentConformsTo).
    """
    comptime r = reflect[Parent]
    comptime names = r.field_names()
    comptime field_name = names[FieldIndex]
    comptime parent_type_name = _unqualified_type_name[Parent]()
    comptime types = r.field_types()
    comptime FieldType = types[FieldIndex]

    # Construct a message like:
    #     Could not derive Equatable for Point - member field `x: Int`
    #     does not implement Equatable
    # For MLIR types, omit the type name since `_unqualified_type_name`
    # can't handle non-struct types.
    comptime if reflect[FieldType].is_struct():
        comptime field_type_name = _unqualified_type_name[FieldType]()
        return StaticString(
            _get_kgen_string[
                "Could not derive ",
                ParentConformsTo,
                " for ",
                parent_type_name,
                " - member field `",
                field_name,
                ": ",
                field_type_name,
                "` does not implement ",
                FieldConformsTo,
            ]()
        )
    else:
        return StaticString(
            _get_kgen_string[
                "Could not derive ",
                ParentConformsTo,
                " for ",
                parent_type_name,
                " - member field `",
                field_name,
                "` does not implement ",
                FieldConformsTo,
            ]()
        )
