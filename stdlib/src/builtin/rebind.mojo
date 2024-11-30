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
"""Implements type rebind.

These are Mojo built-ins, so you don't need to import them.
"""


@always_inline("nodebug")
fn rebind[
    src_type: AnyTrivialRegType, //,
    dest_type: AnyTrivialRegType,
](src: src_type) -> dest_type:
    """Statically assert that a parameter input type `src_type` resolves to the
    same type as a parameter result type `dest_type` after function
    instantiation and "rebind" the input to the result type.

    This function is meant to be used in uncommon cases where a parametric type
    depends on the value of a constrained parameter in order to manually refine
    the type with the constrained parameter value.

    Parameters:
        src_type: The original type.
        dest_type: The type to rebind to.

    Args:
        src: The value to rebind.

    Returns:
        The rebound value of `dest_type`.
    """
    return __mlir_op.`kgen.rebind`[_type=dest_type](src)


@always_inline("nodebug")
fn rebind[
    src_type: AnyType, //,
    dest_type: AnyType,
](ref src: src_type) -> ref [src] dest_type:
    """Statically assert that a parameter input type `src_type` resolves to the
    same type as a parameter result type `dest_type` after function
    instantiation and "rebind" the input to the result type, returning a
    reference to the input value with an adjusted type.

    This function is meant to be used in uncommon cases where a parametric type
    depends on the value of a constrained parameter in order to manually refine
    the type with the constrained parameter value.

    Parameters:
        src_type: The original type.
        dest_type: The type to rebind to.

    Args:
        src: The value to rebind.

    Returns:
        A reference to the value rebound as `dest_type`.
    """
    lit = __get_mvalue_as_litref(src)
    rebound = rebind[Pointer[dest_type, __origin_of(src)]._mlir_type](lit)
    return __get_litref_as_mvalue(rebound)
