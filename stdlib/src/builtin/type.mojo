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
"""Defines some type functions.

These are Mojo built-ins, so you don't need to import them.
"""


@value
struct _Type[T: AnyType]:
    alias type = T


@parameter
fn type[T: AnyType, //](v: T) -> __type_of(_Type[__type_of(v)].type):
    """Get the type of the argument.

    Args:
        v: The value

    Returns:
        The type of v.
    """
    return _Type[__type_of(v)].type


# TODO: Should be a parametric alias.
struct Origin[is_mutable: Bool]:
    """This represents a origin reference of potentially parametric type.
    TODO: This should be replaced with a parametric type alias.

    Parameters:
        is_mutable: Whether the origin reference is mutable.
    """

    alias type = __mlir_type[
        `!lit.origin<`,
        is_mutable.value,
        `>`,
    ]


struct _Origin[is_mutable: Bool, //, origin: Origin[is_mutable].type]:
    ...


fn origin[T: AnyType, //](v: T) -> _Origin[__origin_of(v)].origin.type:
    """Get the origin of the argument.

    Args:
        v: The value

    Returns:
        The origin of v.
    """
    return _Origin[__origin_of(v)].origin
