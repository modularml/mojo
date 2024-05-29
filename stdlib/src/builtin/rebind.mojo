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
    dest_type: AnyTrivialRegType,
    src_type: AnyTrivialRegType,
](val: src_type) -> dest_type:
    """Statically assert that a parameter input type `src_type` resolves to the
    same type as a parameter result type `dest_type` after function
    instantiation and "rebind" the input to the result type.

    This function is meant to be used in uncommon cases where a parametric type
    depends on the value of a constrained parameter in order to manually refine
    the type with the constrained parameter value.

    Parameters:
        dest_type: The type to rebind to.
        src_type: The original type.

    Args:
        val: The value to rebind.

    Returns:
        The rebound value of `dest_type`.
    """
    return __mlir_op.`kgen.rebind`[_type=dest_type](val)
