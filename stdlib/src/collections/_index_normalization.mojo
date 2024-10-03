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
"""The utilities provided in this module help normalize the access
to data elements in arrays."""


@always_inline
fn normalize_index[
    ContainerType: Sized, //, container_name: StringLiteral
](idx: Int, container: ContainerType) -> Int:
    """Normalize the given index value to a valid index value for the given container length.

    If the provided value is negative, the `index + container_length` is returned.

    Parameters:
        ContainerType: The type of the container. Must have a `__len__` method.
        container_name: The name of the container. Used for the error message.

    Args:
        idx: The index value to normalize.
        container: The container to normalize the index for.

    Returns:
        The normalized index value.
    """
    debug_assert[assert_mode="safe", cpu_only=True](
        len(container) > 0,
        "indexing into a ",
        container_name,
        " that has 0 elements",
    )
    debug_assert[assert_mode="safe", cpu_only=True](
        -len(container) <= idx < len(container),
        container_name,
        " has length: ",
        len(container),
        " index out of bounds: ",
        idx,
        " should be between ",
        -len(container),
        " and ",
        len(container) - 1,
    )
    if idx >= 0:
        return idx
    return idx + len(container)
