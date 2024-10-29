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
    ContainerType: Sized, //,
    container_name: StringLiteral,
    *,
    ignore_zero_length: Bool = False,
    cap_to_container_length: Bool = True,
](idx: Int, container: ContainerType) -> Int:
    """Normalize the given index value to a valid index value for the given
    container length. If the provided value is negative, the `index +
    container_length` is returned.

    Parameters:
        ContainerType: The type of the container. Must have a `__len__` method.
        container_name: The name of the container. Used for the error message.
        ignore_zero_length: Whether to ignore if the container is of length 0.
        cap_to_container_length: Whether to cap the value to container length.

    Args:
        idx: The index value to normalize.
        container: The container to normalize the index for.

    Returns:
        The normalized index value.

    Notes:
        Setting cap_to_container_length to True does not deactivate the
        debug_assert that verifies that the index does not exceed the limit.
        Only when setting ignore_zero_length to True as well. Then if the
        container length is zero, the function allways returns 0.
    """
    container_length = len(container)

    @parameter
    if not ignore_zero_length:
        debug_assert[assert_mode="safe", cpu_only=True](
            container_length > 0,
            "indexing into a ",
            container_name,
            " that has 0 elements",
        )
        debug_assert[assert_mode="safe", cpu_only=True](
            -container_length <= idx < container_length,
            container_name,
            " has length: ",
            container_length,
            " index out of bounds: ",
            idx,
            " should be between ",
            -container_length,
            " and ",
            container_length - 1,
        )

    @parameter
    if cap_to_container_length:
        value = idx + container_length * int(idx < 0)
        return value * int(
            value < container_length and value > 0
        ) + container_length * int(value >= container_length)
    else:
        return idx + container_length * int(idx < 0)
