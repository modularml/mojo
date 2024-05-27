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

from sys import triple_is_nvidia_cuda


fn get_out_of_bounds_error_message[
    container_name: StringLiteral
](i: Int, container_length: Int) -> String:
    if container_length == 0:
        return (
            "The "
            + container_name
            + " has a length of 0. Thus it's not possible to access its values"
            " with an index but the index value "
            + str(i)
            + " was used. Aborting now to avoid an out-of-bounds access."
        )
    else:
        return (
            "The "
            + container_name
            + " has a length of "
            + str(container_length)
            + ". Thus the index provided should be between "
            + str(-container_length)
            + " (inclusive) and "
            + str(container_length)
            + " (exclusive) but the index value "
            + str(i)
            + " was used. Aborting now to avoid an out-of-bounds access."
        )


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
    var container_length = len(container)
    if not (-container_length <= idx < container_length):

        @parameter
        if triple_is_nvidia_cuda():
            abort()
        else:
            abort(
                get_out_of_bounds_error_message[container_name](
                    idx, container_length
                )
            )
    return idx + int(idx < 0) * container_length
