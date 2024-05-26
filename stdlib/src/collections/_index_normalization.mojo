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


fn get_out_of_bounds_error_message[
    container_name: String
](i: Int, container_length: Int) -> String:
    if container_length == 0:
        return (
            "The "
            + container_name
            + " has a length of 0. "
            + "Thus it's not possible to access its values with an index "
            + "but the index value "
            + str(i)
            + " was used. "
            + "Aborting now to avoid an out-of-bounds access."
        )
    else:
        return (
            "The "
            + container_name
            + " has a length of "
            + str(container_length)
            + ". "
            + "Thus the index provided should be between "
            + str(-container_length)
            + " (inclusive) and "
            + str(container_length)
            + " (exclusive) but the index value "
            + str(i)
            + " was used. "
            + "Aborting now to avoid an out-of-bounds access."
        )


@always_inline
fn normalize_index[
    IndexType: Indexer,
    ContainerType: Sized, //,
    container_name: StringLiteral,
](index_value: IndexType, container: ContainerType) -> Int:
    """Normalize the given index value to a valid index value for the given container length.

    If the provided value is negative, the `index + container_length` is returned.

    Parameters:
        IndexType: The type of the index value. Must have an `__index__` method.
        ContainerType: The type of the container. Must have a `__len__` method.
        container_name: The name of the container. Used for the error message.

    Args:
        index_value: The index value to normalize.
        container: The container to normalize the index for.

    Returns:
        The normalized index value.
    """
    var index_as_int = index(index_value)
    var container_length = len(container)

    if not (-container_length <= index_as_int < container_length):
        # TODO: Get the container_name from the ContainerType when the compiler allows it.
        abort(
            get_out_of_bounds_error_message[container_name](
                index_as_int, container_length
            )
        )
    if index_as_int < 0:
        index_as_int += container_length
    return index_as_int
