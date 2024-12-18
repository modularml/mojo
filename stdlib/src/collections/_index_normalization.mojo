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
    assert_mode: StringLiteral = "none",
](idx: Int, container: ContainerType) -> Int:
    """Normalize the given index value to a valid index value for the given
    container length. If the provided value is negative, the `index +
    container_length` is returned.

    Parameters:
        ContainerType: The type of the container. Must have a `__len__` method.
        container_name: The name of the container. Used for the error message.
        ignore_zero_length: Whether to ignore if the container is of length 0.
        cap_to_container_length: Whether to cap the value to container length.
        assert_mode: The mode in which to do the bounds check asserts.

    Args:
        idx: The index value to normalize.
        container: The container to normalize the index for.

    Returns:
        The normalized index value.

    Notes:
        Setting cap_to_container_length to True does not deactivate the
        debug_assert that warns that the index does not exceed the limit.
        Only when setting ignore_zero_length to True. Then if the container
        length is zero, the function always returns 0.
    """
    var c_len = len(container)

    @parameter
    if not ignore_zero_length:
        debug_assert[assert_mode=assert_mode, cpu_only=True](
            c_len > 0,
            "Indexing into a ",
            container_name,
            " that has 0 elements",
        )
        debug_assert[assert_mode=assert_mode, cpu_only=True](
            -c_len <= idx < c_len,
            container_name,
            " has length: ",
            c_len,
            ". Index out of bounds: ",
            idx,
            " should be between ",
            -c_len,
            " and ",
            c_len - 1,
        )

    var normalize_len = c_len * int(idx < 0)

    @parameter
    if cap_to_container_length:
        var v = idx + normalize_len
        var v_or_zero = v * int(0 < v < c_len)
        var c_end_on_overflow = (c_len - int(c_len != 0)) * int(v >= c_len)
        return v_or_zero + c_end_on_overflow
    else:

        @parameter
        if ignore_zero_length:
            return idx * int(c_len != 0) + normalize_len
        else:
            return idx + normalize_len
