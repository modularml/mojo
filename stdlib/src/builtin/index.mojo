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


trait Indexer:
    """This trait denotes a type that can be used to index a container that
    handles integral index values.

    This solves the issue of being able to index data structures such as `List` with the various
    integral types (with the `int()` function for conversion) without
    being too broad and allowing types that should not be used such as float point values.
    """

    fn __index__(self) -> Int:
        """Return the index value

        Returns:
            The index value of the object
        """
        ...


@always_inline
fn index[indexer: Indexer](idx: indexer) -> Int:
    """Returns the value of `__index__` for the given value.

    Parameters:
        indexer: The type of the given value.

    Args:
        idx: The value to convert to an `Int`.

    Returns:
        An `Int` respresenting the index value.
    """
    return idx.__index__()
