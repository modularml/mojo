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

from collections.dict import Dict


struct Counter[V: KeyElement](
    #    Sized,
    #    CollectionElement,
    #    Boolable
):
    """A container for counting hashable items.

    The value type must be specified statically, unlike a Python
    Counter, which can accept arbitrary value types.

    The value type must implement the `KeyElement` trait, as its values are
    stored in the dictionary as keys. `KeyElement` includes
    `Movable`, `Hashable`, and `EqualityComparable`.

    Usage:

    ```mojo
    from collections import Counter
    var c = Counter[String](["a", "a", "a", "b", "b", "c", "d", "c", "c"])
    print(c["a"]) # prints 3
    print(c["b"]) # prints 2
    ```

    Parameters:
        V: The value type to be counted. Currently must be KeyElement.
    """

    var _data: Dict[V, Int]

    def __init__(inout self):
        """Create a new, empty Counter object.

        Usage:
        ```mojo
        c = Counter()
        ```
        """
        self._data = Dict[V, Int]()

    # TODO: Change List to Iterable when it is supported in Mojo
    def __init__(inout self, items: List[V]):
        """Create a from an input iterable.

        Args:
            items: A list of items to count.

        Usage:
        ```mojo
        c = Counter(['g', 'a', 't', 't', 'a', 'c', 'a'])
        ```
        """
        self._data = Dict[V, Int]()
        for item_ref in items:
            var item = item_ref[]
            self._data[item] = self._data.get(item, 0) + 1
