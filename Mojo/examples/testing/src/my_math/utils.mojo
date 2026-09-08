# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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

"""Implements various mathematical utilities.

You can import these APIs from the `my_math` package. For example:

```mojo
from my_math import inc
```
"""


def inc(n: Int) raises -> Int:
    """Returns an incremented integer value.

    ```mojo
    from my_math import inc
    i = 7
    j = inc(i)  # j = 8
    ```

    However, `inc()` raises an error if it would result in integer overflow:

    ```mojo
    k = 0
    try:
         k = inc(Int.MAX)
    except e:
         print(e)  # inc overflow
    ```

    Args:
         n: The integer value to increment.

    Returns:
         The input value plus one.

    Raises:
         An error if the incremented value exceeds `Int.MAX`.
    """
    if n == Int.MAX:
        raise Error("inc overflow")
    return n + 1


def dec(n: Int) raises -> Int:
    """Returns a decremented integer value.

    ```mojo
    from my_math import dec
    i = 7
    j = dec(i)  # j = 6
    ```

    However, `dec()` raises an error if it would result in integer overflow:

    ```mojo
    k = 0
    try:
         k = dec(Int.MIN)
    except e:
         print(e)  # inc overflow
    ```

    Args:
         n: The integer value to decrement.

    Returns:
         The input value minus one.

    Raises:
         An error if the decremented value is less than `Int.MIN`.

    """
    if n == Int.MIN:
        raise Error("dec overflow")
    return n - 1
