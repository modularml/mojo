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

"""Implements higher-order functions.

You can import these APIs from the `utils.loop` module. For example:

```mojo
from utils import unroll
```
"""


# ===----------------------------------------------------------------------===#
# unroll
# ===----------------------------------------------------------------------===#


@always_inline
fn unroll[
    func: fn[idx: Int] () capturing -> None,
    count: Int,
]():
    """Repeatedly evaluates a function `count` times.

    Parameters:
        func: The function to evaluate. The function should take a single `Int`
          argument, which is the loop index value.
        count: A number of repetitions.
    """

    @parameter
    for i in range(count):
        func[i]()


@always_inline
fn unroll[
    func: fn[idx: Int] () raises capturing -> None,
    count: Int,
]() raises:
    """Repeatedly evaluates a function `count` times.

    Parameters:
        func: The function to evaluate. The function should take a single `Int`
          argument, which is the loop index value.
        count: A number of repetitions.
    """

    @parameter
    for i in range(count):
        func[i]()


# ===----------------------------------------------------------------------===#
# unroll
# ===----------------------------------------------------------------------===#


@always_inline
fn unroll[
    func: fn[idx0: Int, idx1: Int] () capturing -> None,
    dim0: Int,
    dim1: Int,
]():
    """Repeatedly evaluates a 2D nested loop.

    Parameters:
        func: The function to evaluate. The function should take two `Int`
          arguments: the outer and inner loop index values.
        dim0: The first dimension size.
        dim1: The second dimension size.
    """

    @parameter
    for i in range(dim0):

        @parameter
        for j in range(dim1):
            func[i, j]()


# ===----------------------------------------------------------------------===#
# unroll
# ===----------------------------------------------------------------------===#


@always_inline
fn unroll[
    func: fn[idx0: Int, idx1: Int, idx2: Int] () capturing -> None,
    dim0: Int,
    dim1: Int,
    dim2: Int,
]():
    """Repeatedly evaluates a 3D nested loop.

    Parameters:
        func: The function to evaluate. The function should take three `Int`
          arguments: one for each nested loop index value.
        dim0: The first dimension size.
        dim1: The second dimension size.
        dim2: The second dimension size.
    """

    @parameter
    for i in range(dim0):

        @parameter
        for j in range(dim1):

            @parameter
            for k in range(dim2):
                func[i, j, k]()
