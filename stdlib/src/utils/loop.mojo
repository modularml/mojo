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

from builtin.range import _SequentialRange, _StridedRange, _ZeroStartingRange

"""Implements higher-order functions.

You can import these APIs from the `utils.loop` module. For example:

```mojo
from utils import unroll
```
"""


# ===-----------------------------------------------------------------------===#
# unroll
# ===-----------------------------------------------------------------------===#


@always_inline
fn unroll[
    func: fn[idx0: Int, idx1: Int] () capturing [_] -> None,
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


# ===-----------------------------------------------------------------------===#
# unroll
# ===-----------------------------------------------------------------------===#


@always_inline
fn unroll[
    func: fn[idx0: Int, idx1: Int, idx2: Int] () capturing [_] -> None,
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


# ===-----------------------------------------------------------------------===#
# unroll _ZeroStartingRange
# ===-----------------------------------------------------------------------===#


@always_inline
fn unroll[
    func: fn[idx: Int] () capturing [_] -> None,
    zero_starting_range: _ZeroStartingRange,
]():
    """Repeatedly evaluates a function `range` times.

    Parameters:
        func: The function to evaluate. The function should take a single `Int`
          argument, which is the loop index value.
        zero_starting_range: A range representing the number of single step repetitions starting from zero.
    """

    @parameter
    for i in zero_starting_range:
        func[i]()


@always_inline
fn unroll[
    func: fn[idx: Int] () raises capturing [_] -> None,
    zero_starting_range: _ZeroStartingRange,
]() raises:
    """Repeatedly evaluates a function `range` times.

    Parameters:
        func: The function to evaluate. The function should take a single `Int`
          argument, which is the loop index value.
        zero_starting_range: A range representing the number of single step repetitions starting from zero.
    """

    @parameter
    for i in zero_starting_range:
        func[i]()


# ===-----------------------------------------------------------------------===#
# unroll _SequentialRange
# ===-----------------------------------------------------------------------===#
@always_inline
fn unroll[
    func: fn[idx: Int] () capturing [_] -> None,
    sequential_range: _SequentialRange,
]():
    """Repeatedly evaluates a function `range` times.

    Parameters:
        func: The function to evaluate. The function should take a single `Int`
          argument, which is the loop index value.
        sequential_range: A range representing the number of single step repetitions from [start; end).
    """

    @parameter
    for i in sequential_range:
        func[i]()


@always_inline
fn unroll[
    func: fn[idx: Int] () raises capturing [_] -> None,
    sequential_range: _SequentialRange,
]() raises:
    """Repeatedly evaluates a function `range` times.

    Parameters:
        func: The function to evaluate. The function should take a single `Int`
          argument, which is the loop index value.
        sequential_range: A range representing the number of single step repetitions from [start; end).
    """

    @parameter
    for i in sequential_range:
        func[i]()


# ===-----------------------------------------------------------------------===#
# unroll _StridedRange
# ===-----------------------------------------------------------------------===#
@always_inline
fn unroll[
    func: fn[idx: Int] () capturing [_] -> None,
    strided_range: _StridedRange,
]():
    """Repeatedly evaluates a function `range` times.

    Parameters:
        func: The function to evaluate. The function should take a single `Int`
          argument, which is the loop index value.
        strided_range: A range representing the number of strided repetitions from [start; end).
    """

    @parameter
    for i in strided_range:
        func[i]()


@always_inline
fn unroll[
    func: fn[idx: Int] () raises capturing [_] -> None,
    strided_range: _StridedRange,
]() raises:
    """Repeatedly evaluates a function `range` times.

    Parameters:
        func: The function to evaluate. The function should take a single `Int`
          argument, which is the loop index value.
        strided_range: A range representing the number of strided repetitions from [start; end).
    """

    @parameter
    for i in strided_range:
        func[i]()
