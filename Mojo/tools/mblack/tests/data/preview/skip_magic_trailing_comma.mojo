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

# ===----------------------------------------------------------------------=== #
#
# File originates from:
#   Repo:   git@github.com:psf/black.git
#   Commit: d4a85643a465f5fae2113d07d22d021d4af4795a
#   Path:   tests/data/preview/skip_magic_trailing_comma.py
#
# ===----------------------------------------------------------------------=== #

# We should not remove the trailing comma in a single-element subscript.
a: tuple[
    int,
]
b = tuple[
    int,
]

# But commas in multiple element subscripts should be removed.
c: tuple[
    int,
    int,
]
d = tuple[
    int,
    int,
]

# Remove commas for non-subscripts.
small_list = [
    1,
]
list_of_types = [
    tuple[
        int,
    ],
]
small_set = {
    1,
}
set_of_types = {
    tuple[
        int,
    ],
}

# Except single element tuples
small_tuple = (1,)

# Trailing commas in multiple chained non-nested parens.
zero(one,).two(three,).four(
    five,
)

func1(arg1).func2(arg2,).func3(arg3).func4(
    arg4,
).func5(arg5)

(a, b, c, d,) = func1(
    arg1
) and func2(arg2)

func(
    argument1,
    (
        one,
        two,
    ),
    argument4,
    argument5,
    argument6,
)

# output
# We should not remove the trailing comma in a single-element subscript.
a: tuple[int,]
b = tuple[int,]

# But commas in multiple element subscripts should be removed.
c: tuple[int, int]
d = tuple[int, int]

# Remove commas for non-subscripts.
small_list = [1]
list_of_types = [tuple[int,]]
small_set = {1}
set_of_types = {tuple[int,]}

# Except single element tuples
small_tuple = (1,)

# Trailing commas in multiple chained non-nested parens.
zero(one).two(three).four(five)

func1(arg1).func2(arg2).func3(arg3).func4(arg4).func5(arg5)

(a, b, c, d) = func1(arg1) and func2(arg2)

func(argument1, (one, two), argument4, argument5, argument6)
