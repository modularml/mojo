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
#   Path:   tests/data/simple_cases/torture.py
#
# ===----------------------------------------------------------------------=== #

importA
(
    ()
    << 0
    ** 101234234242352525425252352352525234890264906820496920680926538059059209922523523525
)  #

assert sort_by_dependency(
    {
        "1": {"2", "3"},
        "2": {"2a", "2b"},
        "3": {"3a", "3b"},
        "2a": set(),
        "2b": set(),
        "3a": set(),
        "3b": set(),
    }
) == ["2a", "2b", "2", "3a", "3b", "3", "1"]

importA
0
0 ^ 0  #


class A:
    def foo(self):
        for _ in range(10):
            aaaaaaaaaaaaaaaaaaa = (
                bbbbbbbbbbbbbbb.cccccccccc(  # pylint: disable=no-member
                    xxxxxxxxxxxx
                )
            )


def test(self, othr):
    return 1 == 2 and (
        name,
        description,
        self.default,
        self.selected,
        self.auto_generated,
        self.parameters,
        self.meta_data,
        self.schedule,
    ) == (
        name,
        description,
        othr.default,
        othr.selected,
        othr.auto_generated,
        othr.parameters,
        othr.meta_data,
        othr.schedule,
    )


assert a_function(
    very_long_arguments_that_surpass_the_limit,
    which_is_eighty_eight_in_this_case_plus_a_bit_more,
) == {
    "x": "this need to pass the line limit as well",
    "b": "but only by a little bit",
}

# output

importA
(
    ()
    << 0
    ** 101234234242352525425252352352525234890264906820496920680926538059059209922523523525
)  #

assert sort_by_dependency(
    {
        "1": {"2", "3"},
        "2": {"2a", "2b"},
        "3": {"3a", "3b"},
        "2a": set(),
        "2b": set(),
        "3a": set(),
        "3b": set(),
    }
) == ["2a", "2b", "2", "3a", "3b", "3", "1"]

importA
0
0 ^ 0  #


class A:
    def foo(self):
        for _ in range(10):
            aaaaaaaaaaaaaaaaaaa = (
                bbbbbbbbbbbbbbb.cccccccccc(  # pylint: disable=no-member
                    xxxxxxxxxxxx
                )
            )


def test(self, othr):
    return 1 == 2 and (
        name,
        description,
        self.default,
        self.selected,
        self.auto_generated,
        self.parameters,
        self.meta_data,
        self.schedule,
    ) == (
        name,
        description,
        othr.default,
        othr.selected,
        othr.auto_generated,
        othr.parameters,
        othr.meta_data,
        othr.schedule,
    )


assert a_function(
    very_long_arguments_that_surpass_the_limit,
    which_is_eighty_eight_in_this_case_plus_a_bit_more,
) == {
    "x": "this need to pass the line limit as well",
    "b": "but only by a little bit",
}
