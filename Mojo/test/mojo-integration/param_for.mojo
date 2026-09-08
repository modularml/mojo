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

# RUN: %mojo %s


def test_for_list():
    var cnt = 0

    comptime for i in [1, 2, 3]:
        cnt += i
    assert cnt == 6

    # Test for floating point numbers.
    var fp_cnt = 0.0

    comptime for i in [1.0, 2.0, 3.0]:
        fp_cnt += i
    assert fp_cnt == 6.0

    # Test for strings
    var concated = ""

    # MOCO-2091
    comptime for str in ["a", "b", "c"]:
        concated += str
    assert concated == "abc"


# derived from https://github.com/modular/modular/issues/4566
def test_critical_edge():
    var a = 0

    comptime for i in range(10):
        a = i  # Compiler hung here
    assert a == 9


# derived from https://github.com/modular/modular/issues/4836
def test_else_block():
    var a: Int  # Init not required because always assigned in else.

    comptime for i in range(10):
        pass
    else:
        a = 1  # This should execute.
    assert a == 1

    comptime for i in range(10):
        if i == 4:
            break
    else:
        assert False  # This should NOT execute.


def test_tuple_unpack():
    comptime lst0 = [1, 4]
    comptime lst1 = [1, 4]
    comptime lst2 = [(2, 3), (5, 6)]

    var ret = 0

    # Additional loop with `i` to make sure the name shadowing works
    comptime for i in range(0, 1):
        assert i == 0

        comptime for _, i, (j, k) in zip(lst0, lst1, lst2):
            ret = ret * 1000 + i * 100 + j * 10 + k

    assert ret == 123456


def main():
    test_for_list()
    test_critical_edge()
    test_else_block()
    test_tuple_unpack()
