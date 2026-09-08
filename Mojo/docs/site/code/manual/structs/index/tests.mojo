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
from std.testing import assert_equal, assert_true


@fieldwise_init
struct MyPair:
    var first: Int
    var second: Int

    # Construct with first = 1 and value = 2
    def aux_test_local(self, value: Int) raises:
        assert_true(self.first == 1)
        assert_true(value == 2)
        var first = value
        assert_true(first == value)
        assert_true(self.first != value)


def test_initializer_list(pair: MyPair) raises:
    assert_true(pair.first == 5)
    assert_true(pair.second == 6)


def test_mypair_construction() raises:
    var pair = MyPair(1, 2)
    assert_true(pair.first == 1)
    assert_true(pair.second == 2)
    pair.aux_test_local(2)

    var pair2: MyPair = {first = 3, second = 4}
    assert_true(pair2.first == 3)
    assert_true(pair2.second == 4)

    test_initializer_list({first = 5, second = 6})


@fieldwise_init
struct MyStruct:
    var value: Int

    def increment(mut self):
        self.value += 1  # Works: Mutable `self` allows assignment


def test_mutable_self() raises:
    var s = MyStruct(10)
    s.increment()
    assert_true(s.value == 11)


struct Logger:
    def __init__(out self):
        pass

    @staticmethod
    def log_info(message: String) -> String:
        return "Info: " + message


def test_static_method() raises:
    assert_equal(Logger.log_info("Hello"), "Info: Hello")


def main() raises:
    test_mypair_construction()
    test_mutable_self()
    test_static_method()
