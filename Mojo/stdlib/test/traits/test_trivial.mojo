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


from std.traits import (
    IsTriviallyCopyable,
    IsTriviallyDeinitable,
    IsTriviallyMovable,
)
from std.testing import assert_equal, assert_false, assert_true, TestSuite

# ===-----------------------------------------------------------------------===#
# Triviality Struct
# ===-----------------------------------------------------------------------===#

comptime EVENT_TRIVIAL = 0b1  # 1
comptime EVENT_INIT = 0b10  # 2
comptime EVENT_DEL = 0b100  # 4
comptime EVENT_COPY = 0b1000  # 8
comptime EVENT_MOVE = 0b10000  # 16


struct ConditionalTriviality[
    O: MutOrigin,
    //,
    T: Copyable & Deinitable,
](Copyable):
    var events: Pointer[List[Int], Self.O]

    def add_event(mut self, event: Int):
        self.events[].append(event)

    def __init__(out self, ref[Self.O] events: List[Int]):
        self.events = Pointer(to=events)
        self.add_event(EVENT_INIT)

    def __deinit__(deinit self):
        comptime if IsTriviallyDeinitable[Self.T]:
            self.add_event(EVENT_DEL | EVENT_TRIVIAL)
        else:
            self.add_event(EVENT_DEL)

    def __init__(out self, *, copy: Self):
        self.events = copy.events

        comptime if IsTriviallyCopyable[Self.T]:
            self.add_event(EVENT_COPY | EVENT_TRIVIAL)
        else:
            self.add_event(EVENT_COPY)

    def __init__(out self, *, deinit move: Self):
        self.events = move.events

        comptime if IsTriviallyMovable[Self.T]:
            self.add_event(EVENT_MOVE | EVENT_TRIVIAL)
        else:
            self.add_event(EVENT_MOVE)


struct StructInheritTriviality[T: Copyable & Deinitable](Copyable):
    comptime __move_ctor_is_trivial = IsTriviallyMovable[Self.T]
    comptime __copy_ctor_is_trivial = IsTriviallyCopyable[Self.T]
    comptime __del__is_trivial = IsTriviallyDeinitable[Self.T]


# ===-----------------------------------------------------------------------===#
# Individual tests
# ===-----------------------------------------------------------------------===#


def _test_type_trivial[T: Copyable & Deinitable]() raises:
    var events = List[Int]()
    var value = ConditionalTriviality[T](events)
    var value_copy = value.copy()
    var _value_move = value_copy^
    assert_equal(
        events,
        [
            EVENT_INIT,
            EVENT_COPY | EVENT_TRIVIAL,
            EVENT_DEL | EVENT_TRIVIAL,
            EVENT_MOVE | EVENT_TRIVIAL,
            EVENT_DEL | EVENT_TRIVIAL,
        ],
    )


def test_type_trivial() raises:
    _test_type_trivial[Int]()


def _test_type_not_trivial[T: Copyable & Deinitable]() raises:
    var events = List[Int]()
    var value = ConditionalTriviality[T](events)
    var value_copy = value.copy()
    var _value_move = value_copy^
    assert_equal(
        events,
        [
            EVENT_INIT,
            EVENT_COPY,
            EVENT_DEL,
            EVENT_MOVE | EVENT_TRIVIAL,
            EVENT_DEL,
        ],
    )


def test_type_not_trivial() raises:
    _test_type_not_trivial[String]()


def _test_type_inherit_triviality[T: Copyable & Deinitable]() raises:
    var events = List[Int]()
    var value = ConditionalTriviality[StructInheritTriviality[T]](events)
    var value_copy = value.copy()
    var _value_move = value_copy^
    assert_equal(
        events,
        [
            EVENT_INIT,
            EVENT_COPY | EVENT_TRIVIAL,
            EVENT_DEL | EVENT_TRIVIAL,
            EVENT_MOVE | EVENT_TRIVIAL,
            EVENT_DEL | EVENT_TRIVIAL,
        ],
    )


def test_type_inherit_triviality() raises:
    _test_type_inherit_triviality[Float64]()
    # _test_type_inherit_triviality[Array[Array[Int, 4], 4]]()


def _test_type_inherit_non_triviality[T: Copyable & Deinitable]() raises:
    var events = List[Int]()
    var value = ConditionalTriviality[StructInheritTriviality[T]](events)
    var value_copy = value.copy()
    var _value_move = value_copy^
    assert_equal(
        events,
        [
            EVENT_INIT,
            EVENT_COPY,
            EVENT_DEL,
            EVENT_MOVE | EVENT_TRIVIAL,
            EVENT_DEL,
        ],
    )


def test_type_inherit_non_triviality() raises:
    _test_type_inherit_non_triviality[String]()
    # _test_type_inherit_non_triviality[Array[Array[String, 4], 4]]()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
