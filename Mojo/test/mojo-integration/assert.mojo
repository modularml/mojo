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

# RUN: %mojo %s | FileCheck %s


def check_positive(x: Int):
    assert x > 0, "expected positive"


def check_range(x: Int, lo: Int, hi: Int):
    assert x >= lo
    assert x < hi


struct Point:
    var x: Int
    var y: Int

    def __init__(out self, x: Int, y: Int):
        self.x = x
        self.y = y

    def is_valid(self) -> Bool:
        return self.x >= 0 and self.y >= 0


def main():
    # CHECK-LABEL: test_basic
    print("== test_basic")

    # Basic true assertions.
    assert True
    assert True, "this should pass"

    # Expressions that evaluate to true.
    assert 1 == 1
    assert 2 > 1
    assert 3 != 4, "values should differ"

    # CHECK: basic passed
    print("basic passed")

    # CHECK-LABEL: test_boolean_operators
    print("== test_boolean_operators")

    var t = True
    var f = False
    assert t and t
    assert t or f
    assert not f
    assert not (f and t)
    assert t or (f and f)

    # CHECK: boolean operators passed
    print("boolean operators passed")

    # CHECK-LABEL: test_variables
    print("== test_variables")

    var x = 42
    assert x == 42
    assert x > 0, "x should be positive"
    assert x != 0
    assert x <= 100

    # CHECK: variables passed
    print("variables passed")

    # CHECK-LABEL: test_string_messages
    print("== test_string_messages")

    # String literal message.
    assert True, "literal message"

    # String variable message.
    var msg = String("dynamic message")
    assert True, msg

    # Computed string message.
    assert True, String("computed: ") + String(x)

    # Integer message (Writable).
    assert True, 42

    # CHECK: string messages passed
    print("string messages passed")

    # CHECK-LABEL: test_functions
    print("== test_functions")

    check_positive(10)
    check_positive(1)
    check_range(5, 0, 10)
    check_range(0, 0, 100)
    check_range(99, 0, 100)

    # CHECK: functions passed
    print("functions passed")

    # CHECK-LABEL: test_loops
    print("== test_loops")

    for i in range(10):
        assert i >= 0
        assert i < 10

    # Nested loops.
    for i in range(3):
        for j in range(3):
            assert i + j < 6

    # CHECK: loops passed
    print("loops passed")

    # CHECK-LABEL: test_conditionals
    print("== test_conditionals")

    var val = 7
    if val > 0:
        assert val > 0
    else:
        assert val <= 0

    if val % 2 != 0:
        assert val % 2 == 1, "val should be odd"

    # CHECK: conditionals passed
    print("conditionals passed")

    # CHECK-LABEL: test_struct_methods
    print("== test_struct_methods")

    var p = Point(3, 4)
    assert p.is_valid(), "point should be valid"
    assert p.x > 0
    assert p.y > 0

    # CHECK: struct methods passed
    print("struct methods passed")

    # CHECK-LABEL: test_multiple_asserts_sequence
    print("== test_multiple_asserts_sequence")

    # Many asserts in sequence — code after each is reachable.
    assert True
    assert True
    assert True
    assert 1 == 1
    assert 2 > 1
    assert 3 < 4

    # CHECK: sequence passed
    print("sequence passed")

    # CHECK: all asserts passed
    print("all asserts passed")
