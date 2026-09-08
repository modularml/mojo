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
# test_compound_statements.mojo
# Tests for compound-statements.mdx code examples.
from std.sys import size_of
from std.testing import assert_equal, assert_false, assert_true


# --- If statements ---


def test_if_elif_else_chain() raises:
    var result: String
    var x = 5
    if x > 0:
        result = "positive"
    elif x < 0:
        result = "negative"
    elif x == 0:
        result = "zero"
    else:
        result = "you should never get here"
    assert_equal(result, "positive")

    x = -3
    if x > 0:
        result = "positive"
    elif x < 0:
        result = "negative"
    elif x == 0:
        result = "zero"
    assert_equal(result, "negative")

    x = 0
    if x > 0:
        result = "positive"
    elif x < 0:
        result = "negative"
    elif x == 0:
        result = "zero"
    assert_equal(result, "zero")


def test_single_line_if() raises:
    var result = String("placeholder")
    var x = 5
    if x > 0:
        result = "positive"
    assert_equal(result, "positive")


# --- While loops ---


def test_while_basic() raises:
    var count = 0
    var total = 0
    while count < 10:
        total += count
        count += 1
    assert_equal(count, 10)
    assert_equal(total, 45)  # 0+1+...+9


def test_while_break_continue() raises:
    # break exits the loop, continue skips to the next iteration.
    var processed = 0
    var i = 0
    while True:
        if i >= 10:
            break
        if i % 2 == 1:
            i += 1
            continue
        processed += 1
        i += 1
    assert_equal(processed, 5)  # 0, 2, 4, 6, 8


# --- For loops ---


def test_for_conventions() raises:
    var list: List[String] = ["a", "b", "c", "d"]

    for var item in list:
        item = item + "x"  # works. item is mutable copy of list element
        # print(item) # prints "ax", then "bx", "cx", and "dx"
        assert_true(item.endswith("x"))
    assert_true(list == ["a", "b", "c", "d"])  # list is unchanged
    # print(list) # unchanged

    for ref item in list:
        item = item + "x"
        # mutability picked up in reference to list element
        # print(item) # prints "ax", then "bx", "cx", and "dx"
        assert_true(item.endswith("x"))
    assert_true(list == ["ax", "bx", "cx", "dx"])  # list is changed
    # print(list) # changed to ["ax", "bx", "cx", "dx"]


def test_for_range() raises:
    var total = 0
    for i in range(10):
        total += i
    assert_equal(total, 45)


def test_for_destructuring() raises:
    # Tuple unpacking in the loop target.
    var pairs = [("a", 1), ("b", 2), ("c", 3)]
    var total = 0
    for _key, value in pairs:
        total += value
    assert_equal(total, 6)


# --- For/while else clauses ---


def test_for_else_no_break() raises:
    # else runs when the loop completes without break.
    var items = [1, 2, 3]
    var target = 99
    var found = False
    var ran_else = False
    for item in items:
        if item == target:
            found = True
            break
    else:
        ran_else = True
    assert_false(found)
    assert_true(ran_else)


def test_for_else_with_break() raises:
    # else does NOT run when break exits the loop.
    var items = [1, 2, 3]
    var target = 2
    var found = False
    var ran_else = False
    for item in items:
        if item == target:
            found = True
            break
    else:
        ran_else = True
    assert_true(found)
    assert_false(ran_else)


def test_while_else_no_break() raises:
    var i = 0
    var ran_else = False
    assert_false(ran_else)  # sentinel: must transition to True via else
    while i < 5:
        i += 1
    else:
        ran_else = True
    assert_equal(i, 5)
    assert_true(ran_else)


# --- Error handling ---


def _risky() raises:
    raise Error("oops")


def test_try_except_basic() raises:
    var caught = False
    try:
        _risky()
    except:
        caught = True
    assert_true(caught)


def test_try_full_structure_no_error() raises:
    # try -> else -> finally when no error fires.
    var sequence = String("")
    try:
        sequence += "T"
    except _:
        sequence += "E"
    else:
        sequence += "L"
    finally:
        sequence += "F"
    assert_equal(sequence, "TLF")


def test_try_full_structure_with_error() raises:
    # try -> except -> finally when an error fires.
    var sequence = String("")
    try:
        sequence += "T"
        _risky()
    except e:
        sequence += "E"
    else:
        sequence += "L"
    finally:
        sequence += "F"
    assert_equal(sequence, "TEF")


def test_except_with_binding() raises:
    var msg = String("")
    try:
        _risky()
    except e:
        msg = String(e)
    assert_true("oops" in msg)


def test_except_without_binding() raises:
    var caught = False
    try:
        _risky()
    except:
        caught = True
    assert_true(caught)


# --- Typed errors ---


@fieldwise_init
struct NetworkError(ImplicitlyCopyable, Movable, Writable):
    var message: String
    var code: Int

    def write_to[W: Writer](self, mut writer: W):
        writer.write("NetworkError(code=", self.code, "): ", self.message)


def _fetch() raises NetworkError -> String:
    raise NetworkError("HTCPCP", 418)


def test_typed_raises() raises:
    # raises Type infers the bound error's type and allows field access.
    var code = 0
    var msg = String("placeholder")
    try:
        _ = _fetch()
    except e:
        code = e.code
        msg = e.message
    assert_equal(code, 418)
    assert_equal(msg, "HTCPCP")


# --- Context managers ---


struct Scope(ImplicitlyCopyable):
    var label: String

    def __init__(out self, label: String):
        self.label = label

    def __enter__(self) -> Self:
        # perform setup tasks
        # print("entering", self.label) Not needed for testing
        return self

    def __exit__(self):
        # perform cleanup tasks
        # print("exiting", self.label) Not needed for testing
        pass


def test_custom_context_manager() raises:
    with Scope("setup") as s:
        assert_equal(s.label, "setup")


# --- comptime if ---


def test_comptime_if() raises:
    # Either branch may be selected depending on the target.
    var label: String
    comptime if size_of[Int]() == 8:
        label = "64-bit"
    else:
        label = "Probably 32-bit"
    assert_true(label == "64-bit" or label == "Probably 32-bit")


# --- comptime for ---


def test_comptime_for() raises:
    # The loop unrolls at compile time; body emits as total += 0; total += 1; ...
    var total = 0
    comptime for i in range(5):
        total += i
    assert_equal(total, 10)  # 0+1+2+3+4


# --- Closures with capture lists ---


def test_closure_with_capture_list() raises:
    # {mut count} captures by mutable reference.
    var count = 0

    def inner() {mut count}:
        count += 1

    inner()
    inner()
    inner()
    assert_equal(count, 3)


def main() raises:
    test_if_elif_else_chain()
    test_single_line_if()
    test_while_basic()
    test_while_break_continue()
    test_for_conventions()
    test_for_range()
    test_for_destructuring()
    test_for_else_no_break()
    test_for_else_with_break()
    test_while_else_no_break()
    test_try_except_basic()
    test_try_full_structure_no_error()
    test_try_full_structure_with_error()
    test_except_with_binding()
    test_except_without_binding()
    test_typed_raises()
    test_custom_context_manager()
    test_comptime_if()
    test_comptime_for()
    test_closure_with_capture_list()
