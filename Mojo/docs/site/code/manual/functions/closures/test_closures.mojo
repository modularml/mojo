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
# test_closures.mojo
# Tests for closures.mdx code examples.
# Skip: empty capture error example (deliberate compile error),
#        parallelize (threading + signature may vary).
from std.testing import assert_equal, assert_true


# --- Intro: scale ---


def test_scale() raises:
    """Intro example: imm capture with multiplier."""
    var multiplier = 3

    def scale(x: Int) {imm multiplier} -> Int:
        return x * multiplier

    assert_equal(scale(5), 15)


# --- The capture list: mixed imm + mut ---


def test_capture_list_imm_mut() raises:
    """Capture list example: imm x, mut y."""
    var x = 10
    var y = 20

    def inner(z: Int) {imm x, mut y} -> Int:
        y += z
        return x + y

    assert_equal(inner(5), 35)
    assert_equal(y, 25)


# --- Capture by immutable reference: imm ---


def test_imm_threshold() raises:
    """Explicit imm capture: is_over checks threshold."""
    var threshold = 100

    def is_over(x: Int) {imm threshold} -> Bool:
        return x > threshold

    assert_true(not is_over(50))
    assert_true(is_over(200))


def test_imm_sees_live_updates() raises:
    """`imm` captures a reference, so it sees later mutations."""
    var limit = 10

    def check(x: Int) {imm limit} -> Bool:
        return x < limit

    assert_true(check(5))
    limit = 3
    assert_true(not check(5))


def test_imm_implicit() raises:
    """Implicit `imm`: {imm} captures all used outer values."""
    var a = 1
    var b = 2

    def sum_ab() {imm} -> Int:
        return a + b

    assert_equal(sum_ab(), 3)


# --- Capture by mutable reference: mut ---


def test_mut_accumulate() raises:
    """Explicit mut capture: accumulate modifies outer total."""
    var total = 0

    def accumulate(x: Int) {mut total}:
        total += x

    accumulate(10)
    accumulate(20)
    assert_equal(total, 30)


def test_mut_implicit() raises:
    """Implicit `mut`: {mut} captures all used outer values mutably."""
    var count = 0
    var items = List[String]()

    def record(name: String) {mut}:
        items.append(name)
        count += 1

    record("alpha")
    record("beta")
    assert_equal(count, 2)
    assert_equal(len(items), 2)
    assert_equal(items[0], "alpha")
    assert_equal(items[1], "beta")


# --- Capture by copy: var ---


def test_var_snapshot() raises:
    """Explicit `var` capture: closure keeps value at definition time."""
    var snapshot_val = 42

    def frozen() {var snapshot_val} -> Int:
        return snapshot_val

    snapshot_val = 999
    assert_equal(frozen(), 42)
    _ = snapshot_val


def test_var_implicit() raises:
    """Implicit `var`: {var} copies all used outer values."""
    var x = 10
    var y = 20

    def snap() {var} -> Int:
        return x + y

    x = 0
    y = 0
    assert_equal(snap(), 30)
    _ = x
    _ = y


# --- Move capture: var name^ ---


def test_move_capture() raises:
    """Move capture: `var data^` transfers ownership into closure."""
    var data: List[Int] = [1, 2, 3]

    def take_data() {var data^} -> Int:
        return len(data)

    assert_equal(take_data(), 3)
    # data is consumed; can't be used after this point


# --- Copyable closures: var^ ---


def test_var_caret_copyable() raises:
    """Copyable closure: `{var^}` allows assigning closure to new var."""
    var label = "sensor-1"

    def tag() {var^} -> String:
        return label

    var also_tag = tag
    assert_equal(tag(), "sensor-1")
    assert_equal(also_tag(), "sensor-1")


# --- Caller-determined mutability: ref ---


def test_ref_capture() raises:
    """Ref capture: closure adapts to caller's mutability context."""

    def apply_to(ref items: List[Int]) -> Int:
        def first() {ref items} -> Int:
            return items[0]

        return first()

    var nums: List[Int] = [10, 20, 30]
    assert_equal(apply_to(nums), 10)


def test_ref_capture_mutability() raises:
    """Ref capture with comptime if: verifies mut vs immut propagation."""

    def check_mutability(ref items: List[Int]) -> String:
        def report() {ref items} -> String:
            comptime if origin_of(items).mut:
                return "mut"
            else:
                return "immut"

        return report()

    def from_read(imm xs: List[Int]) -> String:
        return check_mutability(xs)

    def from_mut(mut xs: List[Int]) -> String:
        return check_mutability(xs)

    var nums: List[Int] = [10, 20, 30]
    assert_equal(from_read(nums), "immut")
    assert_equal(from_mut(nums), "mut")


# --- Empty capture list: {} ---


def test_empty_capture_list() raises:
    """Empty capture: {} closure uses only its own arguments."""

    def doubled(x: Int) {} -> Int:
        return x * 2

    assert_equal(doubled(5), 10)


# NOTE: The empty capture error example ({} referencing outer var)
# is a deliberate compile error and cannot be tested at runtime.


# --- Mixing capture conventions ---


def test_mixed_conventions() raises:
    """Mixed capture list: `imm` + `mut` + `var` in one closure."""
    var config = "prod"
    var count = 0
    var label = "run-1"

    def process() {imm config, mut count, var label} -> String:
        count += 1
        return config + " " + String(count) + " " + label

    assert_equal(process(), "prod 1 run-1")
    label = "run-2"
    assert_equal(process(), "prod 2 run-1")
    _ = label


def test_bare_name_defaults_to_imm() raises:
    """Bare name in capture list defaults to `imm` convention."""
    var x = 10

    def show() {x} -> Int:
        return x

    assert_equal(show(), 10)


# --- Setting a default convention ---


def test_default_convention() raises:
    """Implicit default + explicit override in one capture list."""
    var a = 1
    var b = 2
    var z = "snapshot"

    def mixed() {mut, var z} -> String:
        a += 10
        b += 20
        return String(a) + " " + String(b) + " " + z

    assert_equal(mixed(), "11 22 snapshot")
    z = "changed"
    assert_equal(mixed(), "21 42 snapshot")
    _ = z


# --- Closures in practice: accumulating state ---


def test_accumulating_log() raises:
    """Practical example: mut capture builds a log across calls."""
    var log = List[String]()

    def record(event: String) {mut log}:
        log.append(event)

    record("started")
    record("processed item")
    record("finished")

    assert_equal(len(log), 3)
    assert_equal(log[0], "started")
    assert_equal(log[1], "processed item")
    assert_equal(log[2], "finished")


# --- Closures in practice: configurable behavior ---


def test_greet_all() raises:
    """Practical example: parametric function accepts closure."""
    var results = List[String]()

    def greet_all[G: def(String) -> None](names: List[String], greet: G):
        for n in names:
            greet(n)

    var names: List[String] = ["Alice", "Bob"]
    var greeting = "Hello"

    def greeter(name: String) {imm greeting, mut results}:
        results.append(greeting + ", " + name + "!")

    greet_all(names, greeter)
    assert_equal(len(results), 2)
    assert_equal(results[0], "Hello, Alice!")
    assert_equal(results[1], "Hello, Bob!")

    greeting = "Hi"
    greet_all(names, greeter)
    assert_equal(len(results), 4)
    assert_equal(results[2], "Hi, Alice!")
    assert_equal(results[3], "Hi, Bob!")


# NOTE: The parallelize example is skipped. It requires threading
# and the exact API signature may vary across stdlib versions.
# However, it works, and here it is for your manual testing:
#
# from max.algorithm import parallelize
#
# def main():
#    var results = List[Int](length=8, fill=0)
#
#    def work(i: Int) {mut results}:
#        results[i] = i * i
#
#    parallelize(work, 8)
#    print(results)  # [0, 1, 4, 9, 16, 25, 36, 49]


def main() raises:
    # Intro
    test_scale()

    # The capture list
    test_capture_list_imm_mut()

    # read
    test_imm_threshold()
    test_imm_sees_live_updates()
    test_imm_implicit()

    # mut
    test_mut_accumulate()
    test_mut_implicit()

    # var
    test_var_snapshot()
    test_var_implicit()

    # Move capture
    test_move_capture()

    # Copyable closures
    test_var_caret_copyable()

    # ref
    test_ref_capture()
    test_ref_capture_mutability()

    # Empty capture
    test_empty_capture_list()

    # Mixed conventions
    test_mixed_conventions()
    test_bare_name_defaults_to_imm()
    test_default_convention()

    # Practice examples
    test_accumulating_log()
    test_greet_all()
