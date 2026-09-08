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
# test_closure_declarations.mojo
# Tests for closure-declarations.mdx code examples.
#
# Skip (compile-error demonstrations and timing-only side effects):
#   - "Could not infer capture convention" diagnostic for `{}` and
#     omitted capture list referencing outer values.
#   - "'^' requires 'var' convention" for `{mut x^}`, `{imm x^}`,
#     `{ref x^}`.
#   - "default capture convention was already specified" for
#     `{mut, imm}`.
#   - "must be mutable for in-place operator destination" for
#     writing to an `imm`-captured name (bare-name default case).
#   - "register passible value can not be captured by 'mut'" for
#     `{mut name}` on an immutable register-passable arg.
#   - "cannot capture <name> by copy or move because it is not
#     register passable" for a non-register-passable `var` capture
#     inside a closure.
#   - "use of uninitialized value '<name>'" after `{var name^}`.
#   - `{var^}` copy-ctor timing assertion (the [copied] print at
#     closure-value assignment, not at declaration). Verified
#     empirically; cannot be asserted without redirecting stdout.
from std.testing import assert_equal, assert_true


# --- Intro example: imm capture ---


def test_imm_intro() raises:
    var multiplier = 3

    def scale(x: Int) {imm multiplier} -> Int:
        return x * multiplier

    assert_equal(scale(5), 15)


# --- imm: live updates ---


def test_imm_live_updates() raises:
    var limit = 10

    def check(x: Int) {imm limit} -> Bool:
        return x < limit

    assert_equal(check(5), True)
    limit = 3
    assert_equal(check(5), False)


# --- imm: implicit form ---


def test_imm_implicit() raises:
    var a = 1
    var b = 2

    def sum_ab() {imm} -> Int:
        return a + b

    assert_equal(sum_ab(), 3)


# --- bare name defaults to imm ---


def test_bare_name_default_imm() raises:
    var x = 10

    def show() {x} -> Int:
        return x

    assert_equal(show(), 10)


# --- mut: explicit form ---


def test_mut_explicit() raises:
    var total = 0

    def accumulate(value: Int) {mut total}:
        total += value

    accumulate(10)
    accumulate(20)
    assert_equal(total, 30)


# --- mut: implicit form ---


def test_mut_implicit() raises:
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


# --- var: explicit form, snapshot at declaration ---


def test_var_explicit_snapshot() raises:
    var snapshot = 42

    def frozen() {var snapshot} -> Int:
        return snapshot

    snapshot = 999
    assert_equal(frozen(), 42)
    assert_equal(snapshot, 999)  # outer survives


# --- var: implicit form ---


def test_var_implicit() raises:
    var x = 10
    var y = 20

    def snap() {var} -> Int:
        return x + y

    x = 0
    y = 0
    assert_equal(snap(), 30)
    _ = x
    _ = y


# --- var name^: move capture ---


def test_var_move_capture() raises:
    var data: List[Int] = [1, 2, 3]

    def take_data() {var data^} -> Int:
        return len(data)

    assert_equal(take_data(), 3)
    # data is consumed; referencing it here would be a compile error


# --- bare name^: move capture (equivalent to var name^) ---


def test_bare_move_capture() raises:
    var data: List[Int] = [1, 2, 3]

    def take_data() {data^} -> Int:
        return len(data)

    assert_equal(take_data(), 3)
    # data is consumed; referencing it here would be a compile error


# --- var^: closure value is Copyable ---


def test_var_caret_copyable_closure() raises:
    var label = "sensor-1"

    def tag() {var^} -> String:
        return label

    var clone = tag
    assert_equal(tag(), "sensor-1")
    assert_equal(clone(), "sensor-1")


# --- ref: parametric mutability ---
#
# Body refactored to return a Bool instead of printing, so the
# closure's view of origin mutability is testable.


def parametric_returns_mut(ref x: List[Int]) -> Bool:
    def borrow() {ref x} -> Bool:
        comptime if origin_of(x).mut:
            return True
        else:
            return False

    return borrow()


def call_with_immut(imm x: List[Int]) -> Bool:
    return parametric_returns_mut(x)


def call_with_mut(mut x: List[Int]) -> Bool:
    return parametric_returns_mut(x)


def test_ref_parametric_mutability() raises:
    var nums: List[Int] = [10, 20, 30]
    assert_equal(call_with_immut(nums), False)
    assert_equal(call_with_mut(nums), True)


# --- Empty capture list `{}` ---


def test_empty_capture_list() raises:
    def doubled(x: Int) {} -> Int:
        return x * 2

    assert_equal(doubled(5), 10)


# --- No capture list (equivalent to `{}`) ---


def test_no_capture_list() raises:
    def doubled(x: Int) -> Int:
        return x * 2

    assert_equal(doubled(6), 12)


# --- Mixing conventions: per-entry independence ---


def test_mixed_conventions() raises:
    var config: String = "prod"
    var count = 0
    var label: String = "run-1"

    def process() {imm config, mut count, var label} -> String:
        count += 1
        return config + ":" + String(count) + ":" + label

    assert_equal(process(), "prod:1:run-1")
    label = "run-2"  # outer mutation
    # closure's `label` is the def-time snapshot
    assert_equal(process(), "prod:2:run-1")
    assert_equal(count, 2)  # mut count flowed through
    _ = label


# --- Default convention with explicit override ---


def test_default_convention_override() raises:
    var a = 1
    var b = 2
    var z: String = "snapshot"

    def mixed() {mut, var z} -> String:
        a += 10  # 'a' uses default: mut
        b += 20  # 'b' uses default: mut
        return String(a) + "," + String(b) + "," + z

    assert_equal(mixed(), "11,22,snapshot")
    z = "changed"
    assert_equal(mixed(), "21,42,snapshot")
    assert_equal(a, 21)
    assert_equal(b, 42)
    _ = z


# --- Order independence of default-convention entry ---


def test_default_position_independent() raises:
    var x = 1
    var y = 100

    # `mut` as default, with explicit `var y` override
    def first() {mut, var y} -> Int:
        x += 1
        return x + y

    # Same list with positions reversed
    def second() {var y, mut} -> Int:
        x += 1
        return x + y

    assert_equal(first(), 2 + 100)  # x bumped to 2
    assert_equal(second(), 3 + 100)  # x bumped to 3
    assert_equal(x, 3)


# --- Parametric closure: simple type parameter ---


def test_parametric_closure_basic() raises:
    def double[T: Intable](x: T) {} -> Int:
        return Int(x) * 2

    assert_equal(double[Int](5), 10)
    assert_equal(double[Int](7), 14)


# --- Parametric closure: captures outer value ---


def test_parametric_closure_with_capture() raises:
    var factor = 4

    def scale_to[T: Intable](x: T) {imm factor} -> Int:
        return Int(x) * factor

    assert_equal(scale_to[Int](3), 12)
    factor = 10
    assert_equal(scale_to[Int](3), 30)


# --- Effects: raises on closure ---


def test_closure_raises() raises:
    var divisor = 2

    def divide(x: Int) raises {var divisor} -> Int:
        if divisor == 0:
            raise Error("divide by zero")
        return x // divisor

    assert_equal(divide(10), 5)


def test_closure_raises_actually_raises() raises:
    var divisor = 0

    def divide(x: Int) raises {var divisor} -> Int:
        if divisor == 0:
            raise Error("divide by zero")
        return x // divisor

    var caught = False
    try:
        _ = divide(10)
    except e:
        caught = True
    assert_true(caught)


# --- Device-passable closure behavior ---


def test_closure_device_passable() raises:
    var base = 10

    def shift(x: Int) {var base} -> Int:
        return x + base

    assert_equal(shift(3), 13)
    assert_equal(shift(7), 17)


# --- Nesting: closure inside a closure ---


def test_nested_closures() raises:
    var y = 4

    def outer_fn() {var y} -> Int:
        def inner_fn() {var y} -> Int:
            return y

        return inner_fn() + y

    assert_equal(outer_fn(), 8)


# --- Nesting: inner closure captures outer closure by name ---


def test_nested_capture_closure_value() raises:
    var n = 3

    def add(x: Int) {var n} -> Int:
        return x + n

    def twice(x: Int) {var add} -> Int:
        return add(add(x))

    assert_equal(twice(5), 11)  # 5 + 3 + 3
    assert_equal(twice(0), 6)  # 0 + 3 + 3


def main() raises:
    test_imm_intro()
    test_imm_live_updates()
    test_imm_implicit()
    test_bare_name_default_imm()
    test_mut_explicit()
    test_mut_implicit()
    test_var_explicit_snapshot()
    test_var_implicit()
    test_var_move_capture()
    test_bare_move_capture()
    test_var_caret_copyable_closure()
    test_ref_parametric_mutability()
    test_empty_capture_list()
    test_no_capture_list()
    test_mixed_conventions()
    test_default_convention_override()
    test_default_position_independent()
    test_parametric_closure_basic()
    test_parametric_closure_with_capture()
    test_closure_raises()
    test_closure_raises_actually_raises()
    test_closure_device_passable()
    test_nested_closures()
    test_nested_capture_closure_value()
