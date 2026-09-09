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

"""End-to-end smoke test for the Mojo `Range` context manager (MXTOOLS-190).

Exercises the high-level struct rather than the raw `KGEN_CompilerRT_Range*`
FFI symbols (those have separate coverage in `test_range_bridge.mojo`).
"""

from std.ffi import external_call
from std.testing import assert_false, assert_true

from profiling_range import Range, is_enabled


def test_disabled_at_startup() raises:
    """The process-wide profiler starts disabled; `is_enabled()` reflects it."""
    assert_false(is_enabled())


def _disabled_scope() raises:
    """Helper: open and close a Range inside a dedicated scope."""
    # The `with` block bounds the Range's lifetime: __enter__ opens the span
    # (a no-op while disabled) and __exit__ closes it at block end.
    with Range("noop-span"):
        pass


def test_range_construction_when_disabled_is_noop() raises:
    """A Range entered while disabled records no span and exits cleanly."""
    assert_false(is_enabled())
    _disabled_scope()
    assert_false(is_enabled())


def _enabled_scope() raises:
    """Helper: open and close a Range that emits begin/end calls.

    Uses a non-zero color to exercise the `color` parameter making it
    through the FFI (in addition to the default-color path already
    covered by `_disabled_scope`).
    """
    with Range("enabled-span", UInt32(0x00FF00)):
        pass
    # Empty name while enabled exercises the RangeBegin null-pointer contract
    # (namePtr must be non-null even when nameLen == 0; see RangeBridge.cpp).
    with Range(""):
        pass


def test_range_when_enabled_does_not_disturb_profiler_state() raises:
    """Enabling, opening a Range, and disabling all behave together.

    True begin/end pairing assertions need a debug-only span-depth probe
    exposed by `RangeBridge.cpp` (tracked in MXTOOLS-190). Today the
    underlying rangeBegin / rangeEnd are stubs, so this test verifies
    only that opening and closing a Range while enabled does not
    change the enable state itself.
    """
    # TODO(MXTOOLS-190): assert begin/end pairing once the debug-only
    # span-depth probe lands; today only the enable-state invariant is checked.
    external_call["KGEN_CompilerRT_RangeEnable", NoneType]()
    assert_true(is_enabled())
    _enabled_scope()
    # The profiler is still enabled afterwards — Range only opens/closes a
    # libkineto span, it does not disable the profiler itself.
    assert_true(is_enabled())
    external_call["KGEN_CompilerRT_RangeDisable", NoneType]()
    assert_false(is_enabled())


def _nested_scope() raises:
    """Helper: nested Range scopes for the disabled case."""
    with Range("outer"):
        with Range("inner"):
            pass


def test_nested_ranges_disabled() raises:
    """Nested disabled Range scopes are safe."""
    assert_false(is_enabled())
    _nested_scope()
    assert_false(is_enabled())


def _raise_inside_enabled_range() raises:
    """Helper: a Range whose guarded block raises while enabled."""
    with Range("raising-span"):
        raise Error("boom")


def test_range_exits_when_body_raises() raises:
    """A raise inside the `with` block still runs `__exit__` (RangeEnd).

    Mojo calls the basic `__exit__(self)` form before propagating the error,
    so an exception in the guarded block cannot leave a libkineto span open.
    Begin/end pairing can't be asserted yet (stubs), so this verifies the
    error unwinds cleanly and the profiler state is intact afterward.
    """
    external_call["KGEN_CompilerRT_RangeEnable", NoneType]()
    assert_true(is_enabled())
    var raised = False
    try:
        _raise_inside_enabled_range()
    except:
        raised = True
    assert_true(raised)
    assert_true(is_enabled())
    external_call["KGEN_CompilerRT_RangeDisable", NoneType]()
    assert_false(is_enabled())


def main() raises:
    test_disabled_at_startup()
    test_range_construction_when_disabled_is_noop()
    test_range_when_enabled_does_not_disturb_profiler_state()
    test_nested_ranges_disabled()
    test_range_exits_when_body_raises()
