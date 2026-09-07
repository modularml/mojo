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
"""Tests for `ReflectedFn` and the `reflect_fn[func]` comptime alias."""

from std.reflection import ReflectedFn, reflect_fn
from std.testing import TestSuite, assert_equal, assert_true


# ===----------------------------------------------------------------------=== #
# Helpers
# ===----------------------------------------------------------------------=== #


def _free_func(x: Int) -> Int:
    return x + 1


def _no_arg_func():
    pass


# ===----------------------------------------------------------------------=== #
# `ReflectedFn` — direct API
# ===----------------------------------------------------------------------=== #


def test_reflect_fn_alias_resolves_to_handle_type() raises:
    """`reflect_fn[func]` is a comptime alias for the `ReflectedFn[func]`
    type, so the two spellings are interchangeable at type-position."""
    var direct = ReflectedFn[_free_func].display_name()
    var via_alias = reflect_fn[_free_func].display_name()
    assert_equal(direct, via_alias)


def test_reflected_fn_display_name() raises:
    assert_equal(reflect_fn[_free_func].display_name(), "_free_func")
    assert_equal(reflect_fn[_no_arg_func].display_name(), "_no_arg_func")


def test_reflected_fn_linkage_name_nonempty() raises:
    """Linkage name is target-mangled; we can't assert the exact value, but
    it must be non-empty and differ from the source name."""
    var linkage = reflect_fn[_free_func].linkage_name()
    assert_true(linkage.byte_length() > 0)
    assert_true(linkage != "_free_func")


def test_reflected_fn_linkage_name_distinct_per_function() raises:
    """Distinct functions must produce distinct linkage names."""
    var lk_free = reflect_fn[_free_func].linkage_name()
    var lk_noarg = reflect_fn[_no_arg_func].linkage_name()
    assert_true(lk_free != lk_noarg)


def _reflect_in_generic[
    func_type: AnyType, //, func: func_type
]() -> StaticString:
    """Generic helper that resolves `reflect_fn[func]` from a forwarded
    value parameter — exercises the elaboration path the type-side has via
    `Reflected[T]` in a generic `T`."""
    return reflect_fn[func].display_name()


def test_reflect_fn_through_generic() raises:
    assert_equal(_reflect_in_generic[_free_func](), "_free_func")
    assert_equal(_reflect_in_generic[_no_arg_func](), "_no_arg_func")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
