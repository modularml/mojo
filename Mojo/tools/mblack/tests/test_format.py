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
#   Path:   tests/test_format.py
#
# ===----------------------------------------------------------------------=== #

from dataclasses import replace
from typing import Any, Iterator
from unittest.mock import patch

import pytest

import mblack
from tests.util import (
    DEFAULT_MODE,
    PY36_VERSIONS,
    all_data_cases,
    assert_format,
    dump_to_stderr,
    mojo_format_str,
    read_data,
)


@pytest.fixture(autouse=True)
def patch_dump_to_file(request: Any) -> Iterator[None]:
    with patch("mblack.dump_to_file", dump_to_stderr):
        yield


def check_file(
    subdir: str, filename: str, mode: mblack.Mode, *, data: bool = True
) -> None:
    source, expected = read_data(subdir, filename, data=data)
    assert_format(source, expected, mode, fast=False)


@pytest.mark.filterwarnings("ignore:invalid escape sequence.*:DeprecationWarning")
@pytest.mark.parametrize("filename", all_data_cases("simple_cases"))
def test_simple_format(filename: str) -> None:
    check_file("simple_cases", filename, DEFAULT_MODE)


@pytest.mark.parametrize("filename", all_data_cases("preview"))
def test_preview_format(filename: str) -> None:
    magic_trailing_comma = filename != "skip_magic_trailing_comma"
    check_file(
        "preview",
        filename,
        mblack.Mode(preview=True, magic_trailing_comma=magic_trailing_comma),
    )


@pytest.mark.parametrize("filename", all_data_cases("preview_39"))
def test_preview_minimum_python_39_format(filename: str) -> None:
    source, expected = read_data("preview_39", filename)
    mode = mblack.Mode(preview=True)
    assert_format(source, expected, mode, minimum_version=(3, 9))


@pytest.mark.parametrize("filename", all_data_cases("preview_310"))
def test_preview_minimum_python_310_format(filename: str) -> None:
    source, expected = read_data("preview_310", filename)
    mode = mblack.Mode(preview=True)
    assert_format(source, expected, mode, minimum_version=(3, 10))


# =============== #
# Complex cases
# ============= #


def test_empty() -> None:
    source = expected = ""
    assert_format(source, expected)


@pytest.mark.parametrize("filename", all_data_cases("py_36"))
def test_python_36(filename: str) -> None:
    source, expected = read_data("py_36", filename)
    mode = mblack.Mode(target_versions=PY36_VERSIONS)
    assert_format(source, expected, mode, minimum_version=(3, 6))


@pytest.mark.parametrize("filename", all_data_cases("py_37"))
def test_python_37(filename: str) -> None:
    source, expected = read_data("py_37", filename)
    mode = mblack.Mode(target_versions={mblack.TargetVersion.PY37})
    assert_format(source, expected, mode, minimum_version=(3, 7))


@pytest.mark.parametrize("filename", all_data_cases("py_38"))
def test_python_38(filename: str) -> None:
    source, expected = read_data("py_38", filename)
    mode = mblack.Mode(target_versions={mblack.TargetVersion.PY38})
    assert_format(source, expected, mode, minimum_version=(3, 8))


@pytest.mark.parametrize("filename", all_data_cases("py_39"))
def test_python_39(filename: str) -> None:
    source, expected = read_data("py_39", filename)
    mode = mblack.Mode(target_versions={mblack.TargetVersion.PY39})
    assert_format(source, expected, mode, minimum_version=(3, 9))


@pytest.mark.parametrize("filename", all_data_cases("py_310"))
def test_python_310(filename: str) -> None:
    source, expected = read_data("py_310", filename)
    mode = mblack.Mode(target_versions={mblack.TargetVersion.PY310})
    assert_format(source, expected, mode, minimum_version=(3, 10))


@pytest.mark.parametrize("filename", all_data_cases("py_310"))
def test_python_310_without_target_version(filename: str) -> None:
    source, expected = read_data("py_310", filename)
    mode = mblack.Mode()
    assert_format(source, expected, mode, minimum_version=(3, 10))


def test_patma_invalid() -> None:
    source, expected = read_data("miscellaneous", "pattern_matching_invalid")
    mode = mblack.Mode(target_versions={mblack.TargetVersion.PY310})
    with pytest.raises(mblack.parsing.InvalidInput) as exc_info:
        assert_format(source, expected, mode, minimum_version=(3, 10))

    exc_info.match("Cannot parse: 10:11")


@pytest.mark.parametrize("filename", all_data_cases("py_311"))
def test_python_311(filename: str) -> None:
    source, expected = read_data("py_311", filename)
    mode = mblack.Mode(target_versions={mblack.TargetVersion.PY311})
    assert_format(source, expected, mode, minimum_version=(3, 11))


@pytest.mark.parametrize("filename", all_data_cases("fast"))
def test_fast_cases(filename: str) -> None:
    source, expected = read_data("fast", filename)
    assert_format(source, expected, fast=True)


def test_python_2_hint() -> None:
    with pytest.raises(mblack.parsing.InvalidInput) as exc_info:
        assert_format("print 'daylily'", "print 'daylily'")
    exc_info.match(mblack.parsing.PY2_HINT)


def test_unexpected_indent_error_message() -> None:
    source = "def main():\n  x = 0\n   if x > 0:\n    x = 2\n"
    with pytest.raises(mblack.parsing.InvalidInput, match="Unexpected indent"):
        mojo_format_str(source)
    with pytest.raises(
        mblack.parsing.InvalidInput,
        match="previous line has 2 spaces but this line has 3 spaces",
    ):
        mojo_format_str(source)


@pytest.mark.filterwarnings("ignore:invalid escape sequence.*:DeprecationWarning")
def test_docstring_no_string_normalization() -> None:
    """Like test_docstring but with string normalization off."""
    source, expected = read_data("miscellaneous", "docstring_no_string_normalization")
    mode = replace(DEFAULT_MODE, string_normalization=False)
    assert_format(source, expected, mode)


def test_preview_docstring_no_string_normalization() -> None:
    """
    Like test_docstring but with string normalization off *and* the preview style
    enabled.
    """
    source, expected = read_data(
        "miscellaneous", "docstring_preview_no_string_normalization"
    )
    mode = replace(DEFAULT_MODE, string_normalization=False, preview=True)
    assert_format(source, expected, mode)


def test_long_strings_flag_disabled() -> None:
    """Tests for turning off the string processing logic."""
    source, expected = read_data("miscellaneous", "long_strings_flag_disabled")
    mode = replace(DEFAULT_MODE, experimental_string_processing=False)
    assert_format(source, expected, mode)


@pytest.mark.skip("stubs are not relevant to mojo")
def test_stub() -> None:
    mode = replace(DEFAULT_MODE, is_pyi=True)
    source, expected = read_data("miscellaneous", "stub.pyi")
    assert_format(source, expected, mode)


def test_power_op_newline() -> None:
    # requires line_length=0
    source, expected = read_data("miscellaneous", "power_op_newline")
    assert_format(source, expected, mode=mblack.Mode(line_length=0))
