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

from tests.util import assert_format, assert_mojo_format

import mblack


def test_mojo_unindents_cases_from_python_style():
    source = """\
def f(x):
    match x:
        case 1:
            return "one"
        case _:
            return "other"
"""
    expected = """\
def f(x):
    match x:
    case 1:
        return "one"
    case _:
        return "other"
"""
    assert_mojo_format(source, expected)


def test_mojo_preserves_unindented_cases():
    source = """\
def f(x):
    match x:
    case 1:
        return "one"
    case _:
        return "other"
"""
    expected = """\
def f(x):
    match x:
    case 1:
        return "one"
    case _:
        return "other"
"""
    assert_mojo_format(source, expected)


def test_mojo_formats_dunder_match():
    source = """\
def f(x):
    __match x:
        case 1:
            return "one"
        case 2 if x > 0:
            return "two"
        case _:
            return "other"
"""
    expected = """\
def f(x):
    __match x:
    case 1:
        return "one"
    case 2 if x > 0:
        return "two"
    case _:
        return "other"
"""
    assert_mojo_format(source, expected)


def test_python_mode_keeps_case_indent():
    source = """\
def f(x):
    match x:
        case 1:
            return "one"
        case _:
            return "other"
"""
    expected = """\
def f(x):
    match x:
        case 1:
            return "one"
        case _:
            return "other"
"""
    assert_format(
        source,
        expected,
        mblack.Mode(target_versions={mblack.TargetVersion.PY310}),
        minimum_version=(3, 10),
    )
