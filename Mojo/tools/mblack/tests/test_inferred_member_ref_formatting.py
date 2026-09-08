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

from tests.util import assert_mojo_format


def test_inferred_member_in_call():
    source = "def f():\n    takes_color(.red)\n"
    expected = "def f():\n    takes_color(.red)\n"
    assert_mojo_format(source, expected)


def test_inferred_dtype_in_simd_subscript():
    source = "def f():\n    var x = SIMD[.int32, 4](1, 2, 3, 4)\n"
    expected = "def f():\n    var x = SIMD[.int32, 4](1, 2, 3, 4)\n"
    assert_mojo_format(source, expected)


def test_inferred_member_annotation():
    source = "def f():\n    var x: Color = .blue\n"
    expected = "def f():\n    var x: Color = .blue\n"
    assert_mojo_format(source, expected)


def test_inferred_member_chained():
    source = "def f():\n    takes_color(.red.opacity(0.5))\n"
    expected = "def f():\n    takes_color(.red.opacity(0.5))\n"
    assert_mojo_format(source, expected)


def test_inferred_member_list_literal():
    source = "def f():\n    takes_colors([.red, .green, .blue])\n"
    expected = "def f():\n    takes_colors([.red, .green, .blue])\n"
    assert_mojo_format(source, expected)
