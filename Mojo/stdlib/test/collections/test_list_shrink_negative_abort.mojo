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
#
# Regression test: `List.shrink` with a negative `new_length` used to pass the
# over-length guard, then run one destructor too many (destroying an element
# twice) and store a negative value in `len()`. It must now abort instead of
# corrupting the list.
#
# ===----------------------------------------------------------------------=== #

from std.testing import _assert_aborts, TestSuite


def test_list_aborts_on_negative_shrink_size() raises:
    var l: List = [1, 2, 3]

    def shrink() raises {mut l} -> None:
        l.shrink(-1)

    _assert_aborts(
        shrink, contains="You are calling List.shrink with a negative"
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
