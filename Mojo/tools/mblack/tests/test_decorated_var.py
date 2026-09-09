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


def test_decorated_var():
    source = (
        "struct Foo:\n"
        "    @doc_hidden\n"
        "    var  x:  Int\n"
    )
    expected = (
        "struct Foo:\n"
        "    @doc_hidden\n"
        "    var x: Int\n"
    )
    assert_mojo_format(source, expected)


def test_decorated_var_multiple_fields():
    source = (
        "struct Foo:\n"
        "    @doc_hidden\n"
        "    var x: Int\n"
        "    @doc_hidden\n"
        "    var y: String\n"
    )
    expected = (
        "struct Foo:\n"
        "    @doc_hidden\n"
        "    var x: Int\n"
        "\n"
        "    @doc_hidden\n"
        "    var y: String\n"
    )
    assert_mojo_format(source, expected)
