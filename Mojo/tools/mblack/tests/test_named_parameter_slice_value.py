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

import pytest

from tests.util import assert_mojo_format

def test_named_subscript_parameter_slice_value():
    """Test using a slice value on a named subscript parameter."""
    source = (
        "@fieldwise_init\n"
        "struct Foo:\n"
        "    def __getitem__(self, par: Slice): pass\n"
        "def main():\n"
        "    var x = Foo()\n"
        "    x[par=0:1]\n"
    )
    expected = (
        "@fieldwise_init\n"
        "struct Foo:\n"
        "    def __getitem__(self, par: Slice):\n"
        "        pass\n"
        "\n"
        "\n"
        "def main():\n"
        "    var x = Foo()\n"
        "    x[par=0:1]\n"
    )
    assert_mojo_format(source, expected)


def test_named_subscript_parameter_slice_variants():
    """Test using different slice forms in named subscript parameters."""
    source = (
        "@fieldwise_init\n"
        "struct Foo:\n"
        "    def __getitem__(self, par: Slice): pass\n"
        "def main():\n"
        "    var x = Foo()\n"
        "    x[par=:]\n"
        "    x[par=1:]\n"
        "    x[par=:5]\n"
        "    x[par=1:5]\n"
        "    x[par=::]\n"
        "    x[par=::2]\n"
        "    x[par=1::2]\n"
        "    x[par=:5:2]\n"
        "    x[par=1:5:2]\n"
        "    x[par=::-1]\n"
    )
    expected = (
        "@fieldwise_init\n"
        "struct Foo:\n"
        "    def __getitem__(self, par: Slice):\n"
        "        pass\n"
        "\n"
        "\n"
        "def main():\n"
        "    var x = Foo()\n"
        "    x[par=:]\n"
        "    x[par=1:]\n"
        "    x[par=:5]\n"
        "    x[par=1:5]\n"
        "    x[par=::]\n"
        "    x[par=::2]\n"
        "    x[par=1::2]\n"
        "    x[par=:5:2]\n"
        "    x[par=1:5:2]\n"
        "    x[par=::-1]\n"
    )
    assert_mojo_format(source, expected)

def test_named_subscript_parameter_slice_variables():
    """Test using variables in slices in named subscript parameters."""
    source = (
        "@fieldwise_init\n"
        "struct Foo:\n"
        "    def __getitem__(self, par: Slice): pass\n"
        "def main():\n"
        "    var x = Foo()\n"
        "    var a = 1\n"
        "    x[par=a:]\n"
        "    x[par=:a*2]\n"
        "    x[par=a:a*2:a]\n"
    )
    expected = (
        "@fieldwise_init\n"
        "struct Foo:\n"
        "    def __getitem__(self, par: Slice):\n"
        "        pass\n"
        "\n"
        "\n"
        "def main():\n"
        "    var x = Foo()\n"
        "    var a = 1\n"
        "    x[par=a:]\n"
        "    x[par = : a * 2]\n"
        "    x[par = a : a * 2 : a]\n"
    )
    assert_mojo_format(source, expected)


def test_named_subscript_parameter_multiple_slices():
    """Test multiple subscripts with slices."""
    source = (
        "@fieldwise_init\n"
        "struct Foo:\n"
        "    def __getitem__(self, par1: Slice, par2: Slice): pass\n"
        "def main():\n"
        "    var x = Foo()\n"
        "    x[par1=0:1, par2=1:5]\n"
        "    x[0:1, par2=1:5]\n"
        "    x[:, par2=1:5]\n"
    )
    expected = (
        "@fieldwise_init\n"
        "struct Foo:\n"
        "    def __getitem__(self, par1: Slice, par2: Slice):\n"
        "        pass\n"
        "\n"
        "\n"
        "def main():\n"
        "    var x = Foo()\n"
        "    x[par1=0:1, par2=1:5]\n"
        "    x[0:1, par2=1:5]\n"
        "    x[:, par2=1:5]\n"
    )
    assert_mojo_format(source, expected)
