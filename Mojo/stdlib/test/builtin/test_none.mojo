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

from std.testing import assert_equal, TestSuite
from test_utils import check_write_to


def test_write_to() raises:
    check_write_to(NoneType(), expected="None", is_repr=False)
    check_write_to(NoneType(), expected="None", is_repr=True)


struct FromNone:
    var value: Int

    @implicit
    def __init__(out self, none: NoneType):
        self.value = -1

    # FIXME: None literal should be of NoneType not !kgen.none.
    @always_inline
    @implicit
    def __init__(out self, none: __mlir_type.`!kgen.none`):
        self = NoneType()

    @implicit
    def __init__(out self, value: Int):
        self.value = value


def test_type_from_none() raises:
    _ = FromNone(5)

    _ = FromNone(None)

    # -------------------------------------
    # Test implicit conversion from `None`
    # -------------------------------------

    def foo(arg: FromNone):
        pass

    # FIXME:
    #   This currently fails, because it requires 2 "hops" of conversion:
    #       1. !kgen.none => NoneType
    #       2. NoneType => FromNone
    # foo(None)
    #
    #   But, interestingly, this does not fail?
    var _obj2: FromNone = None


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
